#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Module
~~~~~~~~~~~~
Defines the abstract agent interface and concrete agent implementations.

Author: AI Assistant
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

from MurmurNet.modules.data_structures import AgentOutput 

if TYPE_CHECKING:
    from MurmurNet.modules.config_manager import ConfigManager 
    from MurmurNet.modules.blackboard import Blackboard
    # Import specific content types if needed for direct use by agent, though usually it gets data via context
    from MurmurNet.modules.blackboard import (
        CONTENT_TYPE_USER_INPUT, CONTENT_TYPE_RAG_RESULT, 
        CONTENT_TYPE_CONVERSATION_CONTEXT, CONTENT_TYPE_SUMMARY
    )

# Placeholder for a generic LLM client type, can be refined later
LLMClientType = Any 

class AbstractAgent(ABC):
    """
    Abstract base class for an agent in the MurmurNet system.
    """
    def __init__(self, 
                 agent_id: str, 
                 role_config: Dict[str, Any], 
                 config_manager: 'ConfigManager', 
                 blackboard: 'Blackboard', 
                 llm_client: LLMClientType):
        self.agent_id = agent_id
        self.role_config = role_config 
        self.config_manager = config_manager 
        self.blackboard = blackboard 
        self.llm_client = llm_client
        # It's good practice for agents to have their own logger instance.
        # Assuming ConfigManager can provide a logger.
        self.logger = config_manager.get_logger(f"Agent.{self.agent_id}")

    def get_id(self) -> str:
        """Returns the agent's unique ID."""
        return self.agent_id

    def get_role_config(self) -> Dict[str, Any]:
        """Returns the agent's current role configuration."""
        return self.role_config

    def update_role_config(self, new_role_config: Dict[str, Any]):
        """Updates the agent's role configuration."""
        self.role_config = new_role_config
        self.logger.info(f"Role config updated for agent {self.agent_id} to: {new_role_config.get('name', 'UnknownRole')}")

    @abstractmethod
    def generate_response(self, current_iteration: int, other_agent_outputs_str: Optional[str] = None) -> 'AgentOutput':
        """
        Generates a response based on the provided context.

        Args:
            current_iteration: The current iteration number of the system.
            other_agent_outputs_str: A string representing formatted outputs from other agents,
                                     if available and required by the prompt template.
        Returns:
            An AgentOutput object containing the agent's response.
        """
        pass


class StandardAgent(AbstractAgent):
    """
    A standard implementation of an agent that interacts with an LLM.
    """
    def __init__(self, 
                 agent_id: str, 
                 role_config: Dict[str, Any], 
                 config_manager: 'ConfigManager', 
                 blackboard: 'Blackboard', 
                 llm_client: LLMClientType):
        super().__init__(agent_id, role_config, config_manager, blackboard, llm_client)
        # Set specific LLM parameters from role_config or defaults from global config
        self.temperature = self.role_config.get('temperature', self.config_manager.model.temperature)
        self.max_tokens = self.role_config.get('max_tokens', self.config_manager.model.max_tokens)
        # Potentially other parameters like 'top_p', 'stop_sequences' could be managed here.

    def _format_prompt_for_agent(self, current_iteration: int, other_agent_outputs_str: Optional[str] = None) -> str:
        """
        Internal helper to format the prompt for this agent.
        """
        # Import here to avoid circular if blackboard imports agent (though unlikely for constants)
        from MurmurNet.modules.blackboard import (
            CONTENT_TYPE_USER_INPUT, CONTENT_TYPE_RAG_RESULT, 
            CONTENT_TYPE_CONVERSATION_CONTEXT, CONTENT_TYPE_SUMMARY
        )

        # 1. Get User Query for the current iteration
        user_query = "ユーザーからの具体的な質問はありませんが、一般的な情報提供をお願いします。" # Default
        user_input_entry = self.blackboard.get_user_input(iteration=current_iteration)
        if user_input_entry and user_input_entry.data:
            if isinstance(user_input_entry.data, dict):
                user_query = user_input_entry.data.get('normalized', str(user_input_entry.data))
            else:
                user_query = str(user_input_entry.data)
        elif self.logger.isEnabledFor(logging.DEBUG):
             self.logger.debug(f"No user input found for iter {current_iteration} for agent {self.agent_id}.")
        
        # 2. Get RAG Info for the current iteration
        rag_info = "関連する追加情報はありません。" # Default
        rag_entries = self.blackboard.get_entries_by_content_type(
            CONTENT_TYPE_RAG_RESULT, iteration=current_iteration, latest_first=True
        )
        if rag_entries and rag_entries[0].data is not None:
            rag_info = str(rag_entries[0].data)

        # 3. Get Conversation Context (latest available)
        conversation_context_text = "過去の会話の記録はありません。" # Default
        context_entries = self.blackboard.get_entries_by_content_type(
            CONTENT_TYPE_CONVERSATION_CONTEXT, latest_first=True
        )
        if context_entries and context_entries[0].data is not None:
            conversation_context_text = str(context_entries[0].data)

        # 4. Get Previous Iteration's Summary (if applicable)
        previous_summary_text = "前回の議論の要約はありません。" # Default
        if current_iteration > 0:
            summary_entries = self.blackboard.get_entries_by_content_type(
                CONTENT_TYPE_SUMMARY, iteration=current_iteration - 1, latest_first=True
            )
            if summary_entries and summary_entries[0].data is not None:
                summary_data = summary_entries[0].data
                previous_summary_text = summary_data.get('text', str(summary_data)) if isinstance(summary_data, dict) else str(summary_data)
        
        # 5. Other Agent Outputs (passed as argument or default)
        other_outputs_final_str = other_agent_outputs_str if other_agent_outputs_str is not None else "他のエージェントからの意見はまだありません。"

        # 6. Get agent's own role & system prompt template
        role_cfg = self.get_role_config()
        system_prompt_template = role_cfg.get("system_prompt_template", 
                                              "あなたはAIアシスタントです。UserQuery: {user_query} RelevantInfo: {rag_info} OtherAgentOutputs: {other_outputs}")
        
        # 7. Format the prompt
        # Basic formatting, ensure values are strings
        # More robust templating (e.g. Jinja2) could be used if prompts become very complex.
        prompt_elements = {
            "user_query": str(user_query),
            "rag_info": str(rag_info),
            "other_outputs": str(other_outputs_final_str),
            "conversation_context": str(conversation_context_text),
            "previous_summary": str(previous_summary_text),
            "agent_role_name": str(role_cfg.get("name", self.agent_id)),
            # Add any other placeholders the template might expect
        }
        
        formatted_main_prompt = system_prompt_template
        for key, value in prompt_elements.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_main_prompt:
                 formatted_main_prompt = formatted_main_prompt.replace(placeholder, value)
            # else:
            #     self.logger.debug(f"Placeholder {placeholder} not found in template for agent {self.agent_id}. Template: {system_prompt_template}")


        # Prepend context and summary if they are not already part of the main template implicitly
        # This depends on how `system_prompt_template` is structured.
        # If templates are designed to include these, this might be redundant.
        # For now, assuming templates might be simpler like in original AgentPoolManager.
        
        # This logic of prepending context/summary might need to be adjusted based on actual template content.
        # If templates are like: "Context: {conversation_context} Summary: {previous_summary} Main: {main_part_of_template}"
        # then this explicit prepending is not needed.
        # The original _format_prompt had a fixed structure. Here, it's more template-driven.
        
        # Let's assume the template handles these, or they are part of `prompt_elements`
        # final_prompt_str = f"これまでの会話の要約:\n{conversation_context_text}\n\n---\n\n前回の議論の要約:\n{previous_summary_text}\n\n---\n\n{formatted_main_prompt}"
        # For now, let the template decide if it uses {conversation_context} and {previous_summary}
        
        final_instructions = role_cfg.get("final_instructions", "\n\n上記を踏まえ、あなたの役割として最高の応答を生成してください:")
        full_prompt = formatted_main_prompt + final_instructions
        
        self.logger.debug(f"Formatted prompt for agent {self.agent_id} (iteration {current_iteration}) - Snippet:\n{full_prompt[:300]}...")
        return full_prompt


    def generate_response(self, current_iteration: int, other_agent_outputs_str: Optional[str] = None) -> AgentOutput:
        """
        Generates a response by formatting the prompt and querying the LLM.
        """
        # formatted_prompt is now generated internally
        formatted_prompt = self._format_prompt_for_agent(current_iteration, other_agent_outputs_str)

        self.logger.debug(f"Generating response for iteration {current_iteration} with role: {self.role_config.get('name', self.agent_id)}")
        # Formatted prompt is already logged by _format_prompt_for_agent

        try:
            raw_llm_response = ""
            if hasattr(self.llm_client, 'generate'):
                 raw_llm_response = self.llm_client.generate(
                    prompt=formatted_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif hasattr(self.llm_client, 'create_chat_completion'):
                response_obj = self.llm_client.create_chat_completion(
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                if hasattr(response_obj, 'choices') and response_obj.choices:
                    raw_llm_response = response_obj.choices[0].message.content
                else: 
                    raw_llm_response = str(response_obj)
            else:
                self.logger.error(f"LLM client for agent {self.agent_id} has no recognized generation method.")
                raise NotImplementedError(f"LLM client for {self.agent_id} not compatible.")

            processed_text = raw_llm_response.strip() if raw_llm_response else ""

            if not processed_text:
                self.logger.warning(f"Agent {self.agent_id} LLM generated an empty or whitespace-only response.")
                processed_text = f"エージェント{self.agent_id}は空の応答を生成しました。"

        except Exception as e:
            self.logger.error(f"Error during LLM call for agent {self.agent_id} (role: {self.role_config.get('name', 'Unknown')}): {e}", exc_info=True)
            processed_text = f"エージェント{self.agent_id}（役割: {self.role_config.get('name', '不明')}）はエラーにより応答できませんでした: 技術的問題が発生しました。"

        return AgentOutput(
            agent_id=self.agent_id,
            text=processed_text,
            role=self.role_config.get('name', self.agent_id), 
            timestamp=time.time(),
            iteration=current_iteration
        )

# Example usage (for testing or understanding)
if __name__ == '__main__':
    import logging # Make sure logging is imported for the example
    print("Agent module defined. Basic test execution (requires mock objects or actual instances for full test).")

    # Define Mock classes for dependencies
    class MockLLMClient:
        def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
            self.logger.info(f"MockLLMClient.generate called with prompt: {prompt[:60]}...")
            return f"Mock LLM response to: {prompt[:50]}"
        
        def create_chat_completion(self, messages: list, temperature: float, max_tokens: int):
            self.logger.info(f"MockLLMClient.create_chat_completion called with messages: {messages}")
            class Choice:
                def __init__(self, content):
                    self.message = type('Message', (), {'content': content})()
            return type('Completion', (), {'choices': [Choice(f"Mock chat response to: {messages[0]['content'][:50]}")]})()


    class MockConfigManager:
        def __init__(self):
            self.model = type('ModelConfig', (), {'temperature': 0.7, 'max_tokens': 150, 'some_other_param': 'value'})()
            # Basic logger setup for the mock
            self.logger_instance = logging.getLogger("MockConfigManager")
            if not self.logger_instance.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger_instance.addHandler(handler)
                self.logger_instance.setLevel(logging.DEBUG)
        
        def get_logger(self, name: str): # Method to provide named loggers
            logger = logging.getLogger(name)
            if not logger.handlers: # Ensure handlers are not added multiple times
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.DEBUG) # Or get level from a mock logging config
            return logger

    class MockBlackboard: # Minimal mock, can be expanded if agents use it more
        pass

    # Instantiate mocks
    mock_config = MockConfigManager()
    mock_bb = MockBlackboard()
    mock_llm_gen = MockLLMClient() # For generate interface
    mock_llm_chat = MockLLMClient() # For create_chat_completion interface

    # Agent role configuration
    agent_role_details = {
        "name": "TestRole",
        "system_prompt_template": "You are a {role_name}. User asks: {user_query}", # Example template
        "temperature": 0.6,
        "max_tokens": 120
    }

    # Create StandardAgent instance with generate-style LLM
    agent1 = StandardAgent("agent001", agent_role_details, mock_config, mock_bb, mock_llm_gen)
    print(f"\nAgent 1 ID: {agent1.get_id()}, Role: {agent1.get_role_config()['name']}")
    
    # Test generate_response for agent1 - now it formats its own prompt
    # To test _format_prompt_for_agent properly, mock_bb needs methods like get_user_input etc.
    # For this illustrative test, we'll see the placeholder/default values used.
    class MockBlackboardWithGetters(MockBlackboard): # Extend mock blackboard
        def get_user_input(self, iteration: int): return None
        def get_entries_by_content_type(self, content_type: str, iteration: Optional[int] = None, latest_first: bool = False): return []
        def get_agent_outputs(self, iteration: int, agent_id: str): return []

    mock_bb_with_getters = MockBlackboardWithGetters()
    agent1.blackboard = mock_bb_with_getters # Update agent's blackboard instance
    
    output1 = agent1.generate_response(current_iteration=1, other_agent_outputs_str="他のエージェントからの意見はありません。")
    print(f"Agent 1 Response: {output1.text}")
    assert output1.agent_id == "agent001"
    assert "Mock LLM response" in output1.text

    # Create StandardAgent instance with chat-style LLM
    agent2 = StandardAgent("agent002", agent_role_details, mock_config, mock_bb_with_getters, mock_llm_chat)
    print(f"\nAgent 2 ID: {agent2.get_id()}, Role: {agent2.get_role_config()['name']}")

    # Test generate_response for agent2
    output2 = agent2.generate_response(current_iteration=2) # other_agent_outputs_str is optional
    print(f"Agent 2 Response: {output2.text}")
    assert output2.agent_id == "agent002"
    assert "Mock chat response" in output2.text
    
    # Test role update
    new_role_details = agent_role_details.copy()
    new_role_details["name"] = "UpdatedTestRole"
    new_role_details["temperature"] = 0.9
    agent1.update_role_config(new_role_details)
    print(f"Agent 1 updated role name: {agent1.get_role_config()['name']}")
    assert agent1.get_role_config()['temperature'] == 0.9

    print("\nBasic tests for AbstractAgent and StandardAgent completed.")
