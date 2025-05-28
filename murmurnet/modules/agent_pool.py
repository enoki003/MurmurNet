#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Pool モジュール
~~~~~~~~~~~~~~~~~~~
複数のエージェントを管理し、並列/逐次実行を制御
各エージェントの生成や実行を統合的に管理

作者: Yuhi Sonoki
"""

import logging
import os
import json
import re
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any, List, Optional, Tuple, Callable # Keep List for available_roles
from MurmurNet.modules.model_factory import get_shared_model
from MurmurNet.modules.common import AgentExecutionError, ThreadSafetyError # Keep for now
from MurmurNet.modules.config_manager import get_config
# New imports for refactoring
from .agent import StandardAgent, AbstractAgent # Agent classes

logger = logging.getLogger('MurmurNet.AgentPool')

# エージェント毎の個別ロック管理
class AgentLockManager:
    """エージェント毎の個別ロック管理"""
    
    def __init__(self, num_agents: int):
        self._locks = {i: threading.Lock() for i in range(num_agents)}
        self._global_lock = threading.RLock()  # 管理用のロック
    
    def get_agent_lock(self, agent_id: int) -> threading.Lock:
        """指定エージェントのロックを取得"""
        with self._global_lock:
            if agent_id not in self._locks:
                self._locks[agent_id] = threading.Lock()
            return self._locks[agent_id]
    
    def cleanup(self):
        """リソースクリーンアップ"""
        with self._global_lock:
            self._locks.clear()

class AgentPoolManager:
    """
    分散SLMにおけるエージェントプールの管理
    
    責務:
    - 複数エージェントの生成と実行管理    - 役割ベースの分担処理
    - 並列/逐次実行の制御
    - メモリ最適化エージェントパターン
    
    属性:
        config_manager: ConfigManager instance
        blackboard: 共有黒板
        num_agents: エージェント数
        agents: Dict[str, AbstractAgent] holding agent instances
        lock_manager: エージェント毎のロック管理 (to be removed later)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, blackboard=None): # Use Optional for config dict
        """
        エージェントプールの初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
            blackboard: 共有黒板インスタンス
        """
        # ConfigManagerから設定を取得
        self.config_manager = get_config() 
        # self.config dict is not strictly needed if self.config_manager is used directly
        self.blackboard = blackboard
        
        # ConfigManagerから直接設定値を取得
        self.debug = self.config_manager.logging.debug
        self.num_agents = self.config_manager.agent.num_agents
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # Determine the LLM client to be used by agents
        self.llm_client_instance: Any # To store the actual LLM client
        self.parallel_mode = self.config_manager.agent.use_parallel # Store for clarity if needed elsewhere
        
        if self.parallel_mode:
            from MurmurNet.modules.model_pool import get_model_pool
            self.llm_client_instance = get_model_pool()
            logger.info("AgentPoolManager initialized with ModelPool for LLM access.")
        else:
            # Pass the config dictionary if get_shared_model expects it
            self.llm_client_instance = get_shared_model(self.config_manager.config)
            logger.info("AgentPoolManager initialized with a shared LLM for LLM access.")
        
        # Initialize agents dictionary instead of roles list
        self.agents: Dict[str, AbstractAgent] = {} # Changed from self.roles
        
        # エージェント毎の個別ロック管理 (kept for now as other methods are not refactored yet)
        self.lock_manager = AgentLockManager(self.num_agents) 
        
        self._load_role_templates() # This defines self.role_templates
        self._load_roles() # This will now populate self.agents
        
        # 並列モードの場合の設定
        if self.parallel_mode:
            logger.info("並列処理モードを初期化しました（個別ロック使用）")
        
        logger.info(f"エージェントプールを初期化しました (エージェント数: {self.num_agents})")

    def _load_role_templates(self) -> None:
        """
        役割テンプレートの初期化（内部メソッド）
        
        質問タイプ別の役割テンプレートを定義
        """
        # 質問タイプ別の役割テンプレート定義
        self.role_templates = {
            # 議論型質問用の役割
            "discussion": [
                {"role": "多角的視点AI", "system": "あなたは多角的思考のスペシャリストです。論点を多面的に分析して議論の全体像を示してください。", "temperature": 0.7},
                {"role": "批判的思考AI", "system": "あなたは批判的思考の専門家です。前提や論理に疑問を投げかけ、新たな視点を提供してください。", "temperature": 0.8},
                {"role": "実証主義AI", "system": "あなたはデータと証拠を重視する科学者です。事実に基づいた分析と検証可能な情報を提供してください。", "temperature": 0.6},
                {"role": "倫理的視点AI", "system": "あなたは倫理学者です。道徳的・倫理的観点から議論を分析し、価値判断の視点を提供してください。", "temperature": 0.7}
            ],
            
            # 計画・構想型質問用の役割
            "planning": [
                {"role": "実用主義AI", "system": "あなたは実用主義の専門家です。実行可能で具体的なアプローチを提案してください。", "temperature": 0.7},
                {"role": "創造的思考AI", "system": "あなたは創造的思考のスペシャリストです。革新的なアイデアと可能性を探索してください。", "temperature": 0.9},
                {"role": "戦略的視点AI", "system": "あなたは戦略家です。長期的な視点と全体像を考慮した計画を立案してください。", "temperature": 0.7},
                {"role": "リスク分析AI", "system": "あなたはリスク管理専門家です。潜在的な問題点と対策を特定してください。", "temperature": 0.6}
            ],
            
            # 情報提供型質問用の役割
            "informational": [
                {"role": "事実提供AI", "system": "あなたは情報の専門家です。正確で検証可能な事実情報を簡潔に提供してください。", "temperature": 0.5},
                {"role": "教育的視点AI", "system": "あなたは教育者です。わかりやすく体系的に情報を整理して説明してください。", "temperature": 0.6},
                {"role": "比較分析AI", "system": "あなたは比較分析の専門家です。異なる視点や選択肢を公平に比較してください。", "temperature": 0.7}
            ],
            
            # 一般会話型質問用の役割
            "conversational": [
                {"role": "共感的リスナーAI", "system": "あなたは共感的なリスナーです。相手の感情や意図を理解し、温かみのある応答をしてください。", "temperature": 0.8},
                {"role": "実用アドバイザーAI", "system": "あなたは日常の実用知識に詳しいアドバイザーです。役立つ情報や提案を提供してください。", "temperature": 0.7}
            ],
            
            # デフォルト役割（どのタイプにも当てはまらない場合）
            "default": [
                {"role": "バランス型AI", "system": "あなたは総合的な分析ができるバランス型AIです。公平で多面的な視点から回答してください。", "temperature": 0.7},
                {"role": "専門知識AI", "system": "あなたは幅広い知識を持つ専門家です。正確でわかりやすい情報を提供してください。", "temperature": 0.6}
            ]
        }

    def _load_roles(self) -> None:
        """Instantiates agents and stores them in self.agents based on role templates."""
        self.agents.clear() # Clear any existing agents
        
        role_type_key = self.config_manager.agent.role_type
        
        selected_role_configs = self.role_templates.get(role_type_key)
        if not selected_role_configs: # Fallback if key is missing or list is empty
            logger.warning(f"Configured role_type '{role_type_key}' not found/empty in templates. Falling back to 'default'.")
            role_type_key = 'default'
            selected_role_configs = self.role_templates.get(role_type_key, []) 
        
        if not selected_role_configs: # Check again after fallback
            logger.error(f"No role configurations found for role_type '{role_type_key}' (even after fallback). Cannot load agents.")
            return

        for i in range(self.num_agents):
            agent_id_str = str(i) # Agent IDs are "0", "1", ...
            # Cycle through available role_configs for the selected type
            role_config_for_agent = dict(selected_role_configs[i % len(selected_role_configs)]) # Use a copy
            
            # Create StandardAgent instance
            # Pass self.config_manager (actual ConfigManager instance) as 'config' to StandardAgent
            agent_instance = StandardAgent(
                agent_id=agent_id_str,
                role_config=role_config_for_agent,
                config=self.config_manager, 
                llm_client=self.llm_client_instance,
                blackboard=self.blackboard
            )
            self.agents[agent_id_str] = agent_instance
            logger.debug(f"Initialized and stored agent {agent_id_str} with role: {role_config_for_agent.get('name', 'UnnamedRole')}")

        if not self.agents and self.num_agents > 0:
             logger.warning(f"Failed to load any agents despite num_agents being {self.num_agents}. Check role configurations for '{role_type_key}'.")
        
        if self.debug:
            # Update debug logging to reflect the new structure
            agent_role_names = [agent.get_role_config().get('name', 'UnknownRole') for agent in self.agents.values()]
            logger.debug(f"Loaded {len(self.agents)} agents. Roles assigned: {', '.join(agent_role_names)}")

    def run_agents(self, blackboard) -> None:
        """
        すべてのエージェントを逐次実行
        
        引数:
            blackboard: 共有黒板
        """
        logger.info(f"エージェントを逐次実行中... (エージェント数: {self.num_agents})")
        
        # 各エージェントを順番に実行
        for i in range(self.num_agents):
            try:
                result = self._agent_task(i)
                blackboard.write(f'agent_{i}_output', result)
                
                if self.debug:                logger.debug(f"エージェント{i}の実行が完了しました")
                    
            except AgentExecutionError as e:
                # エージェント実行エラーの適切な処理
                logger.error(f"エージェント実行エラー: {e}")
                blackboard.write(f'agent_{e.agent_id}_output', f"エージェント{e.agent_id}は実行エラーにより応答できませんでした")
                
            except TimeoutError as e:
                # タイムアウトエラーの処理
                logger.error(f"エージェント{i}がタイムアウトしました: {e}")
                blackboard.write(f'agent_{i}_output', f"エージェント{i}は処理時間制限により応答できませんでした")
                
            except Exception as e:
                # その他の予期しないエラー
                error_msg = f"エージェント{i}の予期しないエラー: {str(e)}"
                logger.error(error_msg)
                blackboard.write(f'agent_{i}_output', f"エージェント{i}は技術的な問題により応答できませんでした")
                
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        logger.info(f"エージェント実行完了 (実行数: {self.num_agents})")

    def _format_prompt(self, agent_id: int) -> str:
        """
        エージェント用のプロンプトをフォーマット（内部メソッド）
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            フォーマットされたプロンプト
        """
        # 入力情報の取得
        input_data = self.blackboard.read('input')
        if isinstance(input_data, dict) and 'normalized' in input_data:
            input_text = input_data['normalized']
        else:
            input_text = str(input_data)
            
        # RAG情報の取得
        rag_info = self.blackboard.read('rag')
        rag_text = str(rag_info) if rag_info else "関連情報はありません。"
        
        # 会話コンテキストの取得
        conversation_context = self.blackboard.read('conversation_context')
        context_text = str(conversation_context) if conversation_context else "過去の会話はありません。"
        
        # エージェントの役割情報
        role = self.roles[agent_id]
        role_name = role.get('role', f"エージェント{agent_id}")
        role_desc = role.get('system', "あなたは質問に答えるAIアシスタントです。")
        
        # 他のエージェントの出力を収集
        other_agents_output = []
        for i in range(self.num_agents):
            if i != agent_id:  # 自分以外のエージェント
                output = self.blackboard.read(f'agent_{i}_output')
                if output:
                    other_role = self.roles[i].get('role', f"エージェント{i}")
                    other_agents_output.append(f"{other_role}の回答: {output[:200]}...")
                    
        other_agents_text = "\n\n".join(other_agents_output) if other_agents_output else "他のエージェントの出力はまだありません。"
                  # プロンプトの構築（話し言葉重視）
        prompt = f"""こんにちは！私は「{role_name}」だよ。

{role_desc}

質問: {input_text}

参考になりそうな情報: {rag_text}

これまでの会話: {context_text}

仲間たちの意見:
{other_agents_text}

お願い:
- 私らしい視点で話すね
- わかりやすく具体的に説明するよ
- みんなの意見も参考にして、より良い答えを考えてみる
- 150〜250文字くらいで話し言葉でお答えするね

それじゃあ、{role_name}として答えるよ:"""

        return prompt
    
    # _agent_task is being refactored. The LLM call logic will eventually move to StandardAgent.
    # For now, it's adapted to be called by execute_agent_task and to use the new _format_prompt.
    def _agent_task(self, agent_id_as_int: int, current_iteration: int) -> str: # Signature updated
        """
        Internal task execution logic for a single agent.
        This method will be called by execute_agent_task.
        It performs the LLM call and returns the raw text response.
        It no longer writes directly to the blackboard; execute_agent_task handles that.

        Args:
            agent_id_as_int: The integer index of the agent (for compatibility with old lock manager).
            current_iteration: The current iteration number.
            
        Returns:
            The raw text response from the LLM or an error message string.
        """
        agent_id_str = str(agent_id_as_int)
        agent = self.agents.get(agent_id_str)

        if not agent: # Should ideally not happen if called from execute_agent_task which checks first
            logger.error(f"Agent with ID '{agent_id_str}' (from int: {agent_id_as_int}) not found in self.agents for _agent_task.")
            # This specific error should be caught by execute_agent_task if agent is None there.
            # If called directly and agent is None, this is a critical internal issue.
            raise AgentExecutionError(agent_id_as_int, f"Agent instance for ID {agent_id_str} not found internally.", None)

        role_config = agent.get_role_config()
        role_name = role_config.get('name', f'エージェント{agent_id_str}')
        temperature = role_config.get('temperature', self.config_manager.model.temperature)

        try:
            # Call the refactored _format_prompt (which will be done in the next step)
            # For now, the old _format_prompt(agent_id: int) is still in place.
            # To make this runnable NOW, we'd call the old one.
            # However, the plan is to refactor _format_prompt to take (agent, current_iteration).
            # So, this call anticipates that change.
            prompt = self._format_prompt(agent, current_iteration) # Pass agent instance and iteration
            
            if not prompt: 
                logger.error(f"エージェント{agent_id_str}のプロンプト生成に失敗 (反復 {current_iteration})")
                raise AgentExecutionError(agent_id_as_int, "プロンプト生成失敗", None)
            
            logger.debug(f"エージェント{agent_id_str} ({role_name}) タスク開始 (反復 {current_iteration})")
            
            start_time = time.time()
            
            llm_for_task = self.llm_client_instance 

            # AgentLockManager is still used here for now.
            agent_lock = self.lock_manager.get_agent_lock(agent_id_as_int)
            with agent_lock:
                if not hasattr(llm_for_task, 'create_chat_completion') and not hasattr(llm_for_task, 'generate'):
                    logger.error(f"LLM client for agent {agent_id_str} does not have a recognized generation method.")
                    raise AgentExecutionError(agent_id_as_int, "LLMクライアント設定エラー", None)

                if hasattr(llm_for_task, 'create_chat_completion'):
                    resp = llm_for_task.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=role_config.get('max_tokens', self.config_manager.model.max_tokens),
                        temperature=temperature,
                        top_p=role_config.get('top_p', 0.9), 
                        stop=role_config.get('stop_sequences', ["。", ".", "\n\n"]) 
                    )
                else: 
                    resp_text_llm = llm_for_task.generate(
                        prompt=prompt,
                        max_tokens=role_config.get('max_tokens', self.config_manager.model.max_tokens),
                        temperature=temperature
                    )
                    resp = {"choices": [{"message": {"content": resp_text_llm}}]}
            
            generation_time = time.time() - start_time
            if generation_time > 20: 
                logger.warning(f"エージェント{agent_id_str}の応答生成に時間がかかりました: {generation_time:.2f}秒")
            
            response_text = ""
            if isinstance(resp, dict) and 'choices' in resp and resp['choices']:
                response_text = resp['choices'][0]['message']['content'].strip()
            elif hasattr(resp, 'choices') and resp.choices: 
                response_text = resp.choices[0].message.content.strip()
            else: # Should not happen with current logic if LLM call succeeds
                logger.error(f"エージェント{agent_id_str}: 予期しないLLMレスポンス形式: {type(resp)}")
                response_text = str(resp).strip() if resp else "" # Fallback

            if not response_text: # Empty response from LLM
                logger.warning(f"エージェント{agent_id_str}が空の応答を生成しました。")
                # Return a specific string indicating empty response, not an error state necessarily
                return f"{role_name}は応答を生成しませんでした。" # This is a valid text output.
            
            # Removed: self.blackboard.write(f'agent_{agent_id}_output', response_text)
            logger.debug(f"エージェント{agent_id_str} ({role_name}) 内部タスク完了 (反復 {current_iteration}). Response length: {len(response_text)}")
            return response_text # Return raw text
            
        except TimeoutError as e: 
            logger.error(f"エージェント{agent_id_str} タイムアウト (反復 {current_iteration}): {e}", exc_info=True)
            raise AgentExecutionError(agent_id_as_int, "タイムアウト", e) 
            
        except MemoryError as e:
            logger.error(f"エージェント{agent_id_str} メモリ不足 (反復 {current_iteration}): {e}", exc_info=True)
            raise AgentExecutionError(agent_id_as_int, "メモリ不足", e) 
            
        except Exception as e: # Catch other potential errors during LLM call or processing
            logger.error(f"エージェント{agent_id_str} 実行エラー (反復 {current_iteration}): {type(e).__name__} - {str(e)}", exc_info=True)
            raise AgentExecutionError(agent_id_as_int, f"予期せぬ実行エラー: {str(e)}", e)


    def execute_agent_task(self, agent_id_str: str, current_iteration: int) -> 'AgentOutput':
        """
        New public method to execute an agent's task.
        It calls the (temporarily refactored) _agent_task, then wraps the result in AgentOutput.
        It also handles writing the AgentOutput to the blackboard.
        """
        from MurmurNet.modules.data_structures import AgentOutput # Ensure AgentOutput is available

        agent = self.agents.get(agent_id_str)
        role_name_for_error = f"Agent_{agent_id_str}" # Fallback role name for error outputs

        if not agent:
            logger.error(f"execute_agent_task: Agent with ID '{agent_id_str}' not found.")
            error_text = f"エージェント{agent_id_str}は存在しません。"
            agent_output_obj = AgentOutput(
                agent_id=agent_id_str,
                text=error_text,
                role=role_name_for_error + "_NotFound", # Specific error role
                iteration=current_iteration
            )
            if self.blackboard: 
                self.blackboard.add_agent_output(agent_output_obj) 
            return agent_output_obj

        # If agent exists, get its configured role name
        role_name_for_error = agent.get_role_config().get('name', f"Agent_{agent_id_str}")
        response_text = ""
        
        try:
            # _agent_task now expects an integer ID and current_iteration
            response_text = self._agent_task(int(agent_id_str), current_iteration)
            # _agent_task returns a string (either response or specific message for empty response)
            
            agent_output_obj = AgentOutput(
                agent_id=agent_id_str,
                text=response_text,
                role=agent.get_role_config().get('name', role_name_for_error), # Get role name from actual agent
                iteration=current_iteration
            )
        except AgentExecutionError as e: 
            logger.error(f"execute_agent_task: AgentExecutionError for agent {agent_id_str} (iter {current_iteration}): {e.message}", exc_info=False) # exc_info=False as _agent_task already logged it
            response_text = f"{role_name_for_error}はエラーにより応答できませんでした: {e.message}" 
            agent_output_obj = AgentOutput(
                agent_id=agent_id_str,
                text=response_text,
                role=role_name_for_error + "_ExecutionError", 
                iteration=current_iteration
            )
        except Exception as e: 
            logger.error(f"execute_agent_task: Unexpected error for agent {agent_id_str} (iter {current_iteration}): {str(e)}", exc_info=True)
            response_text = f"{role_name_for_error}で予期せぬエラーが発生: {str(e)}"
            agent_output_obj = AgentOutput(
                agent_id=agent_id_str,
                text=response_text,
                role=role_name_for_error + "_UnexpectedError",
                iteration=current_iteration
            )

        if self.blackboard: 
            self.blackboard.add_agent_output(agent_output_obj) 
        
        return agent_output_obj


    def get_agent_info(self, agent_id_str: str) -> Dict[str, Any]: # Changed agent_id to str
        """
        エージェントの情報を取得
        
        引数:
            agent_id_str: エージェントID (string)
            
        戻り値:
            エージェント情報の辞書
        """
        # agent_id is now string for consistency with self.agents keys
        agent = self.agents.get(agent_id_str) 
        if not agent: # Check if agent exists
            return {"error": f"無効なエージェントID: {agent_id_str}"}
            
        role_config = agent.get_role_config() # Use the agent method
        
        return {
            "id": agent_id_str,
            "role": role_config.get("name", f"エージェント{agent_id_str}"), # Use 'name' from role_config
            "description": role_config.get("system", "情報なし"), # 'system' is system_prompt_template
            "temperature": role_config.get("temperature", 0.7) # Default from role_config
        }

    def update_roles_based_on_question(self, question: str) -> None:
        """
        質問の内容に基づいて適切な役割を動的に更新
        
        引数:
            question: 入力された質問
        """
        # 質問タイプを判定
        question_type = self._classify_question_type(question)
        
        # 黒板に質問タイプを書き込み (This write should use new blackboard methods if refactored)
        # For now, keeping old write as blackboard itself isn't fully refactored in this step
        # TODO: Update this to use self.blackboard.add_generic_entry when Blackboard is fully refactored.
        if self.blackboard: # Check if blackboard is available
             self.blackboard.write('question_type', question_type) # Old write method
        
        # 質問タイプに基づいて役割を更新
        if question_type in self.role_templates:
            available_roles = self.role_templates[question_type]
        else:
            available_roles = self.role_templates['default']
        
        if not available_roles: # Ensure available_roles is not empty
            logger.error(f"No roles found for question type '{question_type}' or default. Roles not updated.")
            return

        # エージェント数に合わせて役割を再割り当て
        # This now updates the role_config within each StandardAgent object
        for i in range(self.num_agents):
            agent_id_str = str(i)
            agent = self.agents.get(agent_id_str)
            if agent:
                role_index = i % len(available_roles)
                new_role_config = dict(available_roles[role_index]) # Use a copy
                agent.update_role_config(new_role_config)
            else:
                logger.warning(f"Agent {agent_id_str} not found during role update.")
        
        # self.agent_roles = self.roles # self.roles is no longer the primary store
        
        if self.debug:
            updated_roles_info = ", ".join(self.agents[str(i)].get_role_config().get("name", "Unknown") for i in range(self.num_agents) if str(i) in self.agents)
            logger.debug(f"質問タイプ '{question_type}' に基づいて役割を更新: {updated_roles_info}")

    def _classify_question_type(self, question: str) -> str:
        """
        質問のタイプを分類（内部メソッド）
        
        引数:
            question: 分析する質問
            
        戻り値:
            質問タイプ ('discussion', 'planning', 'informational', 'conversational', 'default')
        """
        question_lower = question.lower()
        
        # 議論型のキーワード
        discussion_keywords = [
            'について議論', '議論して', '賛成', '反対', '問題', '課題', '影響',
            'どう思う', 'どう考える', 'メリット', 'デメリット', '比較',
            'discuss', 'debate', 'argue', 'opinion', 'think', 'pros', 'cons'
        ]
        
        # 計画・構想型のキーワード
        planning_keywords = [
            'どうすれば', 'どのように', '方法', '手順', '計画', '戦略',
            '実現', '達成', '解決', '改善', '対策', '案', '提案',
            'how to', 'plan', 'strategy', 'solve', 'achieve', 'implement'
        ]
        
        # 情報提供型のキーワード
        informational_keywords = [
            'とは', 'なに', '何', '定義', '説明', '詳細', '仕組み',
            '原因', '理由', 'なぜ', '歴史', '背景', '特徴',
            'what is', 'define', 'explain', 'why', 'how', 'cause', 'reason'
        ]
        
        # 会話型のキーワード
        conversational_keywords = [
            'こんにちは', 'はじめまして', 'おはよう', 'こんばんは',
            'ありがとう', 'すみません', 'お疲れ', '元気',
            'hello', 'hi', 'good morning', 'thank you', 'sorry'
        ]
        
        # キーワードマッチング
        if any(keyword in question_lower for keyword in discussion_keywords):
            return 'discussion'
        elif any(keyword in question_lower for keyword in planning_keywords):
            return 'planning'
        elif any(keyword in question_lower for keyword in informational_keywords):
            return 'informational'
        elif any(keyword in question_lower for keyword in conversational_keywords):            return 'conversational'
        else:
            return 'default'
    
    # 追加メソッド: 並列実行最適化
    async def run_agent_parallel(self, agent_id: int, current_iteration: int) -> str: # Added current_iteration
        """
        エージェントを並列実行に最適化された方法で実行
        This method needs to be updated to use execute_agent_task and return AgentOutput or its text.
        For now, its call to _agent_task_parallel_safe is problematic as that method itself is not fully refactored.
        
        引数:
            agent_id: エージェントID
            current_iteration: 現在の反復数
            
        戻り値:
            エージェントの応答テキスト (this will change to AgentOutput)
        """
        # This method will be more deeply refactored later.
        # For now, make minimal changes to accommodate current_iteration if _agent_task_parallel_safe needs it.
        agent_id_str = str(agent_id)
        logger.debug(f"run_agent_parallel called for agent {agent_id_str}, iter {current_iteration}")
        try:
            # Non-blocking execution using asyncio event loop
            loop = asyncio.get_event_loop()
            
            # _agent_task_parallel_safe needs refactoring to accept current_iteration
            # For now, passing a placeholder 0. This method is not fully functional in this intermediate step.
            # It also needs to return AgentOutput eventually.
            result_text = await loop.run_in_executor(
                None, 
                self._agent_task_parallel_safe, # This method is not yet refactored for iteration
                agent_id
            )
            
            return result_text # This is old return type, will become AgentOutput
            
        except Exception as e:
            logger.error(f"エージェント{agent_id_str}並列実行エラー (iter {current_iteration}): {e}")
            agent = self.agents.get(agent_id_str)
            role_name = agent.get_role_config().get('name', f'エージェント{agent_id_str}') if agent else f'エージェント{agent_id_str}'
            # role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は並列実行エラーにより応答できませんでした"
            # Old blackboard write - to be removed when this method is fully refactored
            if self.blackboard: self.blackboard.write(f'agent_{agent_id_str}_output', error_msg) 
            return error_msg # This should be an AgentOutput with error info

    def _agent_task_parallel_safe(self, agent_id: int) -> str: # This method will be removed
        """
        並列実行に安全なエージェントタスク（ロックなし版） - **TO BE REMOVED/REPLACED**
        """
        # This method is problematic as it doesn't take current_iteration for _format_prompt.
        # It's being temporarily kept to avoid breaking run_agent_parallel immediately,
        # but the whole chain (_agent_task_parallel_safe -> run_agent_parallel -> execute_agent_task)
        # needs to be streamlined.
        # For this subtask, we focus on _agent_task and execute_agent_task.
        # This method will be removed in a subsequent step of refactoring AgentPoolManager.
        agent_id_str = str(agent_id)
        agent = self.agents.get(agent_id_str)
        if not agent:
            logger.error(f"無効なエージェントID (parallel_safe): {agent_id_str}")
            return f"エージェント{agent_id_str}は設定されていません"
        
        logger.warning(f"_agent_task_parallel_safe for agent {agent_id_str} is using placeholder iteration 0 for _format_prompt. This method is pending removal/full refactor.")
        prompt = self._format_prompt(agent, 0) # Placeholder iteration
        
        role_config = agent.get_role_config()
        role_name = role_config.get('name', f'エージェント{agent_id_str}')
        temperature = role_config.get('temperature', 0.7)

        try:
            llm_for_task = self.llm_client_instance
            if not hasattr(llm_for_task, 'create_chat_completion') and not hasattr(llm_for_task, 'generate'):
                return f"{role_name}はモデルの問題で応答できませんでした"

            if hasattr(llm_for_task, 'create_chat_completion'):
                resp = llm_for_task.create_chat_completion( messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=temperature)
            else:
                resp_text_llm = llm_for_task.generate(prompt=prompt, max_tokens=300, temperature=temperature)
                resp = {"choices": [{"message": {"content": resp_text_llm}}]}
            
            response_text = resp['choices'][0]['message']['content'].strip() if isinstance(resp, dict) and resp.get('choices') else str(resp)
            
            if not response_text: return f"{role_name}は適切な応答を生成できませんでした"
            
            # Old blackboard write - to be removed
            if self.blackboard: self.blackboard.write(f'agent_{agent_id_str}_output', response_text)
            return response_text
        except Exception as e:
            logger.error(f"エージェント{agent_id_str}モデル実行エラー (parallel_safe): {e}")
            error_msg = f"{role_name}はモデル実行エラーにより応答できませんでした"
            if self.blackboard: self.blackboard.write(f'agent_{agent_id_str}_output', error_msg)
            return error_msg
            start_time = time.time()
            
            # エージェント毎の個別ロックを使用
            agent_lock = self.lock_manager.get_agent_lock(agent_id)
            with agent_lock:
                # モデルの可用性チェック
                if not hasattr(self.llm, 'create_chat_completion') and not hasattr(self.llm, 'generate'):
                    logger.error(f"モデルインスタンスが正しく初期化されていません")
                    return f"{role_name}はモデルの問題で応答できませんでした"
                
                # チャット完了を試行
                if hasattr(self.llm, 'create_chat_completion'):
                    resp = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,  # 話し言葉に適したトークン数
                        temperature=temperature,
                        top_p=0.9,
                        stop=["。", ".", "\n\n"]
                    )
                else:
                    # フォールバック：generate メソッドを使用
                    resp_text = self.llm.generate(
                        prompt=prompt,
                        max_tokens=300,
                        temperature=temperature
                    )
                    # 辞書形式に変換
                    resp = {"choices": [{"message": {"content": resp_text}}]}
            
            generation_time = time.time() - start_time
            if generation_time > 20:  # 20秒以上の場合
                logger.warning(f"エージェント{agent_id}の応答生成に時間がかかりました: {generation_time:.2f}秒")
            
            # レスポンスの解析（堅牢化）
            try:
                if isinstance(resp, dict) and 'choices' in resp and resp['choices']:
                    response_text = resp['choices'][0]['message']['content'].strip()
                elif hasattr(resp, 'choices') and resp.choices:
                    response_text = resp.choices[0].message.content.strip()
                else:
                    logger.error(f"予期しないレスポンス形式: {type(resp)}")
                    response_text = str(resp).strip()
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"レスポンス解析エラー: {e}")
                return f"{role_name}は応答の解析に失敗しました"
            
            # 出力の検証とクリーニング
            if not response_text:
                logger.warning(f"エージェント{agent_id}が空の応答を生成")
                return f"{role_name}は適切な応答を生成できませんでした"
            
            # 出力長の制限
            if len(response_text) > 500:
                response_text = response_text[:500]
                logger.debug(f"エージェント{agent_id}の応答を切り詰めました")
            
            # 黒板への書き込み
            self.blackboard.write(f'agent_{agent_id}_output', response_text)
            logger.debug(f"エージェント{agent_id} ({role_name}) タスク完了: {len(response_text)}文字")
            return response_text
            
        except TimeoutError as e:
            # タイムアウト専用処理
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は処理時間の制限により応答できませんでした"
            logger.error(f"エージェント{agent_id} タイムアウト: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            raise AgentExecutionError(agent_id, "タイムアウト", e)
            
        except MemoryError as e:
            # メモリ不足専用処理
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}はメモリ不足のため応答できませんでした"
            logger.error(f"エージェント{agent_id} メモリ不足: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            raise AgentExecutionError(agent_id, "メモリ不足", e)
            
        except Exception as e:
            # その他の例外
            error_type = type(e).__name__
            logger.error(f"エージェント{agent_id} 実行エラー ({error_type}): {str(e)}")
            
            # エラータイプ別の適切な応答
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            
            if "connection" in str(e).lower() or "network" in str(e).lower():
                error_msg = f"{role_name}は接続の問題により応答できませんでした"
            else:
                error_msg = f"{role_name}は技術的な問題により応答できませんでした"
            
            # 黒板にエラー情報を記録
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            
            # カスタム例外として再発生
            raise AgentExecutionError(agent_id, str(e), e)
            return error_msg

    def get_agent_info(self, agent_id: int) -> Dict[str, Any]:
        """
        エージェントの情報を取得
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェント情報の辞書
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            return {"error": "無効なエージェントID"}
            
        role = self.roles[agent_id]
        
        return {
            "id": agent_id,
            "role": role.get("role", f"エージェント{agent_id}"),
            "description": role.get("system", "情報なし"),
            "temperature": role.get("temperature", 0.7)
        }

    def update_roles_based_on_question(self, question: str) -> None:
        """
        質問の内容に基づいて適切な役割を動的に更新
        
        引数:
            question: 入力された質問
        """
        # 質問タイプを判定
        question_type = self._classify_question_type(question)
        
        # 黒板に質問タイプを書き込み
        self.blackboard.write('question_type', question_type)
        
        # 質問タイプに基づいて役割を更新
        if question_type in self.role_templates:
            available_roles = self.role_templates[question_type]
        else:
            available_roles = self.role_templates['default']
        
        # エージェント数に合わせて役割を再割り当て
        self.roles = []
        for i in range(self.num_agents):
            role_index = i % len(available_roles)
            self.roles.append(available_roles[role_index])
        
        # agent_rolesという属性も設定（テストコードで使用されている）
        self.agent_roles = self.roles
        
        if self.debug:
            roles_info = ", ".join(role["role"] for role in self.roles)
            logger.debug(f"質問タイプ '{question_type}' に基づいて役割を更新: {roles_info}")

    def _classify_question_type(self, question: str) -> str:
        """
        質問のタイプを分類（内部メソッド）
        
        引数:
            question: 分析する質問
            
        戻り値:
            質問タイプ ('discussion', 'planning', 'informational', 'conversational', 'default')
        """
        question_lower = question.lower()
        
        # 議論型のキーワード
        discussion_keywords = [
            'について議論', '議論して', '賛成', '反対', '問題', '課題', '影響',
            'どう思う', 'どう考える', 'メリット', 'デメリット', '比較',
            'discuss', 'debate', 'argue', 'opinion', 'think', 'pros', 'cons'
        ]
        
        # 計画・構想型のキーワード
        planning_keywords = [
            'どうすれば', 'どのように', '方法', '手順', '計画', '戦略',
            '実現', '達成', '解決', '改善', '対策', '案', '提案',
            'how to', 'plan', 'strategy', 'solve', 'achieve', 'implement'
        ]
        
        # 情報提供型のキーワード
        informational_keywords = [
            'とは', 'なに', '何', '定義', '説明', '詳細', '仕組み',
            '原因', '理由', 'なぜ', '歴史', '背景', '特徴',
            'what is', 'define', 'explain', 'why', 'how', 'cause', 'reason'
        ]
        
        # 会話型のキーワード
        conversational_keywords = [
            'こんにちは', 'はじめまして', 'おはよう', 'こんばんは',
            'ありがとう', 'すみません', 'お疲れ', '元気',
            'hello', 'hi', 'good morning', 'thank you', 'sorry'
        ]
        
        # キーワードマッチング
        if any(keyword in question_lower for keyword in discussion_keywords):
            return 'discussion'
        elif any(keyword in question_lower for keyword in planning_keywords):
            return 'planning'
        elif any(keyword in question_lower for keyword in informational_keywords):
            return 'informational'
        elif any(keyword in question_lower for keyword in conversational_keywords):            return 'conversational'
        else:
            return 'default'
    
    # 追加メソッド: 並列実行最適化
    async def run_agent_parallel(self, agent_id: int) -> str:
        """
        エージェントを並列実行に最適化された方法で実行
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        try:
            # 非同期で並列実行（ロックなし）
            loop = asyncio.get_event_loop()
            
            # 専用のモデルインスタンスまたは並列対応実行
            result = await loop.run_in_executor(
                None, 
                self._agent_task_parallel_safe, 
                agent_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"エージェント{agent_id}並列実行エラー: {e}")
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は並列実行エラーにより応答できませんでした"
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            return error_msg

    def _agent_task_parallel_safe(self, agent_id: int) -> str:
        """
        並列実行に安全なエージェントタスク（ロックなし版）
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        try:
            # エージェントIDの検証
            if agent_id < 0 or agent_id >= len(self.roles):
                logger.error(f"無効なエージェントID: {agent_id}")
                return f"エージェント{agent_id}は設定されていません"
            
            # プロンプトの構築
            prompt = self._format_prompt(agent_id)
            if not prompt:
                logger.error(f"エージェント{agent_id}のプロンプト生成に失敗")
                return f"エージェント{agent_id}はプロンプトを生成できませんでした"
            
            # エージェントの役割と設定
            role = self.roles[agent_id]
            temperature = max(0.1, min(1.0, role.get('temperature', 0.7)))
            role_name = role.get('role', f'エージェント{agent_id}')
            
            logger.debug(f"エージェント{agent_id} ({role_name}) 並列タスク開始")
            
            # 並列実行用のモデル呼び出し（ロックなし）
            import time
            start_time = time.time()
            
            # ロックを使わずに直接モデル実行
            try:
                # モデルの可用性チェック
                if not hasattr(self.llm, 'create_chat_completion') and not hasattr(self.llm, 'generate'):
                    logger.error(f"モデルインスタンスが正しく初期化されていません")
                    return f"{role_name}はモデルの問題で応答できませんでした"
                
                # 並列安全なモデル呼び出し
                if hasattr(self.llm, 'create_chat_completion'):
                    resp = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=temperature,
                        top_p=0.9,
                        stop=["。", ".", "\n\n"]
                    )
                else:
                    # フォールバック
                    resp_text = self.llm.generate(
                        prompt=prompt,
                        max_tokens=300,
                        temperature=temperature
                    )
                    resp = {"choices": [{"message": {"content": resp_text}}]}
                
            except Exception as model_error:
                logger.error(f"エージェント{agent_id}モデル実行エラー: {model_error}")
                return f"{role_name}はモデル実行エラーにより応答できませんでした"
            
            generation_time = time.time() - start_time
            logger.debug(f"エージェント{agent_id} 生成時間: {generation_time:.2f}秒")
            
            # レスポンスの解析（堅牢化）
            try:
                if isinstance(resp, dict) and 'choices' in resp and resp['choices']:
                    response_text = resp['choices'][0]['message']['content'].strip()
                elif hasattr(resp, 'choices') and resp.choices:
                    response_text = resp.choices[0].message.content.strip()
                else:
                    logger.error(f"予期しないレスポンス形式: {type(resp)}")
                    response_text = str(resp).strip()
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"レスポンス解析エラー: {e}")
                return f"{role_name}は応答の解析に失敗しました"
            
            # 出力の検証とクリーニング
            if not response_text:
                logger.warning(f"エージェント{agent_id}が空の応答を生成")
                return f"{role_name}は適切な応答を生成できませんでした"
            
            # 出力長の制限
            if len(response_text) > 500:
                response_text = response_text[:500]
                logger.debug(f"エージェント{agent_id}の応答を切り詰めました")
            
            # 黒板への書き込み
            self.blackboard.write(f'agent_{agent_id}_output', response_text)
            logger.debug(f"エージェント{agent_id} ({role_name}) 並列タスク完了: {len(response_text)}文字")
            return response_text
            
        except Exception as e:
            # 全般的なエラーハンドリング
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は並列実行エラーにより応答できませんでした"
            logger.error(f"エージェント{agent_id} 並列実行エラー: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            return error_msg