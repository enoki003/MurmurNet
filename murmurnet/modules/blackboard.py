#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blackboard モジュール
~~~~~~~~~~~~~~~~~~~
エージェント間の共有メモリとして機能する黒板パターン実装
データの読み書き、監視、履歴管理を提供 (Refactored for structured data)

作者: Yuhi Sonoki (Original), AI Assistant (Refactor)
"""

import time
import threading
import uuid
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Callable, Union

from .common import BlackboardError # ResourceLimitError might be deprecated if not used
from .config_manager import get_config
from .data_structures import AgentOutput, BlackboardEntry # New data structures

# Content types (can be expanded)
CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_AGENT_OUTPUT = "agent_output"
CONTENT_TYPE_RAG_RESULT = "rag_result"
CONTENT_TYPE_SUMMARY = "summary"
CONTENT_TYPE_ERROR = "error_message"
CONTENT_TYPE_SYSTEM_MESSAGE = "system_message"
CONTENT_TYPE_USER_INPUT = "user_input_details" # Changed from "user_input" to avoid clash with simple text
CONTENT_TYPE_KEY_FACTS = "key_facts"
CONTENT_TYPE_CONVERSATION_CONTEXT = "conversation_context"
CONTENT_TYPE_QUESTION_TYPE_CLASSIFICATION = "question_type_classification"
CONTENT_TYPE_FINAL_RESPONSE = "final_response"


# Sources (can be expanded)
SOURCE_USER = "user"
SOURCE_AGENT_PREFIX = "agent_" # e.g. agent_1, agent_summary
SOURCE_SUMMARY_ENGINE = "summary_engine"
SOURCE_RAG_RETRIEVER = "rag_retriever"
SOURCE_SYSTEM = "system" # For system-generated entries like errors or control messages
SOURCE_CONVERSATION_MEMORY = "conversation_memory"
SOURCE_OUTPUT_AGENT = "output_agent"
SOURCE_AGENT_POOL_MANAGER = "agent_pool_manager"


class Blackboard:
    """
    スレッドセーフな分散エージェント間の共有メモリ実装 using structured data.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None): # Use Optional for config dict
        self.config_manager = get_config()
        # self.config = config or self.config_manager.to_dict() # Not strictly needed if using config_manager
        
        self._memory: OrderedDict[str, BlackboardEntry] = OrderedDict()
        self._lock = threading.RLock() # RLock allows re-entrant locking by the same thread
        
        self.debug = self.config_manager.logging.debug
        
        self.persistent_content_types = {
            CONTENT_TYPE_CONVERSATION_CONTEXT, 
            CONTENT_TYPE_KEY_FACTS,
            # Previous summaries might be context for the next turn, but new summaries are per-iteration.
            # Let's assume summaries relevant for long-term context are handled by ConversationMemory or similar.
            # For now, only keeping direct context types.
        }
        
        self._stats = {
            'add_count': 0,
            'get_count': 0,
            'remove_count': 0,
            'clear_turn_count': 0,
            'error_count': 0
        }
        if self.debug:
            self.logger = self.config_manager.get_logger("Blackboard") # Get a logger instance
            self.logger.info(f"Blackboard Initialized. Persistent content types: {self.persistent_content_types}")
        else: # Ensure logger exists even if not in debug, but might not output much
            self.logger = self.config_manager.get_logger("Blackboard")
            self.logger.setLevel(logging.INFO) # Default to INFO if not debug


    def _add_entry(self, entry: BlackboardEntry) -> BlackboardEntry:
        """Internal method to add an entry, handling logging and stats."""
        with self._lock:
            if entry.entry_id in self._memory:
                if self.debug:
                    self.logger.warning(f"Overwriting entry with ID {entry.entry_id}")
            self._memory[entry.entry_id] = entry
            self._stats['add_count'] += 1
            if self.debug:
                self.logger.debug(f"Added: {entry}")
            return entry

    # --- Add methods ---
    def add_user_input(self, text: str, iteration: Optional[int] = None, raw_input_details: Optional[Dict[str, Any]] = None) -> BlackboardEntry:
        # raw_input_details is expected to be like {'raw': original_text, 'normalized': normalized_text}
        # For consistency, the main `text` parameter should be the normalized one.
        data_payload = {"normalized": text}
        if raw_input_details: # If full dict is provided
            data_payload = raw_input_details
            if 'normalized' not in data_payload and text: # Ensure normalized is there
                 data_payload['normalized'] = text
        elif text: # if only normalized text is given
             data_payload = {'normalized': text, 'raw': text} # Assume raw is same if not specified
        else:
            data_payload = {'normalized': "", 'raw': ""}


        entry = BlackboardEntry(
            source=SOURCE_USER,
            content_type=CONTENT_TYPE_USER_INPUT, 
            data=data_payload,
            iteration=iteration
        )
        return self._add_entry(entry)

    def add_agent_output(self, output: AgentOutput) -> BlackboardEntry:
        entry = BlackboardEntry(
            source=f"{SOURCE_AGENT_PREFIX}{output.agent_id}",
            content_type=CONTENT_TYPE_AGENT_OUTPUT,
            data=output, 
            iteration=output.iteration
        )
        return self._add_entry(entry)

    def add_rag_result(self, rag_data: Any, iteration: Optional[int] = None, source: str = SOURCE_RAG_RETRIEVER) -> BlackboardEntry:
        entry = BlackboardEntry(
            source=source,
            content_type=CONTENT_TYPE_RAG_RESULT,
            data=rag_data,
            iteration=iteration
        )
        return self._add_entry(entry)

    def add_summary(self, summary_text: str, iteration: int, source_ids: Optional[List[str]] = None, source:str = SOURCE_SUMMARY_ENGINE) -> BlackboardEntry:
        data = {"text": summary_text}
        if source_ids: # These are IDs of BlackboardEntry objects that were summarized
            data["source_entry_ids"] = source_ids
        entry = BlackboardEntry(
            source=source,
            content_type=CONTENT_TYPE_SUMMARY,
            data=data,
            iteration=iteration
        )
        return self._add_entry(entry)
        
    def add_key_facts(self, facts: Any, iteration: Optional[int] = None, source: str = SOURCE_CONVERSATION_MEMORY) -> BlackboardEntry:
        entry = BlackboardEntry(
            source=source,
            content_type=CONTENT_TYPE_KEY_FACTS,
            data=facts,
            iteration=iteration # Key facts might be associated with an iteration or be global (None)
        )
        return self._add_entry(entry)

    def add_conversation_context(self, context: str, iteration: Optional[int] = None, source: str = SOURCE_CONVERSATION_MEMORY) -> BlackboardEntry:
        entry = BlackboardEntry(
            source=source,
            content_type=CONTENT_TYPE_CONVERSATION_CONTEXT,
            data=context,
            iteration=iteration
        )
        return self._add_entry(entry)

    def add_generic_entry(self, source: str, content_type: str, data: Any, iteration: Optional[int] = None, entry_id: Optional[str] = None) -> BlackboardEntry:
        # Ensure entry_id is used if provided, otherwise generate new
        final_entry_id = entry_id if entry_id else str(uuid.uuid4())
        entry = BlackboardEntry(
            entry_id=final_entry_id,
            source=source,
            content_type=content_type,
            data=data,
            iteration=iteration
        )
        if content_type == CONTENT_TYPE_ERROR:
            self._stats['error_count'] +=1
        return self._add_entry(entry)

    # --- Get methods ---
    def get_entry(self, entry_id: str) -> Optional[BlackboardEntry]:
        with self._lock:
            self._stats['get_count'] += 1
            entry = self._memory.get(entry_id)
            if self.debug and entry:
                self.logger.debug(f"Get by ID ('{entry_id}'): {entry}")
            return entry

    def get_entries_by_source(self, source: str, iteration: Optional[int] = None, latest_first: bool = False) -> List[BlackboardEntry]:
        with self._lock:
            self._stats['get_count'] += 1
            # Iterate once and filter. reversed() on list if latest_first.
            all_matching_source = [entry for entry in self._memory.values() if entry.source == source]
            
            results = []
            for entry in (reversed(all_matching_source) if latest_first else all_matching_source):
                if iteration is None or entry.iteration == iteration:
                    results.append(entry)
            
            if self.debug and results:
                self.logger.debug(f"Get by Source ('{source}', iter={iteration}, latest_first={latest_first}): Found {len(results)} entries.")
            # If latest_first was true, results are already in that order. If false, they are in insertion order.
            return results

    def get_entries_by_content_type(self, content_type: str, iteration: Optional[int] = None, latest_first: bool = False) -> List[BlackboardEntry]:
        with self._lock:
            self._stats['get_count'] += 1
            all_matching_type = [entry for entry in self._memory.values() if entry.content_type == content_type]
            
            results = []
            for entry in (reversed(all_matching_type) if latest_first else all_matching_type):
                if iteration is None or entry.iteration == iteration:
                    results.append(entry)

            if self.debug and results:
                self.logger.debug(f"Get by ContentType ('{content_type}', iter={iteration}, latest_first={latest_first}): Found {len(results)} entries.")
            return results

    def get_last_n_entries(self, n: int, content_type: Optional[str] = None, iteration: Optional[int] = None) -> List[BlackboardEntry]:
        with self._lock:
            self._stats['get_count'] += 1
            
            # Iterate from newest to oldest to find the N matching entries
            candidate_entries = reversed(list(self._memory.values()))
            
            results_reversed_order = []
            for entry in candidate_entries:
                if content_type and entry.content_type != content_type:
                    continue
                if iteration is not None and entry.iteration != iteration:
                    continue
                results_reversed_order.append(entry)
                if len(results_reversed_order) == n:
                    break
            
            # Restore original insertion order (oldest to newest among the N found)
            final_results = list(reversed(results_reversed_order))

            if self.debug and final_results:
                self.logger.debug(f"Get Last N ({n}, type={content_type}, iter={iteration}): Found {len(final_results)} entries.")
            return final_results

    def get_user_input(self, iteration: Optional[int] = None) -> Optional[BlackboardEntry]:
        # Assuming one primary user input per iteration, get the latest if multiple.
        entries = self.get_entries_by_content_type(CONTENT_TYPE_USER_INPUT, iteration=iteration, latest_first=True)
        return entries[0] if entries else None

    def get_agent_outputs(self, iteration: Optional[int] = None, agent_id: Optional[str] = None) -> List[AgentOutput]:
        results = []
        with self._lock:
            self._stats['get_count'] += 1
            # Iterate through BlackboardEntry objects
            for entry in self._memory.values():
                if entry.content_type == CONTENT_TYPE_AGENT_OUTPUT:
                    # Ensure data is an AgentOutput instance and matches criteria
                    if isinstance(entry.data, AgentOutput):
                        if (iteration is None or entry.iteration == iteration) and \
                           (agent_id is None or entry.data.agent_id == agent_id):
                            results.append(entry.data)
                    elif self.debug:
                         self.logger.warning(f"Entry {entry.entry_id} has content_type AGENT_OUTPUT but data is not AgentOutput instance: {type(entry.data)}")

            if self.debug and results:
                self.logger.debug(f"Get AgentOutputs (iter={iteration}, agent_id={agent_id}): Found {len(results)} outputs.")
        # Results are already in insertion order due to iterating self._memory.values()
        return results
        
    def get_all_entries(self, latest_first: bool = False) -> List[BlackboardEntry]:
        with self._lock:
            self._stats['get_count'] +=1
            entries = list(self._memory.values()) # Values are already in insertion order
            if latest_first:
                return list(reversed(entries)) # Create a new reversed list
            return entries # Return a copy

    # --- Management methods ---
    def clear_all(self) -> None:
        """Clears all entries from the blackboard."""
        with self._lock:
            self._memory.clear()
            if self.debug:
                self.logger.info("Cleared all entries from blackboard.")
            # Reset relevant stats if needed, e.g., self._stats['add_count'] = 0, etc.
            # For now, keeping historical stats like add_count.

    def clear_current_turn(self, completed_iteration_num: int) -> None:
        """
        Clears entries from the specified 'completed_iteration_num' and older,
        unless their content_type is in `self.persistent_content_types`.
        This is typically called after an iteration is fully processed, before starting the next.
        """
        with self._lock:
            ids_to_remove = []
            for entry_id, entry in self._memory.items():
                # Entry must have an iteration number to be considered for turn-based clearing
                if entry.iteration is not None and entry.iteration <= completed_iteration_num:
                    if entry.content_type not in self.persistent_content_types:
                        ids_to_remove.append(entry_id)
            
            for entry_id in ids_to_remove:
                if entry_id in self._memory: # Check if still exists (relevant for concurrent access, though lock helps)
                    del self._memory[entry_id]
                    self._stats['remove_count'] += 1
            
            self._stats['clear_turn_count'] += 1
            if self.debug:
                self.logger.info(f"Cleared turn data up to iteration {completed_iteration_num}. Removed {len(ids_to_remove)} non-persistent entries. Kept types: {self.persistent_content_types}.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the blackboard content to a dictionary."""
        with self._lock:
            from dataclasses import asdict # For converting dataclasses
            
            mem_list = []
            for entry_obj in self._memory.values():
                try:
                    data_repr = entry_obj.data
                    if isinstance(data_repr, AgentOutput): # AgentOutput is a dataclass
                        data_repr = asdict(data_repr)
                    # For other custom objects in entry_obj.data, they might need a .to_dict() method
                    
                    mem_list.append(asdict(entry_obj)) # Convert BlackboardEntry to dict
                except Exception as e:
                    if self.debug:
                        self.logger.error(f"Error serializing entry {entry_obj.entry_id} for to_dict: {e}", exc_info=True)
                    mem_list.append({"entry_id": entry_obj.entry_id, "error_serializing": str(e)})

            return {
                "memory_entries": mem_list,
                "stats": self._stats.copy()
            }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserializes data from a dictionary to populate the blackboard."""
        with self._lock:
            self.clear_all() 
            
            memory_entries_dicts = data.get("memory_entries", [])
            for entry_dict in memory_entries_dicts:
                if "error_serializing" in entry_dict: continue # Skip problematic entries

                # Reconstruct AgentOutput if it's the content type
                entry_data_dict = entry_dict.get('data')
                if entry_dict.get('content_type') == CONTENT_TYPE_AGENT_OUTPUT and isinstance(entry_data_dict, dict):
                    try:
                        # Create AgentOutput from its dict representation
                        # This assumes keys in entry_data_dict match AgentOutput fields.
                        current_data = AgentOutput(**entry_data_dict)
                    except TypeError as te: # Handles missing fields or extra fields if AgentOutput __init__ is strict
                        if self.debug:
                            self.logger.error(f"from_dict: TypeError reconstructing AgentOutput: {te}, data: {entry_data_dict}", exc_info=True)
                        current_data = entry_data_dict # Store as dict if reconstruction fails
                else:
                    current_data = entry_data_dict # For other data types

                # Create BlackboardEntry, excluding 'data' for a moment
                # then set data to handle the potentially reconstructed AgentOutput
                bb_entry_args = {k: v for k, v in entry_dict.items() if k != 'data'}
                try:
                    entry = BlackboardEntry(**bb_entry_args)
                    entry.data = current_data # Assign potentially reconstructed data
                    self._memory[entry.entry_id] = entry
                except TypeError as te:
                     if self.debug:
                        self.logger.error(f"from_dict: TypeError reconstructing BlackboardEntry: {te}, args: {bb_entry_args}", exc_info=True)


            self._stats = data.get("stats", self._stats) 
            if self.debug:
                self.logger.info(f"Blackboard populated from dict. Loaded {len(self._memory)} entries.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns current operational statistics."""
        with self._lock:
            return {
                **self._stats.copy(),
                'memory_entries_count': len(self._memory),
            }

    def log_blackboard_contents(self, full_data: bool = False) -> None:
        """Logs the current contents of the blackboard if debug is enabled."""
        if not self.debug: 
            return
            
        with self._lock:
            self.logger.debug("\n--- Blackboard Contents Start ---")
            if not self._memory:
                self.logger.debug(" [Empty]")
            else:
                for i, entry in enumerate(self._memory.values()):
                    data_preview = str(entry.data)
                    if not full_data and len(data_preview) > 70: # Adjusted preview length
                        data_preview = data_preview[:70] + "..."
                    
                    log_message = (
                        f" Entry {i+1}: ID={entry.entry_id}, Iter={entry.iteration}, "
                        f"TS={entry.timestamp:.0f}, Src='{entry.source}', Type='{entry.content_type}'\n"
                        f"     Data: {data_preview}"
                    )
                    self.logger.debug(log_message)
            self.logger.debug("--- Blackboard Contents End ---\n")

# Example Usage (illustrative, not run by default)
if __name__ == '__main__':
    # This block would require a config setup for get_config() to work.
    # For standalone testing, you'd mock get_config() or ensure config.yaml is available and loaded.
    
    # Assuming config is loaded and provides a logger for Blackboard:
    # class MockConfigManagerForBB:
    #     def __init__(self):
    #         self.logging = type('LoggingConfig', (), {'debug': True})()
    #         self.memory = type('MemoryConfig', (), {'blackboard_history_limit': 100})() # Example if needed
    #     def get_logger(self, name): return logging.getLogger(name) # Basic logger

    # get_config(MockConfigManagerForBB()) # Mock get_config behavior for this example

    # bb = Blackboard()
    # bb.log_blackboard_contents()
    
    # ao = AgentOutput(agent_id="test_agent", text="Hello from agent", role="tester", iteration=0)
    # bb.add_agent_output(ao)
    # bb.add_user_input(text="Test user input", iteration=0, raw_input_details={"raw":"Test user input", "normalized":"test user input"})
    # bb.log_blackboard_contents(full_data=True)

    # print(f"Stats: {bb.get_stats()}")
    pass
