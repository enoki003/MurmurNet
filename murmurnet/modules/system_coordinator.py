#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System Coordinator モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分散SLMシステム全体の調整・制御を担当
責務の分離によりDistributedSLMの肥大化を防ぐ

作者: Yuhi Sonoki
"""

import logging
import asyncio
import time
import multiprocessing
import json
import copy # For deepcopy, if not already present from previous undone refactor
from typing import Dict, Any, List, Optional, Tuple

from MurmurNet.modules.common import MurmurNetError, AgentExecutionError
from MurmurNet.modules.config_manager import get_config
from MurmurNet.modules.performance import time_async_function
# Import new data structures and blackboard constants
from MurmurNet.modules.data_structures import AgentOutput
from MurmurNet.modules.blackboard import (
    CONTENT_TYPE_USER_INPUT, CONTENT_TYPE_SUMMARY, CONTENT_TYPE_ERROR, 
    CONTENT_TYPE_RAG_RESULT, CONTENT_TYPE_CONVERSATION_CONTEXT, CONTENT_TYPE_KEY_FACTS,
    SOURCE_USER, SOURCE_SYSTEM, SOURCE_AGENT_PREFIX, SOURCE_SUMMARY_ENGINE, 
    SOURCE_RAG_RETRIEVER, SOURCE_CONVERSATION_MEMORY, SOURCE_AGENT_POOL_MANAGER,
    SOURCE_OUTPUT_AGENT
)
# Import project constants
from .. import constants as const 

logger = logging.getLogger('MurmurNet.SystemCoordinator')


def _execute_agent_in_process(task_data: Dict[str, Any]) -> Dict[str, Any]: # Signature changed
    """
    別プロセスでエージェントを実行するワーカー関数（グローバル関数）
    Refactored to use new Blackboard structure and AgentOutput.
    
    引数:
        task_data: {
            'agent_id': str, # Changed to str for consistency if AgentPoolManager uses str IDs
            'config_dict': Dict[str, Any],
            'blackboard_snapshot': Dict[str, Any], # From new Blackboard.to_dict()
            'original_input_text': str, # For agent_pool.update_roles_based_on_question
            'iteration_num_for_worker': int
        }
        
    戻り値:
        実行結果辞書, response field contains AgentOutput object on success
    """
    agent_id_str = str(task_data['agent_id']) # Ensure string ID
    config_dict = task_data['config_dict']
    blackboard_snapshot = task_data['blackboard_snapshot']
    original_input_text = task_data['original_input_text']
    current_iteration = task_data['iteration_num_for_worker']

    try:
        # プロセス内でモジュールを再インポート
        from MurmurNet.modules.config_manager import ConfigManager # get_config might not be needed if passing dict
        from MurmurNet.modules.blackboard import Blackboard
        from MurmurNet.modules.agent_pool import AgentPoolManager
        
        # 設定を初期化 (ConfigManager might need to accept a dict directly)
        # For now, assuming get_config can be seeded or works globally post-initialization.
        # A better way might be: config_manager = ConfigManager(config_dict=config_dict)
        # This depends on ConfigManager's design. Using get_config(config_dict) as in previous attempts.
        from MurmurNet.modules.config_manager import get_config as get_config_process_local
        config_manager_instance = get_config_process_local(config_dict)


        # 黒板を初期化し、スナップショットをロード
        blackboard_instance = Blackboard() # Uses its own get_config()
        if blackboard_snapshot:
            blackboard_instance.from_dict(blackboard_snapshot) # Use new from_dict
        
        # AgentPoolManagerを初期化 (Passes its own config_manager and blackboard)
        # AgentPoolManager now uses StandardAgent internally, which returns AgentOutput
        agent_pool_instance = AgentPoolManager(config=config_manager_instance.config, blackboard=blackboard_instance)
        
        # 必要に応じてロールを更新 (This was part of a previous refactor of SystemCoordinator)
        if hasattr(agent_pool_instance, 'update_roles_based_on_question'):
            agent_pool_instance.update_roles_based_on_question(original_input_text)
        
        # エージェントタスクを実行
        # Assuming AgentPoolManager will have execute_agent_task(agent_id_str, current_iteration)
        # and this method will return an AgentOutput object.
        # The old _agent_task from original agent_pool.py returned a string.
        # エージェントタスクを実行 by directly calling the agent's generate_response method.
        agent_instance = agent_pool_instance.agents.get(agent_id_str)

        if not agent_instance:
            logger.error(f"Agent {agent_id_str} not found in agent_pool_instance.agents.")
            # Return error structure, consistent with how other errors are returned
            return {
                'agent_id': agent_id_str,
                'success': False,
                'response': None,
                'error': const.AGENT_NOT_FOUND_ERROR.format(agent_id=agent_id_str)
            }

        # Call StandardAgent.generate_response directly
        # Pass None for other_agent_outputs_str as per subtask instructions for simplification
        agent_output_obj = agent_instance.generate_response(
            current_iteration=current_iteration,
            other_agent_outputs_str=None 
        )
        
        # Determine success based on the content of agent_output_obj.text
        # For example, if text indicates an error, success should be False.
        # The StandardAgent.generate_response is already designed to put error messages into its text.
        # Here, 'success' means the process executed the agent's generation without system error.
        # The content of agent_output_obj.text is the actual agent's response or its internally caught error.
        is_successful_execution = True # Assume success unless agent_output_obj indicates otherwise (e.g. specific error text pattern)
        # Example check (can be refined):
        if "エラーにより応答できませんでした" in agent_output_obj.text or "empty response" in agent_output_obj.text.lower() or "空の応答を生成しました" in agent_output_obj.text:
             # This depends on how StandardAgent formats its internal errors into the text.
             # For now, we assume if generate_response completes, the call itself was "successful" at this level.
             # The content of agent_output_obj.text will carry the actual success/failure/content of the agent's thinking.
             pass


        return {
            'agent_id': agent_id_str,
            'success': is_successful_execution, # Reflects execution success, not necessarily quality of response
            'response': agent_output_obj,   # This is the AgentOutput object from agent_instance.generate_response
            'error': None if is_successful_execution else agent_output_obj.text # If not successful, error can be the text
        }
        
    except Exception as e:
        logger.error(f"エージェント {agent_id_str} の実行でエラー: {e}", exc_info=True)
        # Return structure for error
        return {
            'agent_id': agent_id_str,
            'success': False,
            'response': None, 
            'error': str(e)
        }


class SystemCoordinator:
    """
    システム全体の調整・制御クラス
    
    責務:
    - 反復処理の制御
    - エージェント実行の調整
    - エラー回復処理
    - パフォーマンス監視
    """
    def __init__(self, config: Dict[str, Any] = None, blackboard=None, agent_pool=None, summary_engine=None):
        """
        システム調整器の初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
            blackboard: 共有黒板
            agent_pool: エージェントプール
            summary_engine: 要約エンジン
        """        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
        self.blackboard = blackboard
        self.agent_pool = agent_pool
        self.summary_engine = summary_engine
        
        # ConfigManagerから設定パラメータを取得
        self.num_agents = self.config_manager.agent.num_agents
        self.iterations = self.config_manager.agent.iterations
        self.use_summary = self.config_manager.agent.use_summary
        self.use_parallel = self.config_manager.agent.use_parallel
          # エラー回復設定（デフォルト値）
        self.max_retry_attempts = 2
        self.failed_agents_threshold = 0.5  # 50%        logger.info(f"システム調整器を初期化しました: {self.num_agents}エージェント, {self.iterations}反復")
        logger.info(f"並列処理モード: {'プロセスベース' if self.use_parallel else '逐次実行'}")

    @time_async_function
    async def run_iteration(self, current_iteration_num: int) -> bool: # Signature changed
        """
        単一の反復サイクルを実行
        
        引数:
            current_iteration_num: 現在の反復インデックス (0-indexed)
            
        戻り値:
            成功した場合True
            
        例外:
            MurmurNetError: システムエラー
        """
        logger.info(f"反復 {current_iteration_num + 1}/{self.iterations} を開始")
        
        try:
            # 0. Clear non-persistent entries from previous turns.
            if current_iteration_num > 0:
                self.blackboard.clear_current_turn(current_iteration_num - 1)
                logger.info(f"Blackboard: Cleared non-persistent entries from iteration {current_iteration_num -1}.")
            elif current_iteration_num == 0: # First iteration
                self.blackboard.clear_all()
                logger.info("Blackboard: Cleared all entries for initial iteration.")

            # 1. エージェント実行（並列または逐次）
            if self.use_parallel:
                success = await self._run_agents_parallel(current_iteration_num) # Pass iteration
            else:
                success = self._run_agents_sequential(current_iteration_num) # Pass iteration
            
            if not success:
                logger.warning(f"反復 {current_iteration_num + 1} でエージェント実行に問題が発生しました")
                return False
            
            # 2. エージェント出力収集 for summary (using new blackboard methods)
            # This retrieves AgentOutput objects directly.
            agent_outputs_for_summary = self.blackboard.get_agent_outputs(iteration=current_iteration_num)
            
            # Format for summary_engine if it expects List[Dict[str, Any]] with "text"
            agent_texts_for_summary = [
                {"agent": ao.agent_id, "text": ao.text} 
                for ao in agent_outputs_for_summary 
                if ao.text and not ao.text.endswith("応答できませんでした") # Filter out non-responses / errors
            ]
            
            # 3. 出力の要約（使用する場合）
            if self.use_summary and agent_texts_for_summary:
                # Pass original AgentOutput objects as well, for potential use in _create_iteration_summary (e.g. for source_ids)
                await self._create_iteration_summary(current_iteration_num, agent_texts_for_summary, agent_outputs_for_summary)
            
            logger.info(f"反復 {current_iteration_num + 1} が正常に完了しました")
            return True
            
        except Exception as e:
            logger.error(f"反復 {current_iteration_num + 1} でエラーが発生しました: {e}", exc_info=True)
            # Add a generic error entry to the blackboard for this iteration
            if self.blackboard:
                self.blackboard.add_generic_entry(
                    source=SOURCE_SYSTEM,
                    content_type=CONTENT_TYPE_ERROR,
                    data=const.UNEXPECTED_ERROR_MSG_TEMPLATE.format(role_name=f"run_iteration {current_iteration_num + 1}", error=str(e)),
                    iteration=current_iteration_num
                )
            raise MurmurNetError(f"反復処理エラー: {e}") # This is a custom exception, message can be generic

    async def _run_agents_parallel(self, current_iteration_num: int) -> bool: # Added current_iteration_num
        """
        エージェントをプロセスベース並列実行（GGML assertion error回避）
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info(f"エージェントをプロセスベース並列実行中 (反復 {current_iteration_num + 1})")
        
        try:
            # 黒板スナップショットを準備 using the new to_dict() method
            blackboard_snapshot_dict = self.blackboard.to_dict() if self.blackboard else {}
            
            # 元のユーザー入力を取得 (AgentPoolManagerが質問タイプ分類などに使用)
            user_input_entry = self.blackboard.get_user_input(iteration=current_iteration_num)
            original_input_text = ""
            if user_input_entry and user_input_entry.data:
                # Assuming user_input_entry.data is a dict like {'normalized': "...", 'raw': "..."}
                if isinstance(user_input_entry.data, dict):
                    original_input_text = user_input_entry.data.get('normalized', "")
                else: # Fallback if data is just a string (older format)
                    original_input_text = str(user_input_entry.data)
            
            if not original_input_text and self.debug: # Log if no input found, can be normal in some cases
                 logger.debug(f"No user input text found on blackboard for iteration {current_iteration_num} to guide agent role update in parallel run.")

            # 設定辞書を取得し、ワーカープロセス用に変更 (This part was from a previous refactor for oversubscription)
            base_config_dict = self.config_manager.to_dict()
            worker_config_dict = copy.deepcopy(base_config_dict)
            if 'agent' not in worker_config_dict: worker_config_dict['agent'] = {}
            worker_config_dict['agent']['use_parallel'] = False # Prevent oversubscription

            # タスクデータを準備
            tasks = []
            for i in range(self.num_agents):
                task_data = {
                    'agent_id': str(i), # Use string agent IDs
                    'config_dict': worker_config_dict, 
                    'blackboard_snapshot': blackboard_snapshot_dict, 
                    'original_input_text': original_input_text,
                    'iteration_num_for_worker': current_iteration_num 
                }
                tasks.append(task_data)
            
            # プロセスプールで並列実行
            start_time = time.time()
            
            # CPUコア数に基づいてプロセス数を決定
            num_processes = min(self.num_agents, multiprocessing.cpu_count())
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_execute_agent_in_process, tasks)
            
            execution_time = time.time() - start_time
            
            # 結果を処理
            successful_agents = 0
            for result in results: # result is a dict from _execute_agent_in_process
                agent_id_str = str(result['agent_id']) 
                if result['success'] and result.get('response'):
                    agent_output_obj: AgentOutput = result['response'] # This is an AgentOutput object
                    
                    # Ensure iteration is set if not already by the worker (it should be)
                    if agent_output_obj.iteration is None:
                        agent_output_obj.iteration = current_iteration_num
                    
                    self.blackboard.add_agent_output(agent_output_obj)
                    successful_agents += 1
                else:
                    error_msg = result.get('error', '不明なエラー')
                    logger.error(f"エージェント {agent_id_str} (反復 {current_iteration_num + 1}) の並列実行でエラー: {error_msg}")
                    self.blackboard.add_generic_entry(
                        source=f"{SOURCE_AGENT_PREFIX}{agent_id_str}",
                        content_type=CONTENT_TYPE_ERROR,
                        data=const.AGENT_EXECUTION_ERROR_MSG_TEMPLATE.format(role_name=f"エージェント{agent_id_str}", error_message=error_msg),
                        iteration=current_iteration_num
                    )
            
            success_rate = successful_agents / self.num_agents if self.num_agents > 0 else 0.0
            
            # パフォーマンス情報をログ出力
            logger.info(f"プロセスベース並列実行完了: {execution_time:.2f}秒")
            logger.info(f"成功率 {success_rate:.2%} ({successful_agents}/{self.num_agents})")
            logger.info(f"使用プロセス数: {num_processes}")
            
            return success_rate >= self.failed_agents_threshold
            
        except Exception as e:
            logger.error(f"プロセスベース並列実行全体でエラー: {e}", exc_info=True)
            # エラー時は全エージェントにエラーメッセージを設定
            for i in range(self.num_agents):
                agent_id_str = str(i)
                self.blackboard.add_generic_entry(
                    source=f"{SOURCE_AGENT_PREFIX}{agent_id_str}",
                    content_type=CONTENT_TYPE_ERROR,
                    data=const.UNEXPECTED_ERROR_MSG_TEMPLATE.format(role_name=f"エージェント{agent_id_str} (parallel run)", error=str(e)),
                    iteration=current_iteration_num
                )
            return False
    
    def _run_agents_sequential(self, current_iteration_num: int) -> bool: # Added current_iteration_num
        """
        エージェントを逐次実行
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info(f"エージェントを逐次実行中 (反復 {current_iteration_num + 1})")
        
        success_count = 0
        for i in range(self.num_agents):
            agent_id_str = str(i)
            agent_output_obj = None # Initialize for this scope
            try:
                agent_instance = self.agent_pool.agents.get(agent_id_str)

                if not agent_instance:
                    logger.error(f"Sequential execution: Agent {agent_id_str} not found in agent_pool.")
                    error_text = const.AGENT_NOT_FOUND_ERROR.format(agent_id=agent_id_str)
                    agent_output_obj = AgentOutput(
                        agent_id=agent_id_str,
                        text=error_text,
                        role=const.ROLE_SYSTEM_ERROR, 
                        iteration=current_iteration_num
                    )
                    # Fall through to add_agent_output and non-success counting
                else:
                    # Call StandardAgent.generate_response directly.
                    # StandardAgent._format_prompt_for_agent will handle fetching prior agent outputs
                    # from the blackboard for the current iteration if its template needs them.
                    agent_output_obj = agent_instance.generate_response(
                        current_iteration=current_iteration_num,
                        other_agent_outputs_str=None # Agent will fetch prior outputs from blackboard itself
                    )
                
                # Store the AgentOutput object (success or error) to the blackboard
                if agent_output_obj: # Should always be true due to error handling above
                    self.blackboard.add_agent_output(agent_output_obj)

                # Determine success based on the content of the response
                if agent_output_obj and agent_output_obj.text and \
                   "エラーにより応答できませんでした" not in agent_output_obj.text and \
                   "応答を生成できませんでした" not in agent_output_obj.text and \
                   "空の応答を生成しました" not in agent_output_obj.text and \
                   "見つかりません" not in agent_output_obj.text: # From agent not found case
                    success_count += 1
                    logger.debug(f"エージェント{agent_id_str}の逐次実行が完了し、成功とみなされました。")
                else:
                    logger.warning(f"エージェント{agent_id_str}の逐次実行が完了しましたが、応答内容から失敗/エラーと判断されました。Text: {agent_output_obj.text[:const.DEFAULT_MAX_RESPONSE_SNIPPET_LOG_LENGTH] if agent_output_obj else 'N/A'}")

            except Exception as e: # Catch errors from agent_instance.generate_response or other unexpected issues
                logger.error(f"エージェント{agent_id_str}の逐次実行中に予期しないエラー: {e}", exc_info=True)
                # Ensure an error AgentOutput is created and added to the blackboard
                error_text = const.UNEXPECTED_ERROR_MSG_TEMPLATE.format(role_name=f"エージェント{agent_id_str} (sequential)", error=str(e))
                agent_output_obj = AgentOutput(
                    agent_id=agent_id_str,
                    text=error_text,
                    role=const.ROLE_SYSTEM_ERROR, # More generic error role
                    iteration=current_iteration_num
                )
                self.blackboard.add_agent_output(agent_output_obj)
                # This agent's execution is considered a failure.
        
        success_rate = success_count / self.num_agents if self.num_agents > 0 else 0.0
        logger.info(f"逐次実行完了: 成功率 {success_rate:.2%} ({success_count}/{self.num_agents})")
        
        return success_rate >= self.failed_agents_threshold

    async def _run_single_agent_optimized(self, agent_id: int, current_iteration_num: int) -> bool: # Added current_iteration_num
        """
        単一エージェントを最適化された並列実行
        
        引数:
            agent_id: エージェントID
            current_iteration_num: 現在の反復数
            
        戻り値:
            成功した場合True
        """
        agent_id_str = str(agent_id)
        agent_output_obj = None # Initialize
        try:
            # AgentPoolManager.run_agent_parallel is expected to be an async method
            # that internally calls execute_agent_task (which calls agent.generate_response)
            # and returns an AgentOutput object.
            # It also handles writing to the blackboard.
            agent_output_obj = await self.agent_pool.run_agent_parallel(agent_id_str, current_iteration_num) 
            
            if agent_output_obj and agent_output_obj.text and \
               "エラーにより応答できませんでした" not in agent_output_obj.text and \
               "応答を生成できませんでした" not in agent_output_obj.text and \
               "空の応答を生成しました" not in agent_output_obj.text and \
               "見つかりません" not in agent_output_obj.text:
                logger.debug(f"エージェント{agent_id_str}の最適化並列実行が完了しました。")
                return True
            else:
                # Error text is already in agent_output_obj.text and it's on blackboard.
                logger.warning(f"エージェント{agent_id_str}が有効な結果を返しませんでした（最適化並列実行）。Text: {agent_output_obj.text[:100] if agent_output_obj else 'N/A'}")
                return False
                
        except Exception as e:
            logger.error(f"エージェント{agent_id_str}最適化並列実行中に予期せぬ例外: {e}", exc_info=True)
            # Ensure an error output is on the blackboard if not already handled by run_agent_parallel's internals
            if not agent_output_obj or agent_output_obj.text.find("エラー") == -1 : # Avoid double logging if already an error AO
                error_text = f"最適化並列実行中に予期せぬシステムエラー: {str(e)}"
                final_error_ao = AgentOutput(
                    agent_id=agent_id_str, 
                    text=error_text, 
                    role=const.ROLE_SYSTEM_ERROR + "_Optimized", 
                    iteration=current_iteration_num
                )
                self.blackboard.add_agent_output(final_error_ao)
            return False

    async def _run_single_agent_async(self, agent_id: int, current_iteration_num: int) -> bool:
        """
        単一エージェントを非同期実行
        
        引数:
            agent_id: エージェントID
            current_iteration_num: 現在の反復数
            
        戻り値:
            成功した場合True
        """
        agent_id_str = str(agent_id)
        try:
            agent_id_str = str(agent_id) 
            agent_output_obj = None 
            agent_instance = self.agent_pool.agents.get(agent_id_str)

            if not agent_instance:
                logger.error(f"Async run: Agent {agent_id_str} not found.")
                error_text = const.AGENT_NOT_FOUND_ERROR.format(agent_id=agent_id_str) + "（非同期実行）"
                agent_output_obj = AgentOutput(agent_id=agent_id_str, text=error_text, role=const.ROLE_SYSTEM_ERROR, iteration=current_iteration_num)
            else:
                # StandardAgent.generate_response is synchronous.
                # We use run_in_executor to make this part awaitable.
                loop = asyncio.get_event_loop()
                agent_output_obj = await loop.run_in_executor(
                    None, 
                    agent_instance.generate_response, # Method to call
                    current_iteration_num, # First arg to generate_response
                    None # Second arg (other_agent_outputs_str) for generate_response
                )

            self.blackboard.add_agent_output(agent_output_obj)

            if agent_output_obj.text and "エラーにより応答できませんでした" not in agent_output_obj.text and "応答を生成できませんでした" not in agent_output_obj.text:
                logger.debug(f"エージェント{agent_id_str}の非同期実行が完了しました。")
                return True
            else:
                logger.warning(f"エージェント{agent_id_str}が有効な結果を返しませんでした（非同期実行）。Text: {agent_output_obj.text[:const.DEFAULT_MAX_RESPONSE_SNIPPET_LOG_LENGTH]}")
                return False
                
        except Exception as e:
            logger.error(f"エージェント{agent_id_str}非同期実行中に予期せぬ例外: {e}", exc_info=True)
            error_text = const.UNEXPECTED_ERROR_MSG_TEMPLATE.format(role_name=f"エージェント{agent_id_str} (async)", error=str(e))
            if not agent_output_obj : 
                 agent_output_obj = AgentOutput(agent_id=agent_id_str, text=error_text, role=const.ROLE_SYSTEM_ERROR + "_Async", iteration=current_iteration_num)
            else: 
                agent_output_obj.text = error_text 
            self.blackboard.add_agent_output(agent_output_obj) 
            return False
    
    # _collect_agent_outputs method is now removed as its functionality 
    # is integrated into run_iteration using blackboard.get_agent_outputs().

    async def _create_iteration_summary(self, current_iteration_num: int, agent_texts_for_summary: List[Dict[str, Any]], original_agent_outputs: List[AgentOutput]) -> None: # Signature updated
        """
        反復の要約を作成
        
        引数:
            current_iteration_num: 反復インデックス
            agent_texts_for_summary: エージェント出力のテキストリスト (for summary_engine)
            original_agent_outputs: 元のAgentOutputオブジェクトのリスト (for source_ids)
        """
        try:
            start_time = time.time()
            # summary_engine.summarize_blackboard might expect List[Dict[str, Any]] with "text" key
            summary_text = self.summary_engine.summarize_blackboard(agent_texts_for_summary)
            
            # Use agent_ids from the original_agent_outputs as source_ids for the summary.
            # A more robust approach might involve entry_ids if AgentOutput objects carried them.
            source_ids_for_summary = [ao.agent_id for ao in original_agent_outputs]

            self.blackboard.add_summary(
                summary_text=summary_text, 
                iteration=current_iteration_num, 
                source_ids=source_ids_for_summary,
                source=SOURCE_SUMMARY_ENGINE # Explicitly set source
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"反復 {current_iteration_num + 1} の要約を作成しました (実行時間: {elapsed_time:.4f}秒)")
            
        except Exception as e:
            logger.error(f"反復 {current_iteration_num + 1} の要約作成エラー: {e}", exc_info=True)
            self.blackboard.add_generic_entry(
                source=SOURCE_SUMMARY_ENGINE, 
                content_type=CONTENT_TYPE_ERROR,
                data=const.UNEXPECTED_ERROR_MSG_TEMPLATE.format(role_name="SummaryEngine", error=str(e)),
                iteration=current_iteration_num
            )
    
    # _build_common_prompt method is removed as StandardAgent now formats its own prompts,
    # and it was confirmed to be no longer used by the primary agent execution flows.
    # _write_results_to_blackboard method was also removed previously.

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        実行統計を取得
        
        戻り値:
            実行統計の辞書
        """
        return {
            "num_agents": self.num_agents,
            "iterations": self.iterations,
            "parallel_mode": self.use_parallel,
            "summary_enabled": self.use_summary,
            "failed_agents_threshold": self.failed_agents_threshold,
            "max_retry_attempts": self.max_retry_attempts
        }