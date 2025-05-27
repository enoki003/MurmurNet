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
from typing import Dict, Any, List, Optional, Tuple
from MurmurNet.modules.common import MurmurNetError, AgentExecutionError
from MurmurNet.modules.config_manager import get_config
from MurmurNet.modules.performance import time_async_function

logger = logging.getLogger('MurmurNet.SystemCoordinator')


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
        self.failed_agents_threshold = 0.5  # 50%
        logger.info(f"システム調整器を初期化しました: {self.num_agents}エージェント, {self.iterations}反復")
        logger.info(f"並列処理モード: {'プロセスベース' if self.use_parallel else '逐次実行'}")

    def _execute_agent_in_process(self, task_data: Tuple[int, str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        別プロセスでエージェントを実行するワーカー関数
        
        引数:
            task_data: (agent_id, prompt, blackboard_data) のタプル
            
        戻り値:
            実行結果辞書
        """
        agent_id, prompt, blackboard_data = task_data
        
        try:
            # プロセス内でモジュールを再インポート
            from MurmurNet.modules.config_manager import get_config
            from MurmurNet.modules.model_factory import ModelFactory
            
            # 設定を取得
            config_manager = get_config()
            
            # モデルを初期化
            model_factory = ModelFactory()
            model = model_factory.create_model(config_manager.model_type)
            
            # プロンプト実行
            response = model.generate(prompt)
            
            return {
                'agent_id': agent_id,
                'success': True,
                'response': response,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"エージェント {agent_id} の実行でエラー: {e}")
            return {
                'agent_id': agent_id,
                'success': False,
                'response': None,
                'error': str(e)
            }
    
    @time_async_function
    async def run_iteration(self, iteration: int) -> bool:
        """
        単一の反復サイクルを実行
        
        引数:
            iteration: 現在の反復インデックス
            
        戻り値:
            成功した場合True
            
        例外:
            MurmurNetError: システムエラー
        """
        logger.info(f"反復 {iteration+1}/{self.iterations} を開始")
        
        try:
            # 1. エージェント実行（並列または逐次）
            if self.use_parallel:
                success = await self._run_agents_parallel()
            else:
                success = self._run_agents_sequential()
            
            if not success:
                logger.warning(f"反復 {iteration+1} でエージェント実行に問題が発生しました")
                return False
            
            # 2. エージェント出力収集
            agent_entries = self._collect_agent_outputs()
            
            # 3. 出力の要約（使用する場合）
            if self.use_summary and agent_entries:
                await self._create_iteration_summary(iteration, agent_entries)
            
            logger.info(f"反復 {iteration+1} が正常に完了しました")
            return True
        except Exception as e:
            logger.error(f"反復 {iteration+1} でエラーが発生しました: {e}")
            raise MurmurNetError(f"反復処理エラー: {e}")

    async def _run_agents_parallel(self) -> bool:
        """
        エージェントをプロセスベース並列実行（GGML assertion error回避）
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info("エージェントをプロセスベース並列実行中...")
        
        try:
            # 現在のプロンプトを構築
            prompt = self._build_common_prompt()
            
            # 黒板データを準備
            blackboard_data = {}
            if self.blackboard:
                # 必要な黒板データを抽出
                for key in ['user_input', 'conversation_context', 'search_results']:
                    value = self.blackboard.read(key)
                    if value:
                        blackboard_data[key] = value
            
            # タスクデータを準備
            tasks = []
            for i in range(self.num_agents):
                task_data = (i, prompt, blackboard_data)
                tasks.append(task_data)
            
            # プロセスプールで並列実行
            start_time = time.time()
            
            # CPUコア数に基づいてプロセス数を決定
            num_processes = min(self.num_agents, multiprocessing.cpu_count())
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(self._execute_agent_in_process, tasks)
            
            execution_time = time.time() - start_time
            
            # 結果を処理
            successful_agents = 0
            for result in results:
                agent_id = result['agent_id']
                if result['success']:
                    successful_agents += 1
                    # 成功した場合は結果を黒板に書き込み
                    self.blackboard.write(f'agent_{agent_id}_output', result['response'])
                else:
                    # 失敗した場合はエラーメッセージを黒板に書き込み
                    error_msg = f"エージェント{agent_id}はエラーにより応答できませんでした: {result['error']}"
                    self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            
            success_rate = successful_agents / self.num_agents
            
            # パフォーマンス情報をログ出力
            logger.info(f"プロセスベース並列実行完了: {execution_time:.2f}秒")
            logger.info(f"成功率 {success_rate:.2%} ({successful_agents}/{self.num_agents})")
            logger.info(f"使用プロセス数: {num_processes}")
            
            return success_rate >= self.failed_agents_threshold
            
        except Exception as e:
            logger.error(f"プロセスベース並列実行エラー: {e}")
            # エラー時は全エージェントにエラーメッセージを設定
            for i in range(self.num_agents):
                self.blackboard.write(f'agent_{i}_output', f"エージェント{i}はプロセス実行エラーにより応答できませんでした")
            return False
    
    def _run_agents_sequential(self) -> bool:
        """
        エージェントを逐次実行
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info("エージェントを逐次実行中...")
        
        success_count = 0
        for i in range(self.num_agents):
            try:
                # エージェントタスクを実行
                result = self.agent_pool._agent_task(i)
                if result and not result.endswith("応答できませんでした"):
                    success_count += 1
                    logger.debug(f"エージェント{i}の逐次実行が完了しました")
                
            except AgentExecutionError as e:
                logger.error(f"エージェント{i}実行エラー: {e}")
                # エラーメッセージは既にagent_poolで黒板に書き込み済み
                
            except Exception as e:
                logger.error(f"エージェント{i}予期しないエラー: {e}")
                self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は予期しないエラーにより応答できませんでした")
        
        success_rate = success_count / self.num_agents
        logger.info(f"逐次実行完了: 成功率 {success_rate:.2%} ({success_count}/{self.num_agents})")
        
        return success_rate >= self.failed_agents_threshold

    async def _run_single_agent_optimized(self, agent_id: int) -> bool:
        """
        単一エージェントを最適化された並列実行
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            成功した場合True
        """
        try:
            # 並列実行に最適化されたエージェントタスクを実行
            result = await self.agent_pool.run_agent_parallel(agent_id)
            
            if result and not result.endswith("応答できませんでした"):
                logger.debug(f"エージェント{agent_id}の最適化並列実行が完了しました")
                return True
            else:
                logger.warning(f"エージェント{agent_id}が有効な結果を返しませんでした")
                return False
                
        except Exception as e:
            logger.error(f"エージェント{agent_id}最適化並列実行エラー: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', f"エージェント{agent_id}は並列実行エラーにより応答できませんでした")
            return False

    async def _run_single_agent_async(self, agent_id: int) -> bool:
        """
        単一エージェントを非同期実行
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            成功した場合True
        """
        try:
            # エージェントタスクを非同期で実行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.agent_pool._agent_task, agent_id)
            
            if result and not result.endswith("応答できませんでした"):
                logger.debug(f"エージェント{agent_id}の並列実行が完了しました")
                return True
            else:
                logger.warning(f"エージェント{agent_id}が有効な結果を返しませんでした")
                return False
                
        except Exception as e:
            logger.error(f"エージェント{agent_id}非同期実行エラー: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', f"エージェント{agent_id}は並列実行エラーにより応答できませんでした")
            return False
    
    def _collect_agent_outputs(self) -> List[Dict[str, Any]]:
        """
        エージェント出力を収集
        
        戻り値:
            エージェント出力のリスト
        """
        agent_entries = []
        for i in range(self.num_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            if agent_output and not agent_output.endswith("応答できませんでした"):
                agent_entries.append({"agent": i, "text": agent_output})
        
        logger.debug(f"有効なエージェント出力を{len(agent_entries)}個収集しました")
        return agent_entries
    
    async def _create_iteration_summary(self, iteration: int, agent_entries: List[Dict[str, Any]]) -> None:
        """
        反復の要約を作成
        
        引数:
            iteration: 反復インデックス
            agent_entries: エージェント出力のリスト
        """
        try:
            start_time = time.time()
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            self.blackboard.write(f'summary_{iteration}', summary)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"反復 {iteration+1} の要約を作成しました (実行時間: {elapsed_time:.4f}秒)")
            
        except Exception as e:
            logger.error(f"反復 {iteration+1} の要約作成エラー: {e}")
            self.blackboard.write(f'summary_{iteration}', "要約の作成中にエラーが発生しました")
    
    def _build_common_prompt(self) -> str:
        """
        共通プロンプトを構築
        
        戻り値:
            構築されたプロンプト
        """
        # 黒板からユーザー入力を取得
        user_input = self.blackboard.read('user_input') if self.blackboard else ""
        
        # 会話コンテキストを取得
        conversation_context = self.blackboard.read('conversation_context') if self.blackboard else ""
        
        # RAG検索結果を取得
        search_results = self.blackboard.read('search_results') if self.blackboard else ""
        
        # プロンプトを構築
        prompt_parts = []
        
        if conversation_context:
            prompt_parts.append(f"会話コンテキスト:\n{conversation_context}")
        
        if search_results:
            prompt_parts.append(f"検索結果:\n{search_results}")
        
        if user_input:
            prompt_parts.append(f"ユーザーの質問: {user_input}")
        else:
            prompt_parts.append("システム質問: あなたは分散SLMエージェントです。与えられた情報に基づいて有用な回答を生成してください。")
        return "\n\n".join(prompt_parts)
    
    def _write_results_to_blackboard(self, collected_results) -> None:
        """
        プロセス並列実行結果を黒板に書き込み
        
        引数:
            collected_results: 収集済み結果
        """
        if not self.blackboard:
            logger.warning("黒板が利用できないため、結果の書き込みをスキップします")
            return
        
        # 成功した結果を黒板に書き込み
        for result in collected_results.successful_results:
            self.blackboard.write(f'agent_{result.agent_id}_output', result.output)
          # 失敗した結果についてはエラーメッセージを書き込み
        for result in collected_results.failed_results:
            error_msg = f"エージェント{result.agent_id}はプロセス実行エラーにより応答できませんでした: {result.error}"
            self.blackboard.write(f'agent_{result.agent_id}_output', error_msg)
        
        logger.debug(f"プロセス並列実行結果を黒板に書き込み完了: 成功{len(collected_results.successful_results)}, 失敗{len(collected_results.failed_results)}")

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