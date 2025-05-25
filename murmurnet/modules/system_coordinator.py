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
from typing import Dict, Any, List, Optional
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
        """
        # ConfigManagerから設定を取得
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
        エージェントを並列実行
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info("エージェントを並列実行中...")
        
        # エージェントタスクを並列実行
        tasks = []
        for i in range(self.num_agents):
            task = asyncio.create_task(self._run_single_agent_async(i))
            tasks.append(task)
        
        # すべてのタスクを実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
          # 結果を分析
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"エージェント {i} 並列実行エラー: {result}")
                self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は並列実行エラーにより応答できませんでした")
            elif result:
                success_count += 1
        
        success_rate = success_count / self.num_agents
        logger.info(f"並列実行完了: 成功率 {success_rate:.2%} ({success_count}/{self.num_agents})")
        
        return success_rate >= self.failed_agents_threshold
    
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
            
            if result:
                self.blackboard.write(f'agent_{agent_id}_output', result)
                return True
            else:
                logger.warning(f"エージェント{agent_id}が空の結果を返しました")
                return False
                
        except Exception as e:
            logger.error(f"エージェント{agent_id}非同期実行エラー: {e}")
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
