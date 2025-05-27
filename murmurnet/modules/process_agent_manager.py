#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process Agent Manager モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
単一責任: プロセスベース並列エージェント実行の統合インターフェース

設計原則:
- 単一責任: 並列エージェント実行の調整のみ
- KISS: シンプルなワークフロー管理
- 分離: 各コンポーネントの責任を明確に分離

作者: Yuhi Sonoki
"""

import logging
import time
from typing import List, Dict, Any, Optional
from MurmurNet.modules.process_coordinator import ProcessCoordinator
from MurmurNet.modules.result_collector import ResultCollector, CollectedResults
from MurmurNet.modules.process_agent_worker import AgentTask, AgentResult
from MurmurNet.modules.config_manager import get_config


class ProcessAgentManager:
    """
    単一責任: プロセスベース並列エージェント実行の統合
    
    設計原則:
    - 各コンポーネントの責任を明確に分離
    - シンプルなワークフロー管理
    - 明確なエラーハンドリング
    """
    
    def __init__(self, num_processes: int = None):
        """
        プロセスエージェントマネージャーの初期化
        
        引数:
            num_processes: プロセス数（None時は自動設定）
        """
        self.config_manager = get_config()
        self.logger = logging.getLogger('ProcessAgentManager')
        
        # 責任分離：各コンポーネントを独立して初期化
        self.coordinator = ProcessCoordinator(num_processes)
        self.collector = ResultCollector()
        
        # 設定取得（ConfigManagerから）
        self.config_dict = self.config_manager.to_dict()
        
        self.logger.info(f"ProcessAgentManager initialized with {self.coordinator.num_processes} processes")
    
    def execute_agents_parallel(self, prompts: List[str], roles: List[str] = None, 
                               blackboard_data: Dict[str, Any] = None) -> CollectedResults:
        """
        エージェントを並列実行（KISS原則：シンプルなワークフロー）
        
        引数:
            prompts: プロンプトリスト
            roles: ロールリスト（省略時はdefault）
            blackboard_data: 共有データ（省略時は空辞書）
            
        戻り値:
            収集済み結果
        """
        if not prompts:
            self.logger.warning("No prompts provided")
            return self.collector.collect_and_analyze([])
        
        # デフォルト値設定（KISS原則）
        if roles is None:
            roles = ["default"] * len(prompts)
        if blackboard_data is None:
            blackboard_data = {}
        
        # 入力検証（シンプルな検証）
        if len(roles) != len(prompts):
            roles = roles + ["default"] * (len(prompts) - len(roles))
        
        self.logger.info(f"Starting parallel execution of {len(prompts)} agents")
        
        try:
            # プロセス開始
            if not self.coordinator.start():
                raise RuntimeError("Failed to start worker processes")
            
            # タスク作成と送信（シンプルなループ）
            tasks = self._create_tasks(prompts, roles, blackboard_data)
            success_count = 0
            
            for task in tasks:
                if self.coordinator.submit_task(task):
                    success_count += 1
                else:
                    self.logger.error(f"Failed to submit task for agent {task.agent_id}")
            
            if success_count == 0:
                raise RuntimeError("Failed to submit any tasks")
            
            # 結果収集
            self.logger.info(f"Submitted {success_count} tasks, waiting for results...")
            results = self.coordinator.get_all_results(success_count, timeout=60.0)
            
            # 結果分析
            collected_results = self.collector.collect_and_analyze(results)
            self.collector.log_statistics(collected_results)
            
            return collected_results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return self.collector.collect_and_analyze([])
        
        finally:
            # 確実にプロセス停止
            self.coordinator.stop()
    
    def _create_tasks(self, prompts: List[str], roles: List[str], 
                     blackboard_data: Dict[str, Any]) -> List[AgentTask]:
        """
        エージェントタスクを作成（KISS原則：シンプルなタスク生成）
        
        引数:
            prompts: プロンプトリスト
            roles: ロールリスト
            blackboard_data: 共有データ
            
        戻り値:
            タスクリスト
        """
        tasks = []
        
        for i, (prompt, role) in enumerate(zip(prompts, roles)):
            task = AgentTask(
                agent_id=i,
                prompt=prompt,
                role=role,
                config=self.config_dict,
                blackboard_data=blackboard_data.copy()  # 各プロセスで独立したコピー
            )
            tasks.append(task)
        
        return tasks
    def execute_single_iteration(self, prompt: str, num_agents: int = None, 
                                blackboard_data: Dict[str, Any] = None) -> CollectedResults:
        """
        単一反復での並列実行（KISS原則：シンプルなインターフェース）
        
        引数:
            prompt: 共通プロンプト
            num_agents: エージェント数（省略時は設定値）
            blackboard_data: 共有黒板データ（省略時は空辞書）
            
        戻り値:
            収集済み結果
        """
        if num_agents is None:
            num_agents = self.config_manager.agent.num_agents
        
        if blackboard_data is None:
            blackboard_data = {}
        
        # 全エージェントに同じプロンプトを配布
        prompts = [prompt] * num_agents
        
        # ロール分散（シンプルなローテーション）
        available_roles = ["researcher", "critic", "synthesizer", "default"]
        roles = [available_roles[i % len(available_roles)] for i in range(num_agents)]
        
        return self.execute_agents_parallel(prompts, roles, blackboard_data)
    
    def get_performance_metrics(self, collected_results: CollectedResults) -> Dict[str, Any]:
        """
        パフォーマンス指標を取得（KISS原則：基本的な指標のみ）
        
        引数:
            collected_results: 収集済み結果
            
        戻り値:
            指標辞書
        """
        return {
            "total_agents": collected_results.total_count,
            "successful_agents": len(collected_results.successful_results),
            "failed_agents": len(collected_results.failed_results),
            "success_rate": collected_results.success_rate,
            "average_execution_time": collected_results.average_execution_time,
            "process_count": self.coordinator.num_processes,
            "parallel_efficiency": self._calculate_parallel_efficiency(collected_results)
        }
    
    def _calculate_parallel_efficiency(self, collected_results: CollectedResults) -> float:
        """
        並列効率を計算（KISS原則：シンプルな計算）
        
        引数:
            collected_results: 収集済み結果
            
        戻り値:
            並列効率（0.0-1.0）
        """
        if not collected_results.successful_results:
            return 0.0
        
        # 理想的な並列実行時間 vs 実際の実行時間
        max_execution_time = max(r.execution_time for r in collected_results.successful_results)
        total_execution_time = sum(r.execution_time for r in collected_results.successful_results)
        
        if max_execution_time <= 0:
            return 0.0
        
        # 並列効率 = (総作業時間 / (並列度 × 最大実行時間))
        parallel_efficiency = total_execution_time / (self.coordinator.num_processes * max_execution_time)
        return min(parallel_efficiency, 1.0)  # 1.0を上限とする
