#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Result Collector モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~
単一責任: エージェント結果の収集と統合

設計原則:
- 単一責任: 結果の収集、統合、分析のみ
- KISS: シンプルなデータ統合
- 分離: プロセス管理は含まない

作者: Yuhi Sonoki
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from MurmurNet.modules.process_agent_worker import AgentResult


@dataclass
class CollectedResults:
    """収集された結果（KISS原則：シンプルなデータ構造）"""
    successful_results: List[AgentResult]
    failed_results: List[AgentResult]
    total_count: int
    success_rate: float
    average_execution_time: float
    combined_response: str
    
    def __post_init__(self):
        """データ検証"""
        if self.total_count < 0:
            raise ValueError("total_count must be non-negative")
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError("success_rate must be between 0.0 and 1.0")


class ResultCollector:
    """
    単一責任: エージェント結果の収集と統合
    
    設計原則:
    - 結果の分析と統合のみを担当
    - シンプルな統計計算
    - 明確なインターフェース
    """
    
    def __init__(self):
        """結果コレクターの初期化"""
        self.logger = logging.getLogger('ResultCollector')
    
    def collect_and_analyze(self, results: List[AgentResult]) -> CollectedResults:
        """
        結果を収集して分析（KISS原則：シンプルな統計処理）
        
        引数:
            results: エージェント結果リスト
            
        戻り値:
            分析済み結果
        """
        if not results:
            return CollectedResults(
                successful_results=[],
                failed_results=[],
                total_count=0,
                success_rate=0.0,
                average_execution_time=0.0,
                combined_response=""
            )
        
        # 成功・失敗の分離（シンプルなフィルタリング）
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # 統計計算（KISS原則：基本的な計算のみ）
        total_count = len(results)
        success_rate = len(successful) / total_count if total_count > 0 else 0.0
        
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        # 応答統合（シンプルな文字列結合）
        combined_response = self._combine_responses(successful)
        
        self.logger.info(f"Collected {total_count} results: {len(successful)} successful, {len(failed)} failed")
        
        return CollectedResults(
            successful_results=successful,
            failed_results=failed,
            total_count=total_count,
            success_rate=success_rate,
            average_execution_time=avg_execution_time,
            combined_response=combined_response
        )
    
    def _combine_responses(self, successful_results: List[AgentResult]) -> str:
        """
        成功した応答を統合（KISS原則：シンプルな文字列操作）
        
        引数:
            successful_results: 成功結果リスト
            
        戻り値:
            統合された応答
        """
        if not successful_results:
            return "エージェントからの有効な応答がありませんでした。"
        
        if len(successful_results) == 1:
            return successful_results[0].response
        
        # 複数の応答を番号付きで統合
        combined_parts = []
        for i, result in enumerate(successful_results, 1):
            if result.response.strip():
                combined_parts.append(f"【エージェント{result.agent_id}】\n{result.response.strip()}")
        
        if not combined_parts:
            return "エージェントからの有効な応答がありませんでした。"
        
        return "\n\n".join(combined_parts)
    
    def create_summary(self, collected_results: CollectedResults) -> str:
        """
        結果サマリーを作成（KISS原則：シンプルなテキスト生成）
        
        引数:
            collected_results: 収集済み結果
            
        戻り値:
            サマリーテキスト
        """
        summary_parts = [
            "=== 並列処理結果サマリー ===",
            f"総エージェント数: {collected_results.total_count}",
            f"成功: {len(collected_results.successful_results)}",
            f"失敗: {len(collected_results.failed_results)}",
            f"成功率: {collected_results.success_rate:.1%}",
            f"平均実行時間: {collected_results.average_execution_time:.2f}秒"
        ]
        
        # エラー詳細（失敗がある場合のみ）
        if collected_results.failed_results:
            summary_parts.append("\n--- エラー詳細 ---")
            for result in collected_results.failed_results:
                summary_parts.append(f"エージェント{result.agent_id}: {result.error_message}")
        
        return "\n".join(summary_parts)
    
    def log_statistics(self, collected_results: CollectedResults) -> None:
        """
        統計情報をログ出力（KISS原則：シンプルなログ出力）
        
        引数:
            collected_results: 収集済み結果
        """
        self.logger.info(f"Process execution statistics:")
        self.logger.info(f"  Total agents: {collected_results.total_count}")
        self.logger.info(f"  Successful: {len(collected_results.successful_results)}")
        self.logger.info(f"  Failed: {len(collected_results.failed_results)}")
        self.logger.info(f"  Success rate: {collected_results.success_rate:.1%}")
        self.logger.info(f"  Average execution time: {collected_results.average_execution_time:.2f}s")
        
        # 失敗の詳細ログ
        for result in collected_results.failed_results:
            self.logger.warning(f"Agent {result.agent_id} failed: {result.error_message}")
