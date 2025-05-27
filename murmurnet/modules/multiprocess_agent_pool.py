#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マルチプロセス エージェント プール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
真の並列処理を実現するためのプロセスベース実装

各エージェントが独立したプロセスで実行されるため、
GILの制約やllama-cpp-pythonの内部競合を回避

作者: Yuhi Sonoki
"""

import logging
import multiprocessing as mp
import queue
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.MultiprocessAgentPool')

@dataclass
class AgentTask:
    """エージェントタスクのデータクラス"""
    agent_id: int
    role: str
    system_prompt: str
    user_prompt: str
    temperature: float
    timeout: int

@dataclass
class AgentResult:
    """エージェント結果のデータクラス"""
    agent_id: int
    role: str
    content: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

def agent_worker_process(task: AgentTask) -> AgentResult:
    """
    独立プロセスで実行されるエージェントワーカー
    
    Args:
        task: エージェントタスク
        
    Returns:
        AgentResult: 実行結果
    """
    start_time = time.time()
    
    try:
        # プロセス内で新しいモデルインスタンスを作成
        from MurmurNet.modules.model_factory import ModelFactory
        
        # プロセス固有のモデルインスタンス
        model_factory = ModelFactory()
        model = model_factory.create_model()
        
        # プロンプト構築
        full_prompt = f"System: {task.system_prompt}\n\nUser: {task.user_prompt}"
        
        # 推論実行
        response = model.generate(
            prompt=full_prompt,
            temperature=task.temperature,
            max_tokens=get_config().model.max_tokens
        )
        
        execution_time = time.time() - start_time
        
        return AgentResult(
            agent_id=task.agent_id,
            role=task.role,
            content=response,
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"エージェント {task.agent_id} でエラー: {str(e)}")
        
        return AgentResult(
            agent_id=task.agent_id,
            role=task.role,
            content="",
            execution_time=execution_time,
            success=False,
            error_message=str(e)
        )

class MultiprocessAgentPool:
    """
    真の並列処理を実現するマルチプロセスエージェントプール
    
    特徴:
    - 各エージェントが独立したプロセスで実行
    - GILの制約を回避
    - llama-cpp-pythonの内部競合を回避
    - 真の並列処理を実現
    """
    
    def __init__(self, num_agents: int = 4, max_workers: Optional[int] = None):
        """
        初期化
        
        Args:
            num_agents: エージェント数
            max_workers: 最大ワーカープロセス数（デフォルトはCPUコア数）
        """
        self.num_agents = num_agents
        self.max_workers = max_workers or min(num_agents, mp.cpu_count())
        self.config = get_config()
        
        # 役割テンプレートの読み込み
        self._load_role_templates()
        
        logger.info(f"マルチプロセスエージェントプール初期化完了")
        logger.info(f"エージェント数: {num_agents}, 最大ワーカー数: {self.max_workers}")
    
    def _load_role_templates(self) -> None:
        """役割テンプレートの読み込み"""
        self.role_templates = {
            "discussion": [
                {"role": "多角的視点AI", "system": "あなたは多角的思考のスペシャリストです。論点を多面的に分析して議論の全体像を示してください。", "temperature": 0.7},
                {"role": "批判的思考AI", "system": "あなたは批判的思考の専門家です。前提や論理に疑問を投げかけ、新たな視点を提供してください。", "temperature": 0.8},
                {"role": "実証主義AI", "system": "あなたはデータと証拠を重視する科学者です。事実に基づいた分析と検証可能な情報を提供してください。", "temperature": 0.6},
                {"role": "倫理的視点AI", "system": "あなたは倫理学者です。道徳的・倫理的観点から議論を分析し、価値判断の視点を提供してください。", "temperature": 0.7}
            ],
            "planning": [
                {"role": "実用主義AI", "system": "あなたは実用主義の専門家です。実行可能で具体的なアプローチを提案してください。", "temperature": 0.7},
                {"role": "創造的思考AI", "system": "あなたは創造的思考のスペシャリストです。革新的なアイデアと可能性を探索してください。", "temperature": 0.9},
                {"role": "戦略的視点AI", "system": "あなたは戦略家です。長期的な視点と全体像を考慮した計画を立案してください。", "temperature": 0.7},
                {"role": "リスク分析AI", "system": "あなたはリスク管理専門家です。潜在的な問題点と対策を特定してください。", "temperature": 0.6}
            ],
            "default": [
                {"role": "バランス型AI", "system": "あなたは総合的な分析ができるバランス型AIです。公平で多面的な視点から回答してください。", "temperature": 0.7},
                {"role": "専門知識AI", "system": "あなたは幅広い知識を持つ専門家です。正確でわかりやすい情報を提供してください。", "temperature": 0.6}
            ]
        }
    
    def classify_question_type(self, prompt: str) -> str:
        """質問タイプの分類"""
        discussion_keywords = ["議論", "論争", "賛否", "意見", "討論", "批判", "分析"]
        planning_keywords = ["計画", "戦略", "方法", "アプローチ", "実装", "設計", "構想"]
        
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in discussion_keywords):
            return "discussion"
        elif any(keyword in prompt_lower for keyword in planning_keywords):
            return "planning"
        else:
            return "default"
    
    def execute_parallel(self, prompt: str, timeout: int = 60) -> List[AgentResult]:
        """
        真の並列処理でエージェントを実行
        
        Args:
            prompt: ユーザープロンプト
            timeout: タイムアウト（秒）
            
        Returns:
            List[AgentResult]: 各エージェントの実行結果
        """
        # 質問タイプの分類
        question_type = self.classify_question_type(prompt)
        roles = self.role_templates.get(question_type, self.role_templates["default"])
        
        # タスクの準備
        tasks = []
        for i in range(self.num_agents):
            role_info = roles[i % len(roles)]
            task = AgentTask(
                agent_id=i,
                role=role_info["role"],
                system_prompt=role_info["system"],
                user_prompt=prompt,
                temperature=role_info["temperature"],
                timeout=timeout
            )
            tasks.append(task)
        
        results = []
        
        # ProcessPoolExecutorで真の並列実行
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            logger.info(f"並列処理開始: {self.num_agents}エージェント, {self.max_workers}プロセス")
            
            # 全タスクを同時に提出
            future_to_task = {
                executor.submit(agent_worker_process, task): task 
                for task in tasks
            }
            
            # 結果を收集
            for future in as_completed(future_to_task, timeout=timeout + 10):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"エージェント {result.agent_id} 完了: {result.execution_time:.2f}秒")
                except TimeoutError:
                    logger.error(f"エージェント {task.agent_id} タイムアウト")
                    results.append(AgentResult(
                        agent_id=task.agent_id,
                        role=task.role,
                        content="",
                        execution_time=timeout,
                        success=False,
                        error_message="タイムアウト"
                    ))
                except Exception as e:
                    logger.error(f"エージェント {task.agent_id} 例外: {str(e)}")
                    results.append(AgentResult(
                        agent_id=task.agent_id,
                        role=task.role,
                        content="",
                        execution_time=0.0,
                        success=False,
                        error_message=str(e)
                    ))
        
        # 結果をエージェントIDでソート
        results.sort(key=lambda x: x.agent_id)
        
        successful_results = [r for r in results if r.success]
        logger.info(f"並列処理完了: {len(successful_results)}/{len(results)} 成功")
        
        return results
    
    def format_results(self, results: List[AgentResult]) -> str:
        """
        結果を見やすい形式でフォーマット
        
        Args:
            results: エージェント実行結果のリスト
            
        Returns:
            str: フォーマットされた結果文字列
        """
        output = []
        output.append("=" * 80)
        output.append("🤖 MurmurNet マルチプロセス並列処理結果")
        output.append("=" * 80)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            output.append(f"\n✅ 成功: {len(successful_results)}エージェント")
            output.append("-" * 60)
            
            for result in successful_results:
                output.append(f"\n🔹 【{result.role}】 (実行時間: {result.execution_time:.2f}秒)")
                output.append("-" * 40)
                output.append(result.content)
                output.append("")
        
        if failed_results:
            output.append(f"\n❌ 失敗: {len(failed_results)}エージェント")
            output.append("-" * 60)
            
            for result in failed_results:
                output.append(f"\n🔸 【{result.role}】 エラー: {result.error_message}")
        
        total_time = max([r.execution_time for r in results], default=0.0)
        output.append(f"\n⏱️ 総実行時間: {total_time:.2f}秒")
        output.append("=" * 80)
        
        return "\n".join(output)

# 使用例
def test_multiprocess_agent_pool():
    """テスト用の関数"""
    pool = MultiprocessAgentPool(num_agents=4, max_workers=4)
    
    test_prompt = "人工知能の倫理的な課題について議論してください"
    
    results = pool.execute_parallel(test_prompt, timeout=60)
    formatted_output = pool.format_results(results)
    
    print(formatted_output)
    
    return results

if __name__ == "__main__":
    # マルチプロセス実行時の初期化
    mp.set_start_method('spawn', force=True)
    test_multiprocess_agent_pool()
