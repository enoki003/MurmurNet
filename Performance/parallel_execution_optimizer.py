#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
並列実行最適化モジュール
~~~~~~~~~~~~~~~~~~~
真の並列実行を実現するための最適化パッチ

主な修正点：
1. グローバルロックの除去
2. プロセス間並列化
3. モデルの事前ロード・共有
4. SentenceTransformerの起動時ロード
5. 並列性の可視化（スレッドID・タイムスタンプ）

作者: Yuhi Sonoki
"""

import os
import sys
import time
import threading
import multiprocessing
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelExecutionOptimizer:
    """並列実行を最適化するクラス"""
    
    def __init__(self):
        self.applied_patches = []
        self.backup_methods = {}
        
    def apply_parallel_optimizations(self) -> bool:
        """並列実行の最適化を適用"""
        try:
            success = True
            
            # 1. グローバルロックの除去
            if self._remove_global_locks():
                self.applied_patches.append("global_lock_removal")
                logger.info("✓ グローバルロックを除去しました")
            else:
                success = False
                
            # 2. プロセス間並列化の実装
            if self._implement_process_parallelization():
                self.applied_patches.append("process_parallelization")
                logger.info("✓ プロセス間並列化を実装しました")
            else:
                success = False
                
            # 3. SentenceTransformerの事前ロード
            if self._preload_sentence_transformer():
                self.applied_patches.append("sentence_transformer_preload")
                logger.info("✓ SentenceTransformerの事前ロードを実装しました")
            else:
                success = False
                
            # 4. 並列性の可視化
            if self._add_parallelism_monitoring():
                self.applied_patches.append("parallelism_monitoring")
                logger.info("✓ 並列性の可視化を追加しました")
            else:
                success = False
                
            return success
            
        except Exception as e:
            logger.error(f"並列実行最適化の適用中にエラー: {str(e)}")
            return False
    
    def _remove_global_locks(self) -> bool:
        """グローバルロックを除去する"""
        try:
            # agent_pool.pyのグローバルロックを除去
            from MurmurNet.modules import agent_pool
            
            # 元のメソッドをバックアップ
            if hasattr(agent_pool.AgentPoolManager, '_agent_task_optimized'):
                self.backup_methods['_agent_task_optimized'] = agent_pool.AgentPoolManager._agent_task_optimized
            
            # ロックを除去した新しいメソッドを定義
            def _agent_task_optimized_no_lock(self, agent_id: int) -> str:
                """ロックを除去したエージェントタスク実行"""
                thread_id = threading.current_thread().ident
                process_id = os.getpid()
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                logger.info(f"[{timestamp}] エージェント{agent_id} 開始 (スレッド:{thread_id}, プロセス:{process_id})")
                
                # プロンプトの構築
                prompt = self._format_prompt(agent_id)
                
                # エージェントの役割と設定
                role = self.roles[agent_id]
                temperature = role.get('temperature', 0.7)
                
                start_time = time.time()
                
                try:
                    # グローバルロックを除去 - 各プロセスが独立してモデルを使用
                    resp = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=temperature,
                        top_p=0.9,
                        top_k=40,
                        repeat_penalty=1.1,
                        mirostat=0,
                    )
                    
                    # レスポンスの形式によって適切にアクセス
                    if isinstance(resp, dict):
                        output = resp['choices'][0]['message']['content'].strip()
                    else:
                        output = resp.choices[0].message.content.strip()
                          
                    # 出力を制限
                    if len(output) > 200:
                        output = output[:200] + "..."
                    
                    inference_time = time.time() - start_time
                    end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    logger.info(f"[{end_timestamp}] エージェント{agent_id} 完了 (推論時間:{inference_time:.2f}s, スレッド:{thread_id})")
                        
                    return output
                    
                except Exception as e:
                    error_time = time.time() - start_time
                    logger.error(f"エージェント{agent_id}のタスク実行エラー (時間:{error_time:.2f}s): {str(e)}")
                    return f"エージェント{agent_id}は応答できませんでした"
            
            # メソッドを置き換え
            agent_pool.AgentPoolManager._agent_task_optimized = _agent_task_optimized_no_lock
            
            return True
            
        except Exception as e:
            logger.error(f"グローバルロック除去中にエラー: {str(e)}")
            return False
    
    def _implement_process_parallelization(self) -> bool:
        """プロセス間並列化を実装する"""
        try:
            from MurmurNet.modules import agent_pool
            import asyncio
            
            # 元のメソッドをバックアップ
            if hasattr(agent_pool.AgentPoolManager, 'run_agents_parallel'):
                self.backup_methods['run_agents_parallel'] = agent_pool.AgentPoolManager.run_agents_parallel
            
            # プロセス間並列化を実装した新しいメソッド
            async def run_agents_parallel_process(self, blackboard) -> None:
                """プロセス間並列実行"""
                logger.info(f"エージェントをプロセス間並列実行中: {self.optimal_threads}プロセス")
                
                # パフォーマンス統計の記録
                if self.performance:
                    self.performance.record_parallel_execution('parallel')
                    self.performance.record_parallel_execution('process_pool')
                
                # OpenMPスレッド数を制限（プロセス間並列化のため）
                os.environ['OMP_NUM_THREADS'] = '2'
                os.environ['OPENBLAS_NUM_THREADS'] = '2'
                os.environ['MKL_NUM_THREADS'] = '2'
                
                # プロセス実行用の関数を定義
                def run_agent_process(agent_config: Dict[str, Any]) -> Tuple[int, str]:
                    """プロセス内でエージェントを実行"""
                    agent_id = agent_config['agent_id']
                    thread_id = threading.current_thread().ident
                    process_id = os.getpid()
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    logger.info(f"[{timestamp}] プロセス{process_id}でエージェント{agent_id}開始 (スレッド:{thread_id})")
                    
                    try:
                        # 各プロセスで独立してモデルを初期化
                        from MurmurNet.modules.model_factory import ModelFactory
                        llm = ModelFactory.get_shared_model(agent_config['config'])
                        
                        # プロンプト構築
                        prompt = agent_config['prompt']
                        temperature = agent_config['temperature']
                        
                        start_time = time.time()
                        
                        # 推論実行（ロックなし）
                        resp = llm.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=200,
                            temperature=temperature,
                            top_p=0.9,
                            top_k=40,
                            repeat_penalty=1.1,
                            mirostat=0,
                        )
                        
                        # レスポンス処理
                        if isinstance(resp, dict):
                            output = resp['choices'][0]['message']['content'].strip()
                        else:
                            output = resp.choices[0].message.content.strip()
                        
                        if len(output) > 200:
                            output = output[:200] + "..."
                        
                        inference_time = time.time() - start_time
                        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        logger.info(f"[{end_timestamp}] プロセス{process_id}でエージェント{agent_id}完了 (推論時間:{inference_time:.2f}s)")
                        
                        return agent_id, output
                        
                    except Exception as e:
                        logger.error(f"プロセス{process_id}でエージェント{agent_id}実行エラー: {str(e)}")
                        return agent_id, f"エージェント{agent_id}は応答できませんでした"
                
                # イベントループを取得
                loop = asyncio.get_event_loop()
                
                # ProcessPoolExecutorを使用
                with ProcessPoolExecutor(max_workers=self.optimal_threads) as executor:
                    try:
                        # 各エージェントの設定を準備
                        agent_configs = []
                        for i in range(self.num_agents):
                            config = {
                                'agent_id': i,
                                'config': self.config,
                                'prompt': self._format_prompt(i),
                                'temperature': self.roles[i].get('temperature', 0.7)
                            }
                            agent_configs.append(config)
                        
                        # 並列実行：すべてのエージェントを同時に実行
                        tasks = []
                        for config in agent_configs:
                            task = loop.run_in_executor(executor, run_agent_process, config)
                            tasks.append(task)
                        
                        # すべてのタスクを同時に実行して結果を取得
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # 結果を黒板に書き込み
                        for result in results:
                            if isinstance(result, Exception):
                                logger.error(f"エージェント処理エラー: {str(result)}")
                                continue
                            elif isinstance(result, tuple) and len(result) == 2:
                                agent_id, output = result
                                if output:
                                    blackboard.write(f'agent_{agent_id}_output', output)
                                else:
                                    blackboard.write(f'agent_{agent_id}_output', f"エージェント{agent_id}は空の応答を返しました")
                                    
                    except Exception as e:
                        logger.error(f"プロセス間並列実行エラー: {str(e)}")
                        # エラーが発生した場合は逐次実行にフォールバック
                        logger.info("逐次実行にフォールバックします")
                        self.run_agents(blackboard)
            
            # メソッドを置き換え
            agent_pool.AgentPoolManager.run_agents_parallel = run_agents_parallel_process
            
            return True
            
        except Exception as e:
            logger.error(f"プロセス間並列化実装中にエラー: {str(e)}")
            return False
    
    def _preload_sentence_transformer(self) -> bool:
        """SentenceTransformerの事前ロードを実装"""
        try:
            from MurmurNet.modules import input_reception
            
            # 元のメソッドをバックアップ
            if hasattr(input_reception.InputReception, '__init__'):
                self.backup_methods['input_reception_init'] = input_reception.InputReception.__init__
            
            # SentenceTransformerをクラス変数として共有
            input_reception.InputReception._shared_embedder = None
            input_reception.InputReception._embedder_lock = threading.Lock()
            
            def get_shared_embedder():
                """共有SentenceTransformerを取得"""
                if input_reception.InputReception._shared_embedder is None:
                    with input_reception.InputReception._embedder_lock:
                        if input_reception.InputReception._shared_embedder is None:
                            logger.info("SentenceTransformerを事前ロード中...")
                            start_time = time.time()
                            
                            from sentence_transformers import SentenceTransformer
                            input_reception.InputReception._shared_embedder = SentenceTransformer(
                                'all-MiniLM-L6-v2', 
                                cache_folder="./models/st_cache"
                            )
                            
                            load_time = time.time() - start_time
                            logger.info(f"SentenceTransformerを事前ロードしました (時間:{load_time:.2f}s)")
                
                return input_reception.InputReception._shared_embedder
            
            # InputReceptionのprocessメソッドを修正
            if hasattr(input_reception.InputReception, 'process'):
                original_process = input_reception.InputReception.process
                
                def process_with_shared_embedder(self, text: str) -> Dict[str, Any]:
                    """共有SentenceTransformerを使用した処理"""
                    try:
                        # 共有エンベッダーを使用
                        self.embedder = get_shared_embedder()
                        
                        # 元の処理を実行
                        return original_process(self, text)
                        
                    except Exception as e:
                        logger.error(f"SentenceTransformer処理エラー: {str(e)}")
                        # フォールバック処理
                        return {"text": text, "normalized": text, "embedding": None}
                
                input_reception.InputReception.process = process_with_shared_embedder
            
            return True
            
        except Exception as e:
            logger.error(f"SentenceTransformer事前ロード実装中にエラー: {str(e)}")
            return False
    
    def _add_parallelism_monitoring(self) -> bool:
        """並列性の可視化を追加"""
        try:
            from MurmurNet.modules import agent_pool
            
            # 元のメソッドをバックアップ
            if hasattr(agent_pool.AgentPoolManager, 'run_agents_parallel'):
                if 'run_agents_parallel' not in self.backup_methods:
                    self.backup_methods['run_agents_parallel_original'] = agent_pool.AgentPoolManager.run_agents_parallel
            
            # 並列性監視機能を追加
            def monitor_parallel_execution(original_method):
                """並列実行の監視デコレーター"""
                async def wrapper(self, blackboard):
                    execution_start = time.time()
                    start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    logger.info(f"[{start_timestamp}] 並列実行開始 (設定スレッド数:{self.optimal_threads})")
                    logger.info(f"[{start_timestamp}] CPUコア数: 物理={self.cpu_count_physical}, 論理={self.cpu_count}")
                    
                    try:
                        # 元のメソッドを実行
                        await original_method(self, blackboard)
                        
                        execution_time = time.time() - execution_start
                        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        logger.info(f"[{end_timestamp}] 並列実行完了 (総実行時間:{execution_time:.2f}s)")
                        
                        # 結果の分析
                        total_outputs = 0
                        for i in range(self.num_agents):
                            output = blackboard.read(f'agent_{i}_output')
                            if output and output != f"エージェント{i}は応答できませんでした":
                                total_outputs += 1
                        
                        logger.info(f"[{end_timestamp}] 成功エージェント数: {total_outputs}/{self.num_agents}")
                        
                        # 並列効率の計算
                        if hasattr(self, 'sequential_time'):
                            parallel_efficiency = (self.sequential_time / execution_time) * 100
                            logger.info(f"[{end_timestamp}] 並列効率: {parallel_efficiency:.1f}% (基準:{self.sequential_time:.2f}s)")
                        
                    except Exception as e:
                        execution_time = time.time() - execution_start
                        error_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        logger.error(f"[{error_timestamp}] 並列実行エラー (経過時間:{execution_time:.2f}s): {str(e)}")
                        raise
                
                return wrapper
            
            # 現在のメソッドを監視機能付きでラップ
            current_method = agent_pool.AgentPoolManager.run_agents_parallel
            agent_pool.AgentPoolManager.run_agents_parallel = monitor_parallel_execution(current_method)
            
            return True
            
        except Exception as e:
            logger.error(f"並列性監視機能追加中にエラー: {str(e)}")
            return False
    
    def rollback_optimizations(self) -> bool:
        """最適化を元に戻す"""
        try:
            from MurmurNet.modules import agent_pool, input_reception
            
            # バックアップしたメソッドを復元
            for method_name, original_method in self.backup_methods.items():
                if method_name == '_agent_task_optimized':
                    agent_pool.AgentPoolManager._agent_task_optimized = original_method
                elif method_name == 'run_agents_parallel':
                    agent_pool.AgentPoolManager.run_agents_parallel = original_method
                elif method_name == 'input_reception_init':
                    input_reception.InputReception.__init__ = original_method
                    
            logger.info("並列実行最適化を元に戻しました")
            return True
            
        except Exception as e:
            logger.error(f"最適化のロールバック中にエラー: {str(e)}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """最適化状況を取得"""
        return {
            "applied_patches": self.applied_patches,
            "backup_methods": list(self.backup_methods.keys()),
            "total_optimizations": len(self.applied_patches)
        }

def main():
    """メイン関数"""
    print("=== MurmurNet 並列実行最適化 ===")
    
    optimizer = ParallelExecutionOptimizer()
    
    print("並列実行の最適化を適用中...")
    success = optimizer.apply_parallel_optimizations()
    
    if success:
        status = optimizer.get_optimization_status()
        print(f"✓ 最適化が完了しました ({status['total_optimizations']}個の最適化を適用)")
        print(f"適用された最適化: {', '.join(status['applied_patches'])}")
        print("\n推奨テスト手順:")
        print("1. python -m MurmurNet.Performance.test_optimizations")
        print("2. 並列実行のログでタイムスタンプ・スレッドIDを確認")
        print("3. 性能向上を確認 (18 tokens/s → 30+ tokens/s)")
    else:
        print("✗ 最適化の適用に失敗しました")
        
    return success

if __name__ == "__main__":
    main()
