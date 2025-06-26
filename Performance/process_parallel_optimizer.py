#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
プロセス並列最適化モジュール
~~~~~~~~~~~~~~~~~~~
真の並列推論を実現するためのProcessPool最適化

主な修正点：
1. ThreadPoolExecutor → ProcessPoolExecutor
2. グローバルロック除去
3. モデルの真のSingleton化（@lru_cache）
4. SentenceTransformerの常駐化
5. 並列性の可視化とボトルネック分析

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
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessParallelOptimizer:
    """真の並列実行を実現するProcessPool最適化クラス"""
    
    def __init__(self):
        self.applied_patches = []
        self.backup_methods = {}
        
    def apply_process_optimizations(self) -> bool:
        """プロセス並列最適化を適用"""
        try:
            success = True
            logger.info("=== プロセス並列最適化を開始 ===")
            
            # 1. モデルの真のSingleton化
            if self._implement_model_singleton():
                self.applied_patches.append("model_singleton")
                logger.info("SUCCESS: モデルの真のSingleton化")
            else:
                success = False
                logger.error("FAILED: モデルの真のSingleton化")
                
            # 2. ProcessPoolExecutorの実装
            if self._implement_process_pool():
                self.applied_patches.append("process_pool")
                logger.info("SUCCESS: ProcessPoolExecutorの実装")
            else:
                success = False
                logger.error("FAILED: ProcessPoolExecutorの実装")
                
            # 3. グローバルロックの除去
            if self._remove_global_locks():
                self.applied_patches.append("lock_removal")
                logger.info("SUCCESS: グローバルロックの除去")
            else:
                success = False
                logger.error("FAILED: グローバルロックの除去")
                
            # 4. SentenceTransformerの常駐化
            if self._implement_persistent_embedder():
                self.applied_patches.append("persistent_embedder")
                logger.info("SUCCESS: SentenceTransformerの常駐化")
            else:
                success = False
                logger.error("FAILED: SentenceTransformerの常駐化")
                
            # 5. 並列性可視化
            if self._add_parallelism_visualization():
                self.applied_patches.append("parallelism_visualization")
                logger.info("SUCCESS: 並列性可視化の追加")
            else:
                success = False
                logger.error("FAILED: 並列性可視化の追加")
                
            logger.info(f"=== 最適化完了: {len(self.applied_patches)}/5 ===")
            return success
            
        except Exception as e:
            logger.error(f"プロセス並列最適化中にエラー: {str(e)}")
            return False
    
    def _implement_model_singleton(self) -> bool:
        """モデルの真のSingleton化を実装"""
        try:
            from MurmurNet.modules import model_factory
            
            # 元のメソッドをバックアップ
            if hasattr(model_factory.ModelFactory, 'get_shared_model'):
                self.backup_methods['get_shared_model'] = model_factory.ModelFactory.get_shared_model
            
            # @lru_cacheを使った真のSingleton
            @lru_cache(maxsize=1)
            def get_singleton_model(config_hash: str) -> Any:
                """ハッシュベースのSingletonモデル取得"""
                logger.info(f"モデルをSingleton初期化中 (hash: {config_hash[:8]}...)")
                start_time = time.time()
                
                try:
                    # 設定を復元
                    import json
                    config = json.loads(config_hash)
                    
                    # Llamaモデルの初期化
                    model_path = config.get('model_path', './models/model.gguf')
                    
                    # OpenMPスレッド数制限（プロセス並列化のため）
                    os.environ['OMP_NUM_THREADS'] = '2'
                    os.environ['OPENBLAS_NUM_THREADS'] = '2'
                    os.environ['MKL_NUM_THREADS'] = '2'
                    
                    from llama_cpp import Llama
                    model = Llama(
                        model_path=model_path,
                        n_ctx=config.get('n_ctx', 2048),
                        n_threads=2,  # プロセス並列化のため少なく
                        n_threads_batch=2,
                        use_mmap=True,
                        use_mlock=False,
                        verbose=False
                    )
                    
                    load_time = time.time() - start_time
                    logger.info(f"Singletonモデル初期化完了 (時間: {load_time:.2f}s)")
                    
                    return model
                    
                except Exception as e:
                    logger.error(f"Singletonモデル初期化エラー: {str(e)}")
                    raise
            
            # 新しいget_shared_modelメソッド
            def get_shared_model_singleton(config: Dict[str, Any]) -> Any:
                """Singleton化されたモデル取得"""
                # 設定をハッシュ化してキーとして使用
                import json
                config_hash = json.dumps(config, sort_keys=True)
                return get_singleton_model(config_hash)
            
            # メソッドを置き換え
            model_factory.ModelFactory.get_shared_model = staticmethod(get_shared_model_singleton)
            
            return True
            
        except Exception as e:
            logger.error(f"モデルSingleton化実装中にエラー: {str(e)}")
            return False
    
    def _implement_process_pool(self) -> bool:
        """ProcessPoolExecutorを実装"""
        try:
            from MurmurNet.modules import agent_pool
            
            # 元のメソッドをバックアップ
            if hasattr(agent_pool.AgentPoolManager, 'run_agents_parallel'):
                self.backup_methods['run_agents_parallel'] = agent_pool.AgentPoolManager.run_agents_parallel
            
            # プロセス実行用の独立関数（モジュールレベルで定義）
            def execute_agent_process(agent_config: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
                """独立プロセスでエージェントを実行"""
                agent_id = agent_config['agent_id']
                process_id = os.getpid()
                thread_id = threading.current_thread().ident
                start_time = time.time()
                
                try:
                    # プロセス内でのログ設定
                    import logging
                    logger = logging.getLogger(f'Agent{agent_id}')
                    
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] プロセス{process_id}でエージェント{agent_id}開始")
                    
                    # 各プロセスで独立してモデルを取得（Singleton）
                    from MurmurNet.modules.model_factory import ModelFactory
                    model = ModelFactory.get_shared_model(agent_config['config'])
                    
                    # プロンプト情報
                    prompt = agent_config['prompt']
                    temperature = agent_config['temperature']
                    
                    # 推論実行（ロックなし）
                    inference_start = time.time()
                    
                    resp = model.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=temperature,
                        top_p=0.9,
                        top_k=40,
                        repeat_penalty=1.1,
                        mirostat=0,
                    )
                    
                    inference_time = time.time() - inference_start
                    
                    # レスポンス処理
                    if isinstance(resp, dict):
                        output = resp['choices'][0]['message']['content'].strip()
                    else:
                        output = resp.choices[0].message.content.strip()
                    
                    if len(output) > 200:
                        output = output[:200] + "..."
                    
                    total_time = time.time() - start_time
                    end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    # 統計情報
                    stats = {
                        'process_id': process_id,
                        'thread_id': thread_id,
                        'start_time': start_time,
                        'inference_time': inference_time,
                        'total_time': total_time,
                        'output_length': len(output)
                    }
                    
                    logger.info(f"[{end_timestamp}] エージェント{agent_id}完了 (推論:{inference_time:.2f}s, 総時間:{total_time:.2f}s)")
                    
                    return agent_id, output, stats
                    
                except Exception as e:
                    error_time = time.time() - start_time
                    logger.error(f"エージェント{agent_id}実行エラー (時間:{error_time:.2f}s): {str(e)}")
                    return agent_id, f"エージェント{agent_id}は応答できませんでした", {'error': str(e)}
            
            # 新しいrun_agents_parallel_processメソッド
            async def run_agents_parallel_process(self, blackboard) -> None:
                """ProcessPoolExecutorを使用した真の並列実行"""
                logger.info(f"=== プロセス並列実行開始: {self.optimal_threads}プロセス ===")
                
                # パフォーマンス統計の記録
                if self.performance:
                    self.performance.record_parallel_execution('process_parallel')
                
                execution_start = time.time()
                start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # プロセス実行用の設定準備
                agent_configs = []
                for i in range(self.num_agents):
                    config = {
                        'agent_id': i,
                        'config': self.config,
                        'prompt': self._format_prompt(i),
                        'temperature': self.roles[i].get('temperature', 0.7)
                    }
                    agent_configs.append(config)
                
                # ProcessPoolExecutorで真の並列実行
                import asyncio
                loop = asyncio.get_event_loop()
                
                try:
                    with ProcessPoolExecutor(max_workers=self.optimal_threads) as executor:
                        # 全エージェントを同時並列実行
                        tasks = []
                        for config in agent_configs:
                            task = loop.run_in_executor(executor, execute_agent_process, config)
                            tasks.append(task)
                        
                        # 結果を並列収集
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # 結果の分析と書き込み
                        successful_agents = 0
                        total_inference_time = 0
                        process_stats = []
                        
                        for result in results:
                            if isinstance(result, Exception):
                                logger.error(f"プロセス実行エラー: {str(result)}")
                                continue
                            elif isinstance(result, tuple) and len(result) == 3:
                                agent_id, output, stats = result
                                
                                # 黒板に書き込み
                                blackboard.write(f'agent_{agent_id}_output', output)
                                
                                # 統計情報を収集
                                if 'error' not in stats:
                                    successful_agents += 1
                                    total_inference_time += stats.get('inference_time', 0)
                                    process_stats.append(stats)
                        
                        # 実行結果の分析
                        execution_time = time.time() - execution_start
                        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        # 並列効率の計算
                        if process_stats:
                            avg_inference_time = total_inference_time / len(process_stats)
                            parallel_efficiency = (avg_inference_time / execution_time) * 100
                            
                            # プロセスID一覧
                            process_ids = [stats['process_id'] for stats in process_stats]
                            unique_processes = len(set(process_ids))
                            
                            logger.info(f"[{end_timestamp}] === 並列実行完了 ===")
                            logger.info(f"成功エージェント: {successful_agents}/{self.num_agents}")
                            logger.info(f"総実行時間: {execution_time:.2f}s")
                            logger.info(f"平均推論時間: {avg_inference_time:.2f}s")
                            logger.info(f"並列効率: {parallel_efficiency:.1f}%")
                            logger.info(f"使用プロセス数: {unique_processes}")
                            logger.info(f"プロセスID: {sorted(set(process_ids))}")
                        
                except Exception as e:
                    logger.error(f"ProcessPool並列実行エラー: {str(e)}")
                    # フォールバック処理
                    logger.info("逐次実行にフォールバックします")
                    await self.run_agents(blackboard)
            
            # メソッドを置き換え
            agent_pool.AgentPoolManager.run_agents_parallel = run_agents_parallel_process
            
            # 実行関数をモジュールレベルに追加（pickle可能にするため）
            agent_pool.execute_agent_process = execute_agent_process
            
            return True
            
        except Exception as e:
            logger.error(f"ProcessPool実装中にエラー: {str(e)}")
            return False
    
    def _remove_global_locks(self) -> bool:
        """グローバルロックを除去"""
        try:
            from MurmurNet.modules import agent_pool
            
            # グローバルロックを無効化
            if hasattr(agent_pool, '_global_llama_lock'):
                # ロックを無効化するダミーロックに置き換え
                class DummyLock:
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        pass
                    def acquire(self, blocking=True):
                        return True
                    def release(self):
                        pass
                
                agent_pool._global_llama_lock = DummyLock()
                logger.info("グローバルロックを無効化しました（プロセス並列化のため）")
            
            # _agent_task_optimizedのロック部分を除去
            if hasattr(agent_pool.AgentPoolManager, '_agent_task_optimized'):
                original_method = agent_pool.AgentPoolManager._agent_task_optimized
                
                def _agent_task_no_lock(self, agent_id: int) -> str:
                    """ロック除去版のエージェントタスク実行"""
                    # プロンプトの構築
                    prompt = self._format_prompt(agent_id)
                    
                    # エージェントの役割と設定
                    role = self.roles[agent_id]
                    temperature = role.get('temperature', 0.7)
                    
                    try:
                        # ロックを除去して実行
                        resp = self.llm.create_chat_completion(
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
                            
                        return output
                        
                    except Exception as e:
                        logger.error(f"エージェント{agent_id}のタスク実行エラー: {str(e)}")
                        return f"エージェント{agent_id}は応答できませんでした"
                
                agent_pool.AgentPoolManager._agent_task_optimized = _agent_task_no_lock
            
            return True
            
        except Exception as e:
            logger.error(f"グローバルロック除去中にエラー: {str(e)}")
            return False
    
    def _implement_persistent_embedder(self) -> bool:
        """SentenceTransformerの常駐化を実装"""
        try:
            from MurmurNet.modules import input_reception
            
            # プロセス間で共有するSentenceTransformer
            @lru_cache(maxsize=1)
            def get_persistent_embedder():
                """常駐SentenceTransformerを取得"""
                logger.info("SentenceTransformerを常駐初期化中...")
                start_time = time.time()
                
                try:
                    from sentence_transformers import SentenceTransformer
                    embedder = SentenceTransformer(
                        'all-MiniLM-L6-v2', 
                        cache_folder="./models/st_cache"
                    )
                    
                    load_time = time.time() - start_time
                    logger.info(f"SentenceTransformer常駐化完了 (時間: {load_time:.2f}s)")
                    
                    return embedder
                    
                except Exception as e:
                    logger.error(f"SentenceTransformer常駐化エラー: {str(e)}")
                    raise
            
            # InputReceptionのprocessメソッドを修正
            if hasattr(input_reception.InputReception, 'process'):
                original_process = input_reception.InputReception.process
                
                def process_with_persistent_embedder(self, text: str) -> Dict[str, Any]:
                    """常駐SentenceTransformerを使用した処理"""
                    try:
                        # 常駐エンベッダーを使用
                        self.embedder = get_persistent_embedder()
                        
                        # 元の処理を実行
                        return original_process(self, text)
                        
                    except Exception as e:
                        logger.error(f"常駐SentenceTransformer処理エラー: {str(e)}")
                        # フォールバック処理
                        return {"text": text, "normalized": text, "embedding": None}
                
                input_reception.InputReception.process = process_with_persistent_embedder
            
            return True
            
        except Exception as e:
            logger.error(f"常駐SentenceTransformer実装中にエラー: {str(e)}")
            return False
    
    def _add_parallelism_visualization(self) -> bool:
        """並列性の可視化を追加"""
        try:
            import psutil
            
            # CPU使用率監視用のデコレーター
            def visualize_parallelism(original_method):
                """並列性可視化デコレーター"""
                async def wrapper(self, blackboard):
                    # 監視開始
                    cpu_before = psutil.cpu_percent(interval=None)
                    start_time = time.time()
                    start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    logger.info(f"[{start_timestamp}] === 並列性監視開始 ===")
                    logger.info(f"CPU使用率(開始前): {cpu_before}%")
                    logger.info(f"設定スレッド/プロセス数: {self.optimal_threads}")
                    logger.info(f"物理CPUコア: {self.cpu_count_physical}, 論理CPUコア: {self.cpu_count}")
                    
                    try:
                        # 元のメソッドを実行
                        await original_method(self, blackboard)
                        
                        # 監視終了
                        execution_time = time.time() - start_time
                        cpu_after = psutil.cpu_percent(interval=0.1)
                        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        logger.info(f"[{end_timestamp}] === 並列性監視完了 ===")
                        logger.info(f"CPU使用率(実行後): {cpu_after}%")
                        logger.info(f"総実行時間: {execution_time:.2f}s")
                        
                        # 並列性の評価
                        if cpu_after > cpu_before * 1.5:
                            logger.info("✓ 並列実行によるCPU使用率向上を確認")
                        else:
                            logger.warning("! 並列実行の効果が限定的です")
                        
                        # メモリ使用量
                        memory_info = psutil.virtual_memory()
                        logger.info(f"メモリ使用率: {memory_info.percent}%")
                        
                    except Exception as e:
                        error_time = time.time() - start_time
                        logger.error(f"並列実行監視エラー (経過時間:{error_time:.2f}s): {str(e)}")
                        raise
                
                return wrapper
            
            # 既存のrun_agents_parallelメソッドをラップ
            from MurmurNet.modules import agent_pool
            if hasattr(agent_pool.AgentPoolManager, 'run_agents_parallel'):
                current_method = agent_pool.AgentPoolManager.run_agents_parallel
                agent_pool.AgentPoolManager.run_agents_parallel = visualize_parallelism(current_method)
            
            return True
            
        except Exception as e:
            logger.error(f"並列性可視化追加中にエラー: {str(e)}")
            return False
    
    def rollback_optimizations(self) -> bool:
        """最適化を元に戻す"""
        try:
            from MurmurNet.modules import agent_pool, model_factory, input_reception
            
            # バックアップしたメソッドを復元
            for method_name, original_method in self.backup_methods.items():
                if method_name == 'get_shared_model':
                    model_factory.ModelFactory.get_shared_model = original_method
                elif method_name == 'run_agents_parallel':
                    agent_pool.AgentPoolManager.run_agents_parallel = original_method
                    
            logger.info("プロセス並列最適化を元に戻しました")
            return True
            
        except Exception as e:
            logger.error(f"最適化のロールバック中にエラー: {str(e)}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """最適化状況を取得"""
        return {
            "applied_patches": self.applied_patches,
            "total_optimizations": len(self.applied_patches),
            "success_rate": f"{len(self.applied_patches)}/5 ({len(self.applied_patches)/5*100:.1f}%)"
        }

def main():
    """メイン関数"""
    print("=== MurmurNet プロセス並列最適化 ===")
    print("真の並列推論を実現します...")
    
    optimizer = ProcessParallelOptimizer()
    
    print("\nプロセス並列最適化を適用中...")
    success = optimizer.apply_process_optimizations()
    
    if success:
        status = optimizer.get_optimization_status()
        print(f"\n✓ 最適化が完了しました ({status['success_rate']})")
        print(f"適用された最適化: {', '.join(status['applied_patches'])}")
        
        print("\n=== 推奨テスト手順 ===")
        print("1. python -m MurmurNet.Performance.test_optimizations")
        print("2. ログでプロセスID・タイムスタンプを確認")
        print("3. 性能向上を確認:")
        print("  - 18 tokens/s → 30+ tokens/s")
        print("  - 複数プロセスでの同時実行")
        print("  - CPU使用率の向上")
        print("  - 並列効率の改善")
        
    else:
        print("\n✗ 最適化の適用に失敗しました")
        print("ログを確認してください")
        
    return success

if __name__ == "__main__":
    main()
