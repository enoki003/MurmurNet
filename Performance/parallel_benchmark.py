#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真の並列実行性能ベンチマーク
~~~~~~~~~~~~~~~~~~~
ProcessPool最適化の効果を定量的に測定

測定項目:
1. 実行時間（逐次 vs 並列）
2. tokens/s（推論性能）
3. CPU使用率
4. メモリ使用量
5. プロセス並列度
6. 並列効率

作者: Yuhi Sonoki
"""

import os
import sys
import asyncio
import time
import psutil
import threading
import multiprocessing
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelBenchmark:
    """並列実行性能ベンチマーククラス"""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        return {
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    async def run_benchmark(self, test_iterations: int = 3) -> Dict[str, Any]:
        """ベンチマーク実行"""
        logger.info("=== 真の並列実行性能ベンチマーク開始 ===")
        logger.info(f"システム情報: {self.system_info}")
        
        # 1. 最適化前のベースライン測定
        logger.info("\n1. ベースライン測定（最適化前）...")
        baseline_results = await self._run_baseline_test(test_iterations)
        
        # 2. 最適化適用
        logger.info("\n2. プロセス並列最適化を適用...")
        optimization_success = self._apply_optimizations()
        
        if not optimization_success:
            logger.error("最適化の適用に失敗しました")
            return {'error': 'optimization_failed'}
        
        # 3. 最適化後の性能測定
        logger.info("\n3. 最適化後性能測定...")
        optimized_results = await self._run_optimized_test(test_iterations)
        
        # 4. 結果比較・分析
        logger.info("\n4. 結果分析...")
        comparison = self._analyze_results(baseline_results, optimized_results)
        
        return {
            'system_info': self.system_info,
            'baseline': baseline_results,
            'optimized': optimized_results,
            'comparison': comparison,
            'optimization_success': optimization_success
        }
    
    async def _run_baseline_test(self, iterations: int) -> Dict[str, Any]:
        """ベースライン性能測定"""
        logger.info("ベースライン測定を実行中...")
        
        results = []
        
        for i in range(iterations):
            logger.info(f"ベースライン測定 {i+1}/{iterations}")
            
            # CPU・メモリ監視開始
            cpu_before = psutil.cpu_percent(interval=None)
            memory_before = psutil.virtual_memory().used / (1024**2)  # MB
            
            start_time = time.time()
            
            try:
                # MurmurNetの標準実行（ThreadPool版）
                from MurmurNet.distributed_slm import DistributedSLM
                
                # テスト用設定
                config = {
                    'model_path': './models/model.gguf',
                    'n_ctx': 2048,
                    'num_agents': 2,
                    'use_parallel': True,  # ThreadPool版
                    'debug': False
                }
                
                # 実行
                slm = DistributedSLM(config)
                test_input = "プログラミングにおける並列処理の重要性について説明してください。"
                response = await slm.process_query(test_input)
                
                execution_time = time.time() - start_time
                
                # CPU・メモリ監視終了
                cpu_after = psutil.cpu_percent(interval=0.1)
                memory_after = psutil.virtual_memory().used / (1024**2)  # MB
                
                # レスポンス解析
                response_length = len(str(response)) if response else 0
                tokens_per_second = response_length / execution_time if execution_time > 0 else 0
                
                result = {
                    'iteration': i + 1,
                    'execution_time': execution_time,
                    'response_length': response_length,
                    'tokens_per_second': tokens_per_second,
                    'cpu_usage_before': cpu_before,
                    'cpu_usage_after': cpu_after,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_used_mb': memory_after - memory_before
                }
                
                results.append(result)
                logger.info(f"  時間: {execution_time:.2f}s, Tokens/s: {tokens_per_second:.1f}")
                
                # テスト間のクールダウン
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"ベースライン測定 {i+1} でエラー: {str(e)}")
                continue
        
        # 統計計算
        if results:
            avg_time = sum(r['execution_time'] for r in results) / len(results)
            avg_tokens_per_sec = sum(r['tokens_per_second'] for r in results) / len(results)
            avg_memory_used = sum(r['memory_used_mb'] for r in results) / len(results)
            
            return {
                'results': results,
                'avg_execution_time': avg_time,
                'avg_tokens_per_second': avg_tokens_per_sec,
                'avg_memory_used_mb': avg_memory_used,
                'total_iterations': len(results)
            }
        else:
            return {'error': 'no_successful_baseline_tests'}
    
    def _apply_optimizations(self) -> bool:
        """最適化を適用"""
        try:
            from Performance.process_parallel_optimizer import ProcessParallelOptimizer
            
            optimizer = ProcessParallelOptimizer()
            success = optimizer.apply_process_optimizations()
            
            if success:
                logger.info("✓ プロセス並列最適化が正常に適用されました")
            else:
                logger.error("✗ プロセス並列最適化の適用に失敗しました")
            
            return success
            
        except Exception as e:
            logger.error(f"最適化適用中にエラー: {str(e)}")
            return False
    
    async def _run_optimized_test(self, iterations: int) -> Dict[str, Any]:
        """最適化後性能測定"""
        logger.info("最適化後性能測定を実行中...")
        
        results = []
        
        for i in range(iterations):
            logger.info(f"最適化後測定 {i+1}/{iterations}")
            
            # CPU・メモリ監視開始
            cpu_before = psutil.cpu_percent(interval=None)
            memory_before = psutil.virtual_memory().used / (1024**2)  # MB
            
            start_time = time.time()
            
            try:
                # MurmurNetの最適化実行（ProcessPool版）
                from MurmurNet.distributed_slm import DistributedSLM
                
                # テスト用設定（ProcessPool版）
                config = {
                    'model_path': './models/model.gguf',
                    'n_ctx': 2048,
                    'num_agents': 2,
                    'use_parallel': True,  # ProcessPool版（最適化済み）
                    'debug': False
                }
                
                # 実行
                slm = DistributedSLM(config)
                test_input = "プログラミングにおける並列処理の重要性について説明してください。"
                response = await slm.process_query(test_input)
                
                execution_time = time.time() - start_time
                
                # CPU・メモリ監視終了
                cpu_after = psutil.cpu_percent(interval=0.1)
                memory_after = psutil.virtual_memory().used / (1024**2)  # MB
                
                # レスポンス解析
                response_length = len(str(response)) if response else 0
                tokens_per_second = response_length / execution_time if execution_time > 0 else 0
                
                result = {
                    'iteration': i + 1,
                    'execution_time': execution_time,
                    'response_length': response_length,
                    'tokens_per_second': tokens_per_second,
                    'cpu_usage_before': cpu_before,
                    'cpu_usage_after': cpu_after,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_used_mb': memory_after - memory_before
                }
                
                results.append(result)
                logger.info(f"  時間: {execution_time:.2f}s, Tokens/s: {tokens_per_second:.1f}")
                
                # テスト間のクールダウン
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"最適化後測定 {i+1} でエラー: {str(e)}")
                continue
        
        # 統計計算
        if results:
            avg_time = sum(r['execution_time'] for r in results) / len(results)
            avg_tokens_per_sec = sum(r['tokens_per_second'] for r in results) / len(results)
            avg_memory_used = sum(r['memory_used_mb'] for r in results) / len(results)
            
            return {
                'results': results,
                'avg_execution_time': avg_time,
                'avg_tokens_per_second': avg_tokens_per_sec,
                'avg_memory_used_mb': avg_memory_used,
                'total_iterations': len(results)
            }
        else:
            return {'error': 'no_successful_optimized_tests'}
    
    def _analyze_results(self, baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
        """結果分析・比較"""
        if 'error' in baseline or 'error' in optimized:
            return {'error': 'insufficient_data_for_comparison'}
        
        # 性能改善の計算
        time_improvement = (baseline['avg_execution_time'] - optimized['avg_execution_time']) / baseline['avg_execution_time'] * 100
        throughput_improvement = (optimized['avg_tokens_per_second'] - baseline['avg_tokens_per_second']) / baseline['avg_tokens_per_second'] * 100
        memory_change = optimized['avg_memory_used_mb'] - baseline['avg_memory_used_mb']
        
        # 並列効率
        theoretical_max_speedup = self.system_info['cpu_count_physical']  # 理論上の最大並列化
        actual_speedup = baseline['avg_execution_time'] / optimized['avg_execution_time']
        parallel_efficiency = (actual_speedup / theoretical_max_speedup) * 100
        
        comparison = {
            'execution_time': {
                'baseline': baseline['avg_execution_time'],
                'optimized': optimized['avg_execution_time'],
                'improvement_percent': time_improvement,
                'improvement_absolute': baseline['avg_execution_time'] - optimized['avg_execution_time']
            },
            'throughput': {
                'baseline_tokens_per_sec': baseline['avg_tokens_per_second'],
                'optimized_tokens_per_sec': optimized['avg_tokens_per_second'],
                'improvement_percent': throughput_improvement,
                'improvement_absolute': optimized['avg_tokens_per_second'] - baseline['avg_tokens_per_second']
            },
            'memory': {
                'baseline_mb': baseline['avg_memory_used_mb'],
                'optimized_mb': optimized['avg_memory_used_mb'],
                'change_mb': memory_change,
                'change_percent': (memory_change / baseline['avg_memory_used_mb']) * 100 if baseline['avg_memory_used_mb'] > 0 else 0
            },
            'parallelism': {
                'theoretical_max_speedup': theoretical_max_speedup,
                'actual_speedup': actual_speedup,
                'parallel_efficiency_percent': parallel_efficiency
            }
        }
        
        # 評価
        evaluation = {
            'overall_success': time_improvement > 0 and throughput_improvement > 0,
            'significant_improvement': time_improvement > 30 and throughput_improvement > 50,
            'meets_target': optimized['avg_tokens_per_second'] > 30,  # 目標: 30+ tokens/s
            'memory_efficient': memory_change < 100  # メモリ使用量の増加が100MB未満
        }
        
        comparison['evaluation'] = evaluation
        
        return comparison
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """結果を整形して出力"""
        print("\n" + "="*80)
        print("MurmurNet 真の並列実行性能ベンチマーク結果")
        print("="*80)
        
        if 'error' in results:
            print(f"❌ エラー: {results['error']}")
            return
        
        # システム情報
        print(f"\n【システム情報】")
        sys_info = results['system_info']
        print(f"CPU: {sys_info['cpu_count_physical']}コア/{sys_info['cpu_count_logical']}スレッド")
        print(f"メモリ: {sys_info['memory_total_gb']:.1f}GB")
        
        # ベースライン結果
        if 'baseline' in results and 'error' not in results['baseline']:
            baseline = results['baseline']
            print(f"\n【ベースライン（ThreadPool版）】")
            print(f"平均実行時間: {baseline['avg_execution_time']:.2f}秒")
            print(f"平均スループット: {baseline['avg_tokens_per_second']:.1f} tokens/s")
            print(f"平均メモリ使用量: {baseline['avg_memory_used_mb']:.1f}MB")
        
        # 最適化後結果
        if 'optimized' in results and 'error' not in results['optimized']:
            optimized = results['optimized']
            print(f"\n【最適化後（ProcessPool版）】")
            print(f"平均実行時間: {optimized['avg_execution_time']:.2f}秒")
            print(f"平均スループット: {optimized['avg_tokens_per_second']:.1f} tokens/s")
            print(f"平均メモリ使用量: {optimized['avg_memory_used_mb']:.1f}MB")
        
        # 比較結果
        if 'comparison' in results and 'error' not in results['comparison']:
            comp = results['comparison']
            print(f"\n【性能改善】")
            print(f"実行時間改善: {comp['execution_time']['improvement_percent']:.1f}% "
                  f"({comp['execution_time']['improvement_absolute']:.2f}秒短縮)")
            print(f"スループット向上: {comp['throughput']['improvement_percent']:.1f}% "
                  f"(+{comp['throughput']['improvement_absolute']:.1f} tokens/s)")
            print(f"メモリ使用量変化: {comp['memory']['change_percent']:.1f}% "
                  f"({comp['memory']['change_mb']:.1f}MB)")
            
            print(f"\n【並列性能】")
            print(f"理論最大並列化: {comp['parallelism']['theoretical_max_speedup']:.1f}倍")
            print(f"実現された並列化: {comp['parallelism']['actual_speedup']:.1f}倍")
            print(f"並列効率: {comp['parallelism']['parallel_efficiency_percent']:.1f}%")
            
            # 評価
            eval_results = comp['evaluation']
            print(f"\n【評価】")
            print(f"{'✅' if eval_results['overall_success'] else '❌'} 全体的な成功")
            print(f"{'✅' if eval_results['significant_improvement'] else '❌'} 大幅な改善 (30%+時間, 50%+スループット)")
            print(f"{'✅' if eval_results['meets_target'] else '❌'} 目標達成 (30+ tokens/s)")
            print(f"{'✅' if eval_results['memory_efficient'] else '❌'} メモリ効率 (<100MB増加)")
        
        print("\n" + "="*80)

async def main():
    """メイン関数"""
    print("MurmurNet 真の並列実行性能ベンチマーク")
    print("ProcessPool最適化の効果を測定します...")
    
    benchmark = ParallelBenchmark()
    
    # ベンチマーク実行
    results = await benchmark.run_benchmark(test_iterations=3)
    
    # 結果出力
    benchmark.print_results(results)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
