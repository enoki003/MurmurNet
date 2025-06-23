#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet CPU最適化テストスクリプト
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU最適化の効果を測定し、パフォーマンス指標を評価する

作者: Yuhi Sonoki
"""

import os
import sys
import time
import yaml
import asyncio
import logging
from typing import Dict, Any, List

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MurmurNet.distributed_slm import DistributedSLM
from MurmurNet.modules.performance import PerformanceMonitor

class CPUOptimizationTester:
    """CPU最適化効果のテスト・評価クラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        テスターの初期化
        
        引数:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # ロギング設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CPUOptimizationTester')
        
        # テストケース
        self.test_cases = [
            "日本の人口について教えてください。",
            "AIと機械学習の違いは何ですか？",
            "環境問題の解決策について議論してください。",
            "プログラミングを学ぶ効果的な方法は？",
            "健康的な生活習慣について説明してください。"
        ]
        
        # パフォーマンス監視
        self.performance = PerformanceMonitor(enabled=True, memory_tracking=True)
        
    def load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return {}
    
    async def run_single_test(self, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        単一テストケースの実行
        
        引数:
            question: テスト質問
            config: 設定辞書
            
        戻り値:
            テスト結果
        """
        self.logger.info(f"テスト実行: {question[:50]}...")
        
        # パフォーマンス測定開始
        start_time = time.time()
        initial_memory = self.performance.get_memory_usage()
        initial_cpu = self.performance.get_cpu_usage()
        
        try:
            # MurmurNetインスタンスの作成
            murmur_net = DistributedSLM(config)
            
            # 応答生成
            response = await murmur_net.generate(question)
            
            # 測定終了
            end_time = time.time()
            final_memory = self.performance.get_memory_usage()
            final_cpu = self.performance.get_cpu_usage()
            
            # 結果の整理
            result = {
                'question': question,
                'response': response,
                'execution_time': end_time - start_time,
                'response_length': len(response),
                'tokens_per_second': len(response.split()) / (end_time - start_time) if end_time > start_time else 0,
                'memory_usage': {
                    'initial_rss': initial_memory.get('rss', 0),
                    'final_rss': final_memory.get('rss', 0),
                    'memory_increase': final_memory.get('rss', 0) - initial_memory.get('rss', 0)
                },
                'cpu_usage': {
                    'initial_cpu': initial_cpu.get('system_cpu_percent', 0),
                    'final_cpu': final_cpu.get('system_cpu_percent', 0)
                },
                'config_used': {
                    'num_agents': config.get('num_agents', 2),
                    'iterations': config.get('iterations', 1),
                    'use_parallel': config.get('use_parallel', False),
                    'n_threads': config.get('n_threads', 4)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"テスト実行エラー: {e}")
            return {
                'question': question,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def run_performance_comparison(self) -> Dict[str, Any]:
        """
        並列処理と逐次処理のパフォーマンス比較
        
        戻り値:
            比較結果
        """
        self.logger.info("=== パフォーマンス比較テスト開始 ===")
        
        results = {
            'sequential': [],
            'parallel': [],
            'comparison': {}
        }
        
        # 逐次処理テスト
        self.logger.info("1. 逐次処理テスト")
        sequential_config = self.config.copy()
        sequential_config['use_parallel'] = False
        sequential_config['num_agents'] = 3
        sequential_config['iterations'] = 2
        
        for question in self.test_cases:
            result = await self.run_single_test(question, sequential_config)
            results['sequential'].append(result)
            await asyncio.sleep(1)  # システム安定化のための待機
        
        # 並列処理テスト
        self.logger.info("2. 並列処理テスト")
        parallel_config = self.config.copy()
        parallel_config['use_parallel'] = True
        parallel_config['num_agents'] = 3
        parallel_config['iterations'] = 2
        
        for question in self.test_cases:
            result = await self.run_single_test(question, parallel_config)
            results['parallel'].append(result)
            await asyncio.sleep(1)  # システム安定化のための待機
        
        # 比較分析
        results['comparison'] = self.analyze_results(
            results['sequential'], 
            results['parallel']
        )
        
        return results
    
    def analyze_results(self, sequential_results: List[Dict], parallel_results: List[Dict]) -> Dict[str, Any]:
        """
        テスト結果の分析
        
        引数:
            sequential_results: 逐次処理結果
            parallel_results: 並列処理結果
            
        戻り値:
            分析結果
        """
        def calculate_average(results: List[Dict], key: str) -> float:
            valid_results = [r for r in results if key in r and 'error' not in r]
            if not valid_results:
                return 0.0
            return sum(r[key] for r in valid_results) / len(valid_results)
        
        # 平均実行時間
        seq_avg_time = calculate_average(sequential_results, 'execution_time')
        par_avg_time = calculate_average(parallel_results, 'execution_time')
        
        # 平均トークン/秒
        seq_avg_tps = calculate_average(sequential_results, 'tokens_per_second')
        par_avg_tps = calculate_average(parallel_results, 'tokens_per_second')
        
        # スピードアップ率
        speedup = seq_avg_time / par_avg_time if par_avg_time > 0 else 0
        
        # 効率性（スレッド効率）
        thread_efficiency = speedup / self.config.get('n_threads', 4) * 100
        
        analysis = {
            'sequential_avg_time': seq_avg_time,
            'parallel_avg_time': par_avg_time,
            'speedup_ratio': speedup,
            'sequential_avg_tps': seq_avg_tps,
            'parallel_avg_tps': par_avg_tps,
            'thread_efficiency_percent': thread_efficiency,
            'performance_improvement_percent': ((seq_avg_time - par_avg_time) / seq_avg_time * 100) if seq_avg_time > 0 else 0,
            'meets_target': par_avg_time <= 2.0,  # 目標: 2秒以内
            'cpu_utilization_improvement': par_avg_tps / seq_avg_tps if seq_avg_tps > 0 else 0
        }
        
        return analysis
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        テスト結果の表示
        
        引数:
            results: テスト結果
        """
        print("\n" + "="*80)
        print("MurmurNet CPU最適化テスト結果")
        print("="*80)
        
        comparison = results['comparison']
        
        print(f"\n📊 パフォーマンス比較:")
        print(f"  逐次処理平均時間: {comparison['sequential_avg_time']:.2f}秒")
        print(f"  並列処理平均時間: {comparison['parallel_avg_time']:.2f}秒")
        print(f"  スピードアップ率: {comparison['speedup_ratio']:.2f}倍")
        print(f"  性能向上率: {comparison['performance_improvement_percent']:.1f}%")
        
        print(f"\n🚀 スループット比較:")
        print(f"  逐次処理: {comparison['sequential_avg_tps']:.1f} tokens/sec")
        print(f"  並列処理: {comparison['parallel_avg_tps']:.1f} tokens/sec")
        print(f"  スループット向上: {comparison['cpu_utilization_improvement']:.2f}倍")
        
        print(f"\n⚡ CPU効率:")
        print(f"  スレッド効率: {comparison['thread_efficiency_percent']:.1f}%")
        
        print(f"\n🎯 目標達成状況:")
        target_met = "✅ 達成" if comparison['meets_target'] else "❌ 未達成"
        print(f"  2秒以内応答: {target_met} ({comparison['parallel_avg_time']:.2f}秒)")
        
        # 詳細結果
        print(f"\n📋 詳細結果:")
        print("  逐次処理:")
        for i, result in enumerate(results['sequential'][:3]):  # 最初の3件のみ表示
            if 'error' not in result:
                print(f"    テスト{i+1}: {result['execution_time']:.2f}秒 ({result['tokens_per_second']:.1f} tok/s)")
            else:
                print(f"    テスト{i+1}: エラー - {result['error']}")
        
        print("  並列処理:")
        for i, result in enumerate(results['parallel'][:3]):  # 最初の3件のみ表示
            if 'error' not in result:
                print(f"    テスト{i+1}: {result['execution_time']:.2f}秒 ({result['tokens_per_second']:.1f} tok/s)")
            else:
                print(f"    テスト{i+1}: エラー - {result['error']}")
        
        print("\n" + "="*80)

async def main():
    """メイン実行関数"""
    print("MurmurNet CPU最適化テストを開始します...")
    
    try:
        # テスターの作成
        tester = CPUOptimizationTester()
        
        # パフォーマンス比較テストの実行
        results = await tester.run_performance_comparison()
        
        # 結果の表示
        tester.print_results(results)
        
        # 結果をファイルに保存
        import json
        with open('cpu_optimization_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 詳細結果を 'cpu_optimization_test_results.json' に保存しました。")
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windowsでの非同期実行の最適化
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
