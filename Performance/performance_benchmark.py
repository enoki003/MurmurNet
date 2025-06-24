#!/usr/bin/env python3
"""
パフォーマンス最適化ベンチマークスクリプト
改善効果を定量的に測定し、目標タイムとの比較を行う
"""

import time
import psutil
import os
import json
from datetime import datetime

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.baseline = {
            "総所要時間": 31.28,
            "入力処理": 5.32,
            "ggufロード": 2.97,
            "要約エンジン": 5.03,
            "OutputAgent": 10.07,
            "Peak_MB": 1328
        }
        
        self.targets = {
            "ggufロード": 1.2,      # リパック効果
            "入力処理": 1.0,        # 最適化効果
            "要約エンジン": 0.0,    # 短文スキップ
            "OutputAgent": 7.0,     # 20t/s目標 (430token)
            "総所要時間": 12.0,     # 目標合計
            "Peak_MB": 1078         # メモリ削減目標
        }
    
    def start_timer(self, stage_name):
        """ステージ開始時刻を記録"""
        self.results[f"{stage_name}_start"] = time.time()
        print(f"[BENCHMARK] {stage_name} 開始")
    
    def end_timer(self, stage_name):
        """ステージ終了時刻を記録し、所要時間を計算"""
        if f"{stage_name}_start" not in self.results:
            print(f"警告: {stage_name} の開始時刻が記録されていません")
            return 0
        
        duration = time.time() - self.results[f"{stage_name}_start"]
        self.results[stage_name] = duration
        
        # 目標時間との比較
        target = self.targets.get(stage_name, None)
        if target:
            status = "✓" if duration <= target else "✗"
            print(f"[BENCHMARK] {stage_name}: {duration:.2f}s (目標: {target}s) {status}")
        else:
            print(f"[BENCHMARK] {stage_name}: {duration:.2f}s")
        
        return duration
    
    def measure_memory_peak(self):
        """現在のメモリ使用量を記録"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.results["Peak_MB"] = memory_mb
        
        target = self.targets["Peak_MB"]
        status = "✓" if memory_mb <= target else "✗"
        print(f"[BENCHMARK] Peak Memory: {memory_mb:.0f}MB (目標: {target}MB) {status}")
        
        return memory_mb
    
    def generate_report(self):
        """最適化効果レポートを生成"""
        print("\n" + "="*60)
        print("パフォーマンス最適化結果")
        print("="*60)
        
        print(f"{'指標':<15} {'初回':<8} {'今回':<8} {'目標':<8} {'差分':<10} {'達成'}")
        print("-" * 60)
        
        total_improvement = 0
        achieved_targets = 0
        total_targets = 0
        
        for stage, current in self.results.items():
            if stage.endswith("_start") or stage == "Peak_MB":
                continue
                
            baseline = self.baseline.get(stage, 0)
            target = self.targets.get(stage, None)
            
            if baseline > 0:
                diff = current - baseline
                diff_str = f"▼ {abs(diff):.2f}s" if diff < 0 else f"▲ {diff:.2f}s"
                total_improvement += abs(diff) if diff < 0 else 0
                
                if target:
                    achieved = "✓" if current <= target else "✗"
                    if current <= target:
                        achieved_targets += 1
                    total_targets += 1
                else:
                    achieved = "-"
                
                print(f"{stage:<15} {baseline:<8.2f} {current:<8.2f} {target or '-':<8} {diff_str:<10} {achieved}")
        
        # メモリ使用量
        if "Peak_MB" in self.results:
            memory_current = self.results["Peak_MB"]
            memory_baseline = self.baseline["Peak_MB"]
            memory_target = self.targets["Peak_MB"]
            memory_diff = memory_current - memory_baseline
            memory_diff_str = f"▼ {abs(memory_diff):.0f}MB" if memory_diff < 0 else f"▲ {memory_diff:.0f}MB"
            memory_achieved = "✓" if memory_current <= memory_target else "✗"
            
            print(f"{'Peak Memory':<15} {memory_baseline:<8.0f} {memory_current:<8.0f} {memory_target:<8.0f} {memory_diff_str:<10} {memory_achieved}")
            
            if memory_current <= memory_target:
                achieved_targets += 1
            total_targets += 1
        
        print("-" * 60)
        print(f"目標達成率: {achieved_targets}/{total_targets} ({achieved_targets/total_targets*100:.1f}%)")
        print(f"総改善時間: {total_improvement:.2f}s")
        
        # 結果をJSONで保存
        self.save_results()
        
        return achieved_targets / total_targets if total_targets > 0 else 0
    
    def save_results(self):
        """結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        report_data = {
            "timestamp": timestamp,
            "baseline": self.baseline,
            "targets": self.targets,
            "results": self.results,
            "improvements": {}
        }
        
        # 改善値を計算
        for stage in self.results:
            if stage.endswith("_start"):
                continue
            baseline = self.baseline.get(stage, 0)
            if baseline > 0:
                improvement = baseline - self.results[stage]
                report_data["improvements"][stage] = improvement
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n結果保存: {filename}")

# 使用例
def run_optimized_benchmark():
    """最適化版システムのベンチマーク実行"""
    bench = PerformanceBenchmark()
    
    # 各ステージの測定
    bench.start_timer("ggufロード")
    # ... ggufロード処理 ...
    bench.end_timer("ggufロード")
    
    bench.start_timer("入力処理")
    # ... 入力処理 ...
    bench.end_timer("入力処理")
    
    bench.start_timer("要約エンジン")
    # ... 要約処理（短文スキップ含む）...
    bench.end_timer("要約エンジン")
    
    bench.start_timer("OutputAgent")
    # ... OutputAgent推論 ...
    bench.end_timer("OutputAgent")
    
    # メモリ測定
    bench.measure_memory_peak()
    
    # 総所要時間計算
    total_time = sum(bench.results[k] for k in bench.results if not k.endswith("_start") and k != "Peak_MB")
    bench.results["総所要時間"] = total_time
    
    # レポート生成
    achievement_rate = bench.generate_report()
    
    return achievement_rate

if __name__ == "__main__":
    print("パフォーマンス最適化ベンチマーク開始")
    rate = run_optimized_benchmark()
    print(f"\n目標達成率: {rate*100:.1f}%")