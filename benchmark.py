#!/usr/bin/env python3
"""
パフォーマンスベンチマーク実行スクリプト
"""

import sys
import os
from pathlib import Path

# Performanceディレクトリをパスに追加
performance_dir = Path(__file__).parent / "Performance"
sys.path.insert(0, str(performance_dir))

def main():
    """メイン実行関数"""
    print("=== MurmurNet パフォーマンスベンチマーク ===")
    
    try:
        # ベンチマークモジュールをインポート
        from performance_benchmark import PerformanceBenchmark
        
        # ベンチマークを実行
        benchmark = PerformanceBenchmark()
        benchmark.run_full_benchmark()
        
    except ImportError as e:
        print(f"モジュールのインポートエラー: {e}")
        print("Performance/performance_benchmark.py を確認してください。")
        return 1
    except Exception as e:
        print(f"ベンチマーク実行中にエラーが発生しました: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
