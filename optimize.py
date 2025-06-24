#!/usr/bin/env python3
"""
パフォーマンス最適化統合実行スクリプト
Performanceディレクトリ内の最適化ツールを実行
"""

import sys
import os
from pathlib import Path

# Performanceディレクトリをパスに追加
performance_dir = Path(__file__).parent / "Performance"
sys.path.insert(0, str(performance_dir))

def main():
    """メイン実行関数"""
    print("=== MurmurNet パフォーマンス最適化 ===")
    print("統合最適化を開始します...")
    
    try:
        # 最適化モジュールをインポート
        from run_optimization import SystemOptimizer
        
        # システム最適化を実行
        optimizer = SystemOptimizer()
        optimizer.run_full_optimization()
        
        print("\n最適化が完了しました！")
        print("ベンチマークでパフォーマンス向上を確認してください。")
        
    except ImportError as e:
        print(f"モジュールのインポートエラー: {e}")
        print("Performanceディレクトリ内のファイルを確認してください。")
        return 1
    except Exception as e:
        print(f"最適化実行中にエラーが発生しました: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
