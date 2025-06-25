#!/usr/bin/env python3
"""
ファイル移動後の整合性確認スクリプト
"""

import os
from pathlib import Path

def check_performance_files():
    """Performance ディレクトリのファイルを確認"""
    performance_dir = Path("Performance")
    
    expected_files = [
        "__init__.py",
        "create_repacked_model.py",
        "template_optimizer.py", 
        "summary_optimizer.py",
        "output_agent_optimizer.py",
        "performance_benchmark.py",
        "run_optimization.py"
    ]
    
    print("=== Performance ディレクトリ ファイル確認 ===")
    
    for file in expected_files:
        file_path = performance_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - 見つかりません")
    
    print("\n=== メインディレクトリ 実行スクリプト確認 ===")
    
    main_scripts = [
        "optimize.py",
        "benchmark.py", 
        "create_model.py"
    ]
    
    for script in main_scripts:
        if Path(script).exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} - 見つかりません")

def main():
    print("MurmurNet Performance ファイル構造確認")
    print("=" * 50)
    check_performance_files()
    print("\n整理完了！")
    print("実行方法:")
    print("  python create_model.py  # 事前リパックモデル作成")
    print("  python optimize.py      # 統合最適化実行")
    print("  python benchmark.py     # ベンチマーク測定")

if __name__ == "__main__":
    main()
