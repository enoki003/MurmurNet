#!/usr/bin/env python3
"""
事前リパックモデル作成スクリプト
"""

import sys
import os
from pathlib import Path

# Performanceディレクトリをパスに追加
performance_dir = Path(__file__).parent / "Performance"
sys.path.insert(0, str(performance_dir))

def main():
    """メイン実行関数"""
    print("=== MurmurNet 事前リパックモデル作成 ===")
    
    try:
        # モジュールをインポート
        from create_repacked_model import create_repacked_gguf
        
        # リパック処理を実行
        success = create_repacked_gguf()
        
        if success:
            print("事前リパック済みモデルの作成が完了しました！")
            return 0
        else:
            print("モデル作成中にエラーが発生しました。")
            return 1
            
    except ImportError as e:
        print(f"モジュールのインポートエラー: {e}")
        print("Performance/create_repacked_model.py を確認してください。")
        return 1
    except Exception as e:
        print(f"モデル作成中にエラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
