#!/usr/bin/env python3
"""
事前リパック済みGGUFファイル作成スクリプト
毎回のリパック処理（約2.5s）を削減するため、起動前に一度だけ実行する
"""

import subprocess
import sys
import os
from pathlib import Path

def create_repacked_gguf():
    """元のGGUFファイルをリパックして最適化済みファイルを作成"""
    
    # 入力ファイルパス（既存のGGUFファイル）
    input_file = "gemma-3-1b-it-q4.gguf"  # 元のファイル名に合わせて調整
    output_file = "gemma-3-1b-it-q4_RP.gguf"  # リパック済みファイル
    
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイル {input_file} が見つかりません")
        return False
    
    if os.path.exists(output_file):
        print(f"リパック済みファイル {output_file} は既に存在します")
        return True
    
    print(f"リパック処理開始: {input_file} → {output_file}")
    
    try:
        # convert.py を使用してリパック実行
        cmd = [
            sys.executable, "./convert.py", 
            "--repack", 
            "--outfile", output_file,
            input_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"リパック完了: {output_file}")
            print(f"ファイルサイズ: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
            return True
        else:
            print(f"リパック失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"エラー: {e}")
        return False

if __name__ == "__main__":
    success = create_repacked_gguf()
    if success:
        print("\n次のステップ:")
        print("1. 起動スクリプトのモデルパスを gemma-3-1b-it-q4_RP.gguf に変更")
        print("2. ロード時間が約1秒に短縮されることを確認")
    else:
        print("リパック処理に失敗しました")
        sys.exit(1)