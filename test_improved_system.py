#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改善されたMurmurNetシステムのテスト
"""

import os
import sys

# MurmurNetのパスを追加
sys.path.insert(0, os.path.dirname(__file__))

from MurmurNet.modules.model_factory import clear_model_cache

def main():
    print("=== MurmurNet 改善版テスト ===")
    
    # 古いモデルキャッシュをクリア（n_ctxの変更を反映させるため）
    print("1. モデルキャッシュをクリア中...")
    clear_model_cache()
    
    # システムのインポートとテスト
    from MurmurNet.distributed_slm import DistributedSLMSystem
    
    # 設定ファイルのパス
    config_path = "config.yaml"
    
    # システムの初期化
    print("2. システムを初期化中...")
    system = DistributedSLMSystem(config_path)
    
    # テスト質問
    test_questions = [
        "AIは教育をどう変える？",
        "機械学習の基本概念を教えて"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n=== テスト {i}: {question} ===")
        try:
            response = system.process_input(question)
            print(f"応答: {response}")
            print(f"応答長: {len(response)}文字")
        except Exception as e:
            print(f"エラー: {e}")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()
