#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGリトリーバーのZIMモードテスト
"""

import sys
import os
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# ここでmurmurnetのモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rag_retriever import RAGRetriever

def test_rag_zim():
    """ZIMモードのRAGリトリーバーをテスト"""
    print("RAGリトリーバー ZIMモードテスト開始")
    
    # ZIMファイルのパスを設定
    zim_path = r"C:\Users\園木優陽\AppData\Roaming\kiwix-desktop\wikipedia_en_top_nopic_2025-03.zim"
    
    # 設定
    config = {
        "rag_mode": "zim",
        "zim_path": zim_path,
        "rag_score_threshold": 0.5,
        "rag_top_k": 3,
        "debug": True,
        "embedding_model": "all-MiniLM-L6-v2",
    }
    
    try:
        # RAGリトリーバーの初期化
        print(f"ZIMファイル: {zim_path}")
        print("RAGリトリーバー初期化中...")
        retriever = RAGRetriever(config)
        
        # モードの確認
        print(f"実際の動作モード: {retriever.mode}")
        
        if retriever.mode == "dummy":
            print("警告: dummyモードにフォールバックしました。理由を確認してください。")
            return False
        
        # いくつかのクエリでテスト
        test_queries = [
            "What is artificial intelligence?",
            "太陽系について教えて",
            "Albert Einstein's theory of relativity"
        ]
        
        for query in test_queries:
            print(f"\nクエリ: {query}")
            print("-" * 40)
            result = retriever.retrieve(query)
            print(f"結果: \n{result[:500]}...")
            print("-" * 40)
        
        print("\nRAGリトリーバー ZIMモードテスト完了")
        return True
    
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rag_zim() 