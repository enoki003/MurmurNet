#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡単なRAGテスト
"""
import sys
sys.path.append('.')

from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.modules.config_manager import get_config

def test_rag():
    print("RAGリトリーバーテストを開始...")
    
    config = get_config().to_dict()
    print(f"RAGモード: {config.get('rag_mode', 'unknown')}")
    
    try:
        rag = RAGRetriever(config)
        print(f"実際の動作モード: {rag.mode}")
        
        # 簡単な検索テスト
        result = rag.retrieve("artificial intelligence")
        print(f"検索結果: {result[:200]}...")
        
        print("RAGテスト完了！")
        return True
        
    except Exception as e:
        print(f"RAGテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rag()
