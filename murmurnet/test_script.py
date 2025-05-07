#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet テストスクリプト
~~~~~~~~~~~~~~~~~~~~~~~
分散創発型言語モデルシステムのテスト用スクリプト
- モジュール単体テスト
- 統合テスト
- 性能測定

作者: MurmurNetチーム
"""

import sys
import os
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
import unittest

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("test_log.txt"),
        logging.StreamHandler()
    ]
)

# モジュールパスの設定
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from murmurnet.distributed_slm import DistributedSLM
from murmurnet.modules.blackboard import Blackboard
from murmurnet.modules.input_reception import InputReception
from murmurnet.modules.agent_pool import AgentPoolManager
from murmurnet.modules.rag_retriever import RAGRetriever
from murmurnet.modules.output_agent import OutputAgent

# 標準設定（異なる環境でも動作するように絶対パス）
MODELS_PATH = r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\"
DEFAULT_CONFIG = {
    "num_agents": 2,
    "rag_mode": "dummy",  # dummy モードなら外部ファイルなしで動作
    "rag_score_threshold": 0.5,
    "rag_top_k": 1,
    "debug": True,
    "model_path": MODELS_PATH + r"gemma-3-1b-it-q4_0.gguf",
    "chat_template": MODELS_PATH + r"gemma3_template.txt",
    "params": MODELS_PATH + r"gemma3_params.json"
}

# ========== モジュール単体テスト ==========

class TestModules(unittest.TestCase):
    """各モジュールの基本機能テスト"""
    
    def setUp(self):
        self.config = DEFAULT_CONFIG.copy()
        self.blackboard = Blackboard(self.config)
    
    def test_blackboard(self):
        """ブラックボードの基本機能テスト"""
        # 書き込みテスト
        entry = self.blackboard.write("test_key", "test_value")
        self.assertEqual(entry["key"], "test_key")
        self.assertEqual(entry["value"], "test_value")
        
        # 読み込みテスト
        value = self.blackboard.read("test_key")
        self.assertEqual(value, "test_value")
        
        # 履歴テスト
        history = self.blackboard.get_history("test_key")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["value"], "test_value")
    
    def test_input_reception(self):
        """入力処理モジュールのテスト"""
        input_reception = InputReception(self.config)
        result = input_reception.process("Hello, World!")
        
        # 正規化テスト
        self.assertIn("normalized", result)
        self.assertIsInstance(result["normalized"], str)
        
        # トークン化テスト
        self.assertIn("tokens", result)
        self.assertIsInstance(result["tokens"], list)
        
        # 埋め込みテスト
        self.assertIn("embedding", result)
    
    def test_rag_retriever(self):
        """RAG検索モジュールのテスト（ダミーモード）"""
        rag = RAGRetriever(self.config)
        result = rag.retrieve("テスト用クエリ")
        
        # 何らかの文字列が返却されるはず
        self.assertIsInstance(result, str)

# ========== 統合テスト ==========

async def test_integration():
    """分散SLMの統合テスト"""
    logging.info("統合テスト開始")
    
    # インスタンス生成
    start_time = time.time()
    slm = DistributedSLM(DEFAULT_CONFIG)
    init_time = time.time() - start_time
    logging.info(f"初期化時間: {init_time:.2f}秒")
    
    # 生成処理
    test_queries = [
        "AIは教育をどのように変えますか？",
        "What is the future of technology?",
        "宇宙探査の意義は何ですか？"
    ]
    
    for query in test_queries:
        logging.info(f"テストクエリ: {query}")
        
        start_time = time.time()
        response = await slm.generate(query)
        gen_time = time.time() - start_time
        
        logging.info(f"生成時間: {gen_time:.2f}秒")
        logging.info(f"応答: {response[:100]}...")
        
        # 黒板内容の確認
        bb_entries = len(slm.blackboard.history)
        logging.info(f"黒板エントリ数: {bb_entries}")
    
    logging.info("統合テスト終了")
    return True

# ========== メイン実行部 ==========

def print_header(title):
    """セクションヘッダの表示"""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

async def main():
    """テストスクリプトのメイン関数"""
    print_header("MurmurNet テストスクリプト")
    
    # 単体テスト
    print_header("モジュール単体テスト")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 統合テスト
    print_header("統合テスト")
    success = await test_integration()
    if success:
        print("✓ 統合テスト成功")
    else:
        print("✗ 統合テスト失敗")
    
    # 単一クエリテスト
    print_header("単一クエリテスト")
    config = DEFAULT_CONFIG.copy()
    slm = DistributedSLM(config)
    
    query = "人工知能と人間の関係はどのように発展するでしょうか？"
    print(f"入力: {query}")
    
    response = await slm.generate(query)
    print(f"出力: {response}")
    
    print("\nテスト完了")

if __name__ == "__main__":
    asyncio.run(main())
