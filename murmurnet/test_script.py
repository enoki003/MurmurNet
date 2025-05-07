#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet テストスクリプト
~~~~~~~~~~~~~~~~~~~~~~~
分散創発型言語モデルシステムのテスト用スクリプト
- モジュール単体テスト
- 統合テスト
- 性能測定
- 機能テスト（反復、要約、並列）

作者: Yuhi Sonoki
"""

import sys
import os
import logging
import asyncio
import time
import gc
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
from murmurnet.modules.summary_engine import SummaryEngine

# 標準設定（異なる環境でも動作するように絶対パス）
MODELS_PATH = r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\"
DEFAULT_CONFIG = {
    "num_agents": 2,
    "iterations": 1,
    "use_summary": True,
    "use_parallel": False,
    "rag_mode": "dummy",  # dummy モードなら外部ファイルなしで動作
    "rag_score_threshold": 0.5,
    "rag_top_k": 1,
    "debug": True,
    "model_path": MODELS_PATH + r"gemma-3-1b-it-q4_0.gguf",
    "chat_template": MODELS_PATH + r"gemma3_template.txt",
    "n_threads": 4,  # スレッド数制限
    "n_ctx": 2048,   # コンテキスト縮小
    "params": MODELS_PATH + r"gemma3_params.json"
}

# グローバルSLMインスタンス（再利用のため）
_slm_instance = None

def get_slm_instance(config=None):
    """再利用可能なSLMインスタンスを取得"""
    global _slm_instance
    if _slm_instance is None or config is not None:
        if _slm_instance is not None:
            # メモリクリア
            del _slm_instance
            gc.collect()
        _slm_instance = DistributedSLM(config or DEFAULT_CONFIG)
    return _slm_instance

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
    
    def test_summary_engine(self):
        """要約エンジンのテスト"""
        # 軽量なテスト用の設定
        test_config = self.config.copy()
        test_config["n_ctx"] = 1024
        
        summary_engine = SummaryEngine(test_config)
        entries = [
            {"agent": 0, "text": "これはテスト文章1です。AIの将来性について議論します。"},
            {"agent": 1, "text": "テスト文章2です。技術の発展は人類に恩恵をもたらします。"}
        ]
        result = summary_engine.summarize_blackboard(entries)
        
        # 要約は空でないはず
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

# ========== 統合テスト ==========

async def test_integration():
    """分散SLMの統合テスト"""
    logging.info("統合テスト開始")
    
    # インスタンス生成
    start_time = time.time()
    # グローバルインスタンスを利用
    slm = get_slm_instance()
    init_time = time.time() - start_time
    logging.info(f"初期化時間: {init_time:.2f}秒")
    
    # テスト用クエリの削減
    test_queries = [
        "AIは教育をどのように変えますか？"
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

# ========== 機能テスト ==========

async def test_iterative_summary():
    """反復と要約機能のテスト"""
    print_header("反復と要約テスト")
    
    # 設定: 2回の反復と要約有効
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1  # 反復回数を減らす
    config["use_summary"] = True
    config["num_agents"] = 2
    
    print(f"設定: {config['iterations']}回反復, 要約有効, {config['num_agents']}エージェント")
    
    # 既存インスタンスを更新
    slm = get_slm_instance(config)
    query = "気候変動の解決策について考察してください"
    
    print(f"入力: {query}")
    start_time = time.time()
    response = await slm.generate(query)
    total_time = time.time() - start_time
    
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"出力の一部: {response[:100]}...")
    
    # 中間要約の確認
    for i in range(config["iterations"]):
        summary = slm.blackboard.read(f'summary_{i}')
        if summary:
            print(f"反復{i+1}の要約: {summary[:100]}...")
    
    return True

async def test_parallel_processing():
    """並列処理機能のテスト"""
    print_header("並列処理テスト")
    
    # 設定: 並列処理有効、エージェント数増加
    config = DEFAULT_CONFIG.copy()
    config["use_parallel"] = True
    config["num_agents"] = 2  # エージェント数削減
    config["iterations"] = 1
    
    print(f"設定: 並列処理有効, {config['num_agents']}エージェント")
    
    # 同じインスタンスで設定変更して実行（逐次）
    slm = get_slm_instance(config)
    config["use_parallel"] = False
    slm.use_parallel = False
    
    query = "複雑な哲学的問題: 意識とは何か？"
    print(f"入力: {query}")
    
    # 逐次処理
    print("逐次処理実行中...")
    start_time = time.time()
    response_normal = await slm.generate(query)
    normal_time = time.time() - start_time
    print(f"逐次処理時間: {normal_time:.2f}秒")
    
    # 同じインスタンスで並列処理に切り替え
    slm.use_parallel = True
    
    # 並列処理
    print("並列処理実行中...")
    start_time = time.time()
    response_parallel = await slm.generate(query)
    parallel_time = time.time() - start_time
    print(f"並列処理時間: {parallel_time:.2f}秒")
    
    # 速度比較
    if normal_time > 0:
        speedup = normal_time / parallel_time
        print(f"速度向上率: {speedup:.2f}倍")
    
    return True

# ========== RAG ZIMモードテスト ==========

async def test_rag_zim_mode():
    """RAGリトリーバーのZIMモードをテスト"""
    print_header("RAGリトリーバー ZIMモードテスト")
    
    # 設定: ZIMモード有効
    config = DEFAULT_CONFIG.copy()
    config["rag_mode"] = "zim"
    config["zim_path"] = r"C:\Users\園木優陽\AppData\Roaming\kiwix-desktop\wikipedia_en_top_nopic_2025-03.zim"
    config["rag_score_threshold"] = 0.5
    config["rag_top_k"] = 3
    config["debug"] = True
    
    try:
        # RAGリトリーバーの初期化
        print(f"ZIMファイル: {config['zim_path']}")
        print("RAGリトリーバー初期化中...")
        rag = RAGRetriever(config)
        
        # モードの確認
        print(f"実際の動作モード: {rag.mode}")
        
        if rag.mode == "dummy":
            print("警告: ZIMモードではなくdummyモードで動作しています。以下の理由が考えられます:")
            print("- ZIMファイルが存在しない")
            print("- libzimがインストールされていない")
            print("- sentence-transformersがインストールされていない")
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
            result = rag.retrieve(query)
            print(f"結果: \n{result[:300]}...")  # 長い結果は省略
            print("-" * 40)
        
        print("\nZIMモードテスト完了")
        return True
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== メイン実行部 ==========

def print_header(title):
    """セクションヘッダの表示"""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

async def main():
    """テストスクリプトのメイン関数"""
    print_header("MurmurNet テストスクリプト")
    
    try:
        # RAG ZIMモードのテスト（追加）
        print_header("RAG ZIMモードテスト")
        await test_rag_zim_mode()
        
        # 順序を変更：統合テスト→機能テスト→単体テストの順で
        
        # 統合テスト 
        print_header("統合テスト")
        success = await test_integration()
        if success:
            print("✓ 統合テスト成功")
        else:
            print("✗ 統合テスト失敗")
        
        # 反復と要約のテスト
        await test_iterative_summary()
        
        # 並列処理テスト
        await test_parallel_processing()
        
        # 単一クエリテスト（最終チェック）
        print_header("単一クエリテスト (最終)")
        config = DEFAULT_CONFIG.copy()
        config["iterations"] = 1  # 反復回数を減らす
        config["use_summary"] = True
        config["num_agents"] = 2  # エージェント数削減
        
        # 既存インスタンスを更新
        slm = get_slm_instance(config)
        
        query = "人工知能と人間の関係はどのように発展するでしょうか？"
        print(f"入力: {query}")
        
        response = await slm.generate(query)
        print(f"出力: {response}")
        
        # 最後に単体テスト（LLMを使わない軽量テスト）
        print_header("モジュール単体テスト")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
        # リソース解放
        global _slm_instance
        if _slm_instance is not None:
            del _slm_instance
            _slm_instance = None
            gc.collect()
        
        print("\nテスト完了")
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
