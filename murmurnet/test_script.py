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
- 黒板アーキテクチャと会話記憶の統合テスト

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
# インポートパスを修正
from MurmurNet.distributed_slm import DistributedSLM
from MurmurNet.modules.blackboard import Blackboard
from MurmurNet.modules.input_reception import InputReception
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.modules.output_agent import OutputAgent
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.conversation_memory import ConversationMemory

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
        
        if (rag.mode == "dummy"):
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

# ========== 新機能テスト ==========

async def test_answer_quality():
    """回答の適切さ機能テスト"""
    print_header("質問適切性テスト")
    
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1
    config["num_agents"] = 2
    
    slm = get_slm_instance(config)
    
    # 質問の適切性をテストする質問リスト
    test_questions = [
        "AIは教育をどのように変えると思う？",  # 実行タスクの例
        "地球温暖化の主な原因は何ですか？",     # 明確な事実質問
        "量子コンピュータの将来性について",     # 技術予測質問
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n質問 {i+1}: {question}")
        print("-" * 40)
        
        start_time = time.time()
        response = await slm.generate(question)
        gen_time = time.time() - start_time
        
        print(f"応答時間: {gen_time:.2f}秒")
        print(f"応答: \n{response}")
        print("-" * 40)
    
    return True

async def test_conversation_memory():
    """会話記憶機能テスト"""
    print_header("会話履歴記憶テスト")
    
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1
    config["num_agents"] = 2
    config["use_memory"] = True  # 会話履歴機能を有効化
    
    # 新しいインスタンスで記憶をクリーンな状態から始める
    slm = get_slm_instance(config)
    
    # 記憶をリセット（直接conversation_memoryを使用）
    if hasattr(slm, 'conversation_memory'):
        slm.conversation_memory.clear_memory()
        print("会話履歴をリセットしました")
    
    # 会話の流れをテストする質問シーケンス
    conversation = [
        "こんにちは、私の名前は園木です。",
        "私の趣味について知りたいですか？",
        "私はプログラミングとピアノが好きです。",
        "私の名前は何でしたか？",  # 以前の会話を覚えているかテスト
    ]
    
    for i, message in enumerate(conversation):
        print(f"\n会話ターン {i+1}: {message}")
        print("-" * 40)
        
        response = await slm.generate(message)
        print(f"応答: {response}")
        
        # 会話コンテキストの状態を表示
        if i > 0 and 'conversation_context' in slm.blackboard.memory:
            context = slm.blackboard.read('conversation_context')
            print(f"会話コンテキスト: {context}")
    
    return True

async def test_role_assignment():
    """役割振り分けモジュールテスト"""
    print_header("役割振り分けモジュールテスト")
    
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1
    config["num_agents"] = 3  # より多くのエージェントで多様な役割を確認
    
    slm = get_slm_instance(config)
    
    # 異なるタイプの質問でテスト
    test_questions = [
        {"text": "AIと人間の協調と競争について議論してください", "type": "discussion"},
        {"text": "新しいモバイルアプリのビジネスプランを考えてください", "type": "planning"},
        {"text": "量子力学の基本原理を簡単に説明してください", "type": "informational"},
        {"text": "今日の気分はどうですか？", "type": "conversational"},
    ]
    
    for i, question_data in enumerate(test_questions):
        question = question_data["text"]
        expected_type = question_data["type"]
        
        print(f"\n質問 {i+1}: {question}")
        print(f"期待される質問タイプ: {expected_type}")
        print("-" * 40)
        
        # 質問タイプと役割の判定
        normalized_question = {"normalized": question}
        slm.blackboard.write('input', normalized_question)
        slm.agent_pool.update_roles_based_on_question(question)
        
        # 実際に判定された質問タイプを表示
        actual_type = slm.blackboard.read('question_type')
        print(f"判定された質問タイプ: {actual_type}")
        
        # 選択されたエージェント役割を表示
        print("選択されたエージェント役割:")
        for j in range(slm.num_agents):
            role_idx = j % len(slm.agent_pool.agent_roles)
            role = slm.agent_pool.agent_roles[role_idx]
            print(f"  エージェント{j+1}: {role['role']}")
        
        # 応答生成
        response = await slm.generate(question)
        print(f"応答: {response[:150]}...")  # 長い場合は省略
    
    return True

async def test_blackboard_conversation_memory():
    """黒板と統合した会話記憶テスト"""
    print_header("黒板と統合した会話記憶テスト")
    
    # 設定
    config = DEFAULT_CONFIG.copy()
    config["use_memory"] = True
    
    # 新しい黒板インスタンスを作成
    blackboard = Blackboard(config)
    
    # 会話記憶インスタンスを黒板と統合して作成
    conversation_memory = ConversationMemory(config, blackboard=blackboard)
    
    print("会話記憶モジュールを黒板と統合しました")
    
    # テスト用の会話データ
    test_conversations = [
        # (ユーザー入力, システム応答)
        ("こんにちは、私の名前は太郎です。", "太郎さん、こんにちは！お元気ですか？"),
        ("東京に住んでいます。", "東京は素晴らしい都市ですね。何か東京について質問はありますか？"),
        ("趣味はプログラミングです。", "プログラミングは素晴らしい趣味ですね。どんな言語を使いますか？"),
        ("Pythonが好きです。", "Pythonは汎用性が高く素晴らしい言語ですね！何か作っているものはありますか？")
    ]
    
    # 会話を順番に追加
    print("\n会話の追加:")
    for i, (user_input, system_response) in enumerate(test_conversations):
        print(f"\n会話 {i+1}:")
        print(f"ユーザー: {user_input}")
        print(f"システム: {system_response}")
        
        # 会話を記憶に追加
        conversation_memory.add_conversation_entry(user_input, system_response)
        
        # 黒板に保存された内容を確認
        print("\n黒板の状態確認:")
        if blackboard.read("conversation_history"):
            print(f"- 履歴エントリ数: {len(blackboard.read('conversation_history'))}")
        if blackboard.read("conversation_context"):
            context = blackboard.read("conversation_context")
            print(f"- コンテキスト: {context[:100]}...")
        if blackboard.read("conversation_key_facts"):
            facts = blackboard.read("conversation_key_facts")
            print("- 抽出された重要な情報:")
            for category, items in facts.items():
                if items:
                    print(f"  {category}: {', '.join(items)}")
    
    # コンテキスト取得テスト
    context = conversation_memory.get_conversation_context()
    print("\n最終的な会話コンテキスト:")
    print(f"{context[:200]}...")
    
    # 重要情報の取得テスト
    key_facts = conversation_memory.get_key_facts()
    print("\n抽出された重要情報:")
    for category, items in key_facts.items():
        if items:
            print(f"{category}: {', '.join(items)}")
    
    # 新しいインスタンスを作成して読み込みテスト
    print("\n新インスタンスで黒板からの読み込みテスト:")
    new_memory = ConversationMemory(config, blackboard=blackboard)
    print(f"- 履歴エントリ数: {len(new_memory.conversation_history)}")
    print(f"- 名前の記憶: {new_memory.key_facts.get('names', [])}")
    
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
    
    try:
        # 黒板と会話記憶の統合テスト (新規追加)
        await test_blackboard_conversation_memory()
        
        # 新機能テスト（追加）
        print_header("新実装機能テスト")
        
        # 1. 質問適切性テスト
        await test_answer_quality()
        
        # 2. 会話記憶テスト
        await test_conversation_memory()
        
        # 3. 役割振り分けテスト
        await test_role_assignment()
        
        # 既存テスト
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
        config["use_memory"] = True  # 会話履歴を有効化
        
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
        if (_slm_instance is not None):
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
