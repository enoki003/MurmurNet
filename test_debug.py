#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
デバッグ用のテストスクリプト
各コンポーネントの動作を個別に確認する
"""
import sys
import os
from pathlib import Path
import logging

# パス設定
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_rag_retriever():
    """RAGリトリーバーのテスト"""
    print("=== RAGリトリーバーのテスト ===")
    try:
        from MurmurNet.modules.rag_retriever import RAGRetriever
        from MurmurNet.modules.config_manager import get_config
        
        config_manager = get_config()
        print(f"設定されたRAGモード: {config_manager.rag.rag_mode}")
        
        rag = RAGRetriever()
        
        test_query = "AIは教育をどう変える？"
        print(f"テストクエリ: {test_query}")
        
        result = rag.retrieve(test_query)
        print(f"RAG検索結果:\n{result}")
        print("="*50)
        
    except Exception as e:
        print(f"RAGリトリーバーテストでエラー: {e}")
        import traceback
        traceback.print_exc()

def test_input_processing():
    """入力処理のテスト"""
    print("=== 入力処理のテスト ===")
    try:
        from MurmurNet.modules.input_reception import InputReception
        
        input_reception = InputReception()
        test_input = "AIは教育をどう変える？"
        
        print(f"テスト入力: {test_input}")
        processed = input_reception.process(test_input)
        print(f"処理結果: {processed}")
        print("="*50)
        
    except Exception as e:
        print(f"入力処理テストでエラー: {e}")
        import traceback
        traceback.print_exc()

def test_agent_pool():
    """エージェントプールのテスト"""
    print("=== エージェントプールのテスト ===")
    try:
        from MurmurNet.modules.agent_pool import AgentPoolManager
        from MurmurNet.modules.blackboard import Blackboard
        
        config = {
            'model': {
                'model_path': './models/gemma3.gguf',
                'model_type': 'gemma3',
                'context_length': 8192,
                'temperature': 0.8,
                'max_tokens': 512
            },
            'system': {
                'use_parallel': True,
                'max_agents': 2
            }
        }
        
        blackboard = Blackboard()
        
        # テストデータを黒板に書き込み
        blackboard.write('input', {'normalized': 'AIは教育をどう変える？'})
        blackboard.write('rag', 'AI（人工知能）は教育分野で革新的な変化をもたらしています。個別化学習、自動評価、パーソナライズされた学習体験などが可能になります。')
        
        agent_pool = AgentPoolManager(2, config, blackboard)
        
        # 質問タイプに基づく役割更新
        agent_pool.update_roles_based_on_question("AIは教育をどう変える？")
        print(f"設定された役割: {[role['role'] for role in agent_pool.roles]}")
        
        # プロンプト生成テスト
        for i in range(2):
            prompt = agent_pool._format_prompt(i)
            print(f"\nエージェント{i}のプロンプト:")
            print("-"*30)
            print(prompt)
            print("-"*30)
        
        print("="*50)
        
    except Exception as e:
        print(f"エージェントプールテストでエラー: {e}")
        import traceback
        traceback.print_exc()

def test_model_generation():
    """モデル生成のテスト"""
    print("=== モデル生成のテスト ===")
    try:
        from MurmurNet.modules.model_factory import LlamaModel
        
        model = LlamaModel('./models/gemma3.gguf')
        
        test_prompt = """こんにちは！私は「事実提供AI」だよ。

あなたは情報の専門家です。正確で検証可能な事実情報を簡潔に提供してください。

質問: AIは教育をどう変える？

参考になりそうな情報: AI（人工知能）は教育分野で革新的な変化をもたらしています。

これまでの会話: 過去の会話はありません。

仲間たちの意見:
他のエージェントの出力はまだありません。

お願い:
- 私らしい視点で話すね
- わかりやすく具体的に説明するよ
- みんなの意見も参考にして、より良い答えを考えてみる
- 150〜250文字くらいで話し言葉でお答えするね

それじゃあ、事実提供AIとして答えるよ:"""
        
        print("テストプロンプト:")
        print(test_prompt)
        print("\n" + "="*30 + " 生成結果 " + "="*30)
        
        result = model.generate(test_prompt, max_tokens=300, temperature=0.7)
        print(f"生成結果: {result}")
        print("="*50)
        
    except Exception as e:
        print(f"モデル生成テストでエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("MurmurNetデバッグテストを開始します")
    print("="*70)
    
    test_input_processing()
    test_rag_retriever()
    test_agent_pool()
    test_model_generation()
    
    print("デバッグテスト完了")
