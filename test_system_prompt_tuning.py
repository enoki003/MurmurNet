#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
システムプロンプト調整テスト
モデルの応答スタイルを確認
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MurmurNet'))

from MurmurNet.modules.model_factory import create_model_from_args
from MurmurNet.modules.prompt_manager import get_prompt_manager
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_prompt_styles():
    """異なる役割のシステムプロンプトをテスト"""
    
    model_name = "llm-jp/llm-jp-3-150m-instruct3"
    logger.info(f"モデル読み込み: {model_name}")
    
    # モデル作成
    model = create_model_from_args(model_type="huggingface", model_name=model_name)
    
    # プロンプトマネージャー作成
    prompt_manager = get_prompt_manager("huggingface", model_name)
    
    # テスト用の質問
    test_questions = [
        "Pythonで効率的にファイルを読む方法を教えてください。",
        "機械学習とディープラーニングの違いは何ですか？",
        "チームワークを向上させるには何が重要ですか？"
    ]
    
    # 異なる役割での応答をテスト
    roles = ["assistant", "researcher", "critic", "writer"]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"質問: {question}")
        print('='*60)
        
        for role in roles:
            print(f"\n--- {role}としての応答 ---")
            
            # プロンプト構築
            prompt = prompt_manager.build_prompt(question, role=role)
            print(f"プロンプト:\n{prompt}\n")
            
            try:
                # 応答生成
                response = model.generate(prompt, max_new_tokens=100, temperature=0.7)
                print(f"応答: {response}")
                
                # 応答の分析
                if response.startswith("申し訳ありませんが"):
                    print("⚠️  免責事項から開始")
                elif len(response.strip()) < 20:
                    print("⚠️  短すぎる応答")
                else:
                    print("✅ 適切な長さの応答")
                    
            except Exception as e:
                print(f"❌ エラー: {e}")
            
            print("-" * 40)
        
        # 少し休憩
        import time
        time.sleep(1)

def test_specific_prompt_adjustments():
    """特定のプロンプト調整をテスト"""
    
    model_name = "llm-jp/llm-jp-3-150m-instruct3"
    model = create_model_from_args(model_type="huggingface", model_name=model_name)
    
    # 異なるシステムプロンプトでテスト
    test_prompts = [
        "あなたは親切で知識豊富なアシスタントです。質問に対して直接的で具体的な回答をしてください。",
        "あなたは実用的なアドバイスを提供する専門家です。簡潔で有用な情報を教えてください。",
        "質問に対して、要点を整理して分かりやすく答えてください。",
        "以下の質問について、具体例を交えて説明してください。"
    ]
    
    test_question = "Pythonでリストをソートする方法を教えてください。"
    
    print(f"\n{'='*60}")
    print(f"質問: {test_question}")
    print('='*60)
    
    for i, system_prompt in enumerate(test_prompts, 1):
        print(f"\n--- パターン {i} ---")
        print(f"システムプロンプト: {system_prompt}")
        
        # llm-jp形式でプロンプト構築
        full_prompt = f"<|system|>{system_prompt}</s><|user|>{test_question}</s><|assistant|>"
        
        try:
            response = model.generate(full_prompt, max_new_tokens=120, temperature=0.7)
            print(f"応答: {response}")
            
            # 応答品質チェック
            if "申し訳" in response or "すみません" in response:
                print("⚠️  過度に謝罪的")
            if len(response.strip()) > 50:
                print("✅ 十分な長さ")
            if "sort" in response.lower() or "ソート" in response:
                print("✅ 関連性あり")
                
        except Exception as e:
            print(f"❌ エラー: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("システムプロンプト調整テスト開始")
    
    try:
        print("\n1. 役割別プロンプトテスト")
        test_system_prompt_styles()
        
        print("\n\n2. 特定プロンプト調整テスト")  
        test_specific_prompt_adjustments()
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nテスト完了")
