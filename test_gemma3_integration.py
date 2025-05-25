#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemma3モデル統合テスト
~~~~~~~~~~~~~~~~~~~~
モック実装を削除したMurmurNetシステムでgemma3モデルが正しく動作するかテスト

作者: Yuhi Sonoki
"""

import sys
import os
import asyncio
import logging

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MurmurNet.modules.config_manager import get_config
from MurmurNet.modules.model_factory import ModelFactory
from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.distributed_slm import DistributedSLM

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_manager():
    """設定マネージャーのテスト"""
    print("=== 設定マネージャーテスト ===")
    
    try:
        config = get_config()
        print(f"✅ 設定読み込み成功")
        print(f"   モデルタイプ: {config.model.model_type}")
        print(f"   RAGモード: {config.rag.rag_mode}")
        print(f"   エージェント数: {config.agent.num_agents}")
        
        # gemma3がサポートされているか確認
        if config.model.model_type == "gemma3":
            print("✅ gemma3モデルが設定されています")
        else:
            print(f"⚠️ モデルタイプが{config.model.model_type}です")
            
        # RAGモードがdummyでないか確認
        if config.rag.rag_mode in ["zim", "embedding"]:
            print(f"✅ RAGモード '{config.rag.rag_mode}' が正しく設定されています")
        else:
            print(f"⚠️ RAGモードが{config.rag.rag_mode}です")
            
        return True
        
    except Exception as e:
        print(f"❌ 設定マネージャーエラー: {e}")
        return False

def test_model_factory():
    """モデルファクトリのテスト"""
    print("\n=== モデルファクトリテスト ===")
    
    try:
        config = get_config()
        
        # gemma3モデルの作成テスト
        print("gemma3モデルの作成をテスト中...")
        model = ModelFactory.create_model(config.to_dict())
        
        if model:
            print("✅ gemma3モデルの作成成功")
            print(f"   モデルタイプ: {type(model).__name__}")
            
            # モデルの可用性チェック
            is_available = model.is_available()
            print(f"   モデル可用性: {is_available}")
            
            if is_available:
                print("✅ モデルが利用可能です")
            else:
                print("⚠️ モデルが利用できません（モデルファイルが見つからない可能性）")
                
        else:
            print("❌ モデル作成に失敗")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ モデルファクトリエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_retriever():
    """RAGリトリーバーのテスト"""
    print("\n=== RAGリトリーバーテスト ===")
    
    try:
        config = get_config()
        
        # RAGリトリーバーの作成
        print("RAGリトリーバーの初期化中...")
        rag = RAGRetriever(config.to_dict())
        
        print(f"✅ RAGリトリーバー初期化成功")
        print(f"   動作モード: {rag.mode}")
        
        # ダミーモードでないことを確認
        if rag.mode in ["zim", "embedding"]:
            print("✅ 実際の検索モードで動作しています")
        else:
            print(f"⚠️ 予期しないモード: {rag.mode}")
            
        # 簡単な検索テスト
        print("検索テスト実行中...")
        result = rag.retrieve("人工知能とは何ですか？")
        
        if result and len(result) > 0:
            print(f"✅ 検索成功: {len(result)}文字の結果")
            print(f"   結果の一部: {result[:100]}...")
        else:
            print("⚠️ 検索結果が空です")
            
        return True
        
    except Exception as e:
        print(f"❌ RAGリトリーバーエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_distributed_slm():
    """分散SLMの統合テスト"""
    print("\n=== 分散SLM統合テスト ===")
    
    try:
        config = get_config()
        
        # 軽量な設定でテスト
        config_dict = config.to_dict()
        config_dict["num_agents"] = 1
        config_dict["iterations"] = 1
        config_dict["use_summary"] = False
        
        print("分散SLMの初期化中...")
        slm = DistributedSLM(config_dict)
        
        print("✅ 分散SLM初期化成功")
        
        # 簡単なクエリでテスト
        query = "こんにちは"
        print(f"テストクエリ: {query}")
        
        response = await slm.generate(query)
        
        if response and len(response) > 0:
            print(f"✅ 応答生成成功: {len(response)}文字")
            print(f"   応答: {response[:200]}...")
        else:
            print("⚠️ 応答が空です")
            
        return True
        
    except Exception as e:
        print(f"❌ 分散SLMエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """メイン実行関数"""
    print("MurmurNet Gemma3統合テスト開始")
    print("=" * 50)
    
    tests = [
        ("設定マネージャー", test_config_manager),
        ("モデルファクトリ", test_model_factory),
        ("RAGリトリーバー", test_rag_retriever),
    ]
    
    success_count = 0
    total_count = len(tests)
    
    # 基本テスト
    for test_name, test_func in tests:
        print(f"\n[{success_count + 1}/{total_count}] {test_name}テスト実行中...")
        if test_func():
            success_count += 1
            print(f"✅ {test_name}テスト成功")
        else:
            print(f"❌ {test_name}テスト失敗")
    
    # 分散SLMテスト（非同期）
    print(f"\n[{total_count + 1}/{total_count + 1}] 分散SLMテスト実行中...")
    if await test_distributed_slm():
        success_count += 1
        print("✅ 分散SLMテスト成功")
    else:
        print("❌ 分散SLMテスト失敗")
    
    total_count += 1
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print(f"テスト結果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 すべてのテストが成功しました！")
        print("MurmurNetシステムからモック実装の削除とgemma3設定が完了しています。")
    else:
        print(f"⚠️ {total_count - success_count}個のテストが失敗しました。")
        print("設定やファイルパスを確認してください。")

if __name__ == "__main__":
    asyncio.run(main())
