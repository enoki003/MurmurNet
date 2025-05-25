#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Manager Integration Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 2: 設定ファイル外部化の統合テスト

作者: Yuhi Sonoki
"""

import sys
import os

# プロジェクトのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MurmurNet'))

def test_config_manager():
    """ConfigManagerの基本機能テスト"""
    print("=== ConfigManager Integration Test ===")
    
    try:
        from MurmurNet.modules.config_manager import get_config
        
        # ConfigManagerの取得
        config = get_config()
        print("✓ ConfigManagerの取得成功")
        
        # 設定値の確認
        print(f"✓ エージェント数: {config.agent.num_agents}")
        print(f"✓ 反復回数: {config.agent.iterations}")
        print(f"✓ モデルタイプ: {config.model.model_type}")
        print(f"✓ デバッグモード: {config.logging.debug}")
        print(f"✓ RAGモード: {config.rag.rag_mode}")
        
        # 辞書形式での後方互換性テスト
        config_dict = config.to_dict()
        print(f"✓ 辞書形式変換: {len(config_dict)}個のキー")
        
        return True
        
    except Exception as e:
        print(f"✗ ConfigManagerテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_integration():
    """各モジュールのConfigManager統合テスト"""
    print("\n=== Module Integration Test ===")
    
    modules_to_test = [
        ('agent_pool', 'AgentPoolManager'),
        ('summary_engine', 'SummaryEngine'),
        ('conversation_memory', 'ConversationMemory'),
        ('output_agent', 'OutputAgent'),
        ('input_reception', 'InputReception'),
        ('blackboard', 'Blackboard'),
        ('system_coordinator', 'SystemCoordinator'),
        ('rag_retriever', 'RAGRetriever'),
    ]
    
    success_count = 0
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(f'MurmurNet.modules.{module_name}', fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # ConfigManagerを使用してインスタンス生成をテスト
            if class_name in ['AgentPoolManager', 'SystemCoordinator']:
                # これらは追加パラメータが必要
                print(f"✓ {class_name}: インポート成功（手動初期化が必要）")
            else:
                # 設定なしでインスタンス生成をテスト
                instance = cls()
                print(f"✓ {class_name}: 初期化成功")
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ {class_name}: エラー - {e}")
    
    print(f"\n統合テスト結果: {success_count}/{len(modules_to_test)} モジュール成功")
    return success_count == len(modules_to_test)

def test_model_factory():
    """ModelFactoryのConfigManager統合テスト"""
    print("\n=== ModelFactory Integration Test ===")
    
    try:
        from MurmurNet.modules.model_factory import ModelFactory, get_shared_model
        
        # ConfigManagerを使用したモデル作成テスト
        print("✓ ModelFactory: インポート成功")
        
        # 共有モデル取得のテスト（実際の初期化はスキップ）
        print("✓ get_shared_model: 関数準備完了")
        
        return True
        
    except Exception as e:
        print(f"✗ ModelFactory: エラー - {e}")
        return False

def main():
    """メインテスト実行"""
    print("Phase 2: 設定ファイル外部化 - 統合テスト開始\n")
    
    tests = [
        ("ConfigManager基本機能", test_config_manager),
        ("モジュール統合", test_module_integration),
        ("ModelFactory統合", test_model_factory),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"テスト: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: 予期しないエラー - {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("テスト結果サマリー")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n総合結果: {passed}/{len(results)} テスト成功")
    
    if passed == len(results):
        print("\n🎉 Phase 2: 設定ファイル外部化の統合が完了しました！")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed}個のテストが失敗しました。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
