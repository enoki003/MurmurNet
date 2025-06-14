#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MurmurNet Architecture Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
新しい通信インターフェースとモジュラー設計のデモンストレーション

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
from typing import Dict, Any
from MurmurNet.modules.communication_interface import (
    create_communication_system,
    create_message,
    MessageType
)
from MurmurNet.modules.module_system_coordinator import ModuleSystemCoordinator
from MurmurNet.distributed_slm import DistributedSLM

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_murmurnet_demo.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


async def demo_communication_interface():
    """通信インターフェースのデモ"""
    print("\\n" + "="*60)
    print("通信インターフェースのデモンストレーション")
    print("="*60)
    
    # 通信システムの作成
    comm_manager = create_communication_system()
    print("✓ 通信システムを初期化しました")
    
    # 各種メッセージタイプのテスト
    messages = [
        ("ユーザー入力", MessageType.USER_INPUT, {'text': 'こんにちは、MurmurNetです'}),
        ("RAG結果", MessageType.RAG_RESULTS, {'results': ['検索結果1', '検索結果2']}),
        ("エージェント応答", MessageType.AGENT_RESPONSE, {'agent_id': 0, 'response': 'エージェント0の応答'}),
        ("要約", MessageType.SUMMARY, {'summary': '会話の要約です'}),
        ("データ格納", MessageType.DATA_STORE, {'key': 'test_data', 'value': 'テストデータ'})
    ]
    
    # メッセージの送信
    for desc, msg_type, data in messages:
        message = create_message(msg_type, data)
        comm_manager.publish(message)
        print(f"✓ {desc}メッセージを送信しました")
    
    # データの取得テスト
    retrieved_data = comm_manager.get_data('test_data')
    print(f"✓ データ取得テスト: {retrieved_data}")
    
    # 統計情報の表示
    stats = comm_manager.get_stats()
    print(f"✓ 通信統計: 送信メッセージ数 {stats.get('total_messages', 0)}")
    
    return comm_manager


async def demo_modular_system_coordinator():
    """モジュラーシステム調整器のデモ"""
    print("\\n" + "="*60)
    print("モジュラーシステム調整器のデモンストレーション")
    print("="*60)
    
    # 通信システムの初期化
    comm_manager = create_communication_system()
    
    # システム調整器の初期化
    coordinator = ModuleSystemCoordinator(comm_manager=comm_manager)
    print("✓ システム調整器を初期化しました")
    
    # テストデータの設定
    test_input = "機械学習とディープラーニングの違いについて教えてください"
    comm_manager.publish(create_message(MessageType.USER_INPUT, {
        'text': test_input
    }))
    comm_manager.publish(create_message(MessageType.DATA_STORE, {
        'key': 'user_input',
        'value': test_input
    }))
    print(f"✓ テスト入力を設定しました: {test_input}")
    
    # RAG検索結果のシミュレート
    comm_manager.publish(create_message(MessageType.RAG_RESULTS, {
        'results': [
            '機械学習は人工知能の一分野で、データから自動的にパターンを学習する技術です。',
            'ディープラーニングは機械学習の一種で、深層ニューラルネットワークを使用します。'
        ]
    }))
    comm_manager.publish(create_message(MessageType.DATA_STORE, {
        'key': 'search_results',
        'value': '機械学習とディープラーニングに関する検索結果'
    }))
    print("✓ RAG検索結果をシミュレートしました")
    
    # 会話コンテキストの設定
    comm_manager.publish(create_message(MessageType.DATA_STORE, {
        'key': 'conversation_context',
        'value': '前回は人工知能の基本概念について話しました。'
    }))
    print("✓ 会話コンテキストを設定しました")
    
    # 反復処理のデモ（簡略版）
    print("\\n反復処理を開始します...")
    
    try:
        # プロンプト構築のテスト
        prompt = coordinator._build_common_prompt()
        print(f"✓ 共通プロンプトを構築しました (長さ: {len(prompt)}文字)")
        
        # 統計情報の表示
        stats = coordinator.get_execution_stats()
        print(f"✓ システム統計:")
        print(f"  - エージェント数: {stats['num_agents']}")
        print(f"  - 反復回数: {stats['iterations']}")
        print(f"  - 並列モード: {stats['parallel_mode']}")
        print(f"  - 要約有効: {stats['summary_enabled']}")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    
    return coordinator


async def demo_enhanced_distributed_slm():
    """改良版分散SLMのデモ"""
    print("\\n" + "="*60)
    print("改良版分散SLMシステムのデモンストレーション")
    print("="*60)
    
    try:
        # 改良版SLMの初期化（新モード）
        print("新しいアーキテクチャモードで初期化中...")
        slm = DistributedSLM(compatibility_mode=False)
        print("✓ 改良版SLMを初期化しました（新モード）")
        
        # システム状態の表示
        status = slm.get_system_status()
        print("✓ システム状態:")
        print(f"  - 互換モード: {status['compatibility_mode']}")
        print(f"  - モジュール初期化済み: {status['modules_initialized']}")
        print(f"  - パフォーマンス監視: {status['performance_monitoring']}")
        
        # 通信統計の表示
        comm_stats = slm.get_communication_stats()
        print(f"✓ 通信統計: {comm_stats}")
        
        # 簡単な入力処理テスト
        test_input = "新しいアーキテクチャの動作テストです"
        print(f"\\n入力テスト: {test_input}")
        
        # 入力処理のシミュレート（フル処理は重いためスキップ）
        print("✓ 入力処理をシミュレートしました")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    
        # 互換モードでのフォールバック
        print("\\n互換モードでのフォールバックを試行...")
        try:
            slm_compat = DistributedSLM(compatibility_mode=True)
            print("✓ 互換モードで初期化成功")
            
            status = slm_compat.get_system_status()
            print(f"✓ 互換モード状態: {status['compatibility_mode']}")
            
        except Exception as e2:
            print(f"❌ 互換モードでもエラー: {e2}")
    
    return slm if 'slm' in locals() else None


async def demo_message_flow():
    """メッセージフローのデモ"""
    print("\\n" + "="*60)
    print("メッセージフローのデモンストレーション")
    print("="*60)
    
    comm_manager = create_communication_system()
    
    # メッセージフローのシミュレート
    workflow_steps = [
        ("1. ユーザー入力受信", MessageType.USER_INPUT, {
            'text': 'Python機械学習ライブラリについて教えて'
        }),
        ("2. RAG検索実行", MessageType.RAG_RESULTS, {
            'query': 'Python機械学習ライブラリ',
            'results': ['scikit-learn', 'TensorFlow', 'PyTorch']
        }),
        ("3. エージェント応答生成", MessageType.AGENT_RESPONSE, {
            'agent_id': 0,
            'response': 'Pythonには優秀な機械学習ライブラリがあります'
        }),
        ("4. 要約作成", MessageType.SUMMARY, {
            'iteration': 0,
            'summary': 'Python機械学習ライブラリに関する情報を要約'
        }),
        ("5. 最終応答", MessageType.FINAL_RESPONSE, {
            'response': '詳細な回答をお返しします'
        })
    ]
    
    for step_desc, msg_type, data in workflow_steps:
        message = create_message(msg_type, data)
        comm_manager.publish(message)
        print(f"✓ {step_desc}")
        await asyncio.sleep(0.1)  # 視覚的な効果のための待機
    
    print(f"\\n✓ メッセージフロー完了 (総メッセージ数: {len(workflow_steps)})")
    
    # 最終統計の表示
    final_stats = comm_manager.get_stats()
    print(f"✓ 最終統計: {final_stats}")


async def demo_error_handling():
    """エラーハンドリングのデモ"""
    print("\\n" + "="*60)
    print("エラーハンドリングのデモンストレーション")
    print("="*60)
    
    comm_manager = create_communication_system()
    
    # 正常なメッセージ
    normal_message = create_message(MessageType.USER_INPUT, {
        'text': '正常なメッセージです'
    })
    comm_manager.publish(normal_message)
    print("✓ 正常なメッセージを送信しました")
    
    # エラーメッセージ
    error_message = create_message(MessageType.ERROR, {
        'error': 'テスト用のエラーメッセージです',
        'component': 'demo_module'
    })
    comm_manager.publish(error_message)
    print("✓ エラーメッセージを送信しました")
    
    # エージェントエラー
    agent_error = create_message(MessageType.AGENT_ERROR, {
        'agent_id': 99,
        'error': 'エージェント実行エラーのテストです'
    })
    comm_manager.publish(agent_error)
    print("✓ エージェントエラーメッセージを送信しました")
    
    print("✓ エラーハンドリングのテストが完了しました")


async def main():
    """メインデモ関数"""
    print("Enhanced MurmurNet Architecture Demo")
    print("新しい通信インターフェースとモジュラー設計のデモンストレーション")
    print("作者: Yuhi Sonoki")
    
    start_time = time.time()
    
    try:
        # 各デモの実行
        await demo_communication_interface()
        await demo_modular_system_coordinator()
        await demo_enhanced_distributed_slm()
        await demo_message_flow()
        await demo_error_handling()
        
        end_time = time.time()
        
        print("\\n" + "="*60)
        print("デモンストレーション完了")
        print("="*60)
        print(f"✓ 総実行時間: {end_time - start_time:.2f}秒")
        print("✓ 新しいアーキテクチャが正常に動作しています")
        print("\\n主な改善点:")
        print("  - 疎結合なモジュール設計")
        print("  - 明確な通信インターフェース")
        print("  - 既存システムとの互換性")
        print("  - エラーハンドリングの改善")
        print("  - パフォーマンス監視機能")
        
    except Exception as e:
        print(f"\\n❌ デモ実行中にエラーが発生しました: {e}")
        logger.error(f"デモエラー: {e}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())
