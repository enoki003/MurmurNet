#!/usr/bin/env python3
"""
MurmurNet完全システムテスト
SystemCoordinatorのrun_iterationメソッドを使用して全体的な動作を確認
"""

import logging
import sys
import os
import asyncio

# パスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from MurmurNet.modules.system_coordinator_new import SystemCoordinator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('full_system_test.log')
    ]
)

logger = logging.getLogger(__name__)

async def test_full_system():
    """完全システムテスト"""
    print("=" * 80)
    print("MurmurNet 完全システムテスト開始")
    print("=" * 80)
    
    try:
        # 1. SystemCoordinatorの初期化
        print("\n1. SystemCoordinatorの初期化...")
        coordinator = SystemCoordinator()
        print("✓ SystemCoordinator初期化成功")
        
        # 2. テスト用の質問を設定
        test_question = "Pythonの基本的な特徴について教えてください。"
        print(f"\n2. テスト質問: {test_question}")
          # 3. process_queryを実行（質問を直接渡す）
        print("\n3. SystemCoordinator.process_query()実行...")
        try:
            result = await coordinator.process_query(test_question)
            print(f"✓ process_query実行完了")
        except Exception as e:
            print(f"✗ process_query実行エラー: {e}")
            logger.exception("process_query実行中にエラーが発生")
            return False
          # 5. 実行統計の取得
        print("\n5. 実行統計の取得...")
        try:
            stats = coordinator.get_execution_stats()
            print("✓ 実行統計:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"✗ 実行統計取得エラー: {e}")
            logger.exception("実行統計取得中にエラーが発生")
        
        # 6. 結果の詳細確認
        print("\n6. 結果の詳細確認...")
        try:
            if isinstance(result, str):
                print("✓ 応答生成結果:")
                if len(result) > 200:
                    print(f"  応答内容 (最初の200文字): {result[:200]}...")
                    print(f"  応答全長: {len(result)}文字")
                else:
                    print(f"  応答内容: {result}")
                print("✓ 応答形式: 文字列（正常）")
                
                # 最終回答として表示
                print(f"\n7. 最終回答:")
                print("-" * 40)
                print(result)
                print("-" * 40)
            else:
                print(f"✗ 応答形式異常: {type(result)}")
                print(f"  実際の値: {result}")
                print("\n7. 最終回答が取得できません")
                
        except Exception as e:
            print(f"✗ 結果確認エラー: {e}")
            logger.exception("結果確認中にエラーが発生")
                
        except Exception as e:
            print(f"✗ Blackboard結果確認エラー: {e}")
            logger.exception("Blackboard結果確認中にエラーが発生")
        
        print("\n" + "=" * 80)
        print("完全システムテスト完了")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ システムテスト中に重大なエラーが発生: {e}")
        logger.exception("システムテスト中に重大なエラーが発生")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_system())
    sys.exit(0 if success else 1)
