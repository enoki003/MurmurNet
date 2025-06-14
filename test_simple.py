#!/usr/bin/env python3
import sys
import os

# パスを追加
sys.path.append('MurmurNet/modules')

try:
    # 基本的なインポートテスト
    from module_system_coordinator import ModuleSystemCoordinator
    from module_adapters import AgentPoolAdapter, SummaryEngineAdapter, create_communication_system
    print("✓ インポートが成功しました")
    
    # 簡単な初期化テスト
    comm_manager = create_communication_system()
    coordinator = ModuleSystemCoordinator(blackboard=comm_manager.storage)
    print("✓ ModuleSystemCoordinatorの初期化が成功しました")
    
    print(f"✓ エージェント数: {coordinator.num_agents}")
    print(f"✓ 反復回数: {coordinator.iterations}")
    
except Exception as e:
    print(f"✗ エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
