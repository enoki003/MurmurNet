#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
インポートテストスクリプト
"""

try:
    print("Testing imports...")
    
    # 依存関係のテスト
    print("1. Testing config_manager...")
    from MurmurNet.modules.config_manager import get_config
    print("   ✓ config_manager imported")
    
    print("2. Testing process_agent_worker...")
    from MurmurNet.modules.process_agent_worker import AgentTask, AgentResult
    print("   ✓ process_agent_worker imported")
    
    print("3. Testing result_collector...")
    from MurmurNet.modules.result_collector import ResultCollector, CollectedResults
    print("   ✓ result_collector imported")
    
    print("4. Testing process_coordinator...")
    from MurmurNet.modules.process_coordinator import ProcessCoordinator
    print("   ✓ process_coordinator imported")
    
    print("5. Testing process_agent_manager...")
    from MurmurNet.modules.process_agent_manager import ProcessAgentManager
    print("   ✓ process_agent_manager imported")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
