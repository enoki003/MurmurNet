#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test to verify enhanced architecture components
"""

def test_communication_interface():
    """Basic test for communication interface"""
    try:
        from communication_interface import (
            ModuleCommunicationManager,
            MessageType,
            create_communication_system,
            create_message
        )
        
        # Test message creation
        msg = create_message(MessageType.AGENT_REQUEST, {"test": "data"})
        print(f"✓ Message created: {msg}")
        
        # Test communication system creation
        comm_manager = create_communication_system()
        print(f"✓ Communication manager created: {type(comm_manager)}")
        
        return True
    except Exception as e:
        print(f"✗ Communication interface test failed: {e}")
        return False

def test_module_adapters():
    """Basic test for module adapters"""
    try:
        from module_adapters import create_module_adapters
        from communication_interface import create_communication_system
        
        comm_manager = create_communication_system()
        adapters = create_module_adapters(comm_manager)
        
        print(f"✓ Module adapters created: {len(adapters)} adapters")
        return True
    except Exception as e:
        print(f"✗ Module adapters test failed: {e}")
        return False

def test_system_coordinator():
    """Basic test for system coordinator"""
    try:
        from module_system_coordinator import ModuleSystemCoordinator
        from communication_interface import create_communication_system
        from module_adapters import create_module_adapters
        
        comm_manager = create_communication_system()
        adapters = create_module_adapters(comm_manager)
        
        coordinator = ModuleSystemCoordinator(
            communication_manager=comm_manager,
            agent_pool=adapters['agent_pool'],
            summary_engine=adapters['summary_engine']
        )
        
        print(f"✓ System coordinator created: {type(coordinator)}")
        return True
    except Exception as e:
        print(f"✗ System coordinator test failed: {e}")
        return False

def test_enhanced_slm():
    """Basic test for enhanced distributed SLM"""
    try:
        from enhanced_distributed_slm import EnhancedDistributedSLM
        
        enhanced_slm = EnhancedDistributedSLM()
        print(f"✓ Enhanced SLM created: {type(enhanced_slm)}")
        return True
    except Exception as e:
        print(f"✗ Enhanced SLM test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("=== Enhanced Architecture Basic Tests ===")
    
    tests = [
        test_communication_interface,
        test_module_adapters,
        test_system_coordinator,
        test_enhanced_slm
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n--- Running {test.__name__} ---")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
