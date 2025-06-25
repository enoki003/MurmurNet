#!/usr/bin/env python3
"""
最適化効果テストスクリプト
ログ分析で特定された問題の解決を確認
"""

import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_optimizations():
    """最適化効果をテスト"""
    print("MurmurNet Optimization Test")
    print("===========================")
    
    test_results = {}
    
    # 1. モデルキャッシュテスト
    print("\n1. Testing model cache optimization...")
    try:
        from Performance.model_cache_optimizer import ModelCache
        
        start_time = time.time()
        embedder = ModelCache.get_embedder()
        first_load_time = time.time() - start_time
        
        start_time = time.time()
        embedder2 = ModelCache.get_embedder()
        cache_time = time.time() - start_time
        
        same_instance = embedder is embedder2
        
        print(f"   First load time: {first_load_time:.3f}s")
        print(f"   Cache access time: {cache_time:.6f}s")
        print(f"   Same instance: {same_instance}")
        
        test_results['model_cache'] = cache_time < 0.01 and same_instance
        
    except Exception as e:
        print(f"   Error: {e}")
        test_results['model_cache'] = False
      # 2. 並列実行最適化テスト
    print("\n2. Testing parallel execution optimization...")
    try:
        from Performance.parallel_execution_optimizer import ParallelExecutionOptimizer
        
        optimizer = ParallelExecutionOptimizer()
        
        # 最適化前の状態を記録
        start_time = time.time()
        
        # 最適化を適用
        success = optimizer.apply_parallel_optimizations()
        
        apply_time = time.time() - start_time
        
        print(f"   Optimization applied: {success}")
        print(f"   Apply time: {apply_time:.3f}s")
        
        if success:
            status = optimizer.get_optimization_status()
            print(f"   Applied patches: {len(status['applied_patches'])}")
            print(f"   Patches: {', '.join(status['applied_patches'])}")
            
            # 具体的な最適化効果を確認
            if 'global_lock_removal' in status['applied_patches']:
                print("   ✓ Global lock removed")
            if 'process_parallelization' in status['applied_patches']:
                print("   ✓ Process parallelization implemented")
            if 'sentence_transformer_preload' in status['applied_patches']:
                print("   ✓ SentenceTransformer preload implemented")
            if 'parallelism_monitoring' in status['applied_patches']:
                print("   ✓ Parallelism monitoring added")
        
        test_results['parallel_execution'] = success
        
    except Exception as e:
        print(f"   Error: {e}")
        test_results['parallel_execution'] = False
        
    # 3. テンプレート最適化テスト
    print("\n3. Testing template optimization...")
    try:
        from Performance.template_optimizer import patch_template_manager
        success = patch_template_manager()
        test_results['template'] = success
        print(f"   Template patch: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"   Error: {e}")
        test_results['template'] = False
    
    # 4. スレッド最適化テスト
    print("\n4. Testing thread optimization...")
    try:
        from Performance.thread_optimizer import apply_thread_optimizations
        optimal_threads, optimal_agents = apply_thread_optimizations()
        cpu_count = os.cpu_count()
        
        print(f"   CPU cores: {cpu_count}")
        print(f"   Optimal threads: {optimal_threads}")
        print(f"   Optimal agents: {optimal_agents}")
        print(f"   GGML_NUM_THREADS: {os.environ.get('GGML_NUM_THREADS', 'Not set')}")
        
        test_results['thread'] = optimal_threads > 0
        
    except Exception as e:
        print(f"   Error: {e}")
        test_results['thread'] = False
    
    # 5. コンソール修正テスト
    print("\n5. Testing console fixes...")
    try:
        from Performance.console_fix import apply_console_fixes
        success = apply_console_fixes()
        test_results['console'] = success
        print(f"   Console fixes: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"   Error: {e}")
        test_results['console'] = False
    
    # 結果レポート
    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= total_tests * 0.75:
        print("🎉 Optimization test: SUCCESS")
        print("\nRecommendations:")
        print("1. Run: python Performance/comprehensive_optimizer.py")
        print("2. Test: python Console/console_app_fixed.py")
        print("3. Benchmark: python Performance/performance_benchmark.py")
        return True
    else:
        print("⚠️ Optimization test: PARTIAL SUCCESS")
        print("\nSome optimizations may need manual adjustment.")
        return False

if __name__ == "__main__":
    success = test_optimizations()
    sys.exit(0 if success else 1)
