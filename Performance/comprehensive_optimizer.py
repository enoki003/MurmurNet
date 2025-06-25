#!/usr/bin/env python3
"""
MurmurNet統合最適化実行スクリプト（v2）
ログ分析に基づく包括的なパフォーマンス改善を実行

目標:
- 総所要時間: 31.28s → 12s以下
- メモリ使用量: 1328MB → 1078MB以下  
- system role警告解消
- 出力表示問題修正
"""

import logging
import time
import os
import sys

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def run_comprehensive_optimization():
    """包括的な最適化を実行"""
    logger.info("=" * 60)
    logger.info("MurmurNet Comprehensive Optimization Started")
    logger.info("=" * 60)
    
    results = {
        'model_cache': False,
        'template_fix': False,
        'thread_optimization': False,
        'summary_optimization': False,
        'console_fix': False,
        'parallel_execution': False,  # 新規追加
        'overall_success': False
    }
    
    start_time = time.time()
    
    try:
        # 1. モデルキャッシュ最適化（6s → 0.01s削減）
        logger.info("1. Applying model cache optimization...")
        try:
            from model_cache_optimizer import optimize_model_loading
            results['model_cache'] = optimize_model_loading()
        except ImportError:
            logger.warning("Model cache optimizer not available")
            results['model_cache'] = False
        logger.info(f"   Model cache: {'SUCCESS' if results['model_cache'] else 'FAILED'}")
        
        # 2. 並列実行最適化（真の並列化）
        logger.info("2. Applying parallel execution optimization...")
        try:
            from parallel_execution_optimizer import ParallelExecutionOptimizer
            optimizer = ParallelExecutionOptimizer()
            results['parallel_execution'] = optimizer.apply_parallel_optimizations()
        except ImportError:
            logger.warning("Parallel execution optimizer not available")
            results['parallel_execution'] = False
        logger.info(f"   Parallel execution: {'SUCCESS' if results['parallel_execution'] else 'FAILED'}")
        
        # 3. テンプレート最適化（system role警告解消、300ms削減）
        logger.info("3. Applying template optimization...")
        try:
            from template_optimizer import patch_template_manager
            results['template_fix'] = patch_template_manager()
        except ImportError:
            logger.warning("Template optimizer not available")
            results['template_fix'] = False
        logger.info(f"   Template fix: {'SUCCESS' if results['template_fix'] else 'FAILED'}")
        
        # 4. スレッド・RAG・要約最適化（並列化向上）
        logger.info("4. Applying thread and RAG/Summary optimization...")
        try:
            from thread_optimizer import apply_thread_optimizations
            optimal_threads, optimal_agents = apply_thread_optimizations()
            results['thread_optimization'] = optimal_threads > 0
            if results['thread_optimization']:
                logger.info(f"   - Threads: {optimal_threads}")
                logger.info(f"   - Agents: {optimal_agents}")
        except ImportError:
            logger.warning("Thread optimizer not available")
            results['thread_optimization'] = False
        logger.info(f"   Thread optimization: {'✓ SUCCESS' if results['thread_optimization'] else '✗ FAILED'}")
        
        # 5. 要約エンジン個別最適化（5.24s → 0s短文スキップ）
        logger.info("5. Applying summary engine optimization...")
        try:
            from summary_optimizer import apply_summary_optimization
            results['summary_optimization'] = apply_summary_optimization()
        except ImportError:
            logger.warning("Summary optimizer not available")
            results['summary_optimization'] = False
        logger.info(f"   Summary optimization: {'✓ SUCCESS' if results['summary_optimization'] else '✗ FAILED'}")
        
        # 6. コンソール出力修正（統合レスポンス表示）
        logger.info("6. Applying console output fixes...")
        try:
            from console_fix import apply_console_fixes
            results['console_fix'] = apply_console_fixes()
        except ImportError:
            logger.warning("Console fix not available")
            results['console_fix'] = False
        logger.info(f"   Console fix: {'✓ SUCCESS' if results['console_fix'] else '✗ FAILED'}")
        
        # 総合判定
        success_count = sum(results.values())
        total_optimizations = len([k for k in results.keys() if k != 'overall_success'])
        results['overall_success'] = success_count >= (total_optimizations * 0.6)  # 60%以上成功
        
        # 並列実行最適化が成功した場合の追加情報
        if results['parallel_execution']:
            logger.info("   ✓ グローバルロック除去完了")
            logger.info("   ✓ プロセス間並列化実装完了") 
            logger.info("   ✓ SentenceTransformer事前ロード完了")
            logger.info("   ✓ 並列性可視化追加完了")
        
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        results['overall_success'] = False
    
    end_time = time.time()
    optimization_time = end_time - start_time
      # 結果レポート
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    
    for key, value in results.items():
        if key != 'overall_success':
            status = "SUCCESS" if value else "FAILED"
            logger.info(f"{key.replace('_', ' ').title()}: {status}")
    
    logger.info(f"Optimization time: {optimization_time:.2f}s")
    logger.info(f"Overall success: {'SUCCESS' if results['overall_success'] else 'FAILED'}")
    
    # 使用推奨コマンド
    if results['overall_success']:
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDED USAGE")
        logger.info("=" * 60)
        logger.info("# 最適化済みコンソールアプリケーション:")
        logger.info("python Console/console_app_fixed.py")
        logger.info("")
        logger.info("# 高パフォーマンス設定:")
        logger.info("python Console/console_app.py --agents 4 --parallel --threads 6")
        logger.info("")
        logger.info("# ベンチマーク測定:")
        logger.info("python Performance/performance_benchmark.py")
    
    logger.info("=" * 60)
    logger.info("MurmurNet Comprehensive Optimization Completed")
    logger.info("=" * 60)
    
    return results

def check_environment():
    """実行環境をチェック"""
    logger.info("Checking environment...")
    
    # CPU情報
    cpu_count = os.cpu_count()
    logger.info(f"Detected CPU cores: {cpu_count}")
    
    # Python version
    python_version = sys.version
    logger.info(f"Python version: {python_version}")
    
    return True

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization.log'),
            logging.StreamHandler()
        ]
    )
    
    print("MurmurNet Comprehensive Optimization v2")
    print("=======================================")
    
    # 環境チェック
    if not check_environment():
        print("Environment check failed.")
        sys.exit(1)
    
    # 最適化実行
    results = run_comprehensive_optimization()
    
    # 終了コード
    exit_code = 0 if results['overall_success'] else 1
    sys.exit(exit_code)
