#!/usr/bin/env python3
"""
MurmurNet統合最適化実行スクリプト（修正版）
実際のモジュール構造に基づいて最適化を実行
"""

import logging
import time
import os
import sys

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def basic_optimization():
    """基本的な最適化のみ実行（安定版）"""
    logger.info("=" * 60)
    logger.info("MurmurNet Basic Optimization Started")
    logger.info("=" * 60)
    
    results = {}
    start_time = time.time()
    
    try:
        # 1. 環境変数設定（スレッド最適化）
        logger.info("1. Setting optimal thread configuration...")
        cpu_count = os.cpu_count()
        optimal_threads = 6 if cpu_count >= 8 else max(2, cpu_count - 1)
        
        os.environ['GGML_NUM_THREADS'] = str(optimal_threads)
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['LLAMA_CPP_N_THREADS'] = str(optimal_threads)
        
        results['thread_config'] = True
        logger.info(f"   Thread optimization: SUCCESS ({optimal_threads} threads)")
        
        # 2. モデルキャッシュ最適化
        logger.info("2. Applying model cache optimization...")
        try:
            from model_cache_optimizer import ModelCache
            # テスト: エンベッダーキャッシュ
            embedder = ModelCache.get_embedder()
            results['model_cache'] = True
            logger.info("   Model cache: SUCCESS")
        except Exception as e:
            logger.warning(f"   Model cache: FAILED ({e})")
            results['model_cache'] = False
        
        # 3. テンプレート最適化
        logger.info("3. Applying template optimization...")
        try:
            from template_optimizer import patch_template_manager
            results['template_fix'] = patch_template_manager()
            status = "SUCCESS" if results['template_fix'] else "FAILED"
            logger.info(f"   Template fix: {status}")
        except Exception as e:
            logger.warning(f"   Template fix: FAILED ({e})")
            results['template_fix'] = False
        
        # 4. コンソール修正
        logger.info("4. Applying console fixes...")
        try:
            from console_fix import apply_console_fixes
            results['console_fix'] = apply_console_fixes()
            status = "SUCCESS" if results['console_fix'] else "FAILED"
            logger.info(f"   Console fix: {status}")
        except Exception as e:
            logger.warning(f"   Console fix: FAILED ({e})")
            results['console_fix'] = False
        
        # 5. 簡単なRAG・要約スキップ
        logger.info("5. Setting up skip optimizations...")
        # 環境変数で制御
        os.environ['MURMURNET_MIN_TEXT_LENGTH'] = '64'
        os.environ['MURMURNET_SKIP_SHORT_SUMMARY'] = '1'
        results['skip_optimization'] = True
        logger.info("   Skip optimization: SUCCESS")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {}
    
    end_time = time.time()
    
    # 結果レポート
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for key, value in results.items():
        status = "SUCCESS" if value else "FAILED"
        logger.info(f"{key.replace('_', ' ').title()}: {status}")
    
    logger.info(f"Optimization time: {end_time - start_time:.2f}s")
    logger.info(f"Success rate: {success_count}/{total_count}")
    
    # 使用推奨
    if success_count >= total_count * 0.6:
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDED USAGE")
        logger.info("=" * 60)
        logger.info("# Optimized settings:")
        logger.info(f"export GGML_NUM_THREADS={optimal_threads}")
        logger.info("export MURMURNET_SKIP_SHORT_SUMMARY=1")
        logger.info("")
        logger.info("# Run optimized console app:")
        logger.info("python Console/console_app_fixed.py")
        
        results['overall_success'] = True
    else:
        results['overall_success'] = False
    
    logger.info("=" * 60)
    logger.info("MurmurNet Basic Optimization Completed")
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    print("MurmurNet Basic Optimization")
    print("============================")
    
    # CPU情報表示
    cpu_count = os.cpu_count()
    print(f"Detected CPU cores: {cpu_count}")
    
    # 最適化実行
    results = basic_optimization()
    
    # 終了コード
    if results.get('overall_success', False):
        print("\nOptimization completed successfully!")
        exit_code = 0
    else:
        print("\nOptimization completed with some issues.")
        exit_code = 1
    
    sys.exit(exit_code)
