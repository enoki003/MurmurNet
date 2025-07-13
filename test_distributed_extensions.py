#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5-8 分散システム拡張機能テストスクリプト
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分散協調、監視、オートスケーリング、パフォーマンス最適化機能のテスト

作者: Yuhi Sonoki
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from murmurnet.modules.distributed_coordination import create_distributed_coordinator, create_load_balancer
from murmurnet.modules.monitoring import create_metrics_collector, create_alert_manager
from murmurnet.modules.autoscaling import create_autoscaler, ScalingMetrics
from murmurnet.modules.performance_optimization import create_distributed_optimizer

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_distributed_extensions.log')
    ]
)

logger = logging.getLogger(__name__)

async def test_distributed_coordination():
    """分散協調システムのテスト"""
    logger.info("=== Phase 5: 分散協調システムテスト開始 ===")
    
    config = {
        'heartbeat_interval': 2.0,
        'failure_timeout': 6.0,
        'consensus_algorithm': 'simple_majority',
        'load_balance_strategy': 'hybrid'
    }
    
    try:
        # 分散協調システム作成・開始
        coordinator = create_distributed_coordinator(config)
        load_balancer = create_load_balancer(coordinator)
        
        await coordinator.start()
        await load_balancer.start()
        
        # テストタスクを投入
        task_id = await coordinator.submit_task({
            'type': 'test_task',
            'data': 'Hello from distributed system!'
        })
        
        logger.info(f"タスク投入: {task_id}")
        
        # 少し待ってステータス確認
        await asyncio.sleep(3)
        
        status = coordinator.get_cluster_status()
        logger.info(f"クラスター状態: {status}")
        
        # 合意テスト
        proposal = {'action': 'test_consensus', 'value': 42}
        consensus_result = await coordinator.achieve_consensus(proposal)
        logger.info(f"合意結果: {consensus_result}")
        
        # 停止
        await coordinator.stop()
        await load_balancer.stop()
        
        logger.info("分散協調システムテスト完了 ✅")
        return True
        
    except Exception as e:
        logger.error(f"分散協調システムテストエラー: {e}")
        return False

async def test_monitoring_system():
    """監視・メトリクスシステムのテスト"""
    logger.info("=== Phase 6: 監視・メトリクスシステムテスト開始 ===")
    
    config = {
        'enable_prometheus': False,  # テスト環境ではPrometheusサーバーを無効化
        'metrics_collection_interval': 1.0,
        'alert_check_interval': 2.0,
        'notification_channels': []
    }
    
    try:
        # 監視システム作成・開始
        metrics_collector = create_metrics_collector(config)
        alert_manager = create_alert_manager(config, metrics_collector)
        
        await metrics_collector.start()
        await alert_manager.start()
        
        # テストメトリクス記録
        metrics_collector.record_metric('test_cpu_usage', 75.5)
        metrics_collector.record_metric('test_memory_usage', 60.2)
        metrics_collector.record_metric('test_requests_count', 100)
        
        # 少し待ってレポート取得
        await asyncio.sleep(2)
        
        # メトリクス履歴取得
        cpu_history = metrics_collector.get_metric_history('test_cpu_usage', 1)
        logger.info(f"CPU使用率履歴: {len(cpu_history)} データポイント")
        
        # 集約メトリクス取得
        aggregated = metrics_collector.get_aggregated_metrics(1)
        logger.info(f"集約メトリクス: {aggregated}")
        
        # アラート状態確認
        alert_status = alert_manager.get_alert_status()
        logger.info(f"アラート状態: {alert_status}")
        
        # 停止
        await metrics_collector.stop()
        await alert_manager.stop()
        
        logger.info("監視・メトリクスシステムテスト完了 ✅")
        return True
        
    except Exception as e:
        logger.error(f"監視・メトリクスシステムテストエラー: {e}")
        return False

async def test_autoscaling_system():
    """オートスケーリングシステムのテスト"""
    logger.info("=== Phase 7: オートスケーリングシステムテスト開始 ===")
    
    config = {
        'scaling_strategy': 'hybrid',
        'min_workers': 1,
        'max_workers': 3,
        'target_cpu_utilization': 0.7,
        'scale_up_threshold': 0.8,
        'scale_down_threshold': 0.3,
        'evaluation_period': 2,
        'use_kubernetes': False  # テスト環境ではKubernetes無効化
    }
    
    try:
        # オートスケーラー作成・開始
        autoscaler = create_autoscaler(config)
        await autoscaler.start()
        
        # テストメトリクス更新（高負荷状態をシミュレート）
        high_load_metrics = ScalingMetrics(
            cpu_usage=85.0,
            memory_usage=70.0,
            task_queue_length=60,
            active_workers=1,
            throughput=50.0
        )
        
        await autoscaler.update_metrics(high_load_metrics)
        logger.info("高負荷メトリクス送信")
        
        # スケーリング判定を待つ
        await asyncio.sleep(3)
        
        # 状態確認
        status = autoscaler.get_scaling_status()
        logger.info(f"スケーリング状態: {status}")
        
        # 低負荷状態をシミュレート
        low_load_metrics = ScalingMetrics(
            cpu_usage=25.0,
            memory_usage=30.0,
            task_queue_length=5,
            active_workers=3,
            throughput=10.0
        )
        
        await autoscaler.update_metrics(low_load_metrics)
        logger.info("低負荷メトリクス送信")
        
        # 再度スケーリング判定を待つ
        await asyncio.sleep(3)
        
        final_status = autoscaler.get_scaling_status()
        logger.info(f"最終スケーリング状態: {final_status}")
        
        # 停止
        await autoscaler.stop()
        
        logger.info("オートスケーリングシステムテスト完了 ✅")
        return True
        
    except Exception as e:
        logger.error(f"オートスケーリングシステムテストエラー: {e}")
        return False

async def test_performance_optimization():
    """パフォーマンス最適化システムのテスト"""
    logger.info("=== Phase 8: パフォーマンス最適化システムテスト開始 ===")
    
    config = {
        'enable_latency_tracing': True,
        'trace_sample_rate': 1.0,  # テストでは100%サンプリング
        'enable_memory_tracking': True,
        'enable_compression': True,
        'compression_algorithm': 'lz4',
        'enable_batching': True,
        'batch_size': 5,
        'enable_profiling': False,  # テスト環境ではプロファイリング無効化
        'enable_auto_optimization': False  # 手動テストのため無効化
    }
    
    try:
        # パフォーマンス最適化システム作成・開始
        optimizer = create_distributed_optimizer(config)
        await optimizer.start()
        
        # レイテンシ測定テスト
        @optimizer.latency_optimizer.measure_latency("test_function")
        async def test_async_function():
            await asyncio.sleep(0.1)  # 100ms の処理をシミュレート
            return "test_result"
        
        # テスト関数を数回実行
        for i in range(5):
            result = await test_async_function()
            logger.debug(f"テスト関数実行 {i+1}: {result}")
        
        # パフォーマンスレポート取得
        latency_report = optimizer.latency_optimizer.get_performance_report()
        logger.info(f"レイテンシレポート: {latency_report}")
        
        # メモリ最適化テスト
        optimizer.memory_optimizer.track_memory_usage()
        memory_report = optimizer.memory_optimizer.get_memory_report()
        logger.info(f"メモリレポート: {memory_report}")
        
        # ネットワーク最適化テスト
        test_data = {'message': 'Hello, world!', 'numbers': list(range(100))}
        
        # データ圧縮テスト
        compressed_data = optimizer.network_optimizer.compress_data(test_data)
        decompressed_data = optimizer.network_optimizer.decompress_data(compressed_data)
        
        logger.info(f"圧縮テスト: 元データ == 復元データ: {test_data == decompressed_data}")
        
        # ネットワーク統計取得
        network_stats = optimizer.network_optimizer.get_network_stats()
        logger.info(f"ネットワーク統計: {network_stats}")
        
        # 包括的レポート取得
        comprehensive_report = optimizer.get_comprehensive_report()
        logger.info(f"包括的レポート取得完了: {len(comprehensive_report)} セクション")
        
        # 停止
        await optimizer.stop()
        
        logger.info("パフォーマンス最適化システムテスト完了 ✅")
        return True
        
    except Exception as e:
        logger.error(f"パフォーマンス最適化システムテストエラー: {e}")
        return False

async def test_integrated_system():
    """統合システムテスト"""
    logger.info("=== 統合システムテスト開始 ===")
    
    # 統合設定
    config = {
        # 分散協調
        'heartbeat_interval': 3.0,
        'consensus_algorithm': 'simple_majority',
        'load_balance_strategy': 'hybrid',
        
        # 監視
        'enable_prometheus': False,
        'metrics_collection_interval': 2.0,
        'alert_check_interval': 5.0,
        
        # オートスケーリング
        'scaling_strategy': 'hybrid',
        'min_workers': 1,
        'max_workers': 2,
        'evaluation_period': 3,
        'use_kubernetes': False,
        
        # パフォーマンス最適化
        'enable_latency_tracing': True,
        'enable_memory_tracking': True,
        'enable_auto_optimization': True,
        'optimization_interval': 10
    }
    
    try:
        # 全システム初期化
        coordinator = create_distributed_coordinator(config)
        metrics_collector = create_metrics_collector(config)
        alert_manager = create_alert_manager(config, metrics_collector)
        autoscaler = create_autoscaler(config)
        optimizer = create_distributed_optimizer(config)
        
        # 全システム開始
        await coordinator.start()
        await metrics_collector.start()
        await alert_manager.start()
        await autoscaler.start()
        await optimizer.start()
        
        logger.info("全分散システムコンポーネント開始完了")
        
        # システム間連携テスト
        for i in range(3):
            # メトリクス生成
            metrics_collector.record_metric('integrated_test_cpu', 50.0 + i * 10)
            metrics_collector.record_metric('integrated_test_memory', 40.0 + i * 15)
            
            # オートスケーラーにメトリクス送信
            scaling_metrics = ScalingMetrics(
                cpu_usage=50.0 + i * 10,
                memory_usage=40.0 + i * 15,
                task_queue_length=10 + i * 5,
                active_workers=1,
                throughput=20.0 + i * 5
            )
            await autoscaler.update_metrics(scaling_metrics)
            
            # 協調システムにタスク投入
            task_id = await coordinator.submit_task({
                'type': 'integrated_test',
                'iteration': i,
                'timestamp': time.time()
            })
            
            logger.info(f"統合テスト反復 {i+1} 完了 - タスクID: {task_id}")
            await asyncio.sleep(2)
        
        # 最終状態確認
        cluster_status = coordinator.get_cluster_status()
        scaling_status = autoscaler.get_scaling_status()
        alert_status = alert_manager.get_alert_status()
        performance_report = optimizer.get_comprehensive_report()
        
        logger.info(f"最終クラスター状態: アクティブノード数={cluster_status['active_nodes']}")
        logger.info(f"最終スケーリング状態: ワーカー数={scaling_status['current_workers']}")
        logger.info(f"最終アラート状態: アクティブアラート数={alert_status['active_alerts']}")
        logger.info(f"パフォーマンスレポート取得: {len(performance_report)} セクション")
        
        # 全システム停止
        await optimizer.stop()
        await autoscaler.stop()
        await alert_manager.stop()
        await metrics_collector.stop()
        await coordinator.stop()
        
        logger.info("統合システムテスト完了 ✅")
        return True
        
    except Exception as e:
        logger.error(f"統合システムテストエラー: {e}")
        return False

async def main():
    """メインテスト関数"""
    logger.info("🚀 Phase 5-8 分散システム拡張機能テスト開始")
    
    test_results = {}
    
    # 各フェーズのテスト実行
    test_results['distributed_coordination'] = await test_distributed_coordination()
    test_results['monitoring_system'] = await test_monitoring_system()
    test_results['autoscaling_system'] = await test_autoscaling_system()
    test_results['performance_optimization'] = await test_performance_optimization()
    test_results['integrated_system'] = await test_integrated_system()
    
    # 結果サマリー
    logger.info("\n" + "="*60)
    logger.info("📊 テスト結果サマリー")
    logger.info("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info("="*60)
    logger.info(f"総合結果: {passed_tests}/{total_tests} テスト通過")
    
    if passed_tests == total_tests:
        logger.info("🎉 全テスト成功！分散システム拡張機能は正常に動作しています。")
        return 0
    else:
        logger.error("⚠️  一部テストが失敗しました。ログを確認してください。")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("テストが中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        sys.exit(1)
