#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分散システムパフォーマンス最適化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
レイテンシ最適化、メモリ効率向上、ネットワーク通信最適化

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
import gc
import sys
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import weakref
import pickle
try:
    import compression_utils
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
from functools import wraps, lru_cache
import hashlib

# パフォーマンス監視ライブラリ（オプション）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 圧縮ライブラリ（オプション）
try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# プロファイリングライブラリ（オプション）
try:
    import cProfile
    import pstats
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    latency_ms: float = 0.0
    throughput_qps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    cache_hit_rate: float = 0.0
    gc_collections: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationResult:
    """最適化結果"""
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    description: str

class LatencyOptimizer:
    """
    レイテンシ最適化システム
    
    応答時間の測定、分析、最適化
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 設定
        self.enable_tracing = config.get('enable_latency_tracing', True)
        self.trace_sample_rate = config.get('trace_sample_rate', 0.1)
        self.optimization_threshold_ms = config.get('optimization_threshold_ms', 1000)
        
        # パフォーマンス追跡
        self.latency_history: deque = deque(maxlen=1000)
        self.hotspots: Dict[str, List[float]] = defaultdict(list)
        self.optimization_suggestions: List[str] = []
        
        # 統計
        self.total_requests = 0
        self.total_latency = 0.0
        
        self.logger.info("レイテンシ最適化システム初期化完了")

    def measure_latency(self, func_name: str = None):
        """レイテンシ測定デコレータ"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enable_tracing or not self._should_trace():
                    return await func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    self._record_latency(func_name or func.__name__, latency_ms)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enable_tracing or not self._should_trace():
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    self._record_latency(func_name or func.__name__, latency_ms)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def _should_trace(self) -> bool:
        """トレースするかどうかをサンプリング率で判定"""
        import random
        return random.random() < self.trace_sample_rate

    def _record_latency(self, func_name: str, latency_ms: float):
        """レイテンシを記録"""
        self.latency_history.append((func_name, latency_ms, time.time()))
        self.hotspots[func_name].append(latency_ms)
        
        self.total_requests += 1
        self.total_latency += latency_ms
        
        # しきい値を超えた場合は警告
        if latency_ms > self.optimization_threshold_ms:
            self.logger.warning(f"高レイテンシ検出: {func_name} - {latency_ms:.2f}ms")
            self._add_optimization_suggestion(func_name, latency_ms)

    def _add_optimization_suggestion(self, func_name: str, latency_ms: float):
        """最適化提案を追加"""
        suggestion = f"{func_name}の最適化が必要 (レイテンシ: {latency_ms:.2f}ms)"
        if suggestion not in self.optimization_suggestions:
            self.optimization_suggestions.append(suggestion)

    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートを生成"""
        if not self.latency_history:
            return {"status": "データなし"}
        
        # 統計計算
        latencies = [latency for _, latency, _ in self.latency_history]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # パーセンタイル計算
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
        p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        # ホットスポット分析
        hotspot_analysis = {}
        for func_name, func_latencies in self.hotspots.items():
            if func_latencies:
                hotspot_analysis[func_name] = {
                    'avg_latency': sum(func_latencies) / len(func_latencies),
                    'max_latency': max(func_latencies),
                    'call_count': len(func_latencies)
                }
        
        return {
            'summary': {
                'total_requests': self.total_requests,
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency
            },
            'percentiles': {
                'p50': p50,
                'p90': p90,
                'p95': p95,
                'p99': p99
            },
            'hotspots': hotspot_analysis,
            'optimization_suggestions': self.optimization_suggestions.copy()
        }

    def optimize_async_operations(self):
        """非同期操作の最適化"""
        # イベントループの最適化設定
        if hasattr(asyncio, 'set_event_loop_policy'):
            # Windowsの場合、SelectorEventLoopを使用
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # タスクプールサイズの最適化
        optimal_pool_size = min(32, (4 * (psutil.cpu_count() if PSUTIL_AVAILABLE else 4)))
        return ThreadPoolExecutor(max_workers=optimal_pool_size)

class MemoryOptimizer:
    """
    メモリ効率向上システム
    
    メモリ使用量の監視、最適化、ガベージコレクション管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 設定
        self.enable_memory_tracking = config.get('enable_memory_tracking', True)
        self.gc_optimization = config.get('enable_gc_optimization', True)
        self.memory_warning_threshold_mb = config.get('memory_warning_threshold_mb', 1024)
        self.memory_critical_threshold_mb = config.get('memory_critical_threshold_mb', 2048)
        
        # オブジェクトプール
        self.object_pools: Dict[str, deque] = {}
        self.weak_references: weakref.WeakSet = weakref.WeakSet()
        
        # メモリ監視
        self.memory_history: deque = deque(maxlen=100)
        self.gc_stats = {'collections': 0, 'freed_objects': 0}
        
        # ガベージコレクション最適化
        if self.gc_optimization:
            self._optimize_gc()
        
        self.logger.info("メモリ最適化システム初期化完了")

    def _optimize_gc(self):
        """ガベージコレクション最適化"""
        # GCしきい値を調整（より積極的なGC）
        gc.set_threshold(700, 10, 10)
        
        # 手動GCサイクル実行
        self._schedule_gc_cycles()

    def _schedule_gc_cycles(self):
        """定期的なGCサイクルをスケジュール"""
        def gc_cycle():
            try:
                before_objects = len(gc.get_objects())
                collected = gc.collect()
                after_objects = len(gc.get_objects())
                
                self.gc_stats['collections'] += 1
                self.gc_stats['freed_objects'] += (before_objects - after_objects)
                
                self.logger.debug(f"GCサイクル完了: {collected} objects collected")
                
                # 次のGCサイクルをスケジュール
                threading.Timer(60.0, gc_cycle).start()
            except Exception as e:
                self.logger.error(f"GCサイクルエラー: {e}")
        
        # 初回実行
        threading.Timer(60.0, gc_cycle).start()

    def create_object_pool(self, pool_name: str, factory_func: Callable, max_size: int = 100):
        """オブジェクトプールを作成"""
        if pool_name not in self.object_pools:
            self.object_pools[pool_name] = deque(maxlen=max_size)
        
        def get_object():
            pool = self.object_pools[pool_name]
            if pool:
                return pool.popleft()
            else:
                return factory_func()
        
        def return_object(obj):
            pool = self.object_pools[pool_name]
            if len(pool) < max_size:
                # オブジェクトをリセット（必要に応じて）
                if hasattr(obj, 'reset'):
                    obj.reset()
                pool.append(obj)
            # プールが満杯の場合は破棄
        
        return get_object, return_object

    def track_memory_usage(self):
        """メモリ使用量を追跡"""
        if not self.enable_memory_tracking or not PSUTIL_AVAILABLE:
            return
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.memory_history.append((time.time(), memory_mb))
            
            # 閾値チェック
            if memory_mb > self.memory_critical_threshold_mb:
                self.logger.critical(f"メモリ使用量が危険レベル: {memory_mb:.1f}MB")
                self._emergency_memory_cleanup()
            elif memory_mb > self.memory_warning_threshold_mb:
                self.logger.warning(f"メモリ使用量が警告レベル: {memory_mb:.1f}MB")
                self._perform_memory_cleanup()
            
        except Exception as e:
            self.logger.error(f"メモリ監視エラー: {e}")

    def _perform_memory_cleanup(self):
        """メモリクリーンアップを実行"""
        self.logger.info("メモリクリーンアップ開始")
        
        # 手動ガベージコレクション
        before_gc = gc.collect()
        self.logger.debug(f"GC後: {before_gc} objects collected")
        
        # オブジェクトプールのクリアンアップ
        for pool_name, pool in self.object_pools.items():
            cleared = len(pool)
            pool.clear()
            if cleared > 0:
                self.logger.debug(f"オブジェクトプール {pool_name} をクリア: {cleared} objects")

    def _emergency_memory_cleanup(self):
        """緊急メモリクリーンアップ"""
        self.logger.critical("緊急メモリクリーンアップ実行")
        
        # より積極的なクリーンアップ
        self._perform_memory_cleanup()
        
        # 全てのオブジェクトプールをクリア
        for pool in self.object_pools.values():
            pool.clear()
        
        # システムに依存するメモリ最適化
        try:
            if hasattr(gc, 'set_debug'):
                gc.set_debug(gc.DEBUG_STATS)
            gc.collect()
            gc.collect()  # 二回実行
        except Exception as e:
            self.logger.error(f"緊急GCエラー: {e}")

    def get_memory_report(self) -> Dict[str, Any]:
        """メモリレポートを生成"""
        if not PSUTIL_AVAILABLE:
            return {"status": "psutil not available"}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'current_usage_mb': memory_info.rss / 1024 / 1024,
                'virtual_memory_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'gc_stats': self.gc_stats.copy(),
                'object_pools': {
                    name: len(pool) for name, pool in self.object_pools.items()
                },
                'gc_counts': gc.get_count(),
                'gc_thresholds': gc.get_threshold()
            }
        except Exception as e:
            self.logger.error(f"メモリレポート生成エラー: {e}")
            return {"error": str(e)}

class NetworkOptimizer:
    """
    ネットワーク通信最適化システム
    
    データ圧縮、バッチング、接続プーリング
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 設定
        self.enable_compression = config.get('enable_compression', True)
        self.compression_algorithm = config.get('compression_algorithm', 'lz4')
        self.compression_threshold_bytes = config.get('compression_threshold_bytes', 1024)
        self.enable_batching = config.get('enable_batching', True)
        self.batch_size = config.get('batch_size', 10)
        self.batch_timeout_ms = config.get('batch_timeout_ms', 100)
        
        # 圧縮器設定
        self.compressor = self._init_compressor()
        
        # バッチング
        self.pending_batches: Dict[str, List] = defaultdict(list)
        self.batch_timers: Dict[str, threading.Timer] = {}
        
        # 統計
        self.network_stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio': 0.0,
            'batch_count': 0
        }
        
        self.logger.info(f"ネットワーク最適化システム初期化完了 - 圧縮: {self.compression_algorithm}")

    def _init_compressor(self):
        """圧縮器を初期化"""
        if not self.enable_compression:
            return None
        
        if self.compression_algorithm == 'lz4' and LZ4_AVAILABLE:
            return {
                'compress': lz4.compress,
                'decompress': lz4.decompress
            }
        elif self.compression_algorithm == 'zstd' and ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor()
            dctx = zstd.ZstdDecompressor()
            return {
                'compress': cctx.compress,
                'decompress': dctx.decompress
            }
        else:
            # フォールバック: pickle圧縮なし
            self.logger.warning(f"圧縮アルゴリズム {self.compression_algorithm} が利用できません")
            return None

    def compress_data(self, data: Any) -> bytes:
        """データを圧縮"""
        if not self.compressor:
            return pickle.dumps(data)
        
        try:
            # まずpickleでシリアライズ
            serialized = pickle.dumps(data)
            
            # 閾値チェック
            if len(serialized) < self.compression_threshold_bytes:
                return serialized
            
            # 圧縮
            compressed = self.compressor['compress'](serialized)
            
            # 圧縮率を統計に記録
            compression_ratio = len(compressed) / len(serialized)
            self.network_stats['compression_ratio'] = (
                self.network_stats['compression_ratio'] * 0.9 + compression_ratio * 0.1
            )
            
            self.logger.debug(f"データ圧縮: {len(serialized)} -> {len(compressed)} bytes (比率: {compression_ratio:.2f})")
            
            # 圧縮されたデータにヘッダーを追加
            return b'COMPRESSED:' + compressed
            
        except Exception as e:
            self.logger.error(f"データ圧縮エラー: {e}")
            return pickle.dumps(data)

    def decompress_data(self, data: bytes) -> Any:
        """データを展開"""
        if not self.compressor:
            return pickle.loads(data)
        
        try:
            # 圧縮ヘッダーチェック
            if data.startswith(b'COMPRESSED:'):
                compressed_data = data[11:]  # 'COMPRESSED:' を除去
                decompressed = self.compressor['decompress'](compressed_data)
                return pickle.loads(decompressed)
            else:
                # 非圧縮データ
                return pickle.loads(data)
                
        except Exception as e:
            self.logger.error(f"データ展開エラー: {e}")
            # フォールバック
            return pickle.loads(data)

    def add_to_batch(self, batch_key: str, item: Any, callback: Optional[Callable] = None):
        """アイテムをバッチに追加"""
        if not self.enable_batching:
            if callback:
                callback([item])
            return
        
        self.pending_batches[batch_key].append((item, callback))
        
        # バッチサイズに達した場合、即座に処理
        if len(self.pending_batches[batch_key]) >= self.batch_size:
            self._process_batch(batch_key)
        else:
            # タイマーを設定（まだ設定されていない場合）
            if batch_key not in self.batch_timers:
                timer = threading.Timer(
                    self.batch_timeout_ms / 1000.0,
                    lambda: self._process_batch(batch_key)
                )
                self.batch_timers[batch_key] = timer
                timer.start()

    def _process_batch(self, batch_key: str):
        """バッチを処理"""
        if batch_key not in self.pending_batches:
            return
        
        batch = self.pending_batches.pop(batch_key, [])
        if batch_key in self.batch_timers:
            self.batch_timers.pop(batch_key).cancel()
        
        if not batch:
            return
        
        items = [item for item, _ in batch]
        callbacks = [callback for _, callback in batch if callback]
        
        self.network_stats['batch_count'] += 1
        
        self.logger.debug(f"バッチ処理: {batch_key} - {len(items)} items")
        
        # 全てのコールバックを実行
        for callback in callbacks:
            try:
                callback(items)
            except Exception as e:
                self.logger.error(f"バッチコールバックエラー: {e}")

    def optimize_serialization(self, data: Any) -> bytes:
        """シリアライゼーション最適化"""
        # オブジェクトタイプに応じた最適化
        if isinstance(data, dict):
            # 辞書の場合、JSONの方が効率的な場合がある
            try:
                import json
                json_data = json.dumps(data, ensure_ascii=False)
                json_bytes = json_data.encode('utf-8')
                pickle_bytes = pickle.dumps(data)
                
                # JSONの方が小さい場合はJSONを使用
                if len(json_bytes) < len(pickle_bytes):
                    return b'JSON:' + json_bytes
            except Exception:
                pass
        
        # デフォルトはpickle
        return b'PICKLE:' + pickle.dumps(data)

    def optimize_deserialization(self, data: bytes) -> Any:
        """デシリアライゼーション最適化"""
        if data.startswith(b'JSON:'):
            import json
            json_data = data[5:].decode('utf-8')
            return json.loads(json_data)
        elif data.startswith(b'PICKLE:'):
            return pickle.loads(data[7:])
        else:
            # フォールバック
            return pickle.loads(data)

    def get_network_stats(self) -> Dict[str, Any]:
        """ネットワーク統計を取得"""
        return {
            'compression': {
                'enabled': self.enable_compression,
                'algorithm': self.compression_algorithm,
                'average_ratio': self.network_stats['compression_ratio']
            },
            'batching': {
                'enabled': self.enable_batching,
                'batch_size': self.batch_size,
                'batch_count': self.network_stats['batch_count'],
                'pending_batches': len(self.pending_batches)
            },
            'traffic': {
                'bytes_sent': self.network_stats['bytes_sent'],
                'bytes_received': self.network_stats['bytes_received']
            }
        }

class PerformanceProfiler:
    """
    パフォーマンスプロファイラー
    
    詳細なパフォーマンス分析とボトルネック特定
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 設定
        self.enable_profiling = config.get('enable_profiling', False)
        self.profile_output_dir = config.get('profile_output_dir', './profiles')
        
        # プロファイラー
        self.profiler = None
        if self.enable_profiling and PROFILING_AVAILABLE:
            self.profiler = cProfile.Profile()
        
        # カスタムメトリクス
        self.custom_metrics: Dict[str, List[float]] = defaultdict(list)
        
        self.logger.info("パフォーマンスプロファイラー初期化完了")

    def start_profiling(self):
        """プロファイリング開始"""
        if self.profiler:
            self.profiler.enable()
            self.logger.info("プロファイリング開始")

    def stop_profiling(self, filename: Optional[str] = None):
        """プロファイリング停止"""
        if not self.profiler:
            return
        
        self.profiler.disable()
        
        if filename:
            # プロファイル結果を保存
            import os
            os.makedirs(self.profile_output_dir, exist_ok=True)
            output_path = os.path.join(self.profile_output_dir, filename)
            self.profiler.dump_stats(output_path)
            self.logger.info(f"プロファイル結果保存: {output_path}")

    def profile_function(self, func_name: str = None):
        """関数プロファイリングデコレータ"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = (end_time - start_time) * 1000  # ms
                    metric_name = func_name or func.__name__
                    self.custom_metrics[metric_name].append(execution_time)
            
            return wrapper
        return decorator

    def record_metric(self, metric_name: str, value: float):
        """カスタムメトリクスを記録"""
        self.custom_metrics[metric_name].append(value)

    def get_profiling_report(self) -> Dict[str, Any]:
        """プロファイリングレポートを生成"""
        report = {
            'profiling_enabled': self.enable_profiling,
            'custom_metrics': {}
        }
        
        # カスタムメトリクスの統計
        for metric_name, values in self.custom_metrics.items():
            if values:
                report['custom_metrics'][metric_name] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        
        return report

class DistributedOptimizer:
    """
    分散システム統合最適化
    
    全体的なパフォーマンス最適化の調整
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 各最適化コンポーネント
        self.latency_optimizer = LatencyOptimizer(config)
        self.memory_optimizer = MemoryOptimizer(config)
        self.network_optimizer = NetworkOptimizer(config)
        self.profiler = PerformanceProfiler(config)
        
        # 最適化履歴
        self.optimization_history: List[OptimizationResult] = []
        
        # 自動最適化設定
        self.enable_auto_optimization = config.get('enable_auto_optimization', True)
        self.optimization_interval = config.get('optimization_interval', 300)  # 5分
        
        self._running = False
        
        self.logger.info("分散システム統合最適化初期化完了")

    async def start(self):
        """最適化システム開始"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("分散システム最適化開始")
        
        # プロファイリング開始
        self.profiler.start_profiling()
        
        # 自動最適化ループ
        if self.enable_auto_optimization:
            asyncio.create_task(self._auto_optimization_loop())

    async def stop(self):
        """最適化システム停止"""
        self._running = False
        
        # プロファイリング停止
        self.profiler.stop_profiling(f"profile_{int(time.time())}.prof")
        
        self.logger.info("分散システム最適化停止")

    async def _auto_optimization_loop(self):
        """自動最適化ループ"""
        while self._running:
            try:
                await self._perform_auto_optimization()
                await asyncio.sleep(self.optimization_interval)
            except Exception as e:
                self.logger.error(f"自動最適化エラー: {e}")

    async def _perform_auto_optimization(self):
        """自動最適化を実行"""
        self.logger.info("自動最適化実行中...")
        
        # 現在のメトリクス取得
        before_metrics = await self._collect_current_metrics()
        
        # メモリ最適化
        self.memory_optimizer.track_memory_usage()
        
        # パフォーマンス分析
        latency_report = self.latency_optimizer.get_performance_report()
        memory_report = self.memory_optimizer.get_memory_report()
        
        # 最適化の提案と実行
        optimizations = []
        
        # 高レイテンシの関数があれば最適化提案
        if 'optimization_suggestions' in latency_report:
            optimizations.extend(latency_report['optimization_suggestions'])
        
        # メモリ使用量が高ければクリーンアップ
        if isinstance(memory_report, dict) and 'current_usage_mb' in memory_report:
            if memory_report['current_usage_mb'] > self.memory_optimizer.memory_warning_threshold_mb:
                self.memory_optimizer._perform_memory_cleanup()
                optimizations.append("メモリクリーンアップ実行")
        
        # 最適化後のメトリクス取得
        after_metrics = await self._collect_current_metrics()
        
        # 最適化結果を記録
        if optimizations:
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            result = OptimizationResult(
                optimization_type="auto_optimization",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                description="; ".join(optimizations)
            )
            self.optimization_history.append(result)
            
            self.logger.info(f"自動最適化完了: {improvement:.1f}% 改善")

    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """現在のメトリクスを収集"""
        metrics = PerformanceMetrics()
        
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                metrics.cpu_usage_percent = process.cpu_percent()
        except Exception as e:
            self.logger.debug(f"メトリクス収集エラー: {e}")
        
        return metrics

    def _calculate_improvement(self, before: PerformanceMetrics, after: PerformanceMetrics) -> float:
        """改善率を計算"""
        improvements = []
        
        # メモリ使用量の改善
        if before.memory_usage_mb > 0:
            memory_improvement = (before.memory_usage_mb - after.memory_usage_mb) / before.memory_usage_mb * 100
            improvements.append(memory_improvement)
        
        # CPU使用率の改善
        if before.cpu_usage_percent > 0:
            cpu_improvement = (before.cpu_usage_percent - after.cpu_usage_percent) / before.cpu_usage_percent * 100
            improvements.append(cpu_improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """包括的なパフォーマンスレポートを生成"""
        return {
            'latency': self.latency_optimizer.get_performance_report(),
            'memory': self.memory_optimizer.get_memory_report(),
            'network': self.network_optimizer.get_network_stats(),
            'profiling': self.profiler.get_profiling_report(),
            'optimization_history': [
                {
                    'type': result.optimization_type,
                    'improvement_percent': result.improvement_percent,
                    'description': result.description
                }
                for result in self.optimization_history[-10:]  # 最新10件
            ],
            'system_info': self._get_system_info()
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        info = {
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info.update({
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    'disk_usage': psutil.disk_usage('/').percent if sys.platform != 'win32' else psutil.disk_usage('C:').percent
                })
            except Exception:
                pass
        
        return info

def create_distributed_optimizer(config: Dict[str, Any]) -> DistributedOptimizer:
    """分散システム統合最適化を作成"""
    return DistributedOptimizer(config)
