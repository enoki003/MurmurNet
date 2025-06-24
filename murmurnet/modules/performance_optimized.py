#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 最適化パフォーマンス監視モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU最適化とシステムのパフォーマンス監視・プロファイリング機能

作者: Yuhi Sonoki
改訂: CPU最適化版
"""

import time
import psutil
import threading
import functools
import multiprocessing
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .common import format_memory_size


@dataclass
class MemorySnapshot:
    """メモリスナップショットのデータクラス"""
    timestamp: float
    rss: int  # Resident Set Size
    vms: int  # Virtual Memory Size
    percent: float
    available: int
    name: str


@dataclass
class CPUPerformanceMetrics:
    """CPU性能メトリクスのデータクラス"""
    timestamp: float
    process_cpu_percent: float
    system_cpu_percent: float
    per_core_usage: List[float]
    cpu_count_logical: int
    cpu_count_physical: int
    cpu_times_user: float
    cpu_times_system: float
    optimal_threads: int
    load_average: Optional[List[float]] = None  # Unix系のみ


class OptimizedPerformanceMonitor:
    """
    最適化パフォーマンス監視クラス
    
    CPU最大活用とメモリ効率を重視したパフォーマンス監視機能
    """
    
    def __init__(self, enabled: bool = True, memory_tracking: bool = True):
        """
        最適化パフォーマンスモニターの初期化
        
        引数:
            enabled: 監視機能の有効/無効
            memory_tracking: メモリ追跡の有効/無効
        """
        self.enabled = enabled
        self.memory_tracking = memory_tracking
        self.timers: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, MemorySnapshot] = {}
        self.cpu_usage_history: List[CPUPerformanceMetrics] = []
        self.start_time = time.time()
        
        # CPU情報取得
        self.cpu_count = psutil.cpu_count(logical=True)
        self.cpu_count_physical = psutil.cpu_count(logical=False)
        
        # プロセス情報の取得
        self.process = psutil.Process()
        
        # スレッドロック
        self.lock = threading.RLock()  # 再帰ロック使用
        
        # 最適スレッド数の計算
        self.optimal_threads = self._calculate_optimal_threads()
        
        # スレッドプール（パフォーマンス監視用）
        self._executor = ThreadPoolExecutor(max_workers=min(4, self.cpu_count))
        
    def _calculate_optimal_threads(self) -> int:
        """
        最適なスレッド数を計算
        
        戻り値:
            推奨スレッド数
        """
        # CPU使用タイプに応じた最適化
        if self.cpu_count_physical >= 6:
            # 高性能CPU: 物理コア数の1.5倍
            optimal = int(self.cpu_count_physical * 1.5)
        elif self.cpu_count_physical >= 4:
            # 中性能CPU: 物理コア数と同じ
            optimal = self.cpu_count_physical
        else:
            # 低性能CPU: 論理コア数
            optimal = self.cpu_count
        
        # 最大値制限（論理コア数を超えない）
        return min(optimal, self.cpu_count)
    
    def start_timer(self, name: str) -> float:
        """
        タイマーを開始
        
        引数:
            name: タイマー名
            
        戻り値:
            開始時刻のタイムスタンプ
        """
        if not self.enabled:
            return time.time()
            
        start_time = time.time()
        with self.lock:
            self.timers[name] = start_time
        return start_time
    
    def end_timer(self, name: str, start_time: Optional[float] = None) -> float:
        """
        タイマーを終了し、経過時間を取得
        
        引数:
            name: タイマー名
            start_time: 開始時刻（省略時はstart_timerで記録された値を使用）
            
        戻り値:
            経過時間（秒）
        """
        if not self.enabled:
            return 0.0
            
        end_time = time.time()
        
        with self.lock:
            if start_time is None:
                start_time = self.timers.get(name, end_time)
            
            elapsed = end_time - start_time
            return elapsed
    
    def take_memory_snapshot(self, name: str) -> Optional[MemorySnapshot]:
        """
        現在のメモリ使用量のスナップショットを取得
        
        引数:
            name: スナップショット名
            
        戻り値:
            メモリスナップショット
        """
        if not self.enabled or not self.memory_tracking:
            return None
            
        try:
            # プロセス情報の取得
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # システム情報の取得
            virtual_memory = psutil.virtual_memory()
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss=memory_info.rss,
                vms=memory_info.vms,
                percent=memory_percent,
                available=virtual_memory.available,
                name=name
            )
            
            with self.lock:
                self.memory_snapshots[name] = snapshot
                
            return snapshot
            
        except Exception:
            # エラーが発生した場合は無視
            return None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        現在のメモリ使用状況を取得
        
        戻り値:
            メモリ使用情報の辞書
        """
        if not self.enabled:
            return {}
            
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            virtual_memory = psutil.virtual_memory()
            
            return {
                "rss": memory_info.rss,
                "rss_formatted": format_memory_size(memory_info.rss),
                "vms": memory_info.vms,
                "vms_formatted": format_memory_size(memory_info.vms),
                "percent": memory_percent,
                "system_available": virtual_memory.available,
                "system_available_formatted": format_memory_size(virtual_memory.available),
                "system_percent": virtual_memory.percent
            }
        except Exception:
            return {}
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """
        現在のCPU使用率を取得（最適化版）
        
        戻り値:
            CPU使用率の詳細情報
        """
        if not self.enabled:
            return {}
            
        try:
            cpu_percent = self.process.cpu_percent()
            cpu_times = self.process.cpu_times()
            system_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # 負荷平均の取得（Unix系のみ）
            load_avg = None
            try:
                if hasattr(psutil, 'getloadavg'):
                    load_avg = list(psutil.getloadavg())
            except:
                pass
            
            metrics = CPUPerformanceMetrics(
                timestamp=time.time(),
                process_cpu_percent=cpu_percent,
                system_cpu_percent=np.mean(system_cpu_percent),
                per_core_usage=system_cpu_percent,
                cpu_count_logical=self.cpu_count,
                cpu_count_physical=self.cpu_count_physical,
                cpu_times_user=cpu_times.user,
                cpu_times_system=cpu_times.system,
                optimal_threads=self.optimal_threads,
                load_average=load_avg
            )
            
            # 履歴に追加
            with self.lock:
                self.cpu_usage_history.append(metrics)
                # 履歴制限（最大1000件）
                if len(self.cpu_usage_history) > 1000:
                    self.cpu_usage_history = self.cpu_usage_history[-500:]
            
            return {
                'process_cpu_percent': metrics.process_cpu_percent,
                'system_cpu_percent': metrics.system_cpu_percent,
                'per_core_usage': metrics.per_core_usage,
                'cpu_count_logical': metrics.cpu_count_logical,
                'cpu_count_physical': metrics.cpu_count_physical,
                'cpu_times_user': metrics.cpu_times_user,
                'cpu_times_system': metrics.cpu_times_system,
                'optimal_threads': metrics.optimal_threads,
                'load_average': metrics.load_average,
                'timestamp': metrics.timestamp
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_optimal_thread_count(self) -> int:
        """
        最適なスレッド数を取得
        
        戻り値:
            推奨スレッド数
        """
        return self.optimal_threads
    
    def get_cpu_efficiency(self) -> Dict[str, Any]:
        """
        CPU効率の分析を取得
        
        戻り値:
            CPU効率分析情報
        """
        if not self.enabled or len(self.cpu_usage_history) < 2:
            return {}
        
        with self.lock:
            recent_metrics = self.cpu_usage_history[-10:]  # 最近10件
        
        if not recent_metrics:
            return {}
        
        # CPU使用率の統計
        cpu_usage = [m.system_cpu_percent for m in recent_metrics]
        process_usage = [m.process_cpu_percent for m in recent_metrics]
        
        return {
            'avg_system_cpu': np.mean(cpu_usage),
            'max_system_cpu': np.max(cpu_usage),
            'min_system_cpu': np.min(cpu_usage),
            'avg_process_cpu': np.mean(process_usage),
            'max_process_cpu': np.max(process_usage),
            'cpu_utilization_efficiency': np.mean(cpu_usage) / 100.0,
            'optimal_threads': self.optimal_threads,
            'current_thread_recommendation': self._get_dynamic_thread_recommendation()
        }
    
    def _get_dynamic_thread_recommendation(self) -> int:
        """
        動的スレッド数推奨値の計算
        
        戻り値:
            動的推奨スレッド数
        """
        if not self.cpu_usage_history:
            return self.optimal_threads
        
        # 最近のCPU使用率を基に動的調整
        recent_usage = np.mean([m.system_cpu_percent for m in self.cpu_usage_history[-5:]])
        
        if recent_usage < 50:
            # CPU使用率が低い場合、スレッド数を増やす
            return min(self.optimal_threads + 2, self.cpu_count)
        elif recent_usage > 85:
            # CPU使用率が高い場合、スレッド数を減らす
            return max(self.optimal_threads - 1, 2)
        else:
            return self.optimal_threads
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンス統計のサマリを取得
        
        戻り値:
            パフォーマンスサマリ
        """
        if not self.enabled:
            return {'enabled': False}
        
        current_cpu = self.get_cpu_usage()
        current_memory = self.get_memory_usage()
        cpu_efficiency = self.get_cpu_efficiency()
        uptime = time.time() - self.start_time
        
        return {
            'enabled': True,
            'uptime_seconds': uptime,
            'cpu_usage': current_cpu,
            'memory_usage': current_memory,
            'cpu_efficiency': cpu_efficiency,
            'optimal_threads': self.optimal_threads,
            'memory_snapshots_count': len(self.memory_snapshots),
            'cpu_history_count': len(self.cpu_usage_history),
            'performance_recommendations': self._get_performance_recommendations()
        }
    
    def _get_performance_recommendations(self) -> List[str]:
        """
        パフォーマンス改善の推奨事項を取得
        
        戻り値:
            推奨事項のリスト
        """
        recommendations = []
        
        if not self.cpu_usage_history:
            return recommendations
        
        # 最近のメトリクス分析
        recent_cpu = np.mean([m.system_cpu_percent for m in self.cpu_usage_history[-5:]])
        recent_process = np.mean([m.process_cpu_percent for m in self.cpu_usage_history[-5:]])
        
        if recent_cpu < 60:
            recommendations.append("CPU使用率が低めです。並列処理を増やしてパフォーマンスを向上できます。")
        elif recent_cpu > 90:
            recommendations.append("CPU使用率が高すぎます。並列処理を減らすか、処理を分散してください。")
        
        if recent_process > 200:  # プロセスが200%を超える場合（多コア利用）
            recommendations.append("マルチコア活用が良好です。現在の並列設定を維持してください。")
        elif recent_process < 50:
            recommendations.append("プロセスのCPU使用率が低いです。並列処理の改善を検討してください。")
        
        return recommendations
    
    def clear(self):
        """監視データをクリア"""
        with self.lock:
            self.timers.clear()
            self.memory_snapshots.clear()
            self.cpu_usage_history.clear()
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# 既存のPerformanceMonitorクラスも継承して互換性を保つ
class PerformanceMonitor(OptimizedPerformanceMonitor):
    """互換性のためのエイリアス"""
    pass


def time_function(func: Callable) -> Callable:
    """
    関数の実行時間を計測するデコレータ（最適化版）
    
    引数:
        func: 計測対象の関数
        
    戻り値:
        ラップされた関数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # より高精度
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            # ロギングは呼び出し元で行う
    return wrapper


def time_async_function(func: Callable) -> Callable:
    """
    非同期関数の実行時間を計測するデコレータ（最適化版）
    
    引数:
        func: 計測対象の非同期関数
        
    戻り値:
        ラップされた非同期関数
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # より高精度
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            # ロギングは呼び出し元で行う
    return wrapper


def get_system_optimization_info() -> Dict[str, Any]:
    """
    システム最適化情報を取得
    
    戻り値:
        システム最適化情報
    """
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    memory = psutil.virtual_memory()
    
    return {
        'cpu_count_logical': cpu_count,
        'cpu_count_physical': cpu_count_physical,
        'memory_total_gb': memory.total / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'recommended_max_processes': min(cpu_count_physical, 8),
        'recommended_max_threads': min(cpu_count, 16),
        'multiprocessing_optimal': cpu_count_physical >= 4,
        'memory_sufficient': memory.total >= 4 * (1024**3),  # 4GB以上
    }
