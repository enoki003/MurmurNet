#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet パフォーマンス監視モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
システムのパフォーマンス監視とプロファイリング機能

作者: Yuhi Sonoki
"""

import time
import psutil
import threading
import functools
import cProfile
import pstats
import io
import asyncio
from typing import Dict, Any, Optional, Callable, List, Tuple
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
class ProcessingStep:
    """処理ステップの詳細情報"""
    name: str
    start_time: float
    end_time: float
    cpu_usage_start: float
    cpu_usage_end: float
    memory_usage_start: int
    memory_usage_end: int
    thread_id: Optional[int] = None
    process_id: Optional[int] = None


class PerformanceMonitor:
    """
    パフォーマンス監視クラス
    
    システムのメモリ使用量、実行時間などを監視し、
    パフォーマンス分析用のデータを収集する
    """
    
    def __init__(self, enabled: bool = True, memory_tracking: bool = True):
        """
        パフォーマンスモニターの初期化
        
        引数:
            enabled: 監視機能の有効/無効            memory_tracking: メモリ追跡の有効/無効
        """
        self.enabled = enabled
        self.memory_tracking = memory_tracking
        self.timers: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, MemorySnapshot] = {}
        self.cpu_usage_history = []
        self.processing_steps: List[ProcessingStep] = []
        self.start_time = time.time()
        self.process = psutil.Process()
        self.cpu_count = psutil.cpu_count(logical=True) or 8
        self.cpu_count_physical = psutil.cpu_count(logical=False) or 4
        self.lock = threading.Lock()
        
        # プロファイラー
        self.profiler = None
        self.profile_results: Dict[str, Any] = {}
        
        # CPU最適化関連の統計
        self.cpu_optimization_stats = {
            'parallel_executions': 0,
            'sequential_executions': 0,
            'thread_pool_usage': 0,
            'process_pool_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def start_profiling(self, name: str = "default") -> None:
        """プロファイリングを開始"""
        if not self.enabled:
            return
            
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
    def end_profiling(self, name: str = "default") -> Dict[str, Any]:
        """プロファイリングを終了し、結果を取得"""
        if not self.enabled or not self.profiler:
            return {}
            
        self.profiler.disable()
        
        # 結果をStringIOに出力
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # 上位20関数
        
        profile_text = s.getvalue()
        
        # 統計情報を抽出
        profile_stats = {
            'total_calls': ps.total_calls,
            'primitive_calls': ps.prim_calls,
            'total_time': ps.total_tt,
            'profile_text': profile_text,
            'timestamp': time.time()
        }
        
        self.profile_results[name] = profile_stats
        return profile_stats
        
    def start_processing_step(self, step_name: str) -> ProcessingStep:
        """処理ステップの開始を記録"""
        if not self.enabled:
            return ProcessingStep(step_name, time.time(), 0, 0, 0, 0, 0)
            
        current_time = time.time()
        cpu_usage = self.process.cpu_percent()
        memory_usage = self.process.memory_info().rss
        
        step = ProcessingStep(
            name=step_name,
            start_time=current_time,
            end_time=0,
            cpu_usage_start=cpu_usage,
            cpu_usage_end=0,
            memory_usage_start=memory_usage,
            memory_usage_end=0,
            thread_id=threading.get_ident(),
            process_id=self.process.pid
        )
        
        return step
        
    def end_processing_step(self, step: ProcessingStep) -> ProcessingStep:
        """処理ステップの終了を記録"""
        if not self.enabled:
            return step
            
        step.end_time = time.time()
        step.cpu_usage_end = self.process.cpu_percent()
        step.memory_usage_end = self.process.memory_info().rss
        
        with self.lock:
            self.processing_steps.append(step)
            
        return step

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
        現在のCPU使用率を取得
        
        戻り値:
            CPU使用率の詳細情報
        """
        if not self.enabled:
            return {}
            
        try:
            cpu_percent = self.process.cpu_percent()
            cpu_times = self.process.cpu_times()
            system_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            
            cpu_info = {
                'process_cpu_percent': cpu_percent,
                'system_cpu_percent': sum(system_cpu_percent) / len(system_cpu_percent),
                'per_core_usage': system_cpu_percent,
                'cpu_count_logical': self.cpu_count,
                'cpu_count_physical': self.cpu_count_physical,
                'cpu_times_user': cpu_times.user,
                'cpu_times_system': cpu_times.system,
                'timestamp': time.time()
            }
            
            self.cpu_usage_history.append(cpu_info)
            # 履歴が長くなりすぎないように制限
            if len(self.cpu_usage_history) > 1000:
                self.cpu_usage_history = self.cpu_usage_history[-500:]
                
            return cpu_info
        except Exception as e:
            return {'error': str(e)}
    
    def get_optimal_thread_count(self) -> int:
        """
        最適なスレッド数を計算
        
        戻り値:
            推奨スレッド数
        """
        # CPU使用率に基づく動的調整
        current_cpu = self.get_cpu_usage()
        system_cpu_avg = current_cpu.get('system_cpu_percent', 50)
        
        # システムCPU使用率が低い場合はより多くのスレッドを使用
        if system_cpu_avg < 30:
            # 低負荷時：論理コア数まで使用
            optimal = self.cpu_count
        elif system_cpu_avg < 70:
            # 中負荷時：物理コア数×1.5
            optimal = int(self.cpu_count_physical * 1.5)
        else:
            # 高負荷時：物理コア数のみ
            optimal = self.cpu_count_physical
            
        return max(min(optimal, self.cpu_count), 2)  # 最低2、最大論理コア数
    
    def get_optimal_process_count(self) -> int:
        """
        最適なプロセス数を計算（マルチプロセシング用）
        
        戻り値:
            推奨プロセス数
        """
        # プロセス間通信のオーバーヘッドを考慮して物理コア数を基準とする
        return max(min(self.cpu_count_physical, 4), 2)  # 最低2、最大4プロセス
    
    def record_parallel_execution(self, execution_type: str) -> None:
        """並列実行の統計を記録"""
        if not self.enabled:
            return
            
        with self.lock:
            if execution_type == 'parallel':
                self.cpu_optimization_stats['parallel_executions'] += 1
            elif execution_type == 'sequential':
                self.cpu_optimization_stats['sequential_executions'] += 1
            elif execution_type == 'thread_pool':
                self.cpu_optimization_stats['thread_pool_usage'] += 1
            elif execution_type == 'process_pool':
                self.cpu_optimization_stats['process_pool_usage'] += 1
    
    def record_cache_access(self, hit: bool) -> None:
        """キャッシュアクセスの統計を記録"""
        if not self.enabled:
            return
            
        with self.lock:
            if hit:
                self.cpu_optimization_stats['cache_hits'] += 1
            else:
                self.cpu_optimization_stats['cache_misses'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンス統計のサマリを取得
        
        戻り値:
            パフォーマンスサマリ
        """
        if not self.enabled:
            return {}
        
        current_cpu = self.get_cpu_usage()
        current_memory = self.get_memory_usage()
        uptime = time.time() - self.start_time
        
        # 処理ステップの統計
        step_stats = {}
        if self.processing_steps:
            total_steps = len(self.processing_steps)
            avg_step_time = sum(step.end_time - step.start_time for step in self.processing_steps if step.end_time > 0) / max(total_steps, 1)
            step_stats = {
                'total_steps': total_steps,
                'average_step_time': avg_step_time,
                'steps_by_name': {}
            }
            
            # ステップ名別の統計
            for step in self.processing_steps:
                if step.end_time > 0:
                    if step.name not in step_stats['steps_by_name']:
                        step_stats['steps_by_name'][step.name] = {
                            'count': 0,
                            'total_time': 0,
                            'avg_time': 0,
                            'min_time': float('inf'),
                            'max_time': 0
                        }
                    
                    duration = step.end_time - step.start_time
                    stats = step_stats['steps_by_name'][step.name]
                    stats['count'] += 1
                    stats['total_time'] += duration
                    stats['avg_time'] = stats['total_time'] / stats['count']
                    stats['min_time'] = min(stats['min_time'], duration)
                    stats['max_time'] = max(stats['max_time'], duration)
        
        # CPU最適化効率の計算
        cpu_efficiency = 0
        if self.cpu_optimization_stats['parallel_executions'] + self.cpu_optimization_stats['sequential_executions'] > 0:
            cpu_efficiency = self.cpu_optimization_stats['parallel_executions'] / (
                self.cpu_optimization_stats['parallel_executions'] + self.cpu_optimization_stats['sequential_executions']
            ) * 100
        
        # キャッシュヒット率の計算
        cache_hit_rate = 0
        total_cache_access = self.cpu_optimization_stats['cache_hits'] + self.cpu_optimization_stats['cache_misses']
        if total_cache_access > 0:
            cache_hit_rate = self.cpu_optimization_stats['cache_hits'] / total_cache_access * 100
        
        return {
            'uptime_seconds': uptime,
            'cpu_usage': current_cpu,
            'memory_usage': current_memory,
            'optimal_threads': self.get_optimal_thread_count(),
            'optimal_processes': self.get_optimal_process_count(),
            'memory_snapshots_count': len(self.memory_snapshots),
            'cpu_history_count': len(self.cpu_usage_history),
            'step_statistics': step_stats,
            'cpu_optimization_stats': self.cpu_optimization_stats,
            'cpu_efficiency_percent': cpu_efficiency,
            'cache_hit_rate_percent': cache_hit_rate,
            'profile_results_available': len(self.profile_results)
        }
    
    def get_detailed_step_analysis(self) -> Dict[str, Any]:
        """詳細なステップ分析を取得"""
        if not self.enabled or not self.processing_steps:
            return {}
        
        analysis = {
            'total_processing_time': 0,
            'bottleneck_steps': [],
            'parallel_efficiency': 0,
            'memory_growth_steps': [],
            'cpu_intensive_steps': []
        }
        
        completed_steps = [step for step in self.processing_steps if step.end_time > 0]
        if not completed_steps:
            return analysis
        
        # 総処理時間
        analysis['total_processing_time'] = sum(step.end_time - step.start_time for step in completed_steps)
        
        # ボトルネックステップの特定（処理時間が平均の2倍以上）
        step_times = [step.end_time - step.start_time for step in completed_steps]
        avg_time = sum(step_times) / len(step_times)
        threshold = avg_time * 2
        
        analysis['bottleneck_steps'] = [
            {
                'name': step.name,
                'duration': step.end_time - step.start_time,
                'cpu_usage_change': step.cpu_usage_end - step.cpu_usage_start,
                'memory_usage_change': step.memory_usage_end - step.memory_usage_start
            }
            for step in completed_steps
            if (step.end_time - step.start_time) > threshold
        ]
        
        # メモリ使用量が大幅に増加したステップ
        memory_threshold = 100 * 1024 * 1024  # 100MB
        analysis['memory_growth_steps'] = [
            {
                'name': step.name,
                'memory_increase': step.memory_usage_end - step.memory_usage_start,
                'memory_increase_formatted': format_memory_size(step.memory_usage_end - step.memory_usage_start)
            }
            for step in completed_steps
            if (step.memory_usage_end - step.memory_usage_start) > memory_threshold
        ]
        
        # CPU集約的なステップ
        cpu_threshold = 20  # 20%以上のCPU使用率増加
        analysis['cpu_intensive_steps'] = [
            {
                'name': step.name,
                'cpu_increase': step.cpu_usage_end - step.cpu_usage_start,
                'duration': step.end_time - step.start_time
            }
            for step in completed_steps
            if (step.cpu_usage_end - step.cpu_usage_start) > cpu_threshold
        ]
        
        return analysis

    def clear(self):
        """監視データをクリア"""
        with self.lock:
            self.timers.clear()
            self.memory_snapshots.clear()

    def shutdown(self):
        """
        PerformanceMonitorの完全なシャットダウン処理
        
        全ての監視データを記録し、リソースを解放する
        """
        if not self.enabled:
            return
        
        try:
            # 1. 最終統計の記録
            with self.lock:
                final_summary = self.get_performance_summary()
                if hasattr(self, '_logger'):
                    self._logger.info(f"PerformanceMonitor最終統計: {final_summary}")
                
                # 2. メモリスナップショットの最終記録
                if self.memory_tracking:
                    final_snapshot = self.take_memory_snapshot("shutdown_final")
                    if final_snapshot and hasattr(self, '_logger'):
                        self._logger.info(f"最終メモリ使用量: RSS={format_memory_size(final_snapshot.rss)}, "
                                        f"使用率={final_snapshot.percent:.1f}%")
                
                # 3. タイマー統計の出力
                if self.timers:
                    timer_summary = {}
                    for name, times in self.timers.items():
                        if times:
                            timer_summary[name] = {
                                'count': len(times),
                                'total': sum(times),
                                'average': sum(times) / len(times),
                                'min': min(times),
                                'max': max(times)
                            }
                    
                    if timer_summary and hasattr(self, '_logger'):
                        self._logger.info(f"タイマー統計: {timer_summary}")
                
                # 4. データクリア
                self.clear()
                
                # 5. 実行プールのシャットダウン（存在する場合）
                if hasattr(self, 'executor') and self.executor:
                    try:
                        self.executor.shutdown(wait=True, timeout=2.0)
                    except:
                        self.executor.shutdown(wait=False)
                    finally:
                        self.executor = None
                
                # 6. スレッド関連リソースのクリア
                self.process = None
                
        except Exception as e:
            # ログが利用できない場合はprintで出力
            try:
                if hasattr(self, '_logger'):
                    self._logger.error(f"PerformanceMonitorシャットダウンエラー: {e}")
                else:
                    print(f"PerformanceMonitorシャットダウンエラー: {e}")
            except:
                pass

def time_function(func: Callable) -> Callable:
    """
    関数の実行時間を計測するデコレータ
    
    引数:
        func: 計測対象の関数
        
    戻り値:
        ラップされた関数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            # ロギングは呼び出し元で行う
    return wrapper


def time_async_function(func: Callable) -> Callable:
    """
    非同期関数の実行時間を計測するデコレータ
    
    引数:
        func: 計測対象の非同期関数
        
    戻り値:
        ラップされた非同期関数
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            # ロギングは呼び出し元で行う
    return wrapper


def cpu_profile(func: Callable) -> Callable:
    """
    CPU使用率プロファイリング用デコレータ
    
    引数:
        func: プロファイリング対象の関数
        
    戻り値:
        ラップされた関数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 関数名からperformance monitorを取得できる場合のみプロファイリング
        try:
            if hasattr(args[0], 'performance') and isinstance(args[0].performance, PerformanceMonitor):
                monitor = args[0].performance
                step = monitor.start_processing_step(func.__name__)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    monitor.end_processing_step(step)
            else:
                return func(*args, **kwargs)
        except:
            return func(*args, **kwargs)
    return wrapper


def cpu_profile_async(func: Callable) -> Callable:
    """
    非同期関数用CPU使用率プロファイリングデコレータ
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if hasattr(args[0], 'performance') and isinstance(args[0].performance, PerformanceMonitor):
                monitor = args[0].performance
                step = monitor.start_processing_step(func.__name__)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    monitor.end_processing_step(step)
            else:
                return await func(*args, **kwargs)
        except:
            return await func(*args, **kwargs)
    return wrapper