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
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
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
            enabled: 監視機能の有効/無効
            memory_tracking: メモリ追跡の有効/無効
        """
        self.enabled = enabled
        self.memory_tracking = memory_tracking
        self.timers: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, MemorySnapshot] = {}
        self.lock = threading.Lock()
        
        # プロセス情報の取得
        self.process = psutil.Process()
        
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンス監視の要約を取得
        
        戻り値:
            パフォーマンス要約の辞書
        """
        if not self.enabled:
            return {"enabled": False}
            
        with self.lock:
            return {
                "enabled": True,
                "memory_tracking": self.memory_tracking,
                "timer_count": len(self.timers),
                "snapshot_count": len(self.memory_snapshots),
                "current_memory": self.get_memory_usage()
            }
    
    def clear(self):
        """監視データをクリア"""
        with self.lock:
            self.timers.clear()
            self.memory_snapshots.clear()


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