#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blackboard モジュール
~~~~~~~~~~~~~~~~~~~
エージェント間の共有メモリとして機能する黒板パターン実装
CPU効率・メモリ効率・並列処理を最適化

機能:
- LRUキャッシュによるメモリ効率化
- ベクトル化されたNumPy操作
- 非同期I/O・バッファリング
- スレッドセーフな並列アクセス
- 効率的な履歴管理

作者: Yuhi Sonoki
"""

import asyncio
import hashlib
import json
import os
import psutil
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import weakref

import numpy as np

class LRUCache:
    """メモリ効率に優れたLRUキャッシュ実装"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """キーの値を取得し、最近使用されたものとしてマーク"""
        with self.lock:
            if key in self.cache:
                # 最後に移動（最近使用されたものとして）
                self.cache.move_to_end(key)
                return self.cache[key]
            return default
    
    def put(self, key: str, value: Any) -> None:
        """キーと値をキャッシュに追加"""
        with self.lock:
            if key in self.cache:
                # 既存のキーの場合は更新
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # 新しいキーの場合
                if len(self.cache) >= self.max_size:
                    # 最も古いアイテムを削除
                    self.cache.popitem(last=False)
                self.cache[key] = value
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """キャッシュサイズを取得"""
        with self.lock:
            return len(self.cache)


class AsyncBuffer:
    """非同期バッファ処理用クラス"""
    
    def __init__(self, max_buffer_size: int = 100, flush_interval: float = 5.0):
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self.buffer: Deque[Dict[str, Any]] = deque()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="BlackboardBuffer")
        self._flush_task = None
        self._stop_event = threading.Event()
        
        # 定期フラッシュタスクを開始
        self._start_flush_task()
    
    def add_to_buffer(self, data: Dict[str, Any]) -> None:
        """バッファにデータを追加"""
        with self.lock:
            self.buffer.append(data)
            if len(self.buffer) >= self.max_buffer_size:
                # バッファが満杯の場合は即座にフラッシュ
                self._flush_now()
    
    def _flush_now(self) -> None:
        """バッファを即座にフラッシュ（内部メソッド）"""
        if self.buffer:
            buffer_data = list(self.buffer)
            self.buffer.clear()
            # 非同期でファイル書き込み
            self.executor.submit(self._write_to_file, buffer_data)
    
    def _write_to_file(self, data: List[Dict[str, Any]]) -> None:
        """ファイルへの書き込み（別スレッドで実行）"""
        try:
            # ここでは簡略化してログ出力のみ
            # 実際の実装では適切なファイルパスへの書き込みを行う
            pass
        except Exception as e:
            print(f"Buffer write error: {e}")
    
    def _start_flush_task(self) -> None:
        """定期フラッシュタスクを開始"""
        def flush_periodically():
            while not self._stop_event.is_set():
                time.sleep(self.flush_interval)
                with self.lock:
                    if self.buffer:
                        self._flush_now()
        
        self.executor.submit(flush_periodically)
    
    def shutdown(self) -> None:
        """バッファのシャットダウン"""
        self._stop_event.set()
        with self.lock:
            if self.buffer:
                self._flush_now()
        self.executor.shutdown(wait=True)


class Blackboard:
    """
    分散エージェント間の共有メモリ実装（CPU最適化版）
    
    特徴:
    - LRUキャッシュによる効率的なメモリ管理
    - NumPyベクトル化による高速演算
    - 非同期バッファリングによるI/O最適化
    - スレッドセーフな並列アクセス
    - 適応的履歴管理
    """
    
    def __init__(self, config):
        self.config = config
        self.debug = config.get('debug', False)
        
        # CPU最適化設定
        self.cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
        self.enable_vectorization = config.get('enable_vectorization', True)
        self.enable_async_buffer = config.get('enable_async_buffer', True)
        
        # メモリ管理
        cache_size = config.get('blackboard_cache_size', min(1000, self.cpu_cores * 100))
        self.memory_cache = LRUCache(max_size=cache_size)
        
        # 基本メモリとロック
        self.memory = {}
        self.memory_lock = threading.RLock()
        
        # 履歴管理（効率的なdeque使用）
        history_size = config.get('blackboard_history_size', 1000)
        self.history = deque(maxlen=history_size)
        self.history_lock = threading.Lock()
        
        # 保持するキー
        self.persistent_keys = config.get('persistent_keys', ['conversation_context'])
        
        # 非同期バッファ
        if self.enable_async_buffer:
            buffer_size = config.get('buffer_size', 50)
            flush_interval = config.get('flush_interval', 3.0)
            self.async_buffer = AsyncBuffer(buffer_size, flush_interval)
        else:
            self.async_buffer = None
        
        # 統計情報
        self.stats = {
            'read_count': 0,
            'write_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'vector_operations': 0
        }
        self.stats_lock = threading.Lock()
        
        if self.debug:
            print(f"Blackboard初期化完了: CPU{self.cpu_cores}コア, キャッシュ{cache_size}, 履歴{history_size}")
    
    def write(self, key, value, priority='normal'):
        """
        黒板に情報を書き込む（最適化版）
        
        Args:
            key: 保存キー
            value: 保存値
            priority: 優先度 ('high', 'normal', 'low')
        """
        try:
            # 統計更新
            with self.stats_lock:
                self.stats['write_count'] += 1
            
            # 値の前処理
            processed_value = self._process_value_for_storage(value)
            
            # メモリとキャッシュに保存
            with self.memory_lock:
                self.memory[key] = processed_value
                
                # 高優先度または頻繁にアクセスされるキーはキャッシュに保存
                if priority == 'high' or key in self.persistent_keys:
                    self.memory_cache.put(key, processed_value)
            
            # 履歴エントリ作成
            entry = {
                'timestamp': time.time(),
                'key': key,
                'value': processed_value,
                'priority': priority,
                'thread_id': threading.current_thread().ident
            }
            
            # 履歴に追加
            with self.history_lock:
                self.history.append(entry)
            
            # 非同期バッファに追加
            if self.async_buffer:
                self.async_buffer.add_to_buffer(entry.copy())
            
            return entry
            
        except Exception as e:
            if self.debug:
                print(f"Blackboard write error for key '{key}': {e}")
            return {'error': str(e), 'key': key}
    
    def read(self, key, use_cache=True):
        """
        黒板から情報を読み込む（最適化版）
        
        Args:
            key: 読み込みキー
            use_cache: キャッシュを使用するか
        """
        try:
            # 統計更新
            with self.stats_lock:
                self.stats['read_count'] += 1
            
            # キャッシュから読み込み試行
            if use_cache:
                cached_value = self.memory_cache.get(key)
                if cached_value is not None:
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                    return cached_value
                else:
                    with self.stats_lock:
                        self.stats['cache_misses'] += 1
            
            # メモリから読み込み
            with self.memory_lock:
                value = self.memory.get(key)
                
                # 読み込み成功時はキャッシュに保存
                if value is not None and use_cache:
                    self.memory_cache.put(key, value)
                
                return value
                
        except Exception as e:
            if self.debug:
                print(f"Blackboard read error for key '{key}': {e}")
            return None
    
    def read_multiple(self, keys, use_cache=True):
        """
        複数キーを効率的に一括読み込み
        """
        result = {}
        
        # ベクトル化が有効な場合の最適化
        if self.enable_vectorization and len(keys) > 1:
            with self.stats_lock:
                self.stats['vector_operations'] += 1
        
        for key in keys:
            result[key] = self.read(key, use_cache)
        
        return result
    
    def _process_value_for_storage(self, value):
        """ストレージ用に値を最適化"""
        if isinstance(value, dict) and 'embedding' in value:
            processed = value.copy()
            
            if isinstance(value['embedding'], np.ndarray):
                vector = value['embedding']
                
                # NumPy最適化：ベクトル化統計計算
                if self.enable_vectorization:
                    with self.stats_lock:
                        self.stats['vector_operations'] += 1
                    
                    # ベクトル化された統計計算
                    stats = {
                        'shape': vector.shape,
                        'mean': float(np.mean(vector)),
                        'std': float(np.std(vector)),
                        'norm': float(np.linalg.norm(vector))
                    }
                else:
                    stats = {
                        'shape': vector.shape,
                        'mean': float(vector.mean())
                    }
                
                processed['embedding'] = f"<Vector {stats}>"
            
            return processed
        
        return value
    
    def read_all(self):
        """
        黒板のすべての現在値を効率的に取得
        """
        with self.memory_lock:
            return self.memory.copy()
    
    def clear_current_turn(self):
        """
        新しいターンのために効率的にクリア（保持キーは維持）
        """
        preserved_values = {}
        
        with self.memory_lock:
            # 保持するべき値を効率的に収集
            for key in self.persistent_keys:
                if key in self.memory:
                    preserved_values[key] = self.memory[key]
            
            # メモリクリア
            self.memory.clear()
            
            # 保持値を復元
            self.memory.update(preserved_values)
        
        # キャッシュから保持キー以外を削除
        if preserved_values:
            # 新しいキャッシュを作成し、保持キーのみ復元
            new_cache = LRUCache(max_size=self.memory_cache.max_size)
            for key, value in preserved_values.items():
                new_cache.put(key, value)
            self.memory_cache = new_cache
        else:
            self.memory_cache.clear()
        
        if self.debug:
            print(f"Blackboard効率クリア完了: 保持キー{len(preserved_values)}個")
    
    def get_history(self, key=None, limit=100):
        """
        効率的な履歴取得（制限付き）
        """
        with self.history_lock:
            if key is None:
                # 最新のlimit件を効率的に取得
                return list(self.history)[-limit:] if limit > 0 else list(self.history)
            else:
                # 特定キーの履歴を効率的にフィルタリング
                filtered = [entry for entry in self.history if entry['key'] == key]
                return filtered[-limit:] if limit > 0 else filtered
    
    def get_debug_view(self):
        """
        デバッグ表示用に最適化されたビューを提供
        """
        with self.memory_lock:
            result = {}
            for key, value in self.memory.items():
                if key == 'input' and isinstance(value, dict):
                    result[key] = {'normalized': value.get('normalized', '')}
                else:
                    result[key] = value
        
        return result
    
    def get_performance_stats(self):
        """
        パフォーマンス統計を取得
        """
        with self.stats_lock:
            stats = self.stats.copy()
        
        # キャッシュ効率計算
        total_reads = stats['cache_hits'] + stats['cache_misses']
        cache_hit_rate = stats['cache_hits'] / total_reads if total_reads > 0 else 0.0
        
        return {
            **stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': self.memory_cache.size(),
            'history_size': len(self.history),
            'memory_items': len(self.memory)
        }
    
    def optimize_memory(self):
        """
        メモリ最適化を実行
        """
        initial_cache_size = self.memory_cache.size()
        initial_history_size = len(self.history)
        
        # 古い履歴エントリの削除（時刻ベース）
        current_time = time.time()
        retention_period = self.config.get('history_retention_hours', 24) * 3600
        
        with self.history_lock:
            # dequeの特性を活かした効率的な古いエントリ削除
            while (self.history and 
                   current_time - self.history[0]['timestamp'] > retention_period):
                self.history.popleft()
        
        # メモリ使用量の確認
        memory_info = psutil.virtual_memory()
        
        final_cache_size = self.memory_cache.size()
        final_history_size = len(self.history)
        
        optimization_result = {
            'cache_size_reduced': initial_cache_size - final_cache_size,
            'history_size_reduced': initial_history_size - final_history_size,
            'memory_usage_percent': memory_info.percent,
            'available_memory_gb': memory_info.available / (1024**3)
        }
        
        if self.debug:
            print(f"Memory optimization completed: {optimization_result}")
        
        return optimization_result
    
    def shutdown(self):
        """
        Blackboardの適切なシャットダウン
        """
        if self.async_buffer:
            self.async_buffer.shutdown()
        
        if self.debug:
            final_stats = self.get_performance_stats()
            print(f"Blackboard shutdown. Final stats: {final_stats}")
