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
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import weakref

import numpy as np

# Redis サポート（オプション）
try:
    import redis
    import redis.exceptions
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

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


class RedisBlackboard:
    """
    Redis分散ブラックボード
    
    分散環境での共有メモリとして機能し、複数のプロセス・ホスト間で
    データの読み書きを行う。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        RedisBlackboard初期化
        
        Args:
            config: 設定辞書
                - redis_host: Redisホスト (default: localhost)
                - redis_port: Redisポート (default: 6379)
                - redis_db: Redis DB番号 (default: 0)
                - redis_password: Redis認証パスワード
                - key_prefix: キープレフィックス (default: murmurnet:)
                - default_ttl: デフォルトTTL秒 (default: 3600)
        """
        if not HAS_REDIS:
            raise ImportError("Redis is required for distributed blackboard. Install with: pip install redis")
        
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        
        # Redis接続設定
        self.redis_host = self.config.get('redis_host', 'localhost')
        self.redis_port = self.config.get('redis_port', 6379)
        self.redis_db = self.config.get('redis_db', 0)
        self.redis_password = self.config.get('redis_password', None)
        
        # キー管理
        self.key_prefix = self.config.get('key_prefix', 'murmurnet:')
        self.session_id = str(uuid.uuid4())[:8]  # セッション識別子
        self.default_ttl = self.config.get('default_ttl', 3600)  # 1時間
        
        # Redis接続
        self.redis_client = None
        self._connect()
        
        # Luaスクリプト準備
        self._prepare_lua_scripts()
        
        if self.debug:
            print(f"RedisBlackboard初期化完了: {self.redis_host}:{self.redis_port}, session={self.session_id}")
    
    def _connect(self):
        """Redis接続確立"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 接続テスト
            self.redis_client.ping()
            if self.debug:
                print("Redis接続成功")
            
        except redis.exceptions.ConnectionError as e:
            if self.debug:
                print(f"Redis接続失敗: {e}")
            raise
        except Exception as e:
            if self.debug:
                print(f"Redis初期化エラー: {e}")
            raise
    
    def _prepare_lua_scripts(self):
        """Luaスクリプトの準備"""
        # アトミックな読み書きスクリプト
        self.lua_atomic_write = self.redis_client.register_script("""
            local key = KEYS[1]
            local value = ARGV[1]
            local ttl = tonumber(ARGV[2])
            
            redis.call('HSET', key, 'value', value)
            redis.call('HSET', key, 'timestamp', redis.call('TIME')[1])
            redis.call('HSET', key, 'session', ARGV[3])
            
            if ttl > 0 then
                redis.call('EXPIRE', key, ttl)
            end
            
            return 1
        """)
        
        # バッチ書き込みスクリプト
        self.lua_batch_write = self.redis_client.register_script("""
            local ttl = tonumber(ARGV[1])
            local session = ARGV[2]
            local timestamp = redis.call('TIME')[1]
            
            for i = 3, #ARGV, 2 do
                local key = ARGV[i]
                local value = ARGV[i + 1]
                
                redis.call('HSET', key, 'value', value)
                redis.call('HSET', key, 'timestamp', timestamp)
                redis.call('HSET', key, 'session', session)
                
                if ttl > 0 then
                    redis.call('EXPIRE', key, ttl)
                end
            end
            
            return 1
        """)
    
    def _make_key(self, key: str) -> str:
        """内部キー作成"""
        return f"{self.key_prefix}{key}"
    
    def write(self, key: str, value: Any, priority: str = 'normal', ttl: Optional[int] = None) -> None:
        """
        値を書き込み
        
        Args:
            key: キー
            value: 値
            priority: 優先度（互換性のため、Redisでは使用しない）
            ttl: 有効期限（秒）
        """
        if ttl is None:
            ttl = self.default_ttl
            
        redis_key = self._make_key(key)
        
        try:
            # JSON形式でシリアライズ
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            else:
                serialized_value = str(value)
            
            # Luaスクリプトでアトミック書き込み
            self.lua_atomic_write(
                keys=[redis_key],
                args=[serialized_value, ttl, self.session_id]
            )
            
            if self.debug:
                print(f"Redis書き込み成功: {key} -> {len(serialized_value)}文字")
            
        except Exception as e:
            if self.debug:
                print(f"Redis書き込みエラー ({key}): {e}")
            raise
    
    def read(self, key: str, use_cache: bool = True) -> Any:
        """
        値を読み取り
        
        Args:
            key: キー
            use_cache: キャッシュ使用フラグ（Redis版では常にサーバから取得）
            
        Returns:
            読み取った値、存在しない場合はNone
        """
        redis_key = self._make_key(key)
        
        try:
            # Redis HASHから取得
            data = self.redis_client.hmget(redis_key, 'value', 'timestamp', 'session')
            
            if data[0] is not None:  # value が存在
                raw_value = data[0]
                
                # JSON復元を試行
                try:
                    return json.loads(raw_value)
                except (json.JSONDecodeError, TypeError):
                    # JSONでない場合はそのまま返す
                    return raw_value
            else:
                return None
                
        except Exception as e:
            if self.debug:
                print(f"Redis読み取りエラー ({key}): {e}")
            return None
    
    def read_multiple(self, keys: List[str], use_cache: bool = True) -> Dict[str, Any]:
        """
        複数キーを一括読み取り
        
        Args:
            keys: キーのリスト
            use_cache: キャッシュ使用フラグ
            
        Returns:
            キー-値の辞書
        """
        result = {}
        
        try:
            # Redis PIPELINEで効率的に一括取得
            with self.redis_client.pipeline() as pipe:
                for key in keys:
                    redis_key = self._make_key(key)
                    pipe.hmget(redis_key, 'value', 'timestamp', 'session')
                
                pipeline_results = pipe.execute()
                
                for i, key in enumerate(keys):
                    data = pipeline_results[i]
                    if data and data[0] is not None:
                        raw_value = data[0]
                        try:
                            result[key] = json.loads(raw_value)
                        except (json.JSONDecodeError, TypeError):
                            result[key] = raw_value
                    else:
                        result[key] = None
                        
        except Exception as e:
            if self.debug:
                print(f"Redis一括読み取りエラー: {e}")
            # フォールバック：個別読み取り
            for key in keys:
                result[key] = self.read(key, use_cache)
        
        return result
    
    def read_all(self) -> Dict[str, Any]:
        """
        全キーを読み取り
        
        Returns:
            全キー-値の辞書
        """
        try:
            # プレフィックスマッチでキー一覧取得
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return {}
            
            result = {}
            
            # 一括取得
            with self.redis_client.pipeline() as pipe:
                for redis_key in keys:
                    pipe.hmget(redis_key, 'value', 'timestamp', 'session')
                
                pipeline_results = pipe.execute()
                
                for i, redis_key in enumerate(keys):
                    # キープレフィックスを除去
                    clean_key = redis_key[len(self.key_prefix):]
                    data = pipeline_results[i]
                    
                    if data and data[0] is not None:
                        raw_value = data[0]
                        try:
                            result[clean_key] = json.loads(raw_value)
                        except (json.JSONDecodeError, TypeError):
                            result[clean_key] = raw_value
                    else:
                        result[clean_key] = None
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"Redis全読み取りエラー: {e}")
            return {}
    
    def write_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        複数キーを一括書き込み
        
        Args:
            data: キー-値の辞書
            ttl: 有効期限（秒）
        """
        if not data:
            return
            
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            # Luaスクリプト用の引数準備
            args = [ttl, self.session_id]
            
            for key, value in data.items():
                redis_key = self._make_key(key)
                
                # シリアライズ
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, ensure_ascii=False)
                else:
                    serialized_value = str(value)
                
                args.extend([redis_key, serialized_value])
            
            # バッチ書き込み実行
            self.lua_batch_write(keys=[], args=args)
            
            if self.debug:
                print(f"Redis一括書き込み成功: {len(data)}件")
            
        except Exception as e:
            if self.debug:
                print(f"Redis一括書き込みエラー: {e}")
            # フォールバック：個別書き込み
            for key, value in data.items():
                try:
                    self.write(key, value, ttl=ttl)
                except Exception as individual_error:
                    if self.debug:
                        print(f"個別書き込みエラー ({key}): {individual_error}")
    
    def clear_current_turn(self, keep_keys: Optional[List[str]] = None) -> None:
        """
        現在のターンをクリア（指定キー以外）
        
        Args:
            keep_keys: 保持するキーのリスト
        """
        try:
            # 全キー取得
            pattern = f"{self.key_prefix}*"
            all_keys = self.redis_client.keys(pattern)
            
            if not all_keys:
                return
            
            # 保持キーのセット作成
            keep_redis_keys = set()
            if keep_keys:
                for key in keep_keys:
                    keep_redis_keys.add(self._make_key(key))
            
            # 削除対象キー抽出
            keys_to_delete = [key for key in all_keys if key not in keep_redis_keys]
            
            if keys_to_delete:
                # 一括削除
                self.redis_client.delete(*keys_to_delete)
                if self.debug:
                    print(f"Redis黒板クリア: {len(keys_to_delete)}キー削除, {len(keep_redis_keys)}キー保持")
            
        except Exception as e:
            if self.debug:
                print(f"Redisクリアエラー: {e}")
    
    def get_debug_view(self) -> Dict[str, Any]:
        """
        デバッグ用の黒板内容表示
        
        Returns:
            デバッグ情報辞書
        """
        debug_info = {
            "redis_host": f"{self.redis_host}:{self.redis_port}",
            "session_id": self.session_id,
            "key_prefix": self.key_prefix,
            "total_keys": 0,
            "data_preview": {}
        }
        
        try:
            # Redis情報取得
            info = self.redis_client.info()
            debug_info["redis_memory"] = info.get('used_memory_human', 'N/A')
            debug_info["redis_clients"] = info.get('connected_clients', 0)
            
            # 全データ取得（制限付き）
            all_data = self.read_all()
            debug_info["total_keys"] = len(all_data)
            
            # プレビュー作成（最大5件、文字数制限）
            preview_count = 0
            for key, value in all_data.items():
                if preview_count >= 5:
                    break
                
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                
                debug_info["data_preview"][key] = value_str
                preview_count += 1
                
        except Exception as e:
            debug_info["error"] = str(e)
        
        return debug_info
    
    async def shutdown(self) -> None:
        """
        Redis接続のシャットダウン
        """
        try:
            if self.redis_client:
                # セッション関連キーの削除（オプション）
                if hasattr(self, 'session_id'):
                    pattern = f"{self.key_prefix}*"
                    keys = self.redis_client.keys(pattern)
                    
                    session_keys = []
                    for key in keys:
                        data = self.redis_client.hmget(key, 'session')
                        if data[0] == self.session_id:
                            session_keys.append(key)
                    
                    if session_keys:
                        self.redis_client.delete(*session_keys)
                        if self.debug:
                            print(f"セッション {self.session_id} のキーを削除: {len(session_keys)}件")
                
                # 接続クローズ
                self.redis_client.close()
                if self.debug:
                    print("Redis接続をクローズしました")
                
        except Exception as e:
            if self.debug:
                print(f"Redisシャットダウンエラー: {e}")


def create_blackboard(config: Dict[str, Any] = None) -> Union[Blackboard, RedisBlackboard]:
    """
    設定に応じて適切なブラックボードを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        ブラックボードインスタンス
    """
    if config is None:
        config = {}
    
    # Redis使用フラグ
    use_redis = config.get('use_redis_blackboard', False)
    
    if use_redis and HAS_REDIS:
        try:
            return RedisBlackboard(config)
        except Exception as e:
            if config.get('debug', False):
                print(f"Redis接続失敗、ローカルモードにフォールバック: {e}")
            return Blackboard(config)
    else:
        return Blackboard(config)
