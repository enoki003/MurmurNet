#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local Distributed Blackboard モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ローカル分散処理用の高速メモリ共有黒板実装
同一マシン内でのマルチプロセス・スレッド間データ共有

機能:
- multiprocessing.Manager による高速プロセス間通信
- threading.Lock による排他制御
- メモリベース高速アクセス
- 分散処理最適化

作者: Yuhi Sonoki
"""

import logging
import time
import threading
import multiprocessing
from multiprocessing import Manager, Lock
from typing import Any, Dict, List, Optional
import json
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """
    numpy配列をJSONシリアライズするためのカスタムエンコーダー
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '_type': 'numpy_array',
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class LocalDistributedBlackboard:
    """
    ローカル分散ブラックボード
    
    同一マシン内での分散処理に最適化されたメモリ共有システム
    Redis不要でマルチプロセス・スレッド間でのデータ共有を提供
    """
    
    _manager = None
    _shared_data = None
    _global_lock = None
    
    @classmethod
    def _init_shared_resources(cls):
        """共有リソースの初期化（クラスレベル）"""
        if cls._manager is None:
            cls._manager = Manager()
            cls._shared_data = cls._manager.dict()
            cls._global_lock = cls._manager.Lock()
            logger.info("ローカル分散共有リソース初期化完了")
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        LocalDistributedBlackboard初期化
        
        Args:
            config: 設定辞書
                - key_prefix: キープレフィックス (default: murmurnet:)
                - debug: デバッグモード (default: False)
        """
        self.config = config or {}
        
        # 共有リソース初期化
        self._init_shared_resources()
        
        # インスタンス設定
        self.key_prefix = self.config.get('key_prefix', 'murmurnet:')
        self.session_id = str(uuid.uuid4())[:8]
        self.debug = self.config.get('debug', False)
        
        # ローカルキャッシュ（読み取り高速化）
        self._local_cache = {}
        self._cache_lock = threading.Lock()
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"LocalDistributedBlackboard初期化完了: session={self.session_id}")
    
    def _make_key(self, key: str) -> str:
        """内部キー作成"""
        return f"{self.key_prefix}{key}"
    
    def write(self, key: str, value: Any, priority: str = 'normal', ttl: Optional[int] = None) -> None:
        """
        値を書き込み
        
        Args:
            key: キー
            value: 値
            priority: 優先度（ローカル版では使用しない）
            ttl: 有効期限（ローカル版では使用しない）
        """
        full_key = self._make_key(key)
        
        try:
            # 共有メモリへの書き込み
            with self._global_lock:
                # データのシリアライズ（プロセス間共有のため）
                if isinstance(value, np.ndarray):
                    # numpy配列の特別処理
                    serialized_value = {
                        '_type': 'numpy_array',
                        'data': value.tolist(),
                        'dtype': str(value.dtype),
                        'shape': value.shape
                    }
                    serialized_value = json.dumps(serialized_value, ensure_ascii=False)
                elif isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, ensure_ascii=False, cls=NumpyEncoder)
                else:
                    serialized_value = str(value)
                
                # メタデータと共に保存
                entry = {
                    'value': serialized_value,
                    'timestamp': time.time(),
                    'session': self.session_id,
                    'original_type': type(value).__name__
                }
                
                self._shared_data[full_key] = entry
            
            # ローカルキャッシュ更新
            with self._cache_lock:
                self._local_cache[full_key] = value
            
            if self.debug:
                logger.debug(f"LocalDistributed書き込み成功: {key} -> {len(str(value))}文字")
            
        except Exception as e:
            logger.error(f"LocalDistributed書き込みエラー ({key}): {e}")
            raise
    
    def read(self, key: str, use_cache: bool = True) -> Any:
        """
        値を読み取り
        
        Args:
            key: キー
            use_cache: ローカルキャッシュ使用フラグ
            
        Returns:
            読み取った値、存在しない場合はNone
        """
        full_key = self._make_key(key)
        
        # ローカルキャッシュから高速取得
        if use_cache:
            with self._cache_lock:
                if full_key in self._local_cache:
                    if self.debug:
                        logger.debug(f"キャッシュヒット: {key}")
                    return self._local_cache[full_key]
        
        try:
            # 共有メモリから取得
            with self._global_lock:
                if full_key not in self._shared_data:
                    return None
                
                entry = dict(self._shared_data[full_key])  # コピーを作成
            
            # デシリアライズ
            raw_value = entry['value']
            original_type = entry.get('original_type', 'str')
            
            if original_type in ['dict', 'list'] or isinstance(raw_value, str):
                try:
                    parsed_value = json.loads(raw_value)
                    # numpy配列の復元
                    if isinstance(parsed_value, dict) and parsed_value.get('_type') == 'numpy_array':
                        value = np.array(parsed_value['data'], dtype=parsed_value['dtype']).reshape(parsed_value['shape'])
                    else:
                        value = parsed_value
                except (json.JSONDecodeError, TypeError):
                    value = raw_value
            else:
                value = raw_value
            
            # ローカルキャッシュ更新
            if use_cache:
                with self._cache_lock:
                    self._local_cache[full_key] = value
            
            return value
            
        except Exception as e:
            logger.error(f"LocalDistributed読み取りエラー ({key}): {e}")
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
        
        for key in keys:
            result[key] = self.read(key, use_cache)
        
        return result
    
    def read_all(self) -> Dict[str, Any]:
        """
        全キーを読み取り
        
        Returns:
            全キー-値の辞書
        """
        result = {}
        
        try:
            # 共有メモリから全キー取得
            with self._global_lock:
                all_keys = [k for k in self._shared_data.keys() if k.startswith(self.key_prefix)]
            
            # 各キーの値を取得
            for full_key in all_keys:
                clean_key = full_key[len(self.key_prefix):]
                result[clean_key] = self.read(clean_key, use_cache=False)
            
        except Exception as e:
            logger.error(f"LocalDistributed全読み取りエラー: {e}")
        
        return result
    
    def write_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        複数キーを一括書き込み
        
        Args:
            data: キー-値の辞書
            ttl: 有効期限（ローカル版では使用しない）
        """
        if not data:
            return
        
        try:
            # バッチ書き込み最適化
            with self._global_lock:
                timestamp = time.time()
                
                for key, value in data.items():
                    full_key = self._make_key(key)
                    
                    # シリアライズ
                    if isinstance(value, np.ndarray):
                        # numpy配列の特別処理
                        serialized_value = {
                            '_type': 'numpy_array',
                            'data': value.tolist(),
                            'dtype': str(value.dtype),
                            'shape': value.shape
                        }
                        serialized_value = json.dumps(serialized_value, ensure_ascii=False)
                    elif isinstance(value, (dict, list)):
                        serialized_value = json.dumps(value, ensure_ascii=False, cls=NumpyEncoder)
                    else:
                        serialized_value = str(value)
                    
                    # エントリ作成
                    entry = {
                        'value': serialized_value,
                        'timestamp': timestamp,
                        'session': self.session_id,
                        'original_type': type(value).__name__
                    }
                    
                    self._shared_data[full_key] = entry
            
            # ローカルキャッシュ一括更新
            with self._cache_lock:
                for key, value in data.items():
                    full_key = self._make_key(key)
                    self._local_cache[full_key] = value
            
            if self.debug:
                logger.debug(f"LocalDistributed一括書き込み成功: {len(data)}件")
            
        except Exception as e:
            logger.error(f"LocalDistributed一括書き込みエラー: {e}")
            raise
    
    def clear_current_turn(self, keep_keys: Optional[List[str]] = None) -> None:
        """
        現在のターンをクリア（指定キー以外）
        
        Args:
            keep_keys: 保持するキーのリスト
        """
        try:
            # 保持キーのセット作成
            keep_full_keys = set()
            if keep_keys:
                for key in keep_keys:
                    keep_full_keys.add(self._make_key(key))
            
            # 削除対象キー抽出
            with self._global_lock:
                all_keys = [k for k in self._shared_data.keys() if k.startswith(self.key_prefix)]
                keys_to_delete = [k for k in all_keys if k not in keep_full_keys]
                
                # 共有メモリから削除
                for key in keys_to_delete:
                    del self._shared_data[key]
            
            # ローカルキャッシュからも削除
            with self._cache_lock:
                for key in keys_to_delete:
                    self._local_cache.pop(key, None)
            
            logger.info(f"LocalDistributed黒板クリア: {len(keys_to_delete)}キー削除, {len(keep_full_keys)}キー保持")
            
        except Exception as e:
            logger.error(f"LocalDistributedクリアエラー: {e}")
    
    def get_debug_view(self) -> Dict[str, Any]:
        """
        デバッグ用の黒板内容表示
        
        Returns:
            デバッグ情報辞書
        """
        debug_info = {
            "type": "LocalDistributedBlackboard",
            "session_id": self.session_id,
            "key_prefix": self.key_prefix,
            "total_keys": 0,
            "cache_size": 0,
            "data_preview": {}
        }
        
        try:
            # 統計情報取得
            with self._global_lock:
                all_keys = [k for k in self._shared_data.keys() if k.startswith(self.key_prefix)]
                debug_info["total_keys"] = len(all_keys)
            
            with self._cache_lock:
                debug_info["cache_size"] = len(self._local_cache)
            
            # データプレビュー（最大5件）
            all_data = self.read_all()
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
    
    def invalidate_cache(self) -> None:
        """ローカルキャッシュを無効化"""
        with self._cache_lock:
            self._local_cache.clear()
        logger.debug("ローカルキャッシュをクリアしました")
    
    async def shutdown(self) -> None:
        """シャットダウン処理"""
        try:
            # ローカルキャッシュクリア
            self.invalidate_cache()
            
            # セッション関連データの削除
            with self._global_lock:
                session_keys = []
                for key, entry in self._shared_data.items():
                    if key.startswith(self.key_prefix) and entry.get('session') == self.session_id:
                        session_keys.append(key)
                
                for key in session_keys:
                    del self._shared_data[key]
                    
                logger.info(f"セッション {self.session_id} のデータを削除: {len(session_keys)}件")
            
        except Exception as e:
            logger.error(f"LocalDistributedシャットダウンエラー: {e}")


def create_blackboard(config: Dict[str, Any] = None) -> LocalDistributedBlackboard:
    """
    ローカル分散ブラックボードを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        LocalDistributedBlackboardインスタンス
    """
    if config is None:
        config = {}
    
    return LocalDistributedBlackboard(config)
