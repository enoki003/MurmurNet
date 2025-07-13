#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis Blackboard モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~
分散環境対応のRedisベース黒板実装
複数プロセス・複数ホスト間でのデータ共有を提供

機能:
- Redis HASHによる高速データストレージ
- Luaスクリプトによるアトミック操作
- 分散ロック機能
- 履歴・キャッシュ機能
- 自動期限切れ（TTL）

作者: Yuhi Sonoki
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    import redis
    import redis.exceptions
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    raise ImportError("Redis is required for distributed operation. Install with: pip install redis")

logger = logging.getLogger(__name__)

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
                
        Raises:
            ImportError: Redis未インストール
            ConnectionError: Redis接続失敗
        """
        self.config = config or {}
        
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
        
        logger.info(f"RedisBlackboard初期化完了: {self.redis_host}:{self.redis_port}, session={self.session_id}")
    
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
            logger.info("Redis接続成功")
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis接続失敗: {e}")
            raise
        except Exception as e:
            logger.error(f"Redis初期化エラー: {e}")
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
        
        # 条件付き読み取りスクリプト
        self.lua_conditional_read = self.redis_client.register_script("""
            local key = KEYS[1]
            local min_timestamp = tonumber(ARGV[1]) or 0
            
            local data = redis.call('HMGET', key, 'value', 'timestamp', 'session')
            if data[1] and data[2] and tonumber(data[2]) >= min_timestamp then
                return {data[1], data[2], data[3]}
            else
                return nil
            end
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
            
            logger.debug(f"Redis書き込み成功: {key} -> {len(serialized_value)}文字")
            
        except Exception as e:
            logger.error(f"Redis書き込みエラー ({key}): {e}")
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
            logger.error(f"Redis読み取りエラー ({key}): {e}")
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
            logger.error(f"Redis一括読み取りエラー: {e}")
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
            logger.error(f"Redis全読み取りエラー: {e}")
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
            
            logger.debug(f"Redis一括書き込み成功: {len(data)}件")
            
        except Exception as e:
            logger.error(f"Redis一括書き込みエラー: {e}")
            # フォールバック：個別書き込み
            for key, value in data.items():
                try:
                    self.write(key, value, ttl=ttl)
                except Exception as individual_error:
                    logger.error(f"個別書き込みエラー ({key}): {individual_error}")
    
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
                logger.info(f"Redis黒板クリア: {len(keys_to_delete)}キー削除, {len(keep_redis_keys)}キー保持")
            
        except Exception as e:
            logger.error(f"Redisクリアエラー: {e}")
    
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
                        logger.info(f"セッション {self.session_id} のキーを削除: {len(session_keys)}件")
                
                # 接続クローズ
                self.redis_client.close()
                logger.info("Redis接続をクローズしました")
                
        except Exception as e:
            logger.error(f"Redisシャットダウンエラー: {e}")
    
    def __del__(self):
        """デストラクタ"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                self.redis_client.close()
        except:
            pass


def create_blackboard(config: Dict[str, Any] = None) -> RedisBlackboard:
    """
    Redis分散ブラックボードを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        RedisBlackboardインスタンス
        
    Raises:
        ImportError: Redisライブラリが未インストール
        ConnectionError: Redis接続失敗
    """
    if config is None:
        config = {}
    
    if not HAS_REDIS:
        raise ImportError("Redis is required for distributed operation. Install with: pip install redis")
    
    return RedisBlackboard(config)
