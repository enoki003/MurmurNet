#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ローカル黒板システム
~~~~~~~~~~~~~~~~~~~
Redisを使わずにローカルでblackboard機能を提供

作者: Yuhi Sonoki
"""

import threading
import time
import json
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LocalMessage:
    """ローカルメッセージクラス"""
    id: str
    channel: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    expiry: Optional[float] = None

class LocalBlackboard:
    """Redis不要のローカル黒板実装"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ローカルストレージ
        self._data = {}
        self._channels = defaultdict(deque)
        self._subscribers = defaultdict(list)
        self._lock = threading.RLock()
        self._message_counter = 0
        
        # 設定
        self.max_messages_per_channel = config.get('max_messages_per_channel', 1000)
        self.message_ttl = config.get('message_ttl', 3600)  # 1時間
        
        self.logger.info("ローカル黒板システムを初期化しました")
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """キー値設定"""
        try:
            with self._lock:
                self._data[key] = {
                    'value': value,
                    'timestamp': time.time(),
                    'expiry': time.time() + ex if ex else None
                }
            return True
        except Exception as e:
            self.logger.error(f"set操作エラー: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """キー値取得"""
        try:
            with self._lock:
                if key not in self._data:
                    return None
                
                item = self._data[key]
                
                # 有効期限チェック
                if item['expiry'] and time.time() > item['expiry']:
                    del self._data[key]
                    return None
                
                return item['value']
        except Exception as e:
            self.logger.error(f"get操作エラー: {e}")
            return None
    
    def read(self, key: str) -> Optional[Any]:
        """キー値読み取り（getのエイリアス）"""
        return self.get(key)
    
    def delete(self, key: str) -> bool:
        """キー削除"""
        try:
            with self._lock:
                if key in self._data:
                    del self._data[key]
                    return True
                return False
        except Exception as e:
            self.logger.error(f"delete操作エラー: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """キー存在確認"""
        try:
            with self._lock:
                if key not in self._data:
                    return False
                
                item = self._data[key]
                
                # 有効期限チェック
                if item['expiry'] and time.time() > item['expiry']:
                    del self._data[key]
                    return False
                
                return True
        except Exception as e:
            self.logger.error(f"exists操作エラー: {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """キー一覧取得"""
        try:
            with self._lock:
                current_time = time.time()
                valid_keys = []
                expired_keys = []
                
                for key, item in self._data.items():
                    if item['expiry'] and current_time > item['expiry']:
                        expired_keys.append(key)
                    else:
                        if pattern == "*" or pattern in key:
                            valid_keys.append(key)
                
                # 期限切れキーを削除
                for key in expired_keys:
                    del self._data[key]
                
                return valid_keys
        except Exception as e:
            self.logger.error(f"keys操作エラー: {e}")
            return []
    
    def read(self, key: str) -> Optional[Any]:
        """キー値読み取り（getのエイリアス）"""
        return self.get(key)
    
    def write(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """キー値書き込み（setのエイリアス）"""
        return self.set(key, value, ex)
    
    def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """メッセージ発行"""
        try:
            with self._lock:
                self._message_counter += 1
                msg = LocalMessage(
                    id=str(self._message_counter),
                    channel=channel,
                    data=message,
                    expiry=time.time() + self.message_ttl
                )
                
                # チャネルにメッセージ追加
                self._channels[channel].append(msg)
                
                # メッセージ数制限
                while len(self._channels[channel]) > self.max_messages_per_channel:
                    self._channels[channel].popleft()
                
                # 購読者に通知
                for callback in self._subscribers.get(channel, []):
                    try:
                        callback(channel, message)
                    except Exception as e:
                        self.logger.error(f"購読者コールバックエラー: {e}")
                
                return True
        except Exception as e:
            self.logger.error(f"publish操作エラー: {e}")
            return False
    
    def subscribe(self, channel: str, callback: callable) -> bool:
        """チャネル購読"""
        try:
            with self._lock:
                if callback not in self._subscribers[channel]:
                    self._subscribers[channel].append(callback)
                return True
        except Exception as e:
            self.logger.error(f"subscribe操作エラー: {e}")
            return False
    
    def unsubscribe(self, channel: str, callback: callable) -> bool:
        """購読解除"""
        try:
            with self._lock:
                if callback in self._subscribers[channel]:
                    self._subscribers[channel].remove(callback)
                return True
        except Exception as e:
            self.logger.error(f"unsubscribe操作エラー: {e}")
            return False
    
    def get_messages(self, channel: str, count: int = 10) -> List[Dict[str, Any]]:
        """チャネルメッセージ取得"""
        try:
            with self._lock:
                current_time = time.time()
                messages = []
                
                # 期限切れメッセージを削除
                while (self._channels[channel] and 
                       self._channels[channel][0].expiry and 
                       current_time > self._channels[channel][0].expiry):
                    self._channels[channel].popleft()
                
                # 最新メッセージを取得
                channel_messages = list(self._channels[channel])
                for msg in channel_messages[-count:]:
                    if not msg.expiry or current_time <= msg.expiry:
                        messages.append({
                            'id': msg.id,
                            'data': msg.data,
                            'timestamp': msg.timestamp
                        })
                
                return messages
        except Exception as e:
            self.logger.error(f"get_messages操作エラー: {e}")
            return []
    
    def clear_expired(self):
        """期限切れデータの削除"""
        try:
            with self._lock:
                current_time = time.time()
                
                # キー値ストレージの期限切れ削除
                expired_keys = [
                    key for key, item in self._data.items()
                    if item['expiry'] and current_time > item['expiry']
                ]
                for key in expired_keys:
                    del self._data[key]
                
                # チャネルメッセージの期限切れ削除
                for channel in self._channels:
                    while (self._channels[channel] and 
                           self._channels[channel][0].expiry and 
                           current_time > self._channels[channel][0].expiry):
                        self._channels[channel].popleft()
                
                self.logger.debug(f"期限切れデータを削除: {len(expired_keys)}キー")
        except Exception as e:
            self.logger.error(f"clear_expired操作エラー: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            with self._lock:
                return {
                    'total_keys': len(self._data),
                    'total_channels': len(self._channels),
                    'total_messages': sum(len(msgs) for msgs in self._channels.values()),
                    'total_subscribers': sum(len(subs) for subs in self._subscribers.values()),
                    'memory_usage': len(json.dumps(self._data)),
                    'uptime': time.time() - getattr(self, '_start_time', time.time())
                }
        except Exception as e:
            self.logger.error(f"get_stats操作エラー: {e}")
            return {}
    
    def ping(self) -> bool:
        """接続確認"""
        return True
    
    def clear_current_turn(self):
        """現在のターンのデータをクリア"""
        try:
            with self._lock:
                # 現在のターンに関連するキーを削除
                turn_keys = [key for key in self._data.keys() if 'current_turn' in key or 'turn_data' in key]
                for key in turn_keys:
                    del self._data[key]
                
                # チャネルメッセージもクリア
                turn_channels = [channel for channel in self._channels.keys() if 'turn' in channel]
                for channel in turn_channels:
                    self._channels[channel].clear()
                
                self.logger.debug("現在のターンデータをクリアしました")
        except Exception as e:
            self.logger.error(f"clear_current_turn操作エラー: {e}")
    
    def increment_turn(self) -> int:
        """ターン番号をインクリメント"""
        try:
            with self._lock:
                current_turn = self.get('turn_counter') or 0
                new_turn = current_turn + 1
                self.set('turn_counter', new_turn)
                return new_turn
        except Exception as e:
            self.logger.error(f"increment_turn操作エラー: {e}")
            return 0
    
    def get_turn_counter(self) -> int:
        """現在のターン番号を取得"""
        return self.get('turn_counter') or 0
    
    def set_turn_data(self, key: str, value: Any):
        """ターンデータを設定"""
        turn_key = f"turn_data:{key}"
        self.set(turn_key, value)
    
    def get_turn_data(self, key: str) -> Optional[Any]:
        """ターンデータを取得"""
        turn_key = f"turn_data:{key}"
        return self.get(turn_key)
    
    def add_to_conversation_history(self, role: str, content: str):
        """会話履歴に追加"""
        try:
            with self._lock:
                history = self.get('conversation_history') or []
                history.append({
                    'role': role,
                    'content': content,
                    'timestamp': time.time()
                })
                self.set('conversation_history', history)
        except Exception as e:
            self.logger.error(f"add_to_conversation_history操作エラー: {e}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """会話履歴を取得"""
        return self.get('conversation_history') or []
    
    def clear_conversation_history(self):
        """会話履歴をクリア"""
        self.delete('conversation_history')
    
    def get_agent_data(self, agent_id: str, key: str) -> Optional[Any]:
        """エージェント固有データを取得"""
        agent_key = f"agent:{agent_id}:{key}"
        return self.get(agent_key)
    
    def set_agent_data(self, agent_id: str, key: str, value: Any):
        """エージェント固有データを設定"""
        agent_key = f"agent:{agent_id}:{key}"
        self.set(agent_key, value)
    
    def delete_agent_data(self, agent_id: str, key: str) -> bool:
        """エージェント固有データを削除"""
        agent_key = f"agent:{agent_id}:{key}"
        return self.delete(agent_key)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_debug_view(self) -> Dict[str, Any]:
        """デバッグ用の黒板内容表示"""
        try:
            with self._lock:
                # 期限切れデータのクリーンアップ
                self.clear_expired()
                
                debug_info = {
                    'total_keys': len(self._data),
                    'total_channels': len(self._channels),
                    'total_messages': sum(len(msgs) for msgs in self._channels.values()),
                    'total_subscribers': sum(len(subs) for subs in self._subscribers.values()),
                    'keys': list(self._data.keys()),
                    'channels': list(self._channels.keys()),
                    'data_sample': {},
                    'recent_messages': {}
                }
                
                # データサンプル（最大10件）
                for i, (key, item) in enumerate(self._data.items()):
                    if i >= 10:
                        break
                    value = item['value']
                    # 大きなデータは省略
                    if isinstance(value, (str, dict, list)) and len(str(value)) > 200:
                        debug_info['data_sample'][key] = f"{str(value)[:200]}... (truncated)"
                    else:
                        debug_info['data_sample'][key] = value
                
                # 最近のメッセージ（各チャネルから最大3件）
                for channel, messages in self._channels.items():
                    recent = []
                    for msg in list(messages)[-3:]:
                        recent.append({
                            'id': msg.id,
                            'timestamp': msg.timestamp,
                            'data': msg.data
                        })
                    if recent:
                        debug_info['recent_messages'][channel] = recent
                
                return debug_info
        except Exception as e:
            self.logger.error(f"get_debug_view操作エラー: {e}")
            return {'error': str(e)}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量情報を取得"""
        try:
            import sys
            with self._lock:
                data_size = sys.getsizeof(self._data)
                channels_size = sys.getsizeof(self._channels)
                subscribers_size = sys.getsizeof(self._subscribers)
                
                return {
                    'total_memory_bytes': data_size + channels_size + subscribers_size,
                    'data_memory_bytes': data_size,
                    'channels_memory_bytes': channels_size,
                    'subscribers_memory_bytes': subscribers_size,
                    'total_items': len(self._data),
                    'total_channels': len(self._channels),
                    'total_subscribers': sum(len(subs) for subs in self._subscribers.values())
                }
        except Exception as e:
            self.logger.error(f"get_memory_usage操作エラー: {e}")
            return {'error': str(e)}

class LocalSlotBlackboard(LocalBlackboard):
    """スロット用ローカル黒板"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.slots_data = {}
        self.slot_states = {}
        self.logger.info("ローカルスロット黒板システムを初期化しました")
    
    def update_slot_state(self, slot_name: str, state: str, data: Dict[str, Any] = None):
        """スロット状態更新"""
        try:
            with self._lock:
                self.slot_states[slot_name] = {
                    'state': state,
                    'data': data or {},
                    'timestamp': time.time()
                }
                
                # チャネルに通知
                self.publish(f"slot:{slot_name}", {
                    'type': 'state_update',
                    'slot_name': slot_name,
                    'state': state,
                    'data': data
                })
        except Exception as e:
            self.logger.error(f"update_slot_state操作エラー: {e}")
    
    def get_slot_state(self, slot_name: str) -> Optional[Dict[str, Any]]:
        """スロット状態取得"""
        try:
            with self._lock:
                return self.slot_states.get(slot_name)
        except Exception as e:
            self.logger.error(f"get_slot_state操作エラー: {e}")
            return None
    
    def get_all_slot_states(self) -> Dict[str, Dict[str, Any]]:
        """全スロット状態取得"""
        try:
            with self._lock:
                return self.slot_states.copy()
        except Exception as e:
            self.logger.error(f"get_all_slot_states操作エラー: {e}")
            return {}

    def get_debug_view(self) -> Dict[str, Any]:
        """スロット用デバッグビュー"""
        try:
            # 基底クラスのデバッグビューを取得
            debug_info = super().get_debug_view()
            
            # スロット固有情報を追加
            with self._lock:
                debug_info['slot_states'] = self.slot_states.copy()
                debug_info['total_slots'] = len(self.slot_states)
                debug_info['active_slots'] = len([
                    slot for slot, state in self.slot_states.items() 
                    if state.get('state') == 'active'
                ])
                
                return debug_info
        except Exception as e:
            self.logger.error(f"スロット用get_debug_view操作エラー: {e}")
            return {'error': str(e)}

def create_local_blackboard(config: Dict[str, Any], use_slots: bool = False):
    """ローカル黒板作成"""
    if use_slots:
        return LocalSlotBlackboard(config)
    else:
        return LocalBlackboard(config)
