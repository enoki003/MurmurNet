import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable

class MessageType(Enum):
    INPUT = "input"
    AGENT_OUTPUT = "agent_output"
    SUMMARY = "summary"
    RAG_RESULT = "rag_result"
    CONTEXT = "context"
    ERROR = "error"
    STATUS = "status"
    # テストで使用される追加のメッセージタイプ
    USER_INPUT = "user_input"
    DATA_STORE = "data_store"
    AGENT_RESPONSE = "agent_response"
    AGENT_ERROR = "agent_error"
    RAG_RESULTS = "rag_results"
    SYSTEM_STATUS = "system_status"
    FINAL_RESPONSE = "final_response"
    INITIAL_SUMMARY = "initial_summary"

@dataclass
class Message:
    message_type: MessageType
    sender: str
    recipient: Optional[str] = None
    content: Any = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

@runtime_checkable
class DataStorage(Protocol):
    """データストレージの抽象インターフェース"""
    
    def store(self, key: str, value: Any) -> bool:
        """データを保存"""
        ...
    
    def retrieve(self, key: str) -> Any:
        """データを取得"""
        ...
    
    def exists(self, key: str) -> bool:
        """キーの存在確認"""
        ...
    
    def remove(self, key: str) -> bool:
        """データを削除"""
        ...

@runtime_checkable
class MessageBroker(Protocol):
    """メッセージブローカーの抽象インターフェース"""
    
    def publish(self, message: Message) -> bool:
        """メッセージを発行"""
        ...
    
    def subscribe(self, message_type: MessageType, callback) -> bool:
        """メッセージタイプに対してコールバックを登録"""
        ...
    
    def unsubscribe(self, message_type: MessageType, callback) -> bool:
        """購読を解除"""
        ...

class CommunicationAdapter:
    """BLACKBOARD適応アダプター"""
    
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.subscribers: Dict[MessageType, List] = {}
        self.logger = logging.getLogger('CommunicationAdapter')
        # blackboardがない場合の代替ストレージ
        self._fallback_storage: Dict[str, Any] = {}
        
    def store(self, key: str, value: Any) -> bool:
        try:
            if self.blackboard:
                self.blackboard.write(key, value)
                return True
            else:
                # 代替ストレージを使用
                self._fallback_storage[key] = value
                return True
        except Exception as e:
            self.logger.error(f"ストレージエラー: {e}")
            return False
    
    def retrieve(self, key: str) -> Any:
        try:
            if self.blackboard:
                return self.blackboard.read(key)
            else:
                # 代替ストレージから取得
                return self._fallback_storage.get(key)
        except Exception as e:
            self.logger.error(f"取得エラー: {e}")
            return None
    
    def exists(self, key: str) -> bool:
        try:
            if self.blackboard:
                return self.blackboard.read(key) is not None
            return False
        except Exception as e:
            self.logger.error(f"存在確認エラー: {e}")
            return False
    
    def remove(self, key: str) -> bool:
        try:
            if self.blackboard:
                self.blackboard.write(key, None)
                return True
            return False
        except Exception as e:
            self.logger.error(f"削除エラー: {e}")
            return False
    
    def publish(self, message: Message) -> bool:
        try:
            # DATA_STOREメッセージタイプの場合は特別な処理
            if message.message_type == MessageType.DATA_STORE and isinstance(message.content, dict):
                if 'key' in message.content and 'value' in message.content:
                    self.store(message.content['key'], message.content['value'])
                else:
                    # 通常のキー生成でコンテンツを保存
                    key = self._generate_key(message)
                    self.store(key, message.content)
            else:
                # 通常のメッセージ処理
                key = self._generate_key(message)
                self.store(key, message.content)
            
            if message.message_type in self.subscribers:
                for callback in self.subscribers[message.message_type]:
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(f"コールバックエラー: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"メッセージ発行エラー: {e}")
            return False
    
    def subscribe(self, message_type: MessageType, callback) -> bool:
        try:
            if message_type not in self.subscribers:
                self.subscribers[message_type] = []
            self.subscribers[message_type].append(callback)
            return True
        except Exception as e:
            self.logger.error(f"購読エラー: {e}")
            return False
    
    def unsubscribe(self, message_type: MessageType, callback) -> bool:
        try:
            if message_type in self.subscribers:
                if callback in self.subscribers[message_type]:
                    self.subscribers[message_type].remove(callback)
                    return True
            return False
        except Exception as e:
            self.logger.error(f"購読解除エラー: {e}")
            return False
    
    def _generate_key(self, message: Message) -> str:
        return f"{message.sender}_{message.message_type.value}"
    
    # BlackBoardインターフェース互換性のためのメソッド
    def read(self, key: str) -> Any:
        """BlackBoard互換のreadメソッド"""
        return self.retrieve(key)
    
    def write(self, key: str, value: Any) -> bool:
        """BlackBoard互換のwriteメソッド"""
        return self.store(key, value)

class ModuleCommunicationManager:
    def __init__(self, storage=None, broker=None):
        self.storage = storage
        self.broker = broker
        self.logger = logging.getLogger('ModuleCommunicationManager')
        self.registered_modules = {}
        
    def register_module(self, module_name: str, module_instance: Any) -> bool:
        try:
            self.registered_modules[module_name] = module_instance
            self.logger.info(f"モジュール '{module_name}' を登録しました")
            return True
        except Exception as e:
            self.logger.error(f"モジュール登録エラー: {e}")
            return False
    
    def unregister_module(self, module_name: str) -> bool:
        try:
            if module_name in self.registered_modules:
                del self.registered_modules[module_name]
                self.logger.info(f"モジュール '{module_name}' の登録を解除しました")
                return True
            return False
        except Exception as e:
            self.logger.error(f"モジュール登録解除エラー: {e}")
            return False
    
    def send_message(self, message: Message) -> bool:
        try:
            if self.broker:
                return self.broker.publish(message)
            
            if message.recipient and message.recipient in self.registered_modules:
                recipient_module = self.registered_modules[message.recipient]
                if hasattr(recipient_module, 'receive_message'):
                    recipient_module.receive_message(message)
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"メッセージ送信エラー: {e}")
            return False
    
    def get_data(self, key: str) -> Any:
        try:
            if self.storage:
                return self.storage.retrieve(key)
            return None
        except Exception as e:
            self.logger.error(f"データ取得エラー: {e}")
            return None
    
    def get_all_storage_data(self) -> Dict[str, Any]:
        """ストレージの全データを取得"""
        try:
            if self.storage and hasattr(self.storage, '_fallback_storage'):
                # CommunicationAdapterの代替ストレージから取得
                return self.storage._fallback_storage.copy()
            elif self.storage and hasattr(self.storage, 'blackboard') and self.storage.blackboard:
                # BLACKBOARDから取得する場合
                if hasattr(self.storage.blackboard, 'read_all'):
                    return self.storage.blackboard.read_all()
                else:
                    # フォールバック: 空の辞書を返す
                    return {}
            return {}
        except Exception as e:
            self.logger.error(f"全データ取得エラー: {e}")
            return {}
    
    def set_data(self, key: str, value: Any) -> bool:
        try:
            if self.storage:
                return self.storage.store(key, value)
            return False
        except Exception as e:
            self.logger.error(f"データ設定エラー: {e}")
            return False
    
    def publish(self, message: Message) -> bool:
        """テスト互換性のためのpublishメソッド"""
        return self.send_message(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """通信統計情報を取得"""
        try:
            stats = {
                'registered_modules': len(self.registered_modules),
                'module_names': list(self.registered_modules.keys()),
                'storage_available': self.storage is not None,
                'broker_available': self.broker is not None
            }
            
            # ストレージがある場合は追加の統計情報を取得
            if self.storage and hasattr(self.storage, 'get_stats'):
                stats.update(self.storage.get_stats())
            
            return stats
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {}

def create_communication_system(blackboard=None):
    adapter = CommunicationAdapter(blackboard)
    return ModuleCommunicationManager(storage=adapter, broker=adapter)

def create_message(message_type, sender, content, recipient=None, metadata=None):
    return Message(message_type, sender, recipient, content, metadata)
