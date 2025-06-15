#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Communication Interface モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
モジュール間通信の抽象化レイヤー
BLACKBOARDへの直接依存を削減し、疎結合アーキテクチャを実現

設計原則:
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
- 明確なAPI境界
- 責務の分離

作者: Yuhi Sonoki
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('MurmurNet.CommunicationInterface')


class MessageType(Enum):
    """メッセージタイプの定義"""
    INPUT = "input"
    AGENT_OUTPUT = "agent_output"
    SUMMARY = "summary"
    RAG_RESULT = "rag_result"
    CONTEXT = "context"
    ERROR = "error"
    STATUS = "status"
    # Added from communication_interface.py
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
    """モジュール間通信用メッセージ"""
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
    """
    既存のBLACKBOARDシステムを新しい通信インターフェースに適応

    責務:
    - 既存システムとの互換性維持
    - 段階的なマイグレーション支援
    - パフォーマンスの最適化
    """
    
    def __init__(self, blackboard=None):
        """
        通信アダプターの初期化

        引数:
            blackboard: 既存の黒板インスタンス
        """
        self.blackboard = blackboard
        self.subscribers: Dict[MessageType, List] = {}
        self.logger = logging.getLogger('CommunicationAdapter')
        
    def store(self, key: str, value: Any) -> bool:
        """データストレージインターフェースの実装"""
        try:
            if self.blackboard:
                self.blackboard.write(key, value)
                return True
            return False
        except Exception as e:
            self.logger.error(f"ストレージエラー: {e}")
            return False
    
    def retrieve(self, key: str) -> Any:
        """データ取得インターフェースの実装"""
        try:
            if self.blackboard:
                return self.blackboard.read(key)
            return None
        except Exception as e:
            self.logger.error(f"取得エラー: {e}")
            return None
    
    def exists(self, key: str) -> bool:
        """存在確認インターフェースの実装"""
        try:
            if self.blackboard:
                return self.blackboard.read(key) is not None
            return False
        except Exception as e:
            self.logger.error(f"存在確認エラー: {e}")
            return False
    
    def remove(self, key: str) -> bool:
        """削除インターフェースの実装"""
        try:
            if self.blackboard:
                # 黒板の実装に依存するため、空の値を設定
                self.blackboard.write(key, None)
                return True
            return False
        except Exception as e:
            self.logger.error(f"削除エラー: {e}")
            return False
    
    def publish(self, message: Message) -> bool:
        """メッセージ発行インターフェースの実装"""
        try:
            # メッセージを適切なキーで黒板に書き込み
            key = self._generate_key(message)
            self.store(key, message.content)
            
            # 購読者に通知
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
        """メッセージ購読インターフェースの実装"""
        try:
            if message_type not in self.subscribers:
                self.subscribers[message_type] = []
            self.subscribers[message_type].append(callback)
            return True
        except Exception as e:
            self.logger.error(f"購読エラー: {e}")
            return False
    
    def unsubscribe(self, message_type: MessageType, callback) -> bool:
        """購読解除インターフェースの実装"""
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
        """メッセージからキーを生成"""
        return f"{message.sender}_{message.message_type.value}"


class ModuleCommunicationManager:
    """
    モジュール間通信の統合管理

    責務:
    - 通信チャネルの管理
    - メッセージルーティング
    - エラーハンドリング
    - ログ記録
    """

    def __init__(self, storage: DataStorage = None, broker: MessageBroker = None):
        """
        通信マネージャーの初期化

        引数:
            storage: データストレージ実装
            broker: メッセージブローカー実装
        """
        self.storage = storage
        self.broker = broker
        self.logger = logging.getLogger('ModuleCommunicationManager')

        # モジュール登録
        self.registered_modules: Dict[str, Any] = {}
        
    def register_module(self, module_name: str, module_instance: Any) -> bool:
        """
        モジュールを通信システムに登録

        引数:
            module_name: モジュール名
            module_instance: モジュールインスタンス

        戻り値:
            登録成功の可否
        """
        try:
            self.registered_modules[module_name] = module_instance
            self.logger.info(f"モジュール '{module_name}' を登録しました")
            return True
        except Exception as e:
            self.logger.error(f"モジュール登録エラー: {e}")
            return False
    
    def unregister_module(self, module_name: str) -> bool:
        """
        モジュールの登録を解除

        引数:
            module_name: モジュール名

        戻り値:
            解除成功の可否
        """
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
        """
        メッセージを送信

        引数:
            message: 送信するメッセージ

        戻り値:
            送信成功の可否
        """
        try:
            if self.broker:
                return self.broker.publish(message)
            
            # フォールバック: 直接配信
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
        """
        データを取得

        引数:
            key: データキー

        戻り値:
            取得されたデータ
        """
        try:
            if self.storage:
                return self.storage.retrieve(key)
            return None
        except Exception as e:
            self.logger.error(f"データ取得エラー: {e}")
            return None
    
    def set_data(self, key: str, value: Any) -> bool:
        """
        データを設定

        引数:
            key: データキー
            value: 設定する値

        戻り値:
            設定成功の可否
        """
        try:
            if self.storage:
                return self.storage.store(key, value)
            return False
        except Exception as e:
            self.logger.error(f"データ設定エラー: {e}")
            return False


# ファクトリー関数
def create_communication_system(blackboard=None) -> ModuleCommunicationManager:
    """
    通信システムのファクトリー関数

    引数:
        blackboard: 既存の黒板インスタンス（オプション）

    戻り値:
        設定された通信マネージャー
    """
    adapter = CommunicationAdapter(blackboard)
    return ModuleCommunicationManager(storage=adapter, broker=adapter)


# 便利な関数
def create_message(message_type: MessageType, sender: str, content: Any,
                  recipient: str = None, metadata: Dict[str, Any] = None) -> Message:
    """
    メッセージを作成する便利な関数

    引数:
        message_type: メッセージタイプ
        sender: 送信者
        content: メッセージ内容
        recipient: 受信者（オプション）
        metadata: メタデータ（オプション）

    戻り値:
        作成されたメッセージ
    """
    return Message(
        message_type=message_type,
        sender=sender,
        recipient=recipient,
        content=content,
        metadata=metadata
    )
