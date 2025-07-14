#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
インターフェース定義
~~~~~~~~~~~~~~~~~
MurmurNet分散システムの抽象インターフェース

作者: Yuhi Sonoki
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    """メッセージタイプ"""
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESPONSE = "response"

@dataclass
class Message:
    """メッセージオブジェクト"""
    id: str
    type: MessageType
    source: str
    target: Optional[str]
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None

class IMessageHandler(ABC):
    """メッセージハンドラーインターフェース"""
    
    @abstractmethod
    async def handle_message(self, message: Message) -> Optional[Message]:
        """メッセージを処理"""
        pass
    
    @abstractmethod
    def get_supported_message_types(self) -> List[MessageType]:
        """サポートするメッセージタイプを取得"""
        pass

class IBlackboard(ABC):
    """ブラックボードインターフェース"""
    
    @abstractmethod
    async def write(self, key: str, value: Any) -> bool:
        """データを書き込み"""
        pass
    
    @abstractmethod
    async def read(self, key: str) -> Optional[Any]:
        """データを読み取り"""
        pass
    
    @abstractmethod
    async def subscribe(self, pattern: str, callback: Callable) -> str:
        """パターンにマッチするキーの変更を購読"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """購読を解除"""
        pass

class IAgent(ABC):
    """エージェントインターフェース"""
    
    @abstractmethod
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """入力を処理"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """エージェント開始"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """エージェント停止"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """エージェントの能力を取得"""
        pass

class ICoordinator(ABC):
    """協調システムインターフェース"""
    
    @abstractmethod
    async def register_node(self, node_id: str, capabilities: List[str]) -> bool:
        """ノードを登録"""
        pass
    
    @abstractmethod
    async def unregister_node(self, node_id: str) -> bool:
        """ノードの登録解除"""
        pass
    
    @abstractmethod
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """タスクを投入"""
        pass
    
    @abstractmethod
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """タスク結果を取得"""
        pass

class ILoadBalancer(ABC):
    """負荷分散器インターフェース"""
    
    @abstractmethod
    async def select_node(self, task: Dict[str, Any], available_nodes: List[str]) -> Optional[str]:
        """タスクに適したノードを選択"""
        pass
    
    @abstractmethod
    async def update_node_load(self, node_id: str, load: float) -> bool:
        """ノードの負荷を更新"""
        pass
    
    @abstractmethod
    def get_load_metrics(self) -> Dict[str, float]:
        """負荷メトリクスを取得"""
        pass
