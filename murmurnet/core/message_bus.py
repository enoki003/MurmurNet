#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
メッセージバスシステム
~~~~~~~~~~~~~~~~~~
疎結合通信のためのメッセージベースアーキテクチャ

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Callable, Set
from collections import defaultdict
from dataclasses import dataclass, field

from .interfaces import Message, MessageType, IMessageHandler

logger = logging.getLogger(__name__)

@dataclass
class EventHandler:
    """イベントハンドラー情報"""
    handler_id: str
    handler: IMessageHandler
    message_types: Set[MessageType] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)

class MessageBus:
    """
    メッセージバス
    
    コンポーネント間の疎結合通信を提供
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ハンドラー管理
        self._handlers: Dict[str, EventHandler] = {}
        self._type_handlers: Dict[MessageType, List[str]] = defaultdict(list)
        self._topic_handlers: Dict[str, List[str]] = defaultdict(list)
        
        # メッセージキュー
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        # メトリクス
        self._metrics = {
            'messages_sent': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'handlers_registered': 0
        }
        
        self.logger.info("メッセージバス初期化完了")

    async def start(self):
        """メッセージバス開始"""
        if self._running:
            return
            
        self._running = True
        self.logger.info("メッセージバス開始")
        
        # メッセージ処理ループを開始
        asyncio.create_task(self._message_processing_loop())

    async def stop(self):
        """メッセージバス停止"""
        self._running = False
        self.logger.info("メッセージバス停止")

    def register_handler(self, 
                        handler: IMessageHandler,
                        message_types: Optional[List[MessageType]] = None,
                        topics: Optional[List[str]] = None) -> str:
        """
        メッセージハンドラーを登録
        
        Args:
            handler: メッセージハンドラー
            message_types: 処理するメッセージタイプ
            topics: 購読するトピック
            
        Returns:
            ハンドラーID
        """
        handler_id = str(uuid.uuid4())
        
        # サポートするメッセージタイプを取得
        supported_types = set(handler.get_supported_message_types())
        if message_types:
            supported_types.update(message_types)
        
        # ハンドラー情報を作成
        event_handler = EventHandler(
            handler_id=handler_id,
            handler=handler,
            message_types=supported_types,
            topics=set(topics or [])
        )
        
        # 登録
        self._handlers[handler_id] = event_handler
        
        # タイプ別インデックスに追加
        for msg_type in supported_types:
            self._type_handlers[msg_type].append(handler_id)
        
        # トピック別インデックスに追加
        for topic in event_handler.topics:
            self._topic_handlers[topic].append(handler_id)
        
        self._metrics['handlers_registered'] += 1
        self.logger.info(f"ハンドラー登録: {handler_id}, タイプ: {supported_types}, トピック: {event_handler.topics}")
        
        return handler_id

    def unregister_handler(self, handler_id: str) -> bool:
        """
        ハンドラーの登録解除
        
        Args:
            handler_id: ハンドラーID
            
        Returns:
            成功時True
        """
        if handler_id not in self._handlers:
            return False
        
        event_handler = self._handlers[handler_id]
        
        # タイプ別インデックスから削除
        for msg_type in event_handler.message_types:
            if handler_id in self._type_handlers[msg_type]:
                self._type_handlers[msg_type].remove(handler_id)
        
        # トピック別インデックスから削除
        for topic in event_handler.topics:
            if handler_id in self._topic_handlers[topic]:
                self._topic_handlers[topic].remove(handler_id)
        
        # ハンドラーを削除
        del self._handlers[handler_id]
        
        self.logger.info(f"ハンドラー登録解除: {handler_id}")
        return True

    async def send_message(self, message: Message) -> bool:
        """
        メッセージを送信
        
        Args:
            message: 送信するメッセージ
            
        Returns:
            成功時True
        """
        try:
            await self._message_queue.put(message)
            self._metrics['messages_sent'] += 1
            self.logger.debug(f"メッセージ送信: {message.id} ({message.type})")
            return True
        except Exception as e:
            self.logger.error(f"メッセージ送信エラー: {e}")
            return False

    async def send_command(self, target: str, command: str, payload: Dict = None) -> str:
        """
        コマンドメッセージを送信
        
        Args:
            target: 送信先
            command: コマンド名
            payload: ペイロード
            
        Returns:
            メッセージID
        """
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.COMMAND,
            source="system",
            target=target,
            payload={
                'command': command,
                'data': payload or {}
            },
            timestamp=time.time()
        )
        
        await self.send_message(message)
        return message.id

    async def send_event(self, event: str, payload: Dict = None, topic: Optional[str] = None) -> str:
        """
        イベントメッセージを送信
        
        Args:
            event: イベント名
            payload: ペイロード
            topic: トピック
            
        Returns:
            メッセージID
        """
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            source="system",
            target=topic,
            payload={
                'event': event,
                'data': payload or {}
            },
            timestamp=time.time()
        )
        
        await self.send_message(message)
        return message.id

    async def send_query(self, target: str, query: str, payload: Dict = None) -> str:
        """
        クエリメッセージを送信
        
        Args:
            target: 送信先
            query: クエリ名
            payload: ペイロード
            
        Returns:
            メッセージID
        """
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.QUERY,
            source="system",
            target=target,
            payload={
                'query': query,
                'data': payload or {}
            },
            timestamp=time.time()
        )
        
        await self.send_message(message)
        return message.id

    async def _message_processing_loop(self):
        """メッセージ処理ループ"""
        while self._running:
            try:
                # タイムアウト付きでメッセージを取得
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # タイムアウトは正常（ループ継続）
                continue
            except Exception as e:
                self.logger.error(f"メッセージ処理ループエラー: {e}")

    async def _process_message(self, message: Message):
        """メッセージを処理"""
        try:
            # 適切なハンドラーを見つける
            handler_ids = self._find_handlers(message)
            
            if not handler_ids:
                self.logger.warning(f"ハンドラーが見つかりません: {message.type}, target: {message.target}")
                return
            
            # 各ハンドラーで処理
            for handler_id in handler_ids:
                if handler_id in self._handlers:
                    handler = self._handlers[handler_id].handler
                    try:
                        response = await handler.handle_message(message)
                        if response:
                            # レスポンスがある場合は送信
                            await self.send_message(response)
                    except Exception as e:
                        self.logger.error(f"ハンドラー処理エラー ({handler_id}): {e}")
                        self._metrics['messages_failed'] += 1
            
            self._metrics['messages_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"メッセージ処理エラー: {e}")
            self._metrics['messages_failed'] += 1

    def _find_handlers(self, message: Message) -> List[str]:
        """メッセージに適したハンドラーを検索"""
        handler_ids = set()
        
        # メッセージタイプで検索
        if message.type in self._type_handlers:
            handler_ids.update(self._type_handlers[message.type])
        
        # ターゲット（トピック）で検索
        if message.target and message.target in self._topic_handlers:
            handler_ids.update(self._topic_handlers[message.target])
        
        return list(handler_ids)

    def get_metrics(self) -> Dict[str, int]:
        """メトリクスを取得"""
        return self._metrics.copy()

    def get_status(self) -> Dict[str, any]:
        """ステータスを取得"""
        return {
            'running': self._running,
            'handlers_count': len(self._handlers),
            'queue_size': self._message_queue.qsize(),
            'metrics': self.get_metrics()
        }
