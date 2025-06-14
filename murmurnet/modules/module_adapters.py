#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Adapters
~~~~~~~~~~~~~~~
既存モジュールを新しい通信インターフェースに適応させるアダプター
レガシーコードとの互換性を保ちながら段階的移行を実現

作者: Yuhi Sonoki
"""

import logging
from typing import Dict, Any, List, Optional
from MurmurNet.modules.communication_interface import (
    ModuleCommunicationManager, 
    MessageType, 
    create_message
)

logger = logging.getLogger('MurmurNet.ModuleAdapters')


class AgentPoolAdapter:
    """
    既存のAgentPoolManagerを新しい通信インターフェースに適応させるアダプター
    """
    def __init__(self, agent_pool, comm_manager: ModuleCommunicationManager):
        """
        アダプターの初期化
        
        引数:
            agent_pool: 既存のAgentPoolManagerインスタンス
            comm_manager: 通信管理器
        """
        self.agent_pool = agent_pool
        self.comm_manager = comm_manager
        
    async def run_agent_async(self, agent_id: int, prompt: str) -> str:
        """
        エージェントを非同期実行
        
        引数:
            agent_id: エージェントID
            prompt: プロンプト
        
        戻り値:
            エージェントの応答
        """
        try:            # 既存のAgentPoolManagerの非同期メソッドを使用
            if hasattr(self.agent_pool, 'run_agent_parallel'):
                result = await self.agent_pool.run_agent_parallel(agent_id)
            else:
                # 非同期メソッドがない場合は既存の内部メソッドを使用
                result = self.agent_pool._agent_task(agent_id)
            
            # 結果を通信システムに送信
            if result:
                message = create_message(
                    MessageType.AGENT_RESPONSE,
                    f"agent_{agent_id}",
                    {
                        'agent_id': agent_id,
                        'response': result
                    }
                )
                self.comm_manager.publish(message)
            
            return result
            
        except Exception as e:
            logger.error(f"エージェント{agent_id}実行エラー: {e}")
            # エラーメッセージを通信システムに送信
            message = create_message(
                MessageType.AGENT_ERROR,
                f"agent_{agent_id}",
                {
                    'agent_id': agent_id,
                    'error': str(e)
                }
            )
            self.comm_manager.publish(message)
            return None
    
    def run_agent_sync(self, agent_id: int, prompt: str) -> str:
        """
        エージェントを同期実行
        
        引数:
            agent_id: エージェントID
            prompt: プロンプト
              戻り値:
            エージェントの応答
        """
        try:            # 既存のAgentPoolManagerの内部メソッドを使用
            result = self.agent_pool._agent_task(agent_id)
            
            # 結果を通信システムに送信
            if result:
                message = create_message(
                    MessageType.AGENT_RESPONSE,
                    f"agent_{agent_id}",
                    {
                        'agent_id': agent_id,
                        'response': result
                    }
                )
                self.comm_manager.publish(message)
            
            return result
            
        except Exception as e:
            logger.error(f"エージェント{agent_id}同期実行エラー: {e}")
            # エラーメッセージを通信システムに送信
            message = create_message(
                MessageType.AGENT_ERROR,
                f"agent_{agent_id}",
                {
                    'agent_id': agent_id,
                    'error': str(e)
                }
            )
            
            self.comm_manager.publish(message)
            return None


class SummaryEngineAdapter:
    """
    既存のSummaryEngineを新しい通信インターフェースに適応させるアダプター
    """
    
    def __init__(self, summary_engine, comm_manager: ModuleCommunicationManager):
        """
        アダプターの初期化
        
        引数:
            summary_engine: 既存のSummaryEngineインスタンス
            comm_manager: 通信管理器
        """
        self.summary_engine = summary_engine
        self.comm_manager = comm_manager
        
    def summarize_blackboard(self, agent_entries: List[Dict[str, Any]]) -> str:
        """
        エージェント出力を要約
        
        引数:
            agent_entries: エージェント出力のリスト
              戻り値:
            要約文字列
        """
        try:
            # 既存のSummaryEngineを使用
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            
            # 要約結果を通信システムに送信
            message = create_message(
                MessageType.SUMMARY,
                "summary_engine",
                {
                    'summary': summary,
                    'agent_count': len(agent_entries)
                }
            )
            self.comm_manager.publish(message)
            
            return summary
            
        except Exception as e:
            logger.error(f"要約作成エラー: {e}")
            # エラーメッセージを通信システムに送信
            message = create_message(
                MessageType.ERROR,
                "summary_engine",
                {
                    'error': f"要約作成中にエラーが発生しました: {e}"
                }
            )
            self.comm_manager.publish(message)
            return "要約の作成中にエラーが発生しました"


class BlackboardBridgeAdapter:
    """
    BLACKBOARDと新しい通信システム間のブリッジアダプター
    既存のBLACKBOARDデータを新しい通信システムに同期
    """
    
    def __init__(self, blackboard, comm_manager: ModuleCommunicationManager):
        """
        ブリッジアダプターの初期化
        
        引数:
            blackboard: 既存のBlackboardインスタンス
            comm_manager: 通信管理器
        """
        self.blackboard = blackboard
        self.comm_manager = comm_manager
        self.is_syncing = False
        
    def sync_to_communication_system(self) -> None:
        """
        BLACKBOARDのデータを通信システムに同期
        """
        if self.is_syncing:
            return  # 無限ループを防ぐ
        
        try:
            self.is_syncing = True
            
            # BLACKBOARDの全データを取得
            blackboard_data = self.blackboard.read_all()
              # 各データを通信システムに送信
            for key, value in blackboard_data.items():
                message = create_message(
                    MessageType.DATA_STORE,
                    "blackboard_bridge",
                    {
                        'key': key,
                        'value': value
                    }
                )
                self.comm_manager.publish(message)
            
            logger.debug(f"BLACKBOARDから通信システムに{len(blackboard_data)}件のデータを同期しました")
            
        except Exception as e:
            logger.error(f"BLACKBOARD同期エラー: {e}")
        finally:
            self.is_syncing = False
    
    def sync_from_communication_system(self) -> None:
        """
        通信システムのデータをBLACKBOARDに同期
        """
        if self.is_syncing:
            return  # 無限ループを防ぐ
        
        try:
            self.is_syncing = True
              # 通信システムから全データを取得
            communication_data = self.comm_manager.get_all_storage_data()
            
            # 各データをBLACKBOARDに書き込み
            for key, value in communication_data.items():
                self.blackboard.write(key, value)
            
            logger.debug(f"通信システムからBLACKBOARDに{len(communication_data)}件のデータを同期しました")
            
        except Exception as e:
            logger.error(f"通信システム同期エラー: {e}")
        finally:
            self.is_syncing = False
    
    def bidirectional_sync(self) -> None:
        """
        双方向でデータを同期
        """
        self.sync_to_communication_system()
        self.sync_from_communication_system()


class ConversationMemoryAdapter:
    """
    既存のConversationMemoryを新しい通信インターフェースに適応させるアダプター
    """
    
    def __init__(self, conversation_memory, comm_manager: ModuleCommunicationManager):
        """
        アダプターの初期化
          引数:
            conversation_memory: 既存のConversationMemoryインスタンス
            comm_manager: 通信管理器
        """
        self.conversation_memory = conversation_memory
        self.comm_manager = comm_manager
    
    def update_context(self, user_input: str, agent_responses: List[str]) -> None:
        """
        会話コンテキストを更新
        
        引数:
            user_input: ユーザー入力
            agent_responses: エージェント応答のリスト
        """
        try:
            # 既存のメソッドを使用してコンテキスト更新
            if hasattr(self.conversation_memory, 'update_context'):
                self.conversation_memory.update_context(user_input, agent_responses)
            
            # 更新されたコンテキストを通信システムに送信
            context = self.get_context()
            if context:
                message = create_message(
                    MessageType.DATA_STORE,
                    "conversation_memory",
                    {
                        'key': 'conversation_context',
                        'value': context
                    }
                )
                self.comm_manager.publish(message)
                
        except Exception as e:
            logger.error(f"会話コンテキスト更新エラー: {e}")
    
    def get_context(self) -> str:
        """
        現在の会話コンテキストを取得
        
        戻り値:
            会話コンテキスト文字列
        """
        try:
            if hasattr(self.conversation_memory, 'get_context'):
                return self.conversation_memory.get_context()
            else:
                # フォールバック: 通信システムから取得
                return self.comm_manager.get_data('conversation_context') or ""
        except Exception as e:
            logger.error(f"会話コンテキスト取得エラー: {e}")
            return ""


def create_module_adapters(
    blackboard=None, 
    agent_pool=None, 
    summary_engine=None, 
    conversation_memory=None,
    comm_manager: ModuleCommunicationManager = None
) -> Dict[str, Any]:
    """
    モジュールアダプターを一括作成するファクトリ関数
    
    引数:
        blackboard: Blackboardインスタンス
        agent_pool: AgentPoolManagerインスタンス
        summary_engine: SummaryEngineインスタンス
        conversation_memory: ConversationMemoryインスタンス
        comm_manager: 通信管理器
        
    戻り値:
        アダプター辞書
    """
    if not comm_manager:
        from MurmurNet.modules.communication_interface import create_communication_system
        comm_manager = create_communication_system()
    
    adapters = {}
    
    if blackboard:
        adapters['blackboard_bridge'] = BlackboardBridgeAdapter(blackboard, comm_manager)
    
    if agent_pool:
        adapters['agent_pool'] = AgentPoolAdapter(agent_pool, comm_manager)
    
    if summary_engine:
        adapters['summary_engine'] = SummaryEngineAdapter(summary_engine, comm_manager)
    
    if conversation_memory:
        adapters['conversation_memory'] = ConversationMemoryAdapter(conversation_memory, comm_manager)
    
    logger.info(f"モジュールアダプターを作成しました: {list(adapters.keys())}")
    return adapters
