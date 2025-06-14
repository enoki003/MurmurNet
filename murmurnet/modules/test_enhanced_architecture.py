#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration Tests for Enhanced MurmurNet Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
新しい通信インターフェースとモジュラー設計のテスト

作者: Yuhi Sonoki
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# テスト対象のモジュール
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MurmurNet.modules.communication_interface import (
    ModuleCommunicationManager,
    MessageType,
    create_communication_system,
    create_message
)
from MurmurNet.modules.module_system_coordinator import (
    ModuleSystemCoordinator
)
from MurmurNet.modules.module_adapters import (
    AgentPoolAdapter,
    SummaryEngineAdapter,
    BlackboardBridgeAdapter,
    create_module_adapters
)
from MurmurNet.distributed_slm import DistributedSLM

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestCommunicationInterface:
    """通信インターフェースのテスト"""
    def test_create_communication_system(self):
        """通信システム作成のテスト"""
        comm_manager = create_communication_system()
        assert comm_manager is not None
        assert hasattr(comm_manager, 'publish')
        assert hasattr(comm_manager, 'get_data')
    
    def test_message_creation(self):
        """メッセージ作成のテスト"""
        message = create_message(MessageType.USER_INPUT, "test_sender", {'text': 'テストメッセージ'})
        assert message.message_type == MessageType.USER_INPUT
        assert message.content['text'] == 'テストメッセージ'
        assert message.timestamp > 0
    
    def test_data_storage_and_retrieval(self):
        """データの格納と取得のテスト"""
        comm_manager = create_communication_system()
        
        # データ格納
        message = create_message(MessageType.DATA_STORE, "test_client", {
            'key': 'test_key',
            'value': 'test_value'
        })
        comm_manager.publish(message)
        
        # データ取得
        retrieved_value = comm_manager.get_data('test_key')
        assert retrieved_value == 'test_value'
    
    def test_multiple_message_types(self):
        """複数のメッセージタイプのテスト"""
        comm_manager = create_communication_system()
        
        # 異なるタイプのメッセージを送信
        messages = [
            create_message(MessageType.USER_INPUT, "user", {'text': 'ユーザー入力'}),
            create_message(MessageType.AGENT_RESPONSE, "agent_0", {'agent_id': 0, 'response': 'エージェント応答'}),
            create_message(MessageType.RAG_RESULTS, "rag_system", {'results': ['結果1', '結果2']}),
            create_message(MessageType.SUMMARY, "summary_engine", {'summary': '要約文'})
        ]
        
        for message in messages:
            result = comm_manager.publish(message)
            assert result is True or result is False  # 何らかの結果が返ることを確認


class MockAgentPool:
    """テスト用のモックエージェントプール"""
    
    def __init__(self):
        self.call_count = 0
    
    async def run_agent_async(self, agent_id: int, prompt: str) -> str:
        self.call_count += 1
        await asyncio.sleep(0.01)  # 非同期処理をシミュレート
        return f"エージェント{agent_id}の応答: {prompt[:20]}..."
    
    def run_agent_sync(self, agent_id: int, prompt: str) -> str:
        self.call_count += 1
        return f"エージェント{agent_id}の応答: {prompt[:20]}..."


class MockSummaryEngine:
    """テスト用のモック要約エンジン"""
    
    def __init__(self):
        self.call_count = 0
    
    def summarize_blackboard(self, agent_entries: List[Dict[str, Any]]) -> str:
        self.call_count += 1
        return f"要約: {len(agent_entries)}件のエージェント出力を要約"


class TestModuleAdapters:
    """モジュールアダプターのテスト"""
    
    def test_agent_pool_adapter(self):
        """エージェントプールアダプターのテスト"""
        mock_agent_pool = MockAgentPool()
        comm_manager = create_communication_system()
        adapter = AgentPoolAdapter(mock_agent_pool, comm_manager)
        
        # 同期実行テスト
        result = adapter.run_agent_sync(0, "テストプロンプト")
        assert result is not None
        assert "エージェント0" in result
        assert mock_agent_pool.call_count == 1
    @pytest.mark.asyncio
    async def test_agent_pool_adapter_async(self):
        """エージェントプールアダプターの非同期テスト"""
        mock_agent_pool = MockAgentPool()
        comm_manager = create_communication_system()
        adapter = AgentPoolAdapter(mock_agent_pool, comm_manager)
          # 非同期実行テスト
        result = await adapter.run_agent_async(0, "テストプロンプト")
        assert result is not None
        assert "エージェント0" in result
        assert mock_agent_pool.call_count == 1
    
    def test_summary_engine_adapter(self):
        """要約エンジンアダプターのテスト"""
        mock_summary_engine = MockSummaryEngine()
        comm_manager = create_communication_system()
        adapter = SummaryEngineAdapter(mock_summary_engine, comm_manager)
        
        # 要約テスト
        agent_entries = [
            {'agent_id': 0, 'output': '応答1'},
            {'agent_id': 1, 'output': '応答2'}
        ]
        summary = adapter.summarize_blackboard(agent_entries)
        assert summary is not None
        assert "要約" in summary
        assert mock_summary_engine.call_count == 1
    
    def test_create_module_adapters(self):
        """アダプター一括作成のテスト"""
        mock_agent_pool = MockAgentPool()
        mock_summary_engine = MockSummaryEngine()
        comm_manager = create_communication_system()
        
        adapters = create_module_adapters(
            agent_pool=mock_agent_pool,
            summary_engine=mock_summary_engine,
            comm_manager=comm_manager
        )
        
        assert 'agent_pool' in adapters
        assert 'summary_engine' in adapters
        assert isinstance(adapters['agent_pool'], AgentPoolAdapter)
        assert isinstance(adapters['summary_engine'], SummaryEngineAdapter)


class TestModuleSystemCoordinator:
    """モジュールシステム調整器のテスト"""
    
    def test_coordinator_initialization(self):
        """システム調整器の初期化テスト"""
        comm_manager = create_communication_system()
        coordinator = ModuleSystemCoordinator(blackboard=comm_manager.storage)
        
        # 基本的なプロパティを確認
        assert coordinator.num_agents > 0
        assert coordinator.iterations > 0

    @pytest.mark.asyncio
    async def test_single_iteration(self):
        """単一反復のテスト"""
        mock_agent_pool = MockAgentPool()
        mock_summary_engine = MockSummaryEngine()
        comm_manager = create_communication_system()
        
        # アダプターを作成
        agent_adapter = AgentPoolAdapter(mock_agent_pool, comm_manager)
        summary_adapter = SummaryEngineAdapter(mock_summary_engine, comm_manager)
        
        # システム調整器を初期化
        coordinator = ModuleSystemCoordinator(
            blackboard=comm_manager.storage,
            agent_pool=agent_adapter,
            summary_engine=summary_adapter
        )
        
        # テストデータを設定
        comm_manager.publish(create_message(MessageType.DATA_STORE, "test_client", {
            'key': 'user_input',
            'value': 'テスト質問'
        }))
          # 反復実行
        success = await coordinator.run_iteration(0)
        assert success
        
        # 統計情報を確認
        stats = coordinator.get_execution_stats()
        assert stats['num_agents'] == 2  # デフォルトの値を確認
        assert stats['iterations'] >= 1  # 反復回数を確認
        assert stats['parallel_mode'] == True  # デフォルトは並列実行
    def test_prompt_building(self):
        """プロンプト構築のテスト"""
        comm_manager = create_communication_system()
        coordinator = ModuleSystemCoordinator(blackboard=comm_manager.storage)
        
        # テストデータを設定
        test_data = {
            'user_input': 'テストユーザー入力',
            'conversation_context': 'テスト会話コンテキスト',
            'search_results': 'テスト検索結果'        }
        for key, value in test_data.items():
            comm_manager.publish(create_message(MessageType.DATA_STORE, "test_client", {
                'key': key,
                'value': value
            }))
        
        # プロンプト構築
        prompt = coordinator._build_common_prompt()
        
        assert 'テストユーザー入力' in prompt
        assert 'テスト会話コンテキスト' in prompt
        assert 'テスト検索結果' in prompt


class TestDistributedSLM:
    """改良版分散SLMのテスト"""
    
    @patch('MurmurNet.modules.agent_pool.AgentPoolManager')
    @patch('MurmurNet.modules.summary_engine.SummaryEngine')
    @patch('MurmurNet.modules.blackboard.Blackboard')
    def test_initialization_compatibility_mode(self, mock_blackboard, mock_summary, mock_agent_pool):
        """互換モードでの初期化テスト"""
        # モックの設定
        mock_blackboard.return_value = Mock()
        mock_agent_pool.return_value = MockAgentPool()
        mock_summary.return_value = MockSummaryEngine()
        
        # 改良版SLMの初期化
        slm = DistributedSLM(compatibility_mode=True)
        assert slm is not None
        assert slm.compatibility_mode
    
    def test_initialization_new_mode(self):
        """新モードでの初期化テスト"""
        slm = DistributedSLM(compatibility_mode=False)
        assert slm is not None
        assert not slm.compatibility_mode
    
    def test_system_status(self):
        """システム状態取得のテスト"""
        slm = DistributedSLM(compatibility_mode=False)
        status = slm.get_system_status()
        
        assert 'compatibility_mode' in status
        assert 'modules_initialized' in status
        assert 'performance_monitoring' in status
        assert 'agent_config' in status
    
    def test_communication_stats(self):
        """通信統計取得のテスト"""
        slm = DistributedSLM(compatibility_mode=False)
        stats = slm.get_communication_stats()
        
        assert isinstance(stats, dict)
        # 統計の詳細は通信インターフェースの実装に依存


class TestIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """エンドツーエンドワークフローのテスト"""
        # モックコンポーネントの準備
        mock_agent_pool = MockAgentPool()
        mock_summary_engine = MockSummaryEngine()
        
        # 通信システムの初期化
        comm_manager = create_communication_system()
        
        # アダプターの作成
        adapters = create_module_adapters(
            agent_pool=mock_agent_pool,
            summary_engine=mock_summary_engine,
            comm_manager=comm_manager
        )        # システム調整器の初期化
        coordinator = ModuleSystemCoordinator(
            blackboard=comm_manager.storage,
            agent_pool=adapters['agent_pool'],
            summary_engine=adapters['summary_engine']
        )
        
        # 入力データの設定
        comm_manager.publish(create_message(MessageType.USER_INPUT, "user", {
            'text': 'テスト質問: 機械学習について教えてください'
        }))
        
        # 反復処理の実行
        num_iterations = 2
        for i in range(num_iterations):
            success = await coordinator.run_iteration(i)
            assert success
          # 結果の確認
        stats = coordinator.get_execution_stats()
        assert stats['iterations'] == num_iterations
        assert stats['successful_iterations'] == num_iterations
        
        # エージェントプールとサマリーエンジンが呼び出されたことを確認
        assert mock_agent_pool.call_count > 0
        assert mock_summary_engine.call_count > 0
    def test_performance_monitoring(self):
        """パフォーマンス監視のテスト"""
        slm = DistributedSLM(compatibility_mode=False)
        
        # パフォーマンス監視が有効であることを確認
        assert hasattr(slm, 'performance')
        assert slm.performance is not None
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        comm_manager = create_communication_system()
          # 不正なメッセージを送信
        try:
            invalid_message = create_message(MessageType.ERROR, "error_source", {
                'error': 'テストエラー'
            })
            comm_manager.publish(invalid_message)
            # エラーが正常に処理されることを確認
        except Exception as e:
            pytest.fail(f"エラーメッセージの処理中に例外が発生: {e}")


def run_tests():
    """テストを実行"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
