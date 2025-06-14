#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分散SLMシステムのテストスイート
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
統合されたDistributedSLMクラスの包括的テスト

作者: Yuhi Sonoki
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# プロジェクトのパスを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# テスト対象のモジュールをインポート
from MurmurNet.distributed_slm import DistributedSLM
from MurmurNet.modules.communication_interface import (
    ModuleCommunicationManager,
    MessageType,
    create_communication_system,
    create_message
)
from MurmurNet.modules.module_system_coordinator import ModuleSystemCoordinator
from MurmurNet.modules.module_adapters import create_module_adapters


class MockAgentPool:
    """エージェントプールのモック"""
    
    def __init__(self):
        self.call_count = 0
        self.agents = []
    
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def get_agents(self):
        return self.agents
    
    async def run_agents_async(self, prompt, iteration=0):
        self.call_count += 1
        return [f"エージェント{i}のテスト応答" for i in range(2)]
    
    def run_agents(self, prompt, iteration=0):
        self.call_count += 1
        return [f"エージェント{i}のテスト応答" for i in range(2)]


class MockSummaryEngine:
    """要約エンジンのモック"""
    
    def __init__(self):
        self.call_count = 0
    
    def create_summary(self, entries, iteration=0):
        self.call_count += 1
        return f"テスト要約 - 反復{iteration}"


class TestDistributedSLM:
    """統合されたDistributedSLMのテスト"""
    
    def test_initialization_compatibility_mode(self):
        """互換性モードでの初期化テスト"""
        config = {
            'num_agents': 2,
            'iterations': 1,
            'use_summary': True,
            'use_parallel': False,
            'use_memory': False
        }
        
        slm = DistributedSLM(config, compatibility_mode=True)
        assert slm is not None
        assert slm.compatibility_mode is True
        assert slm.num_agents == 2
        assert slm.iterations == 1
        assert slm.use_summary is True
    
    def test_initialization_new_mode(self):
        """新モードでの初期化テスト"""
        config = {
            'num_agents': 3,
            'iterations': 2,
            'use_summary': False,
            'use_parallel': True,
            'use_memory': True
        }
        
        slm = DistributedSLM(config, compatibility_mode=False)
        assert slm is not None
        assert slm.compatibility_mode is False
        assert slm.num_agents == 3
        assert slm.iterations == 2
        assert slm.use_summary is False
    
    def test_system_status(self):
        """システム状態取得のテスト"""
        slm = DistributedSLM(compatibility_mode=False)
        status = slm.get_system_status()
        
        assert 'compatibility_mode' in status
        assert 'modules_initialized' in status
        assert 'performance_monitoring' in status
        assert 'agent_config' in status
        assert status['compatibility_mode'] is False
    
    def test_communication_stats(self):
        """通信統計取得のテスト"""
        slm = DistributedSLM(compatibility_mode=False)
        stats = slm.get_communication_stats()
        
        assert isinstance(stats, dict)
        # 統計の詳細は通信インターフェースの実装に依存
    
    def test_get_stats(self):
        """統計情報取得のテスト"""
        config = {
            'num_agents': 2,
            'iterations': 1,
            'use_summary': True,
            'use_parallel': False,
            'use_memory': True
        }
        
        slm = DistributedSLM(config, compatibility_mode=True)
        stats = slm.get_stats()
        
        assert stats['agents'] == 2
        assert stats['iterations'] == 1
        assert stats['memory_enabled'] is True
        assert stats['summary_enabled'] is True
        assert stats['parallel_enabled'] is False
        assert stats['compatibility_mode'] is True
    
    def test_reset_memory(self):
        """メモリリセットのテスト"""
        slm = DistributedSLM(compatibility_mode=True)
        
        # メモリリセットが例外を発生させないことを確認
        try:
            slm.reset_memory()
        except Exception as e:
            pytest.fail(f"メモリリセット中に例外が発生: {e}")
    
    def test_cleanup(self):
        """クリーンアップのテスト"""
        slm = DistributedSLM(compatibility_mode=False)
        
        # クリーンアップが例外を発生させないことを確認
        try:
            slm.cleanup()
        except Exception as e:
            pytest.fail(f"クリーンアップ中に例外が発生: {e}")


class TestCommunicationInterface:
    """通信インターフェースのテスト"""
    
    def test_communication_system_creation(self):
        """通信システム作成のテスト"""
        comm_manager = create_communication_system()
        assert comm_manager is not None
        assert isinstance(comm_manager, ModuleCommunicationManager)
    
    def test_message_creation(self):
        """メッセージ作成のテスト"""
        message = create_message(
            MessageType.USER_INPUT,
            "test_sender",
            {"text": "テストメッセージ"}
        )
        
        assert message is not None
        assert message.type == MessageType.USER_INPUT
        assert message.sender == "test_sender"
        assert message.data["text"] == "テストメッセージ"
    
    def test_message_publishing(self):
        """メッセージ配信のテスト"""
        comm_manager = create_communication_system()
        
        message = create_message(
            MessageType.SYSTEM_STATUS,
            "test_system",
            {"status": "running"}
        )
        
        # メッセージ配信が例外を発生させないことを確認
        try:
            comm_manager.publish(message)
        except Exception as e:
            pytest.fail(f"メッセージ配信中に例外が発生: {e}")


class TestModuleSystemCoordinator:
    """モジュールシステム調整器のテスト"""
    
    def test_coordinator_initialization(self):
        """調整器初期化のテスト"""
        coordinator = ModuleSystemCoordinator()
        assert coordinator is not None
    
    def test_execution_stats(self):
        """実行統計のテスト"""
        coordinator = ModuleSystemCoordinator()
        stats = coordinator.get_execution_stats()
        
        assert isinstance(stats, dict)
        assert 'iterations' in stats
        assert 'successful_iterations' in stats
    
    @pytest.mark.asyncio
    async def test_run_iteration(self):
        """反復実行のテスト"""
        coordinator = ModuleSystemCoordinator()
        
        # 反復実行が例外を発生させないことを確認
        try:
            result = await coordinator.run_iteration(0)
            assert isinstance(result, bool)
        except Exception as e:
            pytest.fail(f"反復実行中に例外が発生: {e}")


class TestModuleAdapters:
    """モジュールアダプターのテスト"""
    
    def test_adapter_creation(self):
        """アダプター作成のテスト"""
        mock_agent_pool = MockAgentPool()
        mock_summary_engine = MockSummaryEngine()
        comm_manager = create_communication_system()
        
        adapters = create_module_adapters(
            agent_pool=mock_agent_pool,
            summary_engine=mock_summary_engine,
            comm_manager=comm_manager
        )
        
        assert isinstance(adapters, dict)
        assert 'agent_pool' in adapters
        assert 'summary_engine' in adapters


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
        )
        
        # システム調整器の初期化
        coordinator = ModuleSystemCoordinator(
            blackboard=None,
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
            assert isinstance(success, bool)
        
        # 結果の確認
        stats = coordinator.get_execution_stats()
        assert stats['iterations'] >= num_iterations
    
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
