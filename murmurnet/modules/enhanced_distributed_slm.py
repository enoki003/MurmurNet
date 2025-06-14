#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Distributed SLM System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
新しい通信インターフェースを使用する改良版DistributedSLM
モジュラー設計と疎結合アーキテクチャを実現

作者: Yuhi Sonoki
"""

import os
import yaml
import logging
import asyncio
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union
from MurmurNet.modules.common import setup_logger, PerformanceError, MurmurNetError, ConfigurationError
from MurmurNet.modules.performance import PerformanceMonitor, time_function, time_async_function
from MurmurNet.modules.input_reception import InputReception
from MurmurNet.modules.communication_interface import (
    ModuleCommunicationManager,
    MessageType,
    create_communication_system,
    create_message
)
from MurmurNet.modules.blackboard import Blackboard
from MurmurNet.modules.module_system_coordinator import ModuleSystemCoordinator
from MurmurNet.modules.module_adapters import create_module_adapters, BlackboardBridgeAdapter
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.EnhancedDistributedSLM')


class EnhancedDistributedSLM:
    """
    改良版分散SLMシステム
    
    改善点:
    - 新しい通信インターフェースの採用
    - モジュラー設計による疎結合
    - アダプターパターンによる既存システムとの互換性
    - 明確なAPI境界
    - 依存性注入による柔軟性
    """
    
    def __init__(
        self, 
        config: Dict[str, Any] = None, 
        comm_manager: ModuleCommunicationManager = None,
        compatibility_mode: bool = True
    ):
        """
        改良版分散SLMシステムの初期化
        
        引数:
            config: 設定辞書（オプション）
            comm_manager: 通信管理器（オプション、未指定時は自動作成）
            compatibility_mode: 既存システムとの互換モード
        """
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()
        
        # ロガー設定
        self.logger = setup_logger(self.config)
        
        # パフォーマンスモニタリング
        self.performance = PerformanceMonitor(
            enabled=self.config_manager.logging.performance_monitoring,
            memory_tracking=self.config_manager.logging.memory_tracking
        )
        
        # 初期メモリスナップショット
        self.performance.take_memory_snapshot("enhanced_slm_init_start")
        
        # 通信システムの初期化
        self.comm_manager = comm_manager or create_communication_system()
        
        # 互換性モードの設定
        self.compatibility_mode = compatibility_mode
        
        # 基本動作パラメータ
        self.num_agents = self.config_manager.agent.num_agents
        self.iterations = self.config_manager.agent.iterations
        self.use_summary = self.config_manager.agent.use_summary
        self.use_parallel = self.config_manager.agent.use_parallel
        self.use_memory = self.config_manager.agent.use_memory
        
        # 遅延初期化フラグ
        self._modules_initialized = False
        self._initialization_lock = threading.Lock()
        
        # 基本モジュールの初期化
        self.input_reception = InputReception(self.config)
        
        # レガシーシステム（互換性モード用）
        self._legacy_blackboard = None
        self._bridge_adapter = None
        
        # 新システムのモジュール
        self._system_coordinator = None
        self._module_adapters = {}
          # パフォーマンスステータスを通信システムに記録
        message = create_message(
            MessageType.SYSTEM_STATUS,
            "enhanced_slm",
            {
                'performance_enabled': self.performance.enabled,
                'compatibility_mode': self.compatibility_mode
            }
        )
        self.comm_manager.publish(message)
        
        # 初期化完了時のメモリスナップショット
        self.performance.take_memory_snapshot("enhanced_slm_init_complete")
        
        self.logger.info(f"改良版分散SLMシステムを初期化しました (互換モード: {self.compatibility_mode})")

    def _ensure_modules_initialized(self) -> None:
        """
        モジュールの遅延初期化を確実に実行
        """
        if self._modules_initialized:
            return
        
        with self._initialization_lock:
            if self._modules_initialized:
                return
                
            try:
                self.performance.take_memory_snapshot("module_initialization_start")
                
                # レガシーシステムの初期化（互換モード時）
                if self.compatibility_mode:
                    self._initialize_legacy_system()
                
                # 新システムの初期化
                self._initialize_new_system()
                
                self._modules_initialized = True
                self.performance.take_memory_snapshot("module_initialization_complete")
                
                self.logger.info("全モジュールの初期化が完了しました")
                
            except Exception as e:
                self.logger.error(f"モジュール初期化エラー: {e}")
                raise ConfigurationError(f"モジュール初期化に失敗しました: {e}")

    def _initialize_legacy_system(self) -> None:
        """
        レガシーシステム（BLACKBOARD等）の初期化
        """
        try:
            # 既存のモジュールをインポート（遅延インポート）
            from MurmurNet.modules.blackboard import Blackboard
            from MurmurNet.modules.agent_pool import AgentPoolManager
            from MurmurNet.modules.summary_engine import SummaryEngine
            from MurmurNet.modules.conversation_memory import ConversationMemory
            
            # レガシーモジュールの初期化
            self._legacy_blackboard = Blackboard(self.config)
            legacy_agent_pool = AgentPoolManager(self.config, self._legacy_blackboard)
            legacy_summary_engine = SummaryEngine(self.config, self._legacy_blackboard)
            legacy_conversation_memory = ConversationMemory(self.config, self._legacy_blackboard)
            
            # アダプターを作成
            self._module_adapters = create_module_adapters(
                blackboard=self._legacy_blackboard,
                agent_pool=legacy_agent_pool,
                summary_engine=legacy_summary_engine,
                conversation_memory=legacy_conversation_memory,
                comm_manager=self.comm_manager
            )
            
            # ブリッジアダプターの参照を保持
            self._bridge_adapter = self._module_adapters.get('blackboard_bridge')
            
            self.logger.info("レガシーシステムの初期化が完了しました")
            
        except Exception as e:
            self.logger.error(f"レガシーシステム初期化エラー: {e}")
            raise

    def _initialize_new_system(self) -> None:
        """
        新システム（通信インターフェース）の初期化
        """
        try:
            # 新しいシステム調整器を初期化
            self._system_coordinator = ModuleSystemCoordinator(
                comm_manager=self.comm_manager,
                agent_pool=self._module_adapters.get('agent_pool') if self.compatibility_mode else None,
                summary_engine=self._module_adapters.get('summary_engine') if self.compatibility_mode else None,
                config=self.config
            )
            
            self.logger.info("新システムの初期化が完了しました")
            
        except Exception as e:
            self.logger.error(f"新システム初期化エラー: {e}")
            raise

    @time_async_function
    async def generate(self, input_text: str) -> str:
        """
        メイン生成処理（改良版）
        
        引数:
            input_text: ユーザー入力テキスト
            
        戻り値:
            生成されたレスポンス
        """
        try:
            # モジュールの初期化を確認
            self._ensure_modules_initialized()
            
            # パフォーマンス監視開始
            self.performance.take_memory_snapshot("generate_start")
            
            # 1. 入力処理
            input_data = self.input_reception.process_input(input_text)
            
            # 入力データを通信システムに送信
            message = create_message(
                MessageType.USER_INPUT,
                {
                    'original': input_text,
                    'processed': input_data
                }
            )
            self.comm_manager.publish(message)
            
            # 2. RAG検索の実行
            await self._execute_rag_search(input_text)
            
            # 3. 会話メモリの処理
            if self.use_memory:
                await self._process_conversation_memory(input_text)
            
            # 4. 初期要約の作成
            await self._create_initial_summary(input_text)
            
            # 5. 反復サイクルの実行
            await self._execute_iteration_cycles()
            
            # 6. 最終レスポンスの生成
            final_response = await self._generate_final_response()
            
            # 7. 会話メモリの更新
            if self.use_memory:
                await self._update_conversation_memory(input_text, final_response)
            
            # パフォーマンス監視終了
            self.performance.take_memory_snapshot("generate_end")
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"生成処理エラー: {e}")
            raise MurmurNetError(f"テキスト生成に失敗しました: {e}")

    async def _execute_rag_search(self, input_text: str) -> None:
        """
        RAG検索を実行
        """
        try:
            # RAG検索の実行（遅延インポート）
            from MurmurNet.modules.rag_retriever import RAGRetriever
            
            if not hasattr(self, '_rag_retriever'):
                if self.compatibility_mode and self._legacy_blackboard:
                    self._rag_retriever = RAGRetriever(self.config, self._legacy_blackboard)
                else:
                    # 新システム用のRAGRetrieverが実装されるまでは既存版を使用
                    self._rag_retriever = RAGRetriever(self.config)
            
            # RAG検索実行
            rag_results = self._rag_retriever.search(input_text)
            
            # 結果を通信システムに送信
            message = create_message(
                MessageType.RAG_RESULTS,
                {
                    'query': input_text,
                    'results': rag_results
                }
            )
            self.comm_manager.publish(message)
            
            self.logger.debug("RAG検索が完了しました")
            
        except Exception as e:
            self.logger.error(f"RAG検索エラー: {e}")
            # エラーメッセージを通信システムに送信
            message = create_message(
                MessageType.ERROR,
                {
                    'error': f"RAG検索中にエラーが発生しました: {e}"
                }
            )
            self.comm_manager.publish(message)

    async def _process_conversation_memory(self, input_text: str) -> None:
        """
        会話メモリを処理
        """
        try:
            if self.compatibility_mode and 'conversation_memory' in self._module_adapters:
                adapter = self._module_adapters['conversation_memory']
                context = adapter.get_context()
                
                # コンテキストを通信システムに送信
                message = create_message(
                    MessageType.DATA_STORE,
                    {
                        'key': 'conversation_context',
                        'value': context
                    }
                )
                self.comm_manager.publish(message)
                
            self.logger.debug("会話メモリ処理が完了しました")
            
        except Exception as e:
            self.logger.error(f"会話メモリ処理エラー: {e}")

    async def _create_initial_summary(self, input_text: str) -> None:
        """
        初期要約を作成
        """
        try:
            # 通信システムから必要なデータを取得
            input_data = self.comm_manager.get_data('user_input')
            rag_results = self.comm_manager.get_data('rag_results')
            conversation_context = self.comm_manager.get_data('conversation_context')
            
            # 初期要約を構築
            initial_summary = f"ユーザー入力: {input_text}"
            
            if rag_results:
                initial_summary += f"\\n\\nRAG検索結果: {rag_results}"
            
            if conversation_context:
                initial_summary += f"\\n\\n会話コンテキスト: {conversation_context}"
            
            # 初期要約を通信システムに送信
            message = create_message(
                MessageType.INITIAL_SUMMARY,
                {
                    'summary': initial_summary
                }
            )
            self.comm_manager.publish(message)
            
            self.logger.debug("初期要約を作成しました")
            
        except Exception as e:
            self.logger.error(f"初期要約作成エラー: {e}")

    async def _execute_iteration_cycles(self) -> None:
        """
        反復サイクルを実行
        """
        try:
            # 互換モード時は既存データを同期
            if self.compatibility_mode and self._bridge_adapter:
                self._bridge_adapter.sync_to_communication_system()
            
            # システム調整器による反復実行
            for i in range(self.iterations):
                self.performance.take_memory_snapshot(f"iteration_{i+1}_start")
                
                success = await self._system_coordinator.run_iteration(i)
                if not success:
                    self.logger.warning(f"反復 {i+1} でエラーが発生しましたが処理を継続します")
                
                self.performance.take_memory_snapshot(f"iteration_{i+1}_end")
            
            # 互換モード時は結果を既存システムに同期
            if self.compatibility_mode and self._bridge_adapter:
                self._bridge_adapter.sync_from_communication_system()
                
            self.logger.info(f"{self.iterations}回の反復サイクルが完了しました")
            
        except Exception as e:
            self.logger.error(f"反復サイクル実行エラー: {e}")
            raise

    async def _generate_final_response(self) -> str:
        """
        最終レスポンスを生成
        """
        try:
            # OutputAgentを使用してレスポンス生成（遅延インポート）
            from MurmurNet.modules.output_agent import OutputAgent
            
            if not hasattr(self, '_output_agent'):
                if self.compatibility_mode and self._legacy_blackboard:
                    self._output_agent = OutputAgent(self.config, self._legacy_blackboard)
                else:
                    # 新システム用のOutputAgentが実装されるまでは既存版を使用
                    self._output_agent = OutputAgent(self.config)
            
            # 最終レスポンス生成
            final_response = self._output_agent.generate_response()
            
            # レスポンスを通信システムに送信
            message = create_message(
                MessageType.FINAL_RESPONSE,
                {
                    'response': final_response
                }
            )
            self.comm_manager.publish(message)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"最終レスポンス生成エラー: {e}")
            return "申し訳ございませんが、応答の生成中にエラーが発生しました。"

    async def _update_conversation_memory(self, input_text: str, response: str) -> None:
        """
        会話メモリを更新
        """
        try:
            if self.compatibility_mode and 'conversation_memory' in self._module_adapters:
                adapter = self._module_adapters['conversation_memory']
                adapter.update_context(input_text, [response])
                
            self.logger.debug("会話メモリを更新しました")
            
        except Exception as e:
            self.logger.error(f"会話メモリ更新エラー: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        システム状態を取得
        
        戻り値:
            システム状態辞書
        """
        status = {
            'compatibility_mode': self.compatibility_mode,
            'modules_initialized': self._modules_initialized,
            'performance_monitoring': self.performance.enabled,
            'agent_config': {
                'num_agents': self.num_agents,
                'iterations': self.iterations,
                'use_summary': self.use_summary,
                'use_parallel': self.use_parallel,
                'use_memory': self.use_memory
            }
        }
        
        # システム調整器の統計情報を追加
        if self._system_coordinator:
            status['execution_stats'] = self._system_coordinator.get_execution_stats()
        
        return status

    def get_communication_stats(self) -> Dict[str, Any]:
        """
        通信システムの統計情報を取得
        
        戻り値:
            通信統計辞書
        """
        return self.comm_manager.get_stats()

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ
        """
        try:
            # 通信システムのクリーンアップ
            if hasattr(self.comm_manager, 'cleanup'):
                self.comm_manager.cleanup()
            
            # レガシーシステムのクリーンアップ
            if self._legacy_blackboard and hasattr(self._legacy_blackboard, 'cleanup'):
                self._legacy_blackboard.cleanup()
            
            self.logger.info("リソースのクリーンアップが完了しました")
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")

    def __del__(self):
        """デストラクタ"""
        try:
            self.cleanup()
        except Exception:
            pass  # デストラクタでは例外を無視
