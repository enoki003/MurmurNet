#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 分散SLMシステム
~~~~~~~~~~~~~~~~~~~
複数の小型言語モデルを組み合わせた分散創発型アーキテクチャ
黒板設計・要約・RAGを統合した協調パターン

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
from MurmurNet.modules.system_coordinator_new import SystemCoordinator as NewSystemCoordinator # Alias to avoid immediate name clash if old one is still referenced
from MurmurNet.modules.module_adapters import create_module_adapters, BlackboardBridgeAdapter
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.DistributedSLM')


class DistributedSLM:
    """
    分散創発型言語モデルメインクラス
    
    複数の小規模な言語モデルが協調的に動作する分散型アーキテクチャを通じて
    高度な対話生成機能を提供する中枢システム。
    
    改善点:
    - 新しい通信インターフェースの採用
    - モジュラー設計による疎結合
    - アダプターパターンによる既存システムとの互換性
    - 明確なAPI境界
    - 依存性注入による柔軟性
    
    特徴:
    - 複数の小規模モデルが協調動作
    - 黒板を通じた情報共有
    - 反復的な知識交換で知性を創発
    
    属性:
        config: 設定辞書
        blackboard: 共有黒板
        num_agents: エージェント数
        iterations: 反復回数
    """
    
    def __init__(
        self, 
        config: Dict[str, Any] = None, 
        blackboard=None,
        comm_manager: ModuleCommunicationManager = None,
        compatibility_mode: bool = True
    ):
        """
        分散SLMシステムの初期化
        
        引数:
            config: 設定辞書（オプション）
            blackboard: Blackboardインスタンス（互換性のため）
            comm_manager: 通信管理器（オプション、未指定時は自動作成）
            compatibility_mode: 既存システムとの互換モード
        """        # ConfigManagerから設定を取得
        if isinstance(config, dict):
            # 辞書の場合、一時的に設定ファイルに保存して読み込み
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                temp_config_path = f.name
            try:
                self.settings_obj = get_config(temp_config_path)
            finally:
                os.unlink(temp_config_path)
        else:
            # パスまたはNoneの場合
            self.settings_obj = get_config(config)
        
        # ロガー設定
        self.logger = setup_logger(self.settings_obj.to_dict())
        
        # パフォーマンスモニタリング
        self.performance = PerformanceMonitor(
            enabled=self.settings_obj.logging.performance_monitoring,
            memory_tracking=self.settings_obj.logging.memory_tracking
        )
        
        # 初期メモリスナップショット
        self.performance.take_memory_snapshot("distributed_slm_init_start")
          # 通信システムの初期化
        self.comm_manager = comm_manager or create_communication_system()
        
        # デバッグ: 通信マネージャーの確認
        self.logger.debug(f"通信マネージャーの型: {type(self.comm_manager)}")
        self.logger.debug(f"利用可能なメソッド: {dir(self.comm_manager)}")
        self.logger.debug(f"publishメソッドの存在: {hasattr(self.comm_manager, 'publish')}")
        
        # 互換性モードの設定
        self.compatibility_mode = compatibility_mode
          # 基本動作パラメータ（渡された設定を優先）
        self.num_agents = self.settings_obj.agent.num_agents
        self.iterations = self.settings_obj.agent.iterations
        self.use_summary = self.settings_obj.agent.use_summary
        self.use_parallel = self.settings_obj.agent.use_parallel
        self.use_memory = self.settings_obj.agent.use_memory
        self.memory_question_patterns = self.settings_obj.agent.memory_question_patterns
        
        # 遅延初期化フラグ
        self._modules_initialized = False
        self._initialization_lock = threading.Lock()
        
        # 基本モジュールの初期化
        self.input_reception = InputReception(self.settings_obj.to_dict())
        
        # レガシーシステム（互換性モード用）
        self._legacy_blackboard = blackboard or None
        self._bridge_adapter = None
        
        # 新システムのモジュール
        self._system_coordinator = None
        self._module_adapters = {}
          # パフォーマンスステータスを通信システムに記録
        message = create_message(
            MessageType.SYSTEM_STATUS,
            "distributed_slm",
            {
                'performance_enabled': self.performance.enabled,
                'compatibility_mode': self.compatibility_mode
            }
        )
        
        # デバッグ: publishメソッドの存在確認
        if hasattr(self.comm_manager, 'publish'):
            self.comm_manager.publish(message)
        else:
            self.logger.error(f"通信マネージャーにpublishメソッドがありません。型: {type(self.comm_manager)}")
            self.logger.error(f"利用可能なメソッド: {[m for m in dir(self.comm_manager) if not m.startswith('_')]}")
            # 代替手段としてsend_messageを使用
            if hasattr(self.comm_manager, 'send_message'):
                self.comm_manager.send_message(message)
            else:
                self.logger.warning("send_messageメソッドも利用できません")
        
        # 初期化完了時のメモリスナップショット
        self.performance.take_memory_snapshot("distributed_slm_init_complete")
        
        self.logger.info(f"分散SLMシステムを初期化しました: {self.num_agents}エージェント, {self.iterations}反復 (遅延初期化)")

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
            from MurmurNet.modules.rag_retriever import RAGRetriever
            from MurmurNet.modules.output_agent import OutputAgent
              # レガシーモジュールの初期化
            if not self._legacy_blackboard:
                self._legacy_blackboard = Blackboard(self.settings_obj.to_dict())
            
            legacy_agent_pool = AgentPoolManager(self.settings_obj.to_dict(), self._legacy_blackboard)
            legacy_summary_engine = SummaryEngine(self.settings_obj.to_dict())  # SummaryEngineはconfigのみ受け取る
            legacy_conversation_memory = ConversationMemory(self.settings_obj.to_dict(), self._legacy_blackboard)
              # RAGRetrieverとOutputAgentの初期化（configパラメータのみ）
            self._rag_retriever = RAGRetriever(self.settings_obj.to_dict())
              # OutputAgentの初期化
            self._output_agent = OutputAgent(self.settings_obj.to_dict())
            
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
            self._system_coordinator = NewSystemCoordinator(
                config=self.settings_obj.to_dict(), # Pass the main config dict
                blackboard=self._legacy_blackboard if self.compatibility_mode else None,
                agent_pool=self._module_adapters.get('agent_pool') if self.compatibility_mode else None,
                summary_engine=self._module_adapters.get('summary_engine') if self.compatibility_mode else None
            )
            
            self.logger.info("新システムの初期化が完了しました")
            
        except Exception as e:
            self.logger.error(f"新システム初期化エラー: {e}")
            raise

    def _is_memory_related_question(self, text: str) -> bool:
        """
        入力テキストが会話記憶に関連する質問かどうかを判定
        
        引数:
            text: 入力テキスト
        
        戻り値:
            会話記憶に関連する質問ならTrue
        """
        # いずれかのパターンにマッチしたら記憶関連の質問と判定
        for pattern in self.memory_question_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

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
            
            self.logger.info("応答生成プロセスを開始")
            
            # 互換性モード：黒板をクリア
            if self.compatibility_mode and self._legacy_blackboard:
                self._legacy_blackboard.clear_current_turn()
            
            # 会話記憶に関連する質問かチェック
            if self.use_memory and self._is_memory_related_question(input_text):
                memory_answer = await self._handle_memory_question(input_text)
                if memory_answer:
                    return memory_answer
              # 1. 入力処理
            input_data = self.input_reception.process(input_text)
            
            # 入力データを通信システムに送信
            message = create_message(
                MessageType.USER_INPUT,
                "input_reception",
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
            
            self.logger.info(f"応答生成完了: {len(final_response)}文字")
            return final_response
            
        except Exception as e:
            self.logger.error(f"生成処理エラー: {e}")
            raise MurmurNetError(f"テキスト生成に失敗しました: {e}")

    async def _handle_memory_question(self, input_text: str) -> Optional[str]:
        """
        会話記憶に関連する質問を処理
        """
        try:
            if self.compatibility_mode and 'conversation_memory' in self._module_adapters:
                adapter = self._module_adapters['conversation_memory']
                # ConversationMemoryAdapterにget_answer_for_questionメソッドがあれば使用
                if hasattr(adapter, 'get_answer_for_question'):
                    memory_answer = adapter.get_answer_for_question(input_text)
                    if memory_answer:
                        self.logger.info("会話記憶から回答を取得しました")                        # 会話記憶を更新
                        adapter.update_context(input_text, [memory_answer])
                        return memory_answer
                        
        except Exception as e:
            self.logger.error(f"会話記憶処理エラー: {e}")
            return None
    
    async def _execute_rag_search(self, input_text: str) -> None:
        """
        RAG検索を実行
        """
        try:
            # RAG検索の実行（遅延インポート）
            from MurmurNet.modules.rag_retriever import RAGRetriever
            
            if not hasattr(self, '_rag_retriever'):
                # RAGRetrieverはconfigパラメータのみを受け取る
                self._rag_retriever = RAGRetriever(self.settings_obj.to_dict())
            
            # RAG検索実行
            rag_results = self._rag_retriever.retrieve(input_text)
            
            # 結果を通信システムに送信
            message = create_message(
                MessageType.RAG_RESULTS,
                "rag_retriever",
                {
                    'query': input_text,
                    'results': rag_results
                }
            )
            self.comm_manager.publish(message)
            
            # 互換性モード：黒板にも書き込み
            if self.compatibility_mode and self._legacy_blackboard:
                self._legacy_blackboard.write('rag', rag_results)
            
            self.logger.debug("RAG検索が完了しました")
            
        except Exception as e:
            self.logger.error(f"RAG検索エラー: {e}")
            # エラーメッセージを通信システムに送信
            message = create_message(
                MessageType.ERROR,
                "rag_retriever",
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
                    "conversation_memory",
                    {
                        'key': 'conversation_context',
                        'value': context
                    }
                )
                self.comm_manager.publish(message)
                
                # 互換性モード：黒板にも書き込み
                if self.compatibility_mode and self._legacy_blackboard and context:
                    self._legacy_blackboard.write('conversation_context', context)
                
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
                "summary_engine",
                {
                    'summary': initial_summary
                }
            )
            self.comm_manager.publish(message)
            
            # 互換性モード：黒板にも書き込み
            if self.compatibility_mode and self._legacy_blackboard and self.use_summary:
                self._legacy_blackboard.write('initial_summary', initial_summary)
            
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
                    self._output_agent = OutputAgent(self.settings_obj.to_dict(), self._legacy_blackboard)
                else:
                    # 新システム用のOutputAgentが実装されるまでは既存版を使用
                    self._output_agent = OutputAgent(self.settings_obj.to_dict())
            
            # 最終レスポンス生成
            if self.compatibility_mode and self._legacy_blackboard:
                # 互換性モード：黒板からエントリを収集
                entries = []
                # 各イテレーションの要約を収集
                if self.use_summary:
                    for i in range(self.iterations):
                        summary = self._legacy_blackboard.read(f'summary_{i}')
                        if summary:
                            entries.append({"type": "summary", "iteration": i, "text": summary})
                
                # 最終エージェント出力も収集
                for i in range(self.num_agents):
                    agent_output = self._legacy_blackboard.read(f"agent_{i}_output")
                    if agent_output:
                        entries.append({"type": "agent", "agent": i, "text": agent_output})
                
                final_response = self._output_agent.generate(self._legacy_blackboard, entries)
            else:
                final_response = self._output_agent.generate_response()
            
            # レスポンスを通信システムに送信
            message = create_message(
                MessageType.FINAL_RESPONSE,
                "output_agent",
                {
                    'response': final_response
                }
            )
            self.comm_manager.publish(message)
            
            # 互換性モード：黒板にも書き込み
            if self.compatibility_mode and self._legacy_blackboard:
                self._legacy_blackboard.write('final_response', final_response)
            
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

    def reset_memory(self) -> None:
        """
        会話履歴をリセット
        
        システムの会話記憶を完全にクリアする
        """
        try:
            if self.compatibility_mode and 'conversation_memory' in self._module_adapters:
                adapter = self._module_adapters['conversation_memory']
                if hasattr(adapter, 'clear_memory'):
                    adapter.clear_memory()
                    
            # 黒板のターン関連データもクリア
            if self.compatibility_mode and self._legacy_blackboard:
                self._legacy_blackboard.clear_current_turn()
                
            self.logger.info("会話記憶をクリアしました")
            
        except Exception as e:
            self.logger.error(f"メモリリセットエラー: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        システムの統計情報を取得
        
        戻り値:
            システム統計情報を含む辞書
        """
        stats = {
            "agents": self.num_agents,
            "iterations": self.iterations,
            "memory_enabled": self.use_memory,
            "summary_enabled": self.use_summary,
            "parallel_enabled": self.use_parallel,
            "compatibility_mode": self.compatibility_mode,
            "modules_initialized": self._modules_initialized
        }
        
        # 会話履歴の長さを追加
        if self.compatibility_mode and 'conversation_memory' in self._module_adapters:
            adapter = self._module_adapters['conversation_memory']
            if hasattr(adapter, 'conversation_memory') and hasattr(adapter.conversation_memory, 'conversation_history'):
                stats["conversation_history"] = len(adapter.conversation_memory.conversation_history)
            else:
                stats["conversation_history"] = 0
        else:
            stats["conversation_history"] = 0
        
        return stats

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

    # プロパティアクセッサー（互換性のため）
    @property
    def blackboard(self):
        """Blackboardへのアクセッサー（互換性のため）"""
        self._ensure_modules_initialized()
        return self._legacy_blackboard

    @property
    def agent_pool(self):
        """エージェントプールマネージャー（互換性のため）"""
        self._ensure_modules_initialized()
        return self._module_adapters.get('agent_pool')

    @property
    def rag_retriever(self):
        """RAG検索エンジン（互換性のため）"""
        self._ensure_modules_initialized()
        return getattr(self, '_rag_retriever', None)

    @property
    def summary_engine(self):
        """要約エンジン（互換性のため）"""
        self._ensure_modules_initialized()
        return self._module_adapters.get('summary_engine')

    @property
    def output_agent(self):
        """出力エージェント（互換性のため）"""
        self._ensure_modules_initialized()
        return getattr(self, '_output_agent', None)

    @property
    def conversation_memory(self):
        """会話記憶（互換性のため）"""
        self._ensure_modules_initialized()
        return self._module_adapters.get('conversation_memory')

    @property
    def system_coordinator(self):
        """システム調整器（互換性のため）"""
        self._ensure_modules_initialized()
        return self._system_coordinator

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
