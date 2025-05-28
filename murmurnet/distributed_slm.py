#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 分散SLMシステム
~~~~~~~~~~~~~~~~~~~
複数の小型言語モデルを組み合わせた分散創発型アーキテクチャ
黒板設計・要約・RAGを統合した協調パターン

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
from MurmurNet.modules.blackboard import Blackboard 
# Import new constants from blackboard (assuming they are defined there or in data_structures and exposed)
from MurmurNet.modules.blackboard import (
    CONTENT_TYPE_SUMMARY, CONTENT_TYPE_USER_INPUT, CONTENT_TYPE_RAG_RESULT,
    CONTENT_TYPE_CONVERSATION_CONTEXT, CONTENT_TYPE_FINAL_RESPONSE,
    SOURCE_SYSTEM, SOURCE_USER, SOURCE_RAG_RETRIEVER, SOURCE_CONVERSATION_MEMORY,
    SOURCE_OUTPUT_AGENT
)
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.modules.output_agent import OutputAgent
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.conversation_memory import ConversationMemory
from MurmurNet.modules.system_coordinator import SystemCoordinator
from MurmurNet.modules.config_manager import get_config

class DistributedSLM:
    """
    分散創発型言語モデルメインクラス
    
    複数の小規模な言語モデルが協調的に動作する分散型アーキテクチャを通じて
    高度な対話生成機能を提供する中枢システム。
    
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
    
    def __init__(self, config: Dict[str, Any] = None, blackboard=None):
        """
        分散SLMシステムの初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
            blackboard: Blackboardインスタンス（省略時は内部で作成）
        """
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
        
        # ロガー設定
        self.logger = setup_logger(self.config)
        
        # ConfigManagerからパフォーマンスモニタリング設定を取得
        self.performance = PerformanceMonitor(
            enabled=self.config_manager.logging.performance_monitoring,
            memory_tracking=self.config_manager.logging.memory_tracking
        )
        
        # 初期メモリスナップショット
        self.performance.take_memory_snapshot("distributed_slm_init_start")
          # ConfigManagerから動作パラメータを取得
        self.num_agents = self.config_manager.agent.num_agents
        self.iterations = self.config_manager.agent.iterations
        self.use_summary = self.config_manager.agent.use_summary
        self.use_parallel = self.config_manager.agent.use_parallel
        self.use_memory = self.config_manager.agent.use_memory
        
        # 遅延初期化フラグとロック
        self._modules_initialized = False
        self._initialization_lock = threading.Lock()
        
        # 基本モジュールのみ即座に初期化
        self.blackboard = blackboard if blackboard is not None else Blackboard(self.config)
        self.input_reception = InputReception(self.config)
        
        # 重いモジュールは遅延初期化用にNoneで初期化
        self._agent_pool = None
        self._rag_retriever = None
        self._summary_engine = None
        self._output_agent = None
        self._conversation_memory = None
        self._system_coordinator = None
        
        # パフォーマンスステータスを黒板に記録
        self.blackboard.write('performance_enabled', self.performance.enabled)
        
        # 初期化完了時のメモリスナップショット
        self.performance.take_memory_snapshot("distributed_slm_init_complete")
        
        self.logger.info(f"分散SLMシステムを初期化しました: {self.num_agents}エージェント, {self.iterations}反復 (遅延初期化)")
      

    def _is_memory_related_question(self, text: str) -> bool:
        """
        入力テキストが会話記憶に関連する質問かどうかを判定
        
        引数:
            text: 入力テキスト
        
        戻り値:
            会話記憶に関連する質問ならTrue
        """
        # 記憶関連の質問パターン
        memory_patterns = [
            r"(私|僕|俺|わたし|ぼく|おれ)の名前(は|を)(なに|何|なん)",
            r"(私|僕|俺|わたし|ぼく|おれ)(の|は|を)(なに|何|なん)と(言い|いい|呼び|よび|よん)",
            r"(私|僕|俺|わたし|ぼく|おれ)の名前(は|を)(覚え|おぼえ)",
            r"(私|僕|俺|わたし|ぼく|おれ)の(趣味|しゅみ)(は|を)(なに|何|なん)",
            r"(私|僕|俺|わたし|ぼく|おれ)(は|が)(なに|何|なん)(が好き|を好きと言いました)",
            r"(覚え|おぼえ)てる",
            r"(覚え|おぼえ)てます",
            r"(私|僕|俺|わたし|ぼく|おれ)について",
        ]
        
        # いずれかのパターンにマッチしたら記憶関連の質問と判定
        for pattern in memory_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
        
    async def generate(self, input_text: str) -> str:
        """
        外部公開API: 入力文字列から最終応答を生成（キャッシュ対応）
        
        引数：
            input_text: ユーザー入力文字列
            
        戻り値：
            生成された応答文字列        """
        self.logger.info("応答生成プロセスを開始")
        self.current_iteration = 0 # Initialize current_iteration for this generate call.

        # Blackboard clearing is now handled by SystemCoordinator.run_iteration()
        # No self.blackboard.clear_current_turn() here.
        
        # 会話記憶に関連する質問かチェック
        if self.use_memory and self._is_memory_related_question(input_text):
            # 会話記憶から回答を取得
            memory_answer = self.conversation_memory.get_answer_for_question(input_text)
            if memory_answer:
                self.logger.info("会話記憶から回答を取得しました")
                # 会話記憶を更新
                self.conversation_memory.add_conversation_entry(
                    user_input=input_text,
                    system_response=memory_answer
                )
                return memory_answer
                
        # 1. 入力受付・前処理
        # processed_input_data is expected to be a dict like {'raw': ..., 'normalized': ...}
        processed_input_data = self.input_reception.process(input_text)
        self.blackboard.add_user_input(
            text=processed_input_data.get('normalized', input_text), # Main text is normalized
            raw_input_details=processed_input_data, # Pass the whole dict
            iteration=self.current_iteration
        )
        
        # 2. 会話履歴の追加（使用する場合）
        if self.use_memory:
            conversation_context_text = self.conversation_memory.get_conversation_context()
            if conversation_context_text and conversation_context_text != "過去の会話はありません。":
                self.blackboard.add_conversation_context(
                    context=conversation_context_text, 
                    iteration=self.current_iteration,
                    source=SOURCE_CONVERSATION_MEMORY
                )
                self.logger.debug(f"会話コンテキストを追加 (反復 {self.current_iteration}): {len(conversation_context_text)}文字")
        
        # 3. RAG検索
        rag_content = self.rag_retriever.retrieve(input_text) 
        self.blackboard.add_rag_result(
            rag_data=rag_content, 
            iteration=self.current_iteration,
            source=SOURCE_RAG_RETRIEVER
        )
        
        # 4. 初期要約の実行（オプション） - This is removed. 
        # If a pre-iteration summary is needed, it should be explicitly generated and added.
        # For now, SystemCoordinator._build_common_prompt will use raw entries.
        
        # 5. 反復サイクルの実行（SystemCoordinatorに委譲）
        try:
            for i in range(self.iterations):
                self.current_iteration = i # Update current_iteration for this loop
                self.performance.take_memory_snapshot(f"iteration_{self.current_iteration + 1}_start")
                
                # SystemCoordinatorに反復処理を委譲
                success = await self.system_coordinator.run_iteration(self.current_iteration)
                if not success:
                    self.logger.warning(f"反復 {self.current_iteration + 1} でエラーが発生しましたが処理を継続します")
                
                self.performance.take_memory_snapshot(f"iteration_{i+1}_end")
                
        except Exception as e:
            self.logger.error(f"反復処理中にエラーが発生しました: {e}")
            # エラーが発生しても最終応答生成を試行
            
        # 6. 最終要約とエージェント出力の収集 for OutputAgent
        output_agent_input_entries = [] # This will be List[Dict[str, Any]] for OutputAgent
        
        if self.use_summary:
            for i in range(self.iterations): # Iterate up to the final iteration number
                summary_bb_entries = self.blackboard.get_entries_by_content_type(
                    CONTENT_TYPE_SUMMARY, 
                    iteration=i
                )
                for bb_entry in summary_bb_entries:
                    # Extract summary text, assuming it's in bb_entry.data['text'] or bb_entry.data
                    summary_text = ""
                    if isinstance(bb_entry.data, dict) and 'text' in bb_entry.data:
                        summary_text = bb_entry.data['text']
                    elif isinstance(bb_entry.data, str):
                         summary_text = bb_entry.data
                    
                    if summary_text:
                        output_agent_input_entries.append({
                            "type": "summary", 
                            "iteration": i, 
                            "text": summary_text
                        })

        # Collect agent outputs from the final iteration for OutputAgent
        # self.current_iteration here refers to the last completed iteration index (e.g., self.iterations - 1)
        final_iteration_agent_outputs = self.blackboard.get_agent_outputs(iteration=self.current_iteration)
        for ao in final_iteration_agent_outputs: # ao is an AgentOutput object
            if ao.text and not ao.text.endswith("応答できませんでした"): # Filter out errors/non-responses
                output_agent_input_entries.append({
                    "type": "agent", 
                    "agent": ao.agent_id, 
                    "text": ao.text,
                    "iteration": ao.iteration 
                })

        # 7. 最終応答生成
        # OutputAgent.generate might need to be updated to use new blackboard methods if it reads directly.
        # For now, passing the collected entries in the old format.
        final_response = self.output_agent.generate(self.blackboard, output_agent_input_entries)
        
        self.blackboard.add_generic_entry(
            source=SOURCE_OUTPUT_AGENT,
            content_type=CONTENT_TYPE_FINAL_RESPONSE,
            data=final_response,
            iteration=self.current_iteration 
        )
        
        # 9. 会話履歴に追加（使用する場合）
        if self.use_memory:
            self.conversation_memory.add_conversation_entry(
                user_input=input_text, 
                system_response=final_response
            )
            self.logger.debug("会話記憶を更新しました")
        
        self.logger.info(f"応答生成完了: {len(final_response)}文字")
        return final_response
    
    def reset_memory(self) -> None:
        """
        会話履歴をリセット
        
        システムの会話記憶を完全にクリアする
        """
        if hasattr(self, 'conversation_memory'):
            self.conversation_memory.clear_memory()
            self.logger.info("会話記憶をクリアしました")
        
        # For a full memory reset, clear all blackboard entries too.
        if self.blackboard:
            self.blackboard.clear_all()
            self.logger.info("Blackboard cleared as part of memory reset.")
    
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
            "conversation_history": len(self.conversation_memory.conversation_history) if hasattr(self, 'conversation_memory') else 0
        }
        
        return stats
    
    def _ensure_modules_initialized(self):
        """重いモジュールの遅延初期化を確実に実行（内部メソッド）"""
        if not self._modules_initialized:
            with self._initialization_lock:
                # ダブルチェックロッキング
                if not self._modules_initialized:
                    self.logger.info("重いモジュールの遅延初期化を開始...")
                    
                    try:
                        # 共有モデルを先に初期化（最も重い処理）
                        from MurmurNet.modules.model_factory import get_shared_model
                        get_shared_model(self.config)  # 初回呼び出しで初期化
                          # 各モジュールを順次初期化
                        self._agent_pool = AgentPoolManager(self.config, self.blackboard)
                        self._rag_retriever = RAGRetriever(self.config)
                        self._summary_engine = SummaryEngine(self.config)
                        self._output_agent = OutputAgent(self.config)
                        self._conversation_memory = ConversationMemory(self.config, self.blackboard)
                        
                        # システム調整器を初期化（他のモジュールが初期化済みのため）
                        self._system_coordinator = SystemCoordinator(
                            self.config, 
                            self.blackboard, 
                            self._agent_pool, 
                            self._summary_engine                        )
                        
                        self._modules_initialized = True
                        self.logger.info("遅延初期化が完了しました")
                        
                    except Exception as e:
                        self.logger.error(f"モジュール初期化エラー: {e}")
                        raise RuntimeError(f"重要なモジュールの初期化に失敗しました: {e}")
    
    @property
    def agent_pool(self):
        """エージェントプールマネージャー（遅延初期化）"""
        self._ensure_modules_initialized()
        return self._agent_pool
    
    @property
    def rag_retriever(self):
        """RAG検索エンジン（遅延初期化）"""
        self._ensure_modules_initialized()
        return self._rag_retriever
    
    @property
    def summary_engine(self):
        """要約エンジン（遅延初期化）"""
        self._ensure_modules_initialized()
        return self._summary_engine
    
    @property
    def output_agent(self):
        """出力エージェント（遅延初期化）"""
        self._ensure_modules_initialized()
        return self._output_agent
    
    @property
    def conversation_memory(self):
        """会話記憶（遅延初期化）"""
        self._ensure_modules_initialized()
        return self._conversation_memory
    
    @property
    def system_coordinator(self):
        """システム調整器（遅延初期化）"""
        self._ensure_modules_initialized()
        return self._system_coordinator
