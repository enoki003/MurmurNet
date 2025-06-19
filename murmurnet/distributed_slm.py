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
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union

# Windows環境でのasyncio最適化（ProactorEventLoop回避）
if os.name == "nt":
    # ProactorEventLoopの問題を回避するためSelectorEventLoopを使用
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from MurmurNet.modules.common import setup_logger, PerformanceError
from MurmurNet.modules.performance import PerformanceMonitor, time_function, time_async_function
from MurmurNet.modules.input_reception import InputReception
from MurmurNet.modules.blackboard import Blackboard
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.modules.output_agent import OutputAgent
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.conversation_memory import ConversationMemory

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
            config: 設定辞書（省略時は空の辞書）
            blackboard: Blackboardインスタンス（省略時は内部で作成）
        """
        self.config = config or {}
        
        # ロガー設定
        self.logger = setup_logger(self.config)
        
        # パフォーマンスモニタリング設定
        self.performance = PerformanceMonitor(
            enabled=self.config.get('performance_monitoring', True),
            memory_tracking=self.config.get('memory_tracking', True)
        )
        
        # 初期メモリスナップショット
        self.performance.take_memory_snapshot("distributed_slm_init_start")
        
        # 動作パラメータ
        self.num_agents = self.config.get('num_agents', 2)
        self.iterations = self.config.get('iterations', 1)  # 反復回数
        self.use_summary = self.config.get('use_summary', True)  # 要約を使うか
        self.use_parallel = self.config.get('use_parallel', False)  # 並列処理を使うか
        self.use_memory = self.config.get('use_memory', True)  # 会話履歴を使うか
          # 各モジュールの初期化
        self.logger.info("黒板モジュールを初期化中...")
        self.blackboard = blackboard if blackboard is not None else Blackboard(self.config)
        
        self.logger.info("入力受信モジュールを初期化中...")
        self.input_reception = InputReception(self.config)
        
        self.logger.info("エージェントプールを初期化中...")
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        
        self.logger.info("RAGリトリーバーを初期化中...")
        self.rag_retriever = RAGRetriever(self.config)
        
        self.logger.info("要約エンジンを初期化中...")
        self.summary_engine = SummaryEngine(self.config)
        
        self.logger.info("出力エージェントを初期化中...")
        self.output_agent = OutputAgent(self.config)
        
        self.logger.info("会話記憶モジュールを初期化中...")
        self.conversation_memory = ConversationMemory(self.config, self.blackboard)
        
        # パフォーマンスステータスを黒板に記録
        self.blackboard.write('performance_enabled', self.performance.enabled)
        
        # 初期化完了時のメモリスナップショット
        self.performance.take_memory_snapshot("distributed_slm_init_complete")
        
        self.logger.info(f"分散SLMシステムを初期化しました: {self.num_agents}エージェント, {self.iterations}反復")
        
    @time_async_function
    async def _run_iteration(self, iteration: int) -> None:
        """
        単一の反復サイクルを実行（内部メソッド）
        
        複数エージェントによる協調的な対話生成処理の一連のサイクルを実行する
        
        引数:
            iteration: 現在の反復インデックス
        """
        self.logger.info(f"反復 {iteration+1}/{self.iterations} を開始")
        self.performance.take_memory_snapshot(f"iteration_{iteration+1}_start")
        
        # 1. エージェント実行（並列または逐次）
        if self.use_parallel:
            await self._run_agents_parallel()
        else:
            self.agent_pool.run_agents(self.blackboard)
            
        # 2. エージェント出力収集
        agent_entries = []
        for i in range(self.num_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            if agent_output:
                agent_entries.append({"agent": i, "text": agent_output})
        
        # 3. 出力の要約（使用する場合）
        if self.use_summary and agent_entries:
            summary_start = self.performance.start_timer(f"summary_iteration_{iteration+1}")
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            self.blackboard.write(f'summary_{iteration}', summary)
            summary_time = self.performance.end_timer(f"summary_iteration_{iteration+1}", summary_start)
            self.logger.debug(f"反復 {iteration+1} の要約を作成しました (実行時間: {summary_time:.4f}秒)")
        
        self.performance.take_memory_snapshot(f"iteration_{iteration+1}_end")
    
    async def _run_agents_parallel(self) -> None:
        """
        エージェントを並列実行（内部メソッド）
        
        複数のエージェントを並列に実行し、結果を黒板に書き込む
        """
        self.logger.info("エージェントを並列実行中...")
        
        # スレッドプール内でエージェントタスクを実行するラッパー
        def run_agent_task(agent_id: int) -> str:
            """エージェントタスクのスレッドプールラッパー"""
            try:
                # 共有モデルを使って実行（グローバルロックはagent_task内で使用）
                return self.agent_pool._agent_task(agent_id)
            except Exception as e:
                self.logger.error(f"エージェント {agent_id} 実行エラー: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return f"エージェント{agent_id}は応答できませんでした"
        
        # イベントループを取得
        loop = asyncio.get_event_loop()
        
        try:
            # 真の並列実行：すべてのエージェントを同時に実行
            tasks = []
            for i in range(self.num_agents):
                # 各エージェントのタスクを作成
                task = loop.run_in_executor(None, run_agent_task, i)
                tasks.append(task)
            
            # すべてのタスクを同時に実行して結果を取得
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を黒板に書き込み
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"エージェント {i} 処理エラー: {str(result)}")
                    self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
                elif result:
                    self.blackboard.write(f'agent_{i}_output', result)
                else:
                    self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は空の応答を返しました")
                    
        except Exception as e:
            self.logger.error(f"並列実行エラー: {str(e)}")
            # エラーが発生した場合は逐次実行にフォールバック
            self.logger.info("逐次実行にフォールバックします")
            for i in range(self.num_agents):
                try:
                    result = self.agent_pool._agent_task(i)
                    self.blackboard.write(f'agent_{i}_output', result)
                except Exception as inner_e:
                    self.logger.error(f"エージェント {i} フォールバック実行エラー: {str(inner_e)}")
                    self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
    
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
        外部公開API: 入力文字列から最終応答を生成
        
        引数：
            input_text: ユーザー入力文字列
            
        戻り値：
            生成された応答文字列
        """
        self.logger.info("応答生成プロセスを開始")
        
        # 新しいターンのために黒板をクリア（会話履歴は保持）
        self.blackboard.clear_current_turn()
        
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
        processed = self.input_reception.process(input_text)
        self.blackboard.write('input', processed)
        
        # 2. 会話履歴の追加（使用する場合）
        if self.use_memory:
            conversation_context = self.conversation_memory.get_conversation_context()
            if conversation_context and conversation_context != "過去の会話はありません。":
                # 会話コンテキストを黒板に書き込む
                self.blackboard.write('conversation_context', conversation_context)
                self.logger.debug(f"会話コンテキストを追加: {len(conversation_context)}文字")
                
                # 会話コンテキストを入力処理に統合
                if isinstance(processed, dict) and 'normalized' in processed:
                    processed['context'] = conversation_context
                    self.blackboard.write('input', processed)
        
        # 3. RAG検索
        rag_result = self.rag_retriever.retrieve(input_text)
        self.blackboard.write('rag', rag_result)
        
        # 4. 初期要約の実行（オプション）
        if self.use_summary:
            initial_summary = f"ユーザー入力: {processed['normalized'] if isinstance(processed, dict) else processed}\n\n検索情報: {rag_result}"
            if self.use_memory and 'conversation_context' in self.blackboard.memory:
                initial_summary += f"\n\n会話コンテキスト: {self.blackboard.read('conversation_context')}"
            self.blackboard.write('initial_summary', initial_summary)
            
        # 5. 反復サイクルの実行
        for i in range(self.iterations):
            await self._run_iteration(i)
            
        # 6. 最終要約とエージェント出力の収集
        entries = []
        # 各イテレーションの要約を収集
        if self.use_summary:
            for i in range(self.iterations):
                summary = self.blackboard.read(f'summary_{i}')
                if summary:
                    entries.append({"type": "summary", "iteration": i, "text": summary})
        
        # 最終エージェント出力も収集
        for i in range(self.num_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            if agent_output:
                entries.append({"type": "agent", "agent": i, "text": agent_output})
                
        # 7. 最終応答生成
        final_response = self.output_agent.generate(self.blackboard, entries)
        self.blackboard.write('final_response', final_response)
        
        # 8. 会話履歴に追加（使用する場合）
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
            
        # 黒板のターン関連データもクリア
        self.blackboard.clear_current_turn()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        システムの統計情報を取得
        
        戻り値:
            システム統計情報を含む辞書
        """
        return {
            "agents": self.num_agents,
            "iterations": self.iterations,
            "memory_enabled": self.use_memory,
            "summary_enabled": self.use_summary,
            "parallel_enabled": self.use_parallel,
            "conversation_history": len(self.conversation_memory.conversation_history) if hasattr(self, 'conversation_memory') else 0
        }
