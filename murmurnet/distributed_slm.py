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
from MurmurNet.modules.shutdown_manager import register_for_shutdown, get_shutdown_manager

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
        
        # ShutdownManagerに自身を登録（最高優先度で）
        register_for_shutdown(self, "DistributedSLM", priority=100)
        
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
                agent_entries.append({"agent": i, "text": agent_output})        # 3. 出力の要約（使用する場合かつ有効な場合のみ）
        should_summarize = self.use_summary and agent_entries and self._should_use_summary_iteration(agent_entries)
        if should_summarize:
            summary_start = self.performance.start_timer(f"summary_iteration_{iteration+1}")
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            self.blackboard.write(f'summary_{iteration}', summary)
            summary_time = self.performance.end_timer(f"summary_iteration_{iteration+1}", summary_start)
            self.logger.debug(f"反復 {iteration+1} の要約を作成しました (実行時間: {summary_time:.4f}秒)")
        elif self.use_summary and agent_entries:
            self.logger.info(f"反復 {iteration+1} の要約をスキップしました（短すぎる内容のため）")
        elif self.use_summary:
            self.logger.info(f"反復 {iteration+1} の要約をスキップしました（エージェント出力なし）")
        
        self.performance.take_memory_snapshot(f"iteration_{iteration+1}_end")
    
    async def _run_agents_parallel(self) -> None:
        """
        エージェントを並列実行（内部メソッド）
        
        複数のエージェントを並列に実行し、結果を黒板に書き込む
        CPU最適化版の並列処理を使用
        """
        self.logger.info("エージェントを並列実行中（CPU最適化版）...")
        
        # パフォーマンス統計の記録
        if hasattr(self.performance, 'record_parallel_execution'):
            self.performance.record_parallel_execution('parallel')
        
        try:
            # AgentPoolManagerの最適化された並列実行を使用
            if hasattr(self.agent_pool, 'run_agents_parallel'):
                await self.agent_pool.run_agents_parallel(self.blackboard)
            else:
                # フォールバック: 従来の並列実行
                await self._run_agents_parallel_fallback()
                    
        except Exception as e:
            self.logger.error(f"並列実行エラー: {str(e)}")
            # エラーが発生した場合は逐次実行にフォールバック
            self.logger.info("逐次実行にフォールバックします")
            self.agent_pool.run_agents(self.blackboard)

    async def _run_agents_parallel_fallback(self) -> None:
        """
        フォールバック用の並列実行
        """
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
    
        
    async def generate(self, input_text: str) -> str:
        """
        外部公開API: 入力文字列から最終応答を生成
        
        引数：
            input_text: ユーザー入力文字列
            
        戻り値：
            生成された応答文字列
        """
        import time
        total_start_time = time.time()
        
        self.logger.info("応答生成プロセスを開始")
        
        # 新しいターンのために黒板をクリア（会話履歴は保持）
        self.blackboard.clear_current_turn()
        
        # 会話記憶に関連する質問かチェック
        if self.use_memory and self._is_memory_related_question(input_text):
            # 会話記憶から回答を取得
            memory_answer = self.conversation_memory.get_answer_for_question(input_text)
            if memory_answer:
                elapsed = time.time() - total_start_time
                self.logger.info(f"会話記憶から回答を取得しました (時間: {elapsed:.2f}秒)")
                # 会話記憶を更新
                self.conversation_memory.add_conversation_entry(
                    user_input=input_text,
                    system_response=memory_answer
                )
                return memory_answer
                
        # 1. 入力受付・前処理
        step_start = time.time()
        processed = self.input_reception.process(input_text)
        self.blackboard.write('input', processed)
        step_time = time.time() - step_start
        self.logger.info(f"[ステップ1] 入力処理完了: {step_time:.2f}秒")
        
        # 2. 会話履歴の追加（使用する場合）
        if self.use_memory:
            step_start = time.time()
            conversation_context = self.conversation_memory.get_conversation_context()
            if conversation_context and conversation_context != "過去の会話はありません。":
                # 会話コンテキストを黒板に書き込む
                self.blackboard.write('conversation_context', conversation_context)
                self.logger.debug(f"会話コンテキストを追加: {len(conversation_context)}文字")
                
                # 会話コンテキストを入力処理に統合
                if isinstance(processed, dict) and 'normalized' in processed:
                    processed['context'] = conversation_context
                    self.blackboard.write('input', processed)
            step_time = time.time() - step_start
            self.logger.info(f"[ステップ2] 会話履歴処理完了: {step_time:.2f}秒")
        
        # 3. RAG検索
        step_start = time.time()
        rag_result = self.rag_retriever.retrieve(input_text)
        self.blackboard.write('rag', rag_result)
        step_time = time.time() - step_start
        self.logger.info(f"[ステップ3] RAG検索完了: {step_time:.2f}秒")        # 4. 初期要約の実行（smart判定対応）
        should_use_summary = self._should_use_summary(input_text, rag_result)
        if self.use_summary and should_use_summary:
            step_start = time.time()
            initial_summary = f"ユーザー入力: {processed['normalized'] if isinstance(processed, dict) else processed}\n\n検索情報: {rag_result}"
            if self.use_memory and 'conversation_context' in self.blackboard.memory:
                initial_summary += f"\n\n会話コンテキスト: {self.blackboard.read('conversation_context')}"
            self.blackboard.write('initial_summary', initial_summary)
            step_time = time.time() - step_start
            self.logger.info(f"[ステップ4] 初期要約完了: {step_time:.2f}秒")
        elif self.use_summary:
            self.logger.info(f"[ステップ4] 要約スキップ（簡潔な入力のため）: {len(input_text)}文字")
        else:
            self.logger.info(f"[ステップ4] 要約機能無効")
            
        # 5. 反復サイクルの実行
        step_start = time.time()
        for i in range(self.iterations):
            iteration_start = time.time()
            await self._run_iteration(i)
            iteration_time = time.time() - iteration_start
            self.logger.info(f"[イテレーション{i+1}] 完了: {iteration_time:.2f}秒")
        step_time = time.time() - step_start
        self.logger.info(f"[ステップ5] 全反復サイクル完了: {step_time:.2f}秒")
            
        # 6. 最終要約とエージェント出力の収集
        step_start = time.time()
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
        step_time = time.time() - step_start
        self.logger.info(f"[ステップ6] 出力収集完了: {step_time:.2f}秒")
                
        # 7. 最終応答生成
        step_start = time.time()
        final_response = self.output_agent.generate(self.blackboard, entries)
        self.blackboard.write('final_response', final_response)
        step_time = time.time() - step_start
        self.logger.info(f"[ステップ7] 最終応答生成完了: {step_time:.2f}秒")
        
        # 8. 会話履歴に追加（使用する場合）
        if self.use_memory:
            step_start = time.time()
            self.conversation_memory.add_conversation_entry(
                user_input=input_text, 
                system_response=final_response
            )
            step_time = time.time() - step_start
            self.logger.info(f"[ステップ8] 会話記憶更新完了: {step_time:.2f}秒")
            self.logger.debug("会話記憶を更新しました")
        
        total_time = time.time() - total_start_time
        self.logger.info(f"=== 応答生成完了: {len(final_response)}文字 (合計時間: {total_time:.2f}秒) ===")
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
            "conversation_history": len(self.conversation_memory.conversation_history) if hasattr(self, 'conversation_memory') else 0        }
    
    def _is_memory_related_question(self, input_text: str) -> bool:
        """
        会話記憶に関連する質問かどうかを判定
        
        引数:
            input_text: ユーザー入力テキスト
            
        戻り値:
            会話記憶を使用すべき場合True
        """
        # 会話記憶関連のキーワード
        memory_keywords = [
            "前に", "さっき", "先ほど", "先程", "以前", "前回",
            "前の話", "前の質問", "前の会話", "その話", "それについて",
            "続き", "詳しく", "もう少し", "さらに", "追加",
            "覚えている", "記憶", "話した", "言った", "聞いた"
        ]
        
        input_lower = input_text.lower()
        
        # キーワードマッチング
        for keyword in memory_keywords:
            if keyword in input_lower:
                self.logger.debug(f"会話記憶関連キーワード検出: {keyword}")
                return True
        
        # 短い質問で代名詞が含まれる場合
        if len(input_text) < 50:
            pronouns = ["それ", "これ", "あれ", "その", "この", "あの"]
            for pronoun in pronouns:
                if pronoun in input_text:
                    self.logger.debug(f"代名詞検出: {pronoun}")
                    return True
        
        return False
    
    def _should_use_summary_iteration(self, agent_entries: List[Dict[str, Any]]) -> bool:
        """
        反復内での要約使用判定
        
        エージェント出力が短すぎる場合は要約を行わない
        
        引数:
            agent_entries: エージェント出力のリスト
            
        戻り値:
            要約を使用する場合True
        """
        if not agent_entries:
            return False
            
        # 全エージェント出力の合計文字数をチェック
        total_chars = sum(len(entry.get('text', '')) for entry in agent_entries)
        if total_chars < 64:  # 64文字未満は要約をスキップ
            self.logger.debug(f"エージェント出力が短すぎるため要約をスキップ: {total_chars}文字")
            return False
            
        # 全体の平均文字数をチェック
        avg_chars = total_chars / len(agent_entries)
        if avg_chars < 20:  # 平均20文字未満は要約をスキップ
            self.logger.debug(f"エージェント出力の平均が短すぎるため要約をスキップ: {avg_chars:.1f}文字")
            return False
            
        return True

    def _should_use_summary(self, user_input: str, rag_result: Optional[str]) -> bool:
        """
        要約を使用するかどうかの判定（スマート判定・厳格版）
        
        短い入力や単純な質問に対しては要約を無効化して高速化を図る
        
        引数:
            user_input: ユーザー入力テキスト
            rag_result: RAG検索結果
            
        戻り値:
            要約を使用する場合True
        """
        # 基本的に要約を使用する設定でない場合はFalse
        if not self.use_summary:
            return False
            
        # 入力が短すぎる場合はスキップ（閾値を64文字に厳格化）
        if len(user_input) < 64:
            self.logger.info(f"入力が短すぎるため要約をスキップ: {len(user_input)}文字 < 64文字")
            return False
            

        
        # RAG結果が空の場合で、入力も短い場合はスキップ（閾値を128文字に厳格化）
        if not rag_result and len(user_input) < 128:
            self.logger.info(f"RAG結果なし且つ入力が短いため要約をスキップ: {len(user_input)}文字 < 128文字")
            return False
            
        return True

    # ...existing code...
    
    async def shutdown(self):
        """
        DistributedSLMシステムの完全なシャットダウン処理
        
        全てのモジュールとリソースを適切に終了し、メモリを解放する
        途中で中断されることなく、安全にシステムを停止する
        """
        import time
        self.logger.info("=== DistributedSLMシャットダウン開始 ===")
        shutdown_start_time = time.time()
        
        try:
            # 1. 進行中のタスクの状態を確認
            if hasattr(self, '_current_generation_task'):
                self.logger.info("進行中の生成タスクを停止中...")
                try:
                    self._current_generation_task.cancel()
                    await asyncio.sleep(0.1)  # キャンセル処理の完了を待機
                except:
                    pass
            
            # 2. 各モジュールの個別シャットダウン
            shutdown_tasks = []
            
            # エージェントプールのシャットダウン
            if hasattr(self, 'agent_pool') and self.agent_pool:
                self.logger.info("エージェントプールをシャットダウン中...")
                try:
                    if hasattr(self.agent_pool, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.agent_pool, "AgentPool"))
                    elif hasattr(self.agent_pool, '__del__'):
                        self.agent_pool.__del__()
                except Exception as e:
                    self.logger.warning(f"エージェントプールシャットダウンエラー: {e}")
            
            # RAGリトリーバーのシャットダウン
            if hasattr(self, 'rag_retriever') and self.rag_retriever:
                self.logger.info("RAGリトリーバーをシャットダウン中...")
                try:
                    if hasattr(self.rag_retriever, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.rag_retriever, "RAGRetriever"))
                except Exception as e:
                    self.logger.warning(f"RAGリトリーバーシャットダウンエラー: {e}")
            
            # 要約エンジンのシャットダウン
            if hasattr(self, 'summary_engine') and self.summary_engine:
                self.logger.info("要約エンジンをシャットダウン中...")
                try:
                    if hasattr(self.summary_engine, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.summary_engine, "SummaryEngine"))
                except Exception as e:
                    self.logger.warning(f"要約エンジンシャットダウンエラー: {e}")
            
            # 出力エージェントのシャットダウン
            if hasattr(self, 'output_agent') and self.output_agent:
                self.logger.info("出力エージェントをシャットダウン中...")
                try:
                    if hasattr(self.output_agent, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.output_agent, "OutputAgent"))
                except Exception as e:
                    self.logger.warning(f"出力エージェントシャットダウンエラー: {e}")
            
            # 会話記憶のシャットダウン
            if hasattr(self, 'conversation_memory') and self.conversation_memory:
                self.logger.info("会話記憶をシャットダウン中...")
                try:
                    if hasattr(self.conversation_memory, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.conversation_memory, "ConversationMemory"))
                    else:
                        # 会話記憶の保存処理
                        self.conversation_memory.clear_memory()
                except Exception as e:
                    self.logger.warning(f"会話記憶シャットダウンエラー: {e}")
            
            # 黒板のシャットダウン
            if hasattr(self, 'blackboard') and self.blackboard:
                self.logger.info("黒板をシャットダウン中...")
                try:
                    if hasattr(self.blackboard, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.blackboard, "Blackboard"))
                    else:
                        # 黒板の最終クリア
                        self.blackboard.clear_current_turn()
                except Exception as e:
                    self.logger.warning(f"黒板シャットダウンエラー: {e}")
            
            # パフォーマンスモニターのシャットダウン
            if hasattr(self, 'performance') and self.performance:
                self.logger.info("パフォーマンスモニターをシャットダウン中...")
                try:
                    if hasattr(self.performance, 'shutdown'):
                        shutdown_tasks.append(self._safe_shutdown(self.performance, "PerformanceMonitor"))
                    else:
                        # パフォーマンス統計の最終記録
                        final_summary = self.performance.get_performance_summary()
                        self.logger.info(f"最終パフォーマンス統計: {final_summary}")
                        self.performance.clear()
                except Exception as e:
                    self.logger.warning(f"パフォーマンスモニターシャットダウンエラー: {e}")
            
            # 並列シャットダウンタスクの実行
            if shutdown_tasks:
                self.logger.info(f"並列シャットダウンタスクを実行中: {len(shutdown_tasks)}個")
                try:
                    # 10秒のタイムアウトで並列実行（個別タイムアウト3秒 + バッファ）
                    await asyncio.wait_for(asyncio.gather(*shutdown_tasks, return_exceptions=True), timeout=10.0)
                except asyncio.TimeoutError:
                    self.logger.warning("一部のシャットダウンタスクがタイムアウトしました")
                except Exception as e:
                    self.logger.error(f"並列シャットダウンエラー: {e}")
            
            # 3. モデルキャッシュのクリア
            self.logger.info("モデルキャッシュをクリア中...")
            try:
                from MurmurNet.modules.model_factory import clear_model_cache
                clear_model_cache()
            except Exception as e:
                self.logger.warning(f"モデルキャッシュクリアエラー: {e}")
            
            # 4. メモリの最終クリーンアップ
            self.logger.info("メモリクリーンアップを実行中...")
            try:
                # オブジェクト参照をクリア
                self.agent_pool = None
                self.rag_retriever = None
                self.summary_engine = None
                self.output_agent = None
                self.conversation_memory = None
                self.blackboard = None
                self.performance = None
                self.input_reception = None
                
                # ガベージコレクション
                import gc
                collected = gc.collect()
                self.logger.info(f"ガベージコレクション完了: {collected}個のオブジェクトを回収")
            except Exception as e:
                self.logger.warning(f"メモリクリーンアップエラー: {e}")
            
            # 5. シャットダウン完了
            shutdown_time = time.time() - shutdown_start_time
            self.logger.info(f"=== DistributedSLMシャットダウン完了 (時間: {shutdown_time:.2f}秒) ===")
            
        except Exception as e:
            self.logger.error(f"シャットダウン中に予期しないエラーが発生: {e}")
            # エラーが発生してもシャットダウンは継続
            import traceback
            self.logger.debug(traceback.format_exc())
        finally:
            # 最終的なログフラッシュ
            try:
                for handler in self.logger.handlers:
                    if hasattr(handler, 'flush'):                        handler.flush()
            except:
                pass
    
    async def _safe_shutdown(self, component, name: str, timeout: float = 3.0):
        """
        コンポーネントの安全なシャットダウンを実行
        
        引数:
            component: シャットダウン対象のコンポーネント
            name: コンポーネント名（ログ用）
            timeout: 個別コンポーネントのタイムアウト（秒）
        """
        try:
            if hasattr(component, 'shutdown'):
                if asyncio.iscoroutinefunction(component.shutdown):
                    # 非同期シャットダウンにタイムアウトを適用
                    await asyncio.wait_for(component.shutdown(), timeout=timeout)
                else:
                    # 同期シャットダウンをエクゼキューターで実行してタイムアウトを適用
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, component.shutdown),
                        timeout=timeout
                    )
                self.logger.debug(f"{name}のシャットダウン完了")
            else:
                self.logger.debug(f"{name}にshutdownメソッドがありません")
        except asyncio.TimeoutError:
            self.logger.warning(f"{name}のシャットダウンがタイムアウトしました（{timeout}秒）")
            # タイムアウトエラーは上位に伝播させない
        except Exception as e:
            self.logger.error(f"{name}のシャットダウンエラー: {e}")
            # エラーを上位に伝播させない（他のコンポーネントのシャットダウンを継続）
