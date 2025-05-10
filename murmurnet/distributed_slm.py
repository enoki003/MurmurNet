#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 分散SLMシステム
~~~~~~~~~~~~~~~~~~~
複数の小型言語モデルを組み合わせた分散創発型アーキテクチャ
黒板設計・要約・RAGを統合した協調パターン

作者: Yuhi Sonoki
"""

# distributed_slm.py
import os
import yaml
import logging
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
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
    単一の関数呼び出しで高度な対話生成機能を提供するブラックボックス型モジュール
    
    特徴:
    - 複数の小規模モデルが協調的に動作
    - 黒板を通じた情報共有
    - 反復的な知識交換で知性を創発
    """
    def __init__(self, config: dict = None, blackboard=None):
        """
        各モジュール初期化
        
        引数:
            config: 設定辞書
            blackboard: Blackboardインスタンス（省略時は内部で作成）
        """
        self.config = config or {}
        
        # 動作パラメータ
        self.num_agents = self.config.get('num_agents', 2)
        self.iterations = self.config.get('iterations', 1)  # 反復回数
        self.use_summary = self.config.get('use_summary', True)  # 要約を使うか
        self.use_parallel = self.config.get('use_parallel', False)  # 並列処理を使うか
        self.use_memory = self.config.get('use_memory', True)  # 会話履歴を使うか
        
        # 各モジュールの初期化
        self.blackboard = blackboard if blackboard is not None else Blackboard(self.config)
        self.input_reception = InputReception(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.summary_engine = SummaryEngine(self.config)
        self.output_agent = OutputAgent(self.config)
        self.conversation_memory = ConversationMemory(self.config)
        
        # ロガー設定
        self._setup_logger()
        self.logger.info(f"Initialized with {self.num_agents} agents, {self.iterations} iterations")
        
    def _setup_logger(self):
        """ロガー初期化（内部メソッド）"""
        self.logger = logging.getLogger('DistributedSLM')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if not self.config.get('debug') else logging.DEBUG)

    async def _run_iteration(self, iteration: int) -> None:
        """単一の反復サイクルを実行（内部メソッド）"""
        self.logger.info(f"Starting iteration {iteration}")
        
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
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            self.blackboard.write(f'summary_{iteration}', summary)
            self.logger.debug(f"Created summary for iteration {iteration}")
    
    async def _run_agents_parallel(self) -> None:
        """エージェントを並列実行（内部メソッド）"""
        self.logger.info("Running agents in parallel")
        
        # スレッドプール内でエージェントタスクを実行するラッパー
        def run_agent_task(agent_id):
            try:
                # 共有モデルを使って実行（グローバルロックはagent_task内で使用）
                # すでに保護されているので各エージェントは安全に実行できる
                return self.agent_pool._agent_task(agent_id)
            except Exception as e:
                self.logger.error(f"Agent {agent_id} execution failed: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return f"エージェント{agent_id}は応答できませんでした"
        
        # 実行するエージェント数を調整（最大2つに制限）
        num_agents = self.num_agents
        max_parallel = 1  # 1つずつ処理するよう制限（安全のため）
        
        # イベントループを取得
        loop = asyncio.get_event_loop()
        
        try:
            # エージェントを1つずつ実行
            for i in range(num_agents):
                try:
                    # 同期関数を非同期実行
                    result = await loop.run_in_executor(None, run_agent_task, i)
                    if result:
                        self.blackboard.write(f'agent_{i}_output', result)
                    else:
                        self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は空の応答を返しました")
                except Exception as e:
                    self.logger.error(f"Agent {i} execution error: {str(e)}")
                    self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
                    
        except Exception as e:
            self.logger.error(f"Parallel execution error: {str(e)}")
            # エラーが発生した場合は逐次実行にフォールバック
            for i in range(self.num_agents):
                try:
                    result = self.agent_pool._agent_task(i)
                    self.blackboard.write(f'agent_{i}_output', result)
                except Exception as inner_e:
                    self.logger.error(f"Agent {i} fallback execution failed: {str(inner_e)}")
    
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
        self.logger.info("Starting generation process")
        
        # 新しいターンのために黒板をクリア（会話履歴は保持）
        self.blackboard.clear_current_turn()
        
        # 会話記憶に関連する質問かチェック
        if self.use_memory and self._is_memory_related_question(input_text):
            # 会話記憶から回答を取得
            memory_answer = self.conversation_memory.get_answer_for_question(input_text)
            if memory_answer:
                self.logger.info("Found answer from conversation memory")
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
                self.logger.debug(f"Added conversation context: {len(conversation_context)} chars")
                
                # 会話コンテキストを入力処理に統合
                if isinstance(processed, dict) and 'normalized' in processed:
                    processed['context'] = conversation_context
                    self.blackboard.write('input', processed)
        
        # 3. RAG検索
        rag_result = self.rag_retriever.retrieve(input_text)
        self.blackboard.write('rag', rag_result)
        
        # 4. 初期要約の実行（オプション）
        if self.use_summary:
            initial_summary = f"ユーザー入力: {processed['normalized']}\n\n検索情報: {rag_result}"
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
            self.logger.debug("Updated conversation memory")
        
        return final_response
        
    def reset_memory(self):
        """会話履歴をリセット"""
        if hasattr(self, 'conversation_memory'):
            self.conversation_memory.clear_memory()
            self.logger.info("Conversation memory cleared")