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
from concurrent.futures import ThreadPoolExecutor
from murmurnet.modules.input_reception import InputReception
from murmurnet.modules.blackboard import Blackboard
from murmurnet.modules.agent_pool import AgentPoolManager
from murmurnet.modules.rag_retriever import RAGRetriever
from murmurnet.modules.output_agent import OutputAgent
from murmurnet.modules.summary_engine import SummaryEngine

class DistributedSLM:
    """
    分散創発型言語モデルメインクラス
    単一の関数呼び出しで高度な対話生成機能を提供するブラックボックス型モジュール
    
    特徴:
    - 複数の小規模モデルが協調的に動作
    - 黒板を通じた情報共有
    - 反復的な知識交換で知性を創発
    """
    def __init__(self, config: dict = None):
        """各モジュール初期化"""
        self.config = config or {}
        
        # 動作パラメータ
        self.num_agents = self.config.get('num_agents', 2)
        self.iterations = self.config.get('iterations', 1)  # 反復回数
        self.use_summary = self.config.get('use_summary', True)  # 要約を使うか
        self.use_parallel = self.config.get('use_parallel', False)  # 並列処理を使うか
        
        # 各モジュールの初期化
        self.blackboard = Blackboard(self.config)
        self.input_reception = InputReception(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.summary_engine = SummaryEngine(self.config)
        self.output_agent = OutputAgent(self.config)
        
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
        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            loop = asyncio.get_event_loop()
            futures = []
            for i in range(self.num_agents):
                # エージェントタスクを非同期実行
                future = loop.run_in_executor(
                    executor, 
                    self.agent_pool._agent_task, 
                    i
                )
                futures.append(future)
            # すべての完了を待機
            await asyncio.gather(*futures)
        
    async def generate(self, input_text: str) -> str:
        """
        外部公開API: 入力文字列から最終応答を生成
        引数：
            input_text: ユーザー入力文字列
        戻り値：
            生成された応答文字列
        """
        self.logger.info("Starting generation process")
        
        # 1. 入力受付・前処理
        processed = self.input_reception.process(input_text)
        self.blackboard.write('input', processed)
        
        # 2. RAG検索
        rag_result = self.rag_retriever.retrieve(input_text)
        self.blackboard.write('rag', rag_result)
        
        # 3. 初期要約の実行（オプション）
        if self.use_summary:
            initial_summary = f"ユーザー入力: {processed['normalized']}\n\n検索情報: {rag_result}"
            self.blackboard.write('initial_summary', initial_summary)
            
        # 4. 反復サイクルの実行
        for i in range(self.iterations):
            await self._run_iteration(i)
            
        # 5. 最終要約とエージェント出力の収集
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
                
        # 6. 最終応答生成
        final_response = self.output_agent.generate(self.blackboard, entries)
        self.blackboard.write('final_response', final_response)
        
        return final_response