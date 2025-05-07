# distributed_slm.py
import os
import yaml
import logging
from murmurnet.modules.input_reception import InputReception
from murmurnet.modules.blackboard import Blackboard
from murmurnet.modules.agent_pool import AgentPoolManager
from murmurnet.modules.rag_retriever import RAGRetriever
from murmurnet.modules.output_agent import OutputAgent

class DistributedSLM:
    """
    分散創発型言語モデルメインクラス
    単一の関数呼び出しで高度な対話生成機能を提供するブラックボックス型モジュール
    """
    def __init__(self, config: dict = None):
        """各モジュール初期化"""
        self.config = config or {}
        self.blackboard = Blackboard(self.config)
        self.input_reception = InputReception(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.output_agent = OutputAgent(self.config)
        self._setup_logger()

    def _setup_logger(self):
        """ロガー初期化（内部メソッド）"""
        self.logger = logging.getLogger('DistributedSLM')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if not self.config.get('debug') else logging.DEBUG)

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
        
        # 3. エージェント実行
        self.agent_pool.run_agents(self.blackboard)
        
        # 4. エージェント出力収集
        num_agents = self.config.get('num_agents', 2)
        entries = []
        for i in range(num_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            if agent_output:
                entries.append({"agent": i, "text": agent_output})
                
        # 5. 最終応答生成
        final_response = self.output_agent.generate(self.blackboard, entries)
        
        return final_response