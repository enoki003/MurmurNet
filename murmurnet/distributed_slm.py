# DistributedSLM本体と各モジュールのインターフェースをまとめて用意
import os
from modules.input_reception import InputReception
from modules.blackboard import Blackboard
from modules.summary_engine import SummaryEngine
from modules.agent_pool import AgentPoolManager
from modules.rag_retriever import RAGRetriever
from modules.output_agent import OutputAgent

class DistributedSLM:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.input_reception = InputReception(self.config)
        self.blackboard = Blackboard(self.config)
        self.summary_engine = SummaryEngine(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.output_agent = OutputAgent(self.config)

    async def generate(self, input_text: str) -> str:
        # 入力処理
        input_data = self.input_reception.process(input_text)
        # 黒板に書き込み
        self.blackboard.write('input', input_data)
        # RAG検索
        rag_result = self.rag_retriever.retrieve(input_data)
        self.blackboard.write('rag', rag_result)
        # エージェント実行
        self.agent_pool.run_agents(self.blackboard)
        # 要約
        summary = self.summary_engine.summarize(self.blackboard)
        self.blackboard.write('summary', summary)
        # 出力生成
        response = self.output_agent.generate(self.blackboard)
        return response
