# DistributedSLM本体と各モジュールのインターフェースをまとめて用意
import os
import yaml
from modules.input_reception import InputReception
from modules.blackboard import Blackboard
from modules.summary_engine import SummaryEngine
from modules.agent_pool import AgentPoolManager
from modules.rag_retriever import RAGRetriever
from modules.output_agent import OutputAgent

class DistributedSLM:
    def __init__(self, config: dict = None):
        """コンストラクタ：各モジュール初期化"""
        self.config = config or {}
        self.num_agents = self.config.get('num_agents', 2)
        self.rag_top_k = self.config.get('rag_top_k', 5)
        self.prompt_config = self.load_prompt_config()
        self.input_reception = InputReception(self.config)
        self.blackboard = Blackboard(self.config)
        self.summary_engine = SummaryEngine(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.output_agent = OutputAgent(self.config)
        self.logger = self.setup_logger()

    def setup_logger(self):
        """ログ機能を設定"""
        import logging
        logger = logging.getLogger('DistributedSLM')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_prompt_config(self):
        """外部プロンプト設定を読み込む"""
        try:
            with open('prompt_config.yaml', 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}

    async def generate(self, input_text: str, num_agents: int = None, roles: list = None, instructions: list = None, max_length: int = 512) -> str:
        """
        入力文字列から最終応答を生成
        ・エンドツーエンド非同期呼び出し
        """
        self.logger.info("Starting generation process")

        # 設定を上書き
        num_agents = num_agents or self.num_agents
        roles = roles or self.prompt_config.get('roles', [])
        instructions = instructions or self.prompt_config.get('instructions', [])

        # 黒板に初期入力を書き込む
        self.logger.info("Writing input to blackboard")
        self.blackboard.write('input', {'normalized': input_text})

        # 要約エンジンで要約を生成
        self.logger.info("Generating summary")
        summary = self.summary_engine.summarize(self.blackboard)
        self.blackboard.write('summary', summary)

        # RAGRetrieverでデータを取得し、黒板に書き込む
        self.logger.info("Retrieving data using RAGRetriever")
        rag_data = self.rag_retriever.retrieve(input_text)
        if not rag_data:
            self.logger.warning("No data found by RAGRetriever")
            rag_data = "知識が見つかりませんでした"
        self.blackboard.write('rag', rag_data)

        # エージェントプールを実行
        self.logger.info("Running agent pool")
        self.agent_pool.run_agents(self.blackboard)

        # 最終応答を生成
        self.logger.info("Generating final response")
        final_response = self.output_agent.generate(self.blackboard)

        self.logger.info("Generation process completed")
        return final_response
