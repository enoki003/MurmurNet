# distributed_slm.py
import os
import yaml
from modules.input_reception import InputReception
from modules.blackboard import Blackboard
from modules.agent_pool import AgentPoolManager
from modules.rag_retriever import RAGRetriever
from modules.output_agent import OutputAgent

class DistributedSLM:
    def __init__(self, config: dict = None, blackboard=None):
        """
        各モジュール初期化
        
        引数:
            config: 設定辞書
            blackboard: Blackboardインスタンス（省略時は内部で作成）
        """
        self.config = config or {}
        self.num_agents = self.config.get('num_agents', 2)
        self.iterations = self.config.get('iterations', 1)  # 反復回数
        self.use_summary = self.config.get('use_summary', True)  # 要約を使うかどうか
        self.prompt_config = self.load_prompt_config()
        self.input_reception = InputReception(self.config)
        self.blackboard = blackboard if blackboard is not None else Blackboard(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.output_agent = OutputAgent(self.config)
        self.logger = self.setup_logger()

    def setup_logger(self):
        import logging
        logger = logging.getLogger('DistributedSLM')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_prompt_config(self):
        try:
            with open('prompt_config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    async def generate(self, input_text: str, num_agents: int = None, max_length: int = 512) -> str:
        """
        入力文字列から最終応答を生成
        ・エンドツーエンド非同期呼び出し
        """
        self.logger.info("Starting generation process")

        # 1) パラメータ反映
        num_agents = num_agents or self.num_agents

        # 2) 黒板に初期入力を書き込む
        self.logger.info("Writing input to blackboard")
        processed = self.input_reception.process(input_text)
        self.blackboard.write('input', processed)

        # 3) RAG結果を取得して黒板に書き込む
        self.logger.info("Retrieving RAG results")
        rag_result = self.rag_retriever.retrieve(input_text) or "関連情報が見つかりませんでした"
        self.blackboard.write('rag', rag_result)

        # 4) エージェントプールを実行
        self.logger.info("Running agent pool")
        self.agent_pool.run_agents(self.blackboard)

        # 5) 黒板から各 agent_i_output を収集
        entries = []
        for i in range(num_agents):
            out = self.blackboard.read(f"agent_{i}_output")
            if out:
                entries.append({"agent": i, "text": out})
        self.logger.info(f"Collected entries: {[e['agent'] for e in entries]}")

        # 6) OutputAgent で最終レスポンス生成
        final_response = self.output_agent.generate(self.blackboard, entries)

        self.logger.info("Final response generated")
        return final_response