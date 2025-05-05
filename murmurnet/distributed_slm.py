# DistributedSLM本体と各モジュールのインターフェースをまとめて用意
import os
import yaml
from modules.input_reception import InputReception
from modules.blackboard import Blackboard
from modules.summary_engine import SummaryEngine
from modules.agent_pool import AgentPoolManager
from modules.rag_retriever import RAGRetriever
from modules.output_agent import OutputAgent
from modules.controller import Controller

class DistributedSLM:
    def __init__(self, config: dict = None):
        """コンストラクタ：各モジュール初期化"""
        self.config = config or {}
        self.num_agents = self.config.get('num_agents', 2)
        self.rag_top_k = self.config.get('rag_top_k', 5)
        self.internal_language = self.config.get('internal_language', 'en')  # 内部処理の言語
        self.output_language = self.config.get('output_language', 'ja')      # 出力言語
        self.prompt_config = self.load_prompt_config()
        self.input_reception = InputReception(self.config)
        self.blackboard = Blackboard(self.config)
        self.summary_engine = SummaryEngine(self.config)
        self.agent_pool = AgentPoolManager(self.config, self.blackboard)
        self.rag_retriever = RAGRetriever(self.config)
        self.output_agent = OutputAgent(self.config)
        self.controller = Controller(self.config)
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
    
        # RAG結果を取得して黒板に書き込む  
        self.logger.info("Retrieving RAG results")  
        rag_result = self.rag_retriever.retrieve(input_text)  
        if rag_result is None:  
            self.logger.warning("RAG result is None")  
            rag_result = "関連情報が見つかりませんでした"  
        self.blackboard.write('rag', rag_result)  
    
        # エージェントプールを実行  
        self.logger.info("Running agent pool")  
        self.agent_pool.run_agents(self.blackboard)  
    
        # agent_outputsの収集  
        agent_outputs = []  
        for i in range(num_agents):  
            out = self.blackboard.read(f'agent_{i}_output')  
            if out:  
                agent_outputs.append(out)  
    
        # 最終応答の生成  
        if agent_outputs:  
            final_response = "\n".join(agent_outputs)  
        else:  
            final_response = "申し訳ありませんが、適切な応答を生成できませんでした。"  
    
        self.logger.info("Final response generated")  
        return final_response

