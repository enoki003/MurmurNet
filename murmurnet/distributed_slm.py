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

    async def generate(self, input_text: str, num_agents: int = None, roles: list = None, instructions: list = None, max_length: int = 512, use_multi_stage: bool = False) -> str:
        """
        入力文字列から最終応答を生成
        ・エンドツーエンド非同期呼び出し
        """
        self.logger.info("Starting generation process")

        # 入力テキストから役割を抽出
        import re
        role_match = re.search(r'役割・(\S+)', input_text)
        role = role_match.group(1) if role_match else None

        # 役割の部分を入力テキストから削除
        if role_match:
            input_text = re.sub(r'役割・\S+', '', input_text).strip()

        # 設定を上書き
        num_agents = num_agents or self.num_agents
        roles = roles or self.prompt_config.get('roles', [])
        instructions = instructions or self.prompt_config.get('instructions', [])

        # 黒板に初期入力と役割を書き込む
        self.logger.info("Writing input to blackboard")
        self.blackboard.write('input', {'normalized': input_text, 'role': role})

        if use_multi_stage or role:
            # 新しいマルチステージフロー
            await self.execute_multi_stage_flow()
            final_response = self.blackboard.read('final_response')
        else:
            # 従来のフロー
            self.logger.info("Running agent pool")
            self.agent_pool.run_agents(self.blackboard)

            # agent_outputsの収集
            agent_outputs = []
            for i in range(num_agents or self.num_agents):
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

    async def execute_multi_stage_flow(self):
        # 1. 哲学者としての回答生成
        await self.generate_philosopher_response()

        # 2. RAGによる知識補充
        await self.augment_with_rag()

        # 3. 誤り修正エージェント
        await self.correct_errors()

        # 4. 根拠追記エージェント
        await self.add_evidence()

        # 5. 批判・反論エージェント
        await self.add_critique()

        # 6. 要約と最終回答生成
        await self.generate_final_summary()

    async def generate_philosopher_response(self):
        input_data = self.blackboard.read('input')
        question = input_data['normalized']
        role = input_data.get('role', '哲学者')  # デフォルトは哲学者

        # 哲学者としてのシステムプロンプト
        system_prompt = f"あなたは{role}です。{role}の観点から以下の質問に答えてください。"

        # LLMを使用して回答を生成
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        response = self.output_agent.llm.create_chat_completion(messages=messages)
        philosopher_response = response["choices"][0]["message"]["content"].strip()

        # 黒板に書き込む
        self.blackboard.write('philosopher_response', philosopher_response)

    async def augment_with_rag(self):
        input_data = self.blackboard.read('input')
        question = input_data['normalized']

        # RAGを使用して関連知識を取得
        rag_result = self.rag_retriever.retrieve(question)

        # 黒板に書き込む
        self.blackboard.write('rag_knowledge', rag_result)

    async def correct_errors(self):
        philosopher_response = self.blackboard.read('philosopher_response')
        rag_knowledge = self.blackboard.read('rag_knowledge')

        # 誤り修正のためのプロンプト
        prompt = f"""
        以下の哲学者の回答と関連知識を比較し、哲学者の回答に誤りがあれば修正してください。

        哲学者の回答:
        {philosopher_response}

        関連知識:
        {rag_knowledge}
        """

        # LLMを使用して修正を生成
        messages = [{"role": "user", "content": prompt}]
        response = self.output_agent.llm.create_chat_completion(messages=messages)
        corrections = response["choices"][0]["message"]["content"].strip()

        # 黒板に書き込む
        self.blackboard.write('corrections', corrections)

    async def add_evidence(self):
        corrections = self.blackboard.read('corrections')

        # 根拠追記のためのプロンプト
        prompt = f"""
        以下の修正内容に対する根拠を追加してください。学術的な引用や具体的な事例を含めてください。

        修正内容:
        {corrections}
        """

        # LLMを使用して根拠を生成
        messages = [{"role": "user", "content": prompt}]
        response = self.output_agent.llm.create_chat_completion(messages=messages)
        evidence = response["choices"][0]["message"]["content"].strip()

        # 黒板に書き込む
        self.blackboard.write('evidence', evidence)

    async def add_critique(self):
        corrections = self.blackboard.read('corrections')
        evidence = self.blackboard.read('evidence')

        # 批判・反論のためのプロンプト
        prompt = f"""
        以下の修正内容と根拠に対して批判的な視点から反論してください。

        修正内容:
        {corrections}

        根拠:
        {evidence}
        """

        # LLMを使用して反論を生成
        messages = [{"role": "user", "content": prompt}]
        response = self.output_agent.llm.create_chat_completion(messages=messages)
        critique = response["choices"][0]["message"]["content"].strip()

        # 黒板に書き込む
        self.blackboard.write('critique', critique)

    async def generate_final_summary(self):
        # 黒板から全情報を取得
        input_data = self.blackboard.read('input')
        question = input_data['normalized']
        philosopher_response = self.blackboard.read('philosopher_response')
        rag_knowledge = self.blackboard.read('rag_knowledge')
        corrections = self.blackboard.read('corrections')
        evidence = self.blackboard.read('evidence')
        critique = self.blackboard.read('critique')

        # 英語の内容を集約
        english_content = f"""
        Question: {question}

        Philosopher's response:
        {philosopher_response}

        Related knowledge:
        {rag_knowledge}

        Corrections:
        {corrections}

        Evidence:
        {evidence}

        Critique:
        {critique}
        """

        # 日本語での最終回答生成
        system_prompt = "あなたは翻訳者兼要約者です。以下の英語の内容を日本語に翻訳し、簡潔にまとめてください。必ず日本語で回答してください。"

        prompt = f"{system_prompt}\n\n元の質問: {question}\n\n英語の内容:\n{english_content}\n\n日本語での回答:"

        # LLMを使用して最終回答を生成（日本語）
        messages = [{"role": "user", "content": prompt}]
        response = self.output_agent.llm.create_chat_completion(messages=messages, max_tokens=768)
        final_response = response["choices"][0]["message"]["content"].strip()

        # 応答の完全性チェック
        if not self.check_response_completeness(final_response):
            continuation_prompt = f"以下の文章の続きを短く完結させてください（日本語で）：\n{final_response}"
            continuation_messages = [{"role": "user", "content": continuation_prompt}]
            continuation_response = self.output_agent.llm.create_chat_completion(messages=continuation_messages, max_tokens=100)
            continuation = continuation_response["choices"][0]["message"]["content"].strip()
            final_response = final_response + continuation

        # 黒板に書き込む（日本語）
        self.blackboard.write('final_response', final_response)
