from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
import os

# エージェントプール管理の雛形
class AgentPoolManager:
    def __init__(self, config, blackboard):
        self.config = config
        self.blackboard = blackboard
        # チャットテンプレートをconfigから取得
        chat_template = config.get('chat_template')
        # モデルパスをconfigから取得、なければデフォルト絶対パス
        model_path = config.get('model_path')
        if not model_path:
            # Windows用絶対パス
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf'))
        llama_kwargs = dict(
            model_path=model_path,
            n_ctx=1024,  # Context size updated to maximum supported by gemma3:1b
            n_threads=6,
            use_mmap=True,
            use_mlock=False,
            n_gpu_layers=0,
            seed=42,
            chat_format="gemma"
        )
        if chat_template:
            llama_kwargs['chat_template'] = chat_template
        self.llm = Llama(**llama_kwargs)

    def agent_task(self, blackboard, agent_id):
        # 各エージェントの処理をGemma-3 1Bで実行
        input_data = blackboard.read('input')

        # agentごとに視点・トーンを割り当て
        agent_roles = [
            {"role": "経済視点", "system": "あなたは経済の専門家です。経済的観点から具体的な意見を述べてください。", "temperature": 0.7},
            {"role": "倫理視点", "system": "あなたは倫理の専門家です。倫理的観点から具体的な意見を述べてください。", "temperature": 0.9},
            {"role": "技術視点", "system": "あなたは技術の専門家です。技術的観点から多言語で答えてください。", "temperature": 0.6},
            {"role": "社会視点", "system": "あなたは社会学の専門家です。社会的観点から多言語で答えてください。", "temperature": 0.8}
        ]
        role = agent_roles[agent_id % len(agent_roles)]
        # 入力言語を判定し、system promptを調整
        import re
        def detect_lang(text):
            if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text):
                return 'ja'
            elif re.search(r'[a-zA-Z]', text):
                return 'en'
            else:
                return 'ja'
        lang = detect_lang(input_data['normalized'])
        if lang == 'ja':
            system_prompt = role['system'] + " 必ず日本語で返答してください。"
        else:
            system_prompt = role['system'] + " Always respond in English."
        prompt = f"{system_prompt}\n\n問い: {input_data['normalized']}"

        # 各エージェントの入力データを黒板に記録
        blackboard.write(f'agent_{agent_id}_input', input_data['normalized'])

        messages = [
            {"role": "user", "content": prompt}
        ]
    
        response = self.llm.create_chat_completion(messages=messages, max_tokens=512, temperature=role['temperature'])
        blackboard.write(f'agent_{agent_id}_output', response["choices"][0]["message"]["content"].strip())

    def run_agents(self, blackboard):
        num_agents = self.config.get('num_agents', 2)
        for agent_id in range(num_agents):
            self.agent_task(blackboard, agent_id)