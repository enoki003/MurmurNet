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
            n_ctx=32768,  # Context size updated to maximum supported by gemma3:1b
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

        # エージェントの役割を英語で定義
        agent_roles = [
            {"role": "Economic Perspective", "system": "You are an economics expert. Provide specific opinions from an economic perspective.", "temperature": 0.7},
            {"role": "Ethical Perspective", "system": "You are an ethics expert. Provide specific opinions from an ethical perspective.", "temperature": 0.9},
            {"role": "Technical Perspective", "system": "You are a technology expert. Provide specific opinions from a technical perspective.", "temperature": 0.6},
            {"role": "Social Perspective", "system": "You are a sociology expert. Provide specific opinions from a social perspective.", "temperature": 0.8}
        ]

        role = agent_roles[agent_id % len(agent_roles)]

        # 常に英語で内部処理を行うよう指示
        system_prompt = role['system'] + " Always respond in English for internal processing."
        prompt = f"{system_prompt}\n\nQuestion: {input_data['normalized']}"

        # 各エージェントの入力データを黒板に記録
        blackboard.write(f'agent_{agent_id}_input', input_data['normalized'])

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm.create_chat_completion(messages=messages, max_tokens=512, temperature=role['temperature'])
        output = response["choices"][0]["message"]["content"].strip()

        # 出力をトリムしてフォーマットを整える
        if len(output) > 500:
            output = output[:500] + "... (truncated)"

        blackboard.write(f'agent_{agent_id}_output', output)

    def run_agents(self, blackboard):
        num_agents = self.config.get('num_agents', 2)
        for agent_id in range(num_agents):
            self.agent_task(blackboard, agent_id)
