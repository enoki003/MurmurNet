from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
import os

# エージェントプール管理の雛形
class AgentPoolManager:
    def __init__(self, config, blackboard):
        self.config = config
        self.blackboard = blackboard
        model_path = os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=6,
            use_mmap=True,
            use_mlock=False,
            n_gpu_layers=0,
            seed=42,
            chat_format="gemma"
        )

    def agent_task(self, blackboard, agent_id):
        # 各エージェントの処理をGemma-3 1Bで実行
        input_data = blackboard.read('input')
        prompt = f"Agent {agent_id}, process the following input:\n{input_data['normalized']}"

        # 各エージェントの入力データを黒板に記録
        blackboard.write(f'agent_{agent_id}_input', input_data['normalized'])

        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.llm.create_chat_completion(messages=messages, max_tokens=128)
        blackboard.write(f'agent_{agent_id}_output', response["choices"][0]["message"]["content"].strip())

    def run_agents(self, blackboard):
        num_agents = self.config.get('num_agents', 2)
        for agent_id in range(num_agents):
            self.agent_task(blackboard, agent_id)
