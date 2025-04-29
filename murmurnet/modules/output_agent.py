from llama_cpp import Llama
import os

class OutputAgent:
    def __init__(self, config):
        self.config = config
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

    def generate(self, blackboard):
        # 黒板から情報を取得
        final_summary = blackboard.read('final_summary')  # 修正: get を read に変更
        agent_outputs = blackboard.read('agent_outputs')

        # プロンプトを作成
        prompt = f"Final Summary: {final_summary}\nAgent Outputs: {agent_outputs}"
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Llama モデルで応答を生成
        response = self.llm.create_chat_completion(messages=messages, max_tokens=128)
        return response["choices"][0]["message"]["content"].strip()
