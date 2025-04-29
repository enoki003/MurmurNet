from llama_cpp import Llama
import os

# 出力生成エージェント雛形
class OutputAgent:
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,           # コンテキスト長
            n_threads=6,          # CPUコア数（必要に応じて調整）
            use_mmap=True,        # メモリ節約
            use_mlock=False,      # WindowsなのでFalse（LinuxならTrue推奨）
            n_gpu_layers=0,       # GPUなし
            seed=42,              # 乱数固定
            chat_format="gemma"   # チャット形式を明示
        )

    def generate(self, blackboard):
        # チャット形式でプロンプトを渡す
        messages = [
            {"role": "user", "content": getattr(blackboard, 'prompt', 'こんにちは、調子はどう？')}
        ]
        response = self.llm.create_chat_completion(messages=messages, max_tokens=128)
        return response["choices"][0]["message"]["content"].strip()
