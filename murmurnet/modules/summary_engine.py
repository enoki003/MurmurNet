# summary_engine.py
from llama_cpp import Llama
import os

class SummaryEngine:
    def __init__(self, config: dict = None):
        model_path = config.get('model_path') or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')
        )
        llama_kwargs = dict(
            model_path=model_path,
            n_ctx=32768,
            n_threads=6,
            use_mmap=True,
            use_mlock=False,
            n_gpu_layers=0,
            seed=42,
            chat_format="gemma"
        )
        chat_template = config.get('chat_template')
        if chat_template:
            llama_kwargs['chat_template'] = chat_template

        self.llm = Llama(**llama_kwargs)

    def summarize_blackboard(self, entries: list) -> str:
        # entries: List[{'agent': id, 'text': str}]
        combined = "\n\n".join(e['text'] for e in entries)
        prompt = (
            "以下の各エージェントの出力を統合・要約してください。\n\n" + combined
        )
        resp = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return resp.choices[0].message.content.strip()
