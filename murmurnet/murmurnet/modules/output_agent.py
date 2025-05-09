# output_agent.py
from llama_cpp import Llama
import os
import re

class OutputAgent:
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
        self.config = config or {}

    def generate(self, blackboard, entries: list) -> str:
        # 1) 入力と RAG を取得
        inp = blackboard.read('input')
        user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
        rag = blackboard.read('rag')

        # 2) システムプロンプト設定
        def detect_lang(text):
            if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text):
                return 'ja'
            if re.search(r'[A-Za-z]', text):
                return 'en'
            return 'ja'
        lang = detect_lang(user_input)
        sys_p = (
            "あなたは日本語話者向けの多言語アシスタントです。必ず日本語で返答してください。"
            if lang == 'ja' else
            "You are a general-purpose AI assistant. Answer in English."
        )

        # 3) プロンプト組み立て
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": (
                f"問い: {user_input}\n\n"
                f"参考情報: {rag}\n\n"
                "以下の各エージェントの意見を参考に最適な回答を作成してください。"
            )}
        ]
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        # llama_cpp returns a dict, so access choices via key
        if isinstance(resp, dict):
            answer = resp['choices'][0]['message']['content'].strip()
        else:
            answer = resp.choices[0].message.content.strip()


        return answer
