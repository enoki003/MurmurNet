from llama_cpp import Llama
import os
import re

class OutputAgent:
    def __init__(self, config):
        self.config = config
        chat_template = config.get('chat_template')
        model_path = config.get('model_path', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')))
        rag_db_path = config.get('rag_db_path')
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
        self.rag_db_path = rag_db_path

    def generate(self, blackboard):
        # 1ターン分のみの明示的プロンプト構成
        user_input = blackboard.read('input')
        if isinstance(user_input, dict):
            user_input = user_input.get('normalized', str(user_input))
        rag_info = blackboard.read('rag')

        # 黒板から英語の内部処理結果を取得
        agent_outputs = []
        for k, v in blackboard.memory.items():
            if k.startswith('agent_') and k.endswith('_output'):
                agent_outputs.append(v)

        english_content = "\n".join(agent_outputs)

        # 日本語での最終回答を生成するプロンプト
        system_prompt = "あなたは翻訳者兼要約者です。以下の英語の内容を日本語に翻訳し、簡潔にまとめてください。必ず日本語で回答してください。"

        prompt = f"{system_prompt}\n\n元の質問: {user_input}\n\n英語の内容:\n{english_content}\n\n日本語での回答:"

        messages = [
            {"role": "user", "content": prompt}
        ]

        # max_tokensの増加
        response = self.llm.create_chat_completion(messages=messages, max_tokens=1024, temperature=0.7)  # Adjusted temperature for more deterministic output
        answer = response["choices"][0]["message"]["content"].strip()

        # 応答の完全性チェック機能の追加
        def check_response_completeness(text):
            # 文が途中で切れていないか確認
            if text.endswith(('。', '！', '？', '」', '）', ')', '.', '!', '?', '"')):
                return True
            return False

        # 応答生成後
        if not check_response_completeness(answer):
            # 応答が不完全な場合、続きを生成
            continuation_prompt = f"以下の文章の続きを短く完結させてください（日本語で）：\n{answer}"
            continuation_messages = [{"role": "user", "content": continuation_prompt}]
            continuation_response = self.llm.create_chat_completion(messages=continuation_messages, max_tokens=512)
            continuation = continuation_response["choices"][0]["message"]["content"].strip()
            answer = answer + continuation

        return answer
