from llama_cpp import Llama
import os
import re

class OutputAgent:
    def __init__(self, config):
        self.config = config
        chat_template = config.get('chat_template')
        model_path = config.get('model_path')
        if not model_path:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf'))
        llama_kwargs = dict(
            model_path=model_path,
            n_ctx=4096,
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

    def generate(self, blackboard):
        # 1ターン分のみの明示的プロンプト構成
        user_input = blackboard.read('input')
        if isinstance(user_input, dict):
            user_input = user_input.get('normalized', str(user_input))
        rag_info = blackboard.read('rag')
        # 入力言語を自動判定し、system promptで指示
        def detect_lang(text):
            if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text):
                return 'ja'
            elif re.search(r'[a-zA-Z]', text):
                return 'en'
            else:
                return 'ja'
        lang = detect_lang(user_input)
        if lang == 'ja':
            system_prompt = "あなたはユーザーの入力言語（日本語または英語）に合わせて、同じ言語で丁寧に返答する多言語アシスタントです。必ず日本語で返答してください。"
        else:
            system_prompt = "You are a multilingual assistant. Always respond in English."
        prompt = f"{system_prompt}\n\n問い: {user_input}\n参考情報: {rag_info}\n\n答え:"
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.llm.create_chat_completion(messages=messages, max_tokens=512)
        answer = response["choices"][0]["message"]["content"].strip()
        # fallback: 決まり文句ならsummaryやagent_outputsで再試行
        if answer.strip() in ["ごめんなさい、その質問には答えられませんでした。", "申し訳ありませんが、その質問にはお答えできません。"]:
            final_summary = blackboard.read('final_summary')
            agent_outputs = blackboard.read('agent_outputs')
            if agent_outputs is None:
                agent_outputs = []
                for k, v in blackboard.memory.items():
                    if k.startswith('agent_') and k.endswith('_output'):
                        agent_outputs.append(f"{k}: {v}")
                agent_outputs = '\n'.join(agent_outputs) if agent_outputs else '[No agent outputs]'
            fallback_prompt = f"追加情報: {final_summary}\nエージェント出力: {agent_outputs}\nこの情報も参考にして再度答えてください。"
            fallback_messages = [
                {"role": "user", "content": fallback_prompt}
            ]
            fallback_response = self.llm.create_chat_completion(messages=fallback_messages, max_tokens=512)
            answer = fallback_response["choices"][0]["message"]["content"].strip()
        return answer
