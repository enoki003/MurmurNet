from llama_cpp import Llama
import os

# 要約エンジンの雛形
class SummaryEngine:
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

    def summarize(self, blackboard):
        # 黒板上の情報を要約
        input_text = blackboard.read('input')['normalized']
        rag_result = blackboard.read('rag')
        prompt_summary_1 = f"Summarize the following:\nInput: {input_text}\nRAG: {rag_result}"

        # summary_1 を生成
        messages_summary_1 = [
            {"role": "user", "content": prompt_summary_1}
        ]
        response_summary_1 = self.llm.create_chat_completion(messages=messages_summary_1, max_tokens=128)
        summary_1 = response_summary_1["choices"][0]["message"]["content"].strip()
        blackboard.write('summary_1', summary_1)

        # final_summary を生成
        prompt_final_summary = f"Integrate and refine the following summary:\nSummary_1: {summary_1}"
        messages_final_summary = [
            {"role": "user", "content": prompt_final_summary}
        ]
        response_final_summary = self.llm.create_chat_completion(messages=messages_final_summary, max_tokens=128)
        final_summary = response_final_summary["choices"][0]["message"]["content"].strip()
        blackboard.write('final_summary', final_summary)

        return final_summary
