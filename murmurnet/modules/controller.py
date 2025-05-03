import numpy as np
import os
import json
from collections import Counter

def calc_entropy(text):
    # 文字単位エントロピー
    counter = Counter(text)
    total = len(text)
    probs = [c / total for c in counter.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy

def calc_vocab_score(text):
    # 語彙の多様性スコア（ユニーク単語数/総単語数）
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def calc_duplication(texts):
    # 発言間の重複度（Jaccard類似度の平均）
    sets = [set(t.split()) for t in texts if t]
    if len(sets) < 2:
        return 0.0
    scores = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            inter = sets[i] & sets[j]
            union = sets[i] | sets[j]
            if union:
                scores.append(len(inter)/len(union))
    return np.mean(scores) if scores else 0.0

class Controller:
    def __init__(self, config=None):
        self.config = config or {}

    def evaluate_outputs(self, agent_outputs):
        # agent_outputs: List[str]
        results = []
        for out in agent_outputs:
            entropy = calc_entropy(out)
            vocab_score = calc_vocab_score(out)
            results.append({
                'text': out,
                'embedding_entropy': entropy,
                'vocab_score': vocab_score
            })
        duplication = calc_duplication(agent_outputs)
        return results, duplication

    def select_best(self, results):
        # 情報量（エントロピー×語彙スコア）最大の発言を選択
        best = max(results, key=lambda r: r['embedding_entropy'] * r['vocab_score'])
        return best['text']

    def unify_style(self, text):
        # 文体統一・再言語化（ここではダミーでそのまま返す）
        return text

    def generate_final_output(self, agent_outputs):
        # agent_outputsを統合・要約して最終応答を生成
        from llama_cpp import Llama
        config = self.config or {}
        chat_template = config.get('chat_template')
        model_path = config.get('model_path')
        if not model_path:
            import os
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
        llm = Llama(**llama_kwargs)
        # agent_outputsをまとめて要約・統合
        opinions = "\n---\n".join(agent_outputs)
        prompt = f"以下は複数のエージェントによる意見です。これらを統合・要約し、質問に対して最適な答えを日本語で出してください。\n\n意見一覧:\n{opinions}\n\n統合回答:"
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = llm.create_chat_completion(messages=messages, max_tokens=512)
        final_response = response["choices"][0]["message"]["content"].strip()
        # 途中終了対策
        def is_complete(text):
            return text.strip().endswith(('。', '！', '？', '.', '!', '?'))
        if not is_complete(final_response):
            # 2回目は温度を上げて再生成
            response2 = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=1.0)
            alt_response = response2["choices"][0]["message"]["content"].strip()
            if is_complete(alt_response):
                final_response = alt_response
        # agent_scores, duplicationは従来通り
        results, duplication = self.evaluate_outputs(agent_outputs)
        return final_response, results, duplication

    def _select_best(self, agent_outputs, temperature=0.7):
        # agent_outputsを統合・要約して最適な答えを返す
        from llama_cpp import Llama
        config = self.config or {}
        chat_template = config.get('chat_template')
        model_path = config.get('model_path')
        if not model_path:
            import os
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
        llm = Llama(**llama_kwargs)
        opinions = "\n---\n".join(agent_outputs)
        prompt = f"以下は複数のエージェントによる意見です。これらを統合・要約し、質問に対して最適な答えを日本語で出してください。\n\n意見一覧:\n{opinions}\n\n統合回答:"
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=temperature)
        final_response = response["choices"][0]["message"]["content"].strip()
        def is_complete(text):
            return text.strip().endswith(('。', '！', '？', '.', '!', '?'))
        if not is_complete(final_response):
            response2 = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=1.0)
            alt_response = response2["choices"][0]["message"]["content"].strip()
            if is_complete(alt_response):
                final_response = alt_response
        return final_response
