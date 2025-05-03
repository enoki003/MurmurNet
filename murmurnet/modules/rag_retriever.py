import numpy as np

# RAGリトリーバーモジュール雛形
class RAGRetriever:
    def __init__(self, config):
        self.config = config
        self.mode = config.get('rag_mode', 'dummy')  # 'dummy' or 'external'
        self.knowledge_base = [
            {"text": "AIは人工知能の略であり、機械が人間のように学習する技術です。", "embedding": np.random.rand(384)},
            {"text": "黒板型アーキテクチャは、複数のエージェントが協調して問題を解決するための枠組みです。", "embedding": np.random.rand(384)},
            {"text": "RAGはRetrieval-Augmented Generationの略で、検索と生成を組み合わせた技術です。", "embedding": np.random.rand(384)}
        ]
        self.score_threshold = config.get('rag_score_threshold', 0.5)
        self.top_k = config.get('rag_top_k', 1)
        self.debug = config.get('debug', False)

    def retrieve(self, input_data):
        if self.mode == 'dummy':
            if isinstance(input_data, str):
                input_embedding = np.random.rand(384)  # 仮のエンベディング生成
            else:
                input_embedding = input_data['embedding']
            def cosine_similarity(vec1, vec2):
                return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            scored_chunks = []
            for kb in self.knowledge_base:
                score = cosine_similarity(input_embedding, kb['embedding'])
                scored_chunks.append({"text": kb['text'], "score": score})
            # スコアで降順ソート
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            # 閾値以上のものをtop_k件取得
            filtered = [c for c in scored_chunks if c['score'] >= self.score_threshold][:self.top_k]
            if self.debug:
                print(f"[RAG DEBUG] 取得チャンク数: {len(filtered)}")
                for i, c in enumerate(filtered):
                    print(f"  chunk{i}: score={c['score']:.3f}, text={c['text']}")
                if filtered:
                    print(f"[RAG DEBUG] 最終プロンプト用: {[c['text'] for c in filtered]}")
            if not filtered:
                return "関連情報が見つかりませんでした"
            # 1件ならstr, 複数なら連結
            if len(filtered) == 1:
                return filtered[0]['text']
            else:
                return '\n'.join([c['text'] for c in filtered])
        elif self.mode == 'external':
            return '[外部DB参照: 実装予定]'
        else:
            return '[RAGモード未設定]'

    def retrieve(self, query):
        result, score = self._search_with_score(query)
        if score < 0.7:
            return "関連知識が見つかりませんでした"
        return result

    def _search_with_score(self, query):
        if "AI" in query or "仕事" in query or "雇用" in query or "労働" in query:
            return "AIと雇用・労働に関する知識チャンク（例）", 1.0
        return "黒板型アーキテクチャは、複数のエージェントが協調して問題を解決するための枠組みです。", 0.5
