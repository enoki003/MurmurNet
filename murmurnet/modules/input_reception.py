import re
from sentence_transformers import SentenceTransformer

class InputReception:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 埋め込みモデル

    def process(self, input_text: str):
        # 正規化
        normalized_text = re.sub(r'[^\w\s]', '', input_text).lower()
        # トークナイズ
        tokens = normalized_text.split()
        # 埋め込み取得
        embedding = self.model.encode(normalized_text)
        return {'normalized': normalized_text, 'tokens': tokens, 'embedding': embedding}
