# 入力受付・前処理用モジュール雛形
class InputReception:
    def __init__(self, config):
        self.config = config

    def process(self, input_text: str):
        # ここで正規化・トークナイズ・embedding取得などを行う（ダミー実装）
        return {'normalized': input_text}
