import sys
import os
import unittest
import asyncio

# モジュール検索パスに `murmurnet` ディレクトリを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'murmurnet'))

# 既存のインポート
from murmurnet.distributed_slm import DistributedSLM

class TestDistributedSLM(unittest.TestCase):
    def setUp(self):
        # テスト用の設定を初期化
        self.config = {"num_agents": 2, "rag_top_k": 5}
        self.slm = DistributedSLM(self.config)

    def test_generate(self):
        # 非同期メソッドのテスト
        async def run_test():
            input_text = "AIは教育をどう変えると思う？"
            response = await self.slm.generate(input_text)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

        asyncio.run(run_test())

    def test_generate_responses(self):
        """
        複数の入力に対して応答が自然であるかを確認するテスト。
        """
        async def run_test():
            test_cases = [
                {"input": "AIは教育をどう変えると思う？", "expected_keywords": ["教育", "AI", "効率化"]},
                {"input": "気候変動について教えてください。", "expected_keywords": ["気候", "変動", "対策"]},
                {"input": "宇宙の起源は何ですか？", "expected_keywords": ["宇宙", "起源", "ビッグバン"]}
            ]

            for case in test_cases:
                response = await self.slm.generate(case["input"])
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)

                # 応答が期待されるキーワードを含むか確認
                for keyword in case["expected_keywords"]:
                    self.assertIn(keyword, response, f"応答に期待されるキーワード '{keyword}' が含まれていません。")

        asyncio.run(run_test())

if __name__ == "__main__":
    # unittestの実行結果を明示的に表示
    unittest.main(verbosity=2, argv=['first-arg-is-ignored'], exit=False)