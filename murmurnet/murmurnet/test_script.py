# テスト用スクリプト
# MurmurNetの各モジュールの動作確認用

import sys
import os
import logging

# ログ設定
logging.basicConfig(
    level=logging.WARNING,  # INFOからWARNINGに変更
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("test_log.txt"),
        logging.StreamHandler()
    ]
)

sys.path.append(os.path.dirname(__file__))
from modules import agent_pool, blackboard, input_reception, output_agent, rag_retriever, summary_engine

def main():
    logging.info("MurmurNet テスト開始")
    config = {
        "num_agents": 2,
        "rag_mode": "zim",
        "zim_path": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\課題研究\\KNOWAGE_DATABASE\\wikipedia_en_top_nopic_2025-03.zim",
        "model_path": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\課題研究\\models\\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_template.txt",
        "params": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_params.json"
    }
    try:
        ap = agent_pool.AgentPoolManager(config, None)
        logging.info("agent_pool: OK")
    except Exception as e:
        logging.error(f"agent_pool: NG ({e})", exc_info=True)
    try:
        bb = blackboard.Blackboard(config)
        logging.info("blackboard: OK")
    except Exception as e:
        logging.error(f"blackboard: NG ({e})", exc_info=True)
    try:
        ir = input_reception.InputReception(config)
        logging.info("input_reception: OK")
    except Exception as e:
        logging.error(f"input_reception: NG ({e})", exc_info=True)
    try:
        oa = output_agent.OutputAgent(config)
        logging.info("output_agent: OK")
    except Exception as e:
        logging.error(f"output_agent: NG ({e})", exc_info=True)
    try:
        rag = rag_retriever.RAGRetriever(config)
        logging.info("rag_retriever: OK")
    except Exception as e:
        logging.error(f"rag_retriever: NG ({e})", exc_info=True)
    try:
        se = summary_engine.SummaryEngine(config)
        logging.info("summary_engine: OK")
    except Exception as e:
        logging.error(f"summary_engine: NG ({e})", exc_info=True)
    logging.info("MurmurNet テスト終了")

if __name__ == "__main__":
    main()

# DistributedSLMの逐次テスト
import asyncio
from distributed_slm import DistributedSLM

# 黒板内容のフィルタリング
def print_blackboard(bb):
    print("--- 黒板の内容 ---")
    for k, v in bb.memory.items():
        if k in ['input', 'final_summary', 'response']:  # 必要なキーのみ表示
            print(f"{k}: {v}")
    print("-----------------")

# 冗長な出力を削減
async def main():
    config = {
        "num_agents": 2,
        "model_path": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_template.txt",
        "params": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_params.json"
    }
    slm = DistributedSLM(config)

    input_text = "AIは教育をどう変えると思う？"
    print(f"入力: {input_text}")
    response = await slm.generate(input_text)
    print("\n=== SLM応答 ===")
    print(response)
    print_blackboard(slm.blackboard)

if __name__ == "__main__":
    asyncio.run(main())

import unittest
from modules.input_reception import InputReception

class TestInputReception(unittest.TestCase):
    def setUp(self):
        config = {}
        self.input_reception = InputReception(config)

    def test_process(self):
        input_text = "Hello, World! This is a test."
        result = self.input_reception.process(input_text)

        # 正規化されたテキストの確認
        self.assertEqual(result['normalized'], "hello world this is a test")

        # トークンの確認
        self.assertEqual(result['tokens'], ["hello", "world", "this", "is", "a", "test"])

        # 埋め込みの確認（長さのみ確認）
        self.assertEqual(len(result['embedding']), 384)  # all-MiniLM-L6-v2 の出力次元

if __name__ == "__main__":
    unittest.main()

import asyncio
from murmurnet.distributed_slm import DistributedSLM

def main():
    config = {
        "num_agents": 2,
        "rag_mode": "dummy",
        "rag_score_threshold": 0.5,
        "rag_top_k": 1,
        "debug": True,
        "model_path": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_template.txt"
    }

    slm = DistributedSLM(config)

    async def test():
        input_text = "AIが教育に与える影響について教えてください。"
        response = await slm.generate(input_text)
        print("最終出力:", response)

    asyncio.run(test())

if __name__ == "__main__":
    main()
