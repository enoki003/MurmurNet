import sys
import os
import argparse
import asyncio
from pathlib import Path
import logging

# murmurunetパスを追加
sys.path.append(str(Path(__file__).parent.parent / 'murmurnet'))
from distributed_slm import DistributedSLM

# ログ設定
# --debug 時は DEBUG レベルでコンソール & ファイル出力
parser = argparse.ArgumentParser(description="MurmurNet Console App")
parser.add_argument('--debug', action='store_true', help='デバッグ情報を表示')
parser.add_argument('--log', action='store_true', help='ログをファイルに保存')
args, _ = parser.parse_known_args()
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    filename='console_app.log',
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# コンソールにもログ出力
console = logging.StreamHandler()
console.setLevel(log_level)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console)


def print_debug(slm):
    print("\n[DEBUG] 黒板の内容:")
    for k, v in slm.blackboard.memory.items():
        print(f"  {k}: {v}")
    print("\n[DEBUG] RAG結果:")
    print(f"  {slm.blackboard.read('rag')}")
    print("\n[DEBUG] エージェント出力:")
    for k, v in slm.blackboard.memory.items():
        if k.startswith('agent_') and k.endswith('_output'):
            print(f"  {k}: {v}")
    print()


def get_chat_template():
    return (
        "{ bos_token }"
        "<|start|>system\nあなたはユーザーの入力言語（日本語または英語）に合わせて、同じ言語で丁寧に返答する多言語アシスタントです。\n<|end|>\n"
        "{% for message in messages[1:] %}"
        "<|start|>{{ message['role'] }}\n{{ message['content'] }}<|end|>\n"
        "{% endfor %}"
    )


async def main():
    # 再度引数パース（debug, log フラグを利用）
    # parser は上で定義済み
    args = parser.parse_args()

    chat_template = get_chat_template()
    config = {
        "num_agents": 2,
        "rag_mode": "zim",
        "rag_score_threshold": 0.0,
        "rag_top_k": 5,
        "debug": args.debug,
        "zim_path": r"C:\\Users\\園木優陽\\AppData\\Roaming\\kiwix-desktop\\wikipedia_en_top_nopic_2025-03.zim",
        "model_path": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_template.txt",
        "params": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma3_params.json"
    }
    slm = DistributedSLM(config)
    print(f"MurmurNet Console (type 'exit' to quit, 多言語対応)")
    while True:
        try:
            user_input = input("あなた> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("終了します。")
                break
            response = await slm.generate(user_input)
            print(f"AI> {response}")
            if args.debug:
                print_debug(slm)
            if args.log:
                logging.info(f"User Input: {user_input}")
                logging.info(f"AI Response: {response}")
        except Exception:
            print("エラーが発生しました。詳細はログを確認してください。")
            logging.error("エラー発生", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
