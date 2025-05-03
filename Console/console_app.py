import sys
import os
import argparse
import asyncio
from pathlib import Path

# murmurunetパスを追加
sys.path.append(str(Path(__file__).parent.parent / 'murmurnet'))
from distributed_slm import DistributedSLM

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
        "{{ bos_token }}"
        "<|start|>system\nあなたはユーザーの入力言語（日本語または英語）に合わせて、同じ言語で丁寧に返答する多言語アシスタントです。\n<|end|>\n"
        "{% for message in messages[1:] %}"
        "<|start|>{{ message['role'] }}\n{{ message['content'] }}<|end|>\n"
        "{% endfor %}"
    )

async def main():
    parser = argparse.ArgumentParser(description="MurmurNet Console App")
    parser.add_argument('--debug', action='store_true', help='デバッグ情報を表示')
    args = parser.parse_args()

    chat_template = get_chat_template()
    config = {
        "num_agents": 2,
        "rag_mode": "dummy",
        "model_path": r"c:\\Users\\園木優陽\\OneDrive\\デスクトップ\\models\\gemma-3-1b-it-q4_0.gguf",
        "chat_template": chat_template
    }
    slm = DistributedSLM(config)
    print(f"MurmurNet Console (type 'exit' to quit, 多言語対応)")
    while True:
        user_input = input("あなた> ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("終了します。")
            break
        response = await slm.generate(user_input)
        print(f"AI> {response}")
        if args.debug:
            print_debug(slm)

if __name__ == "__main__":
    asyncio.run(main())
