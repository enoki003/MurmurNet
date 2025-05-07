import sys
import os
import argparse
import asyncio
from pathlib import Path
import logging

# murmurnetパスを追加
sys.path.append(str(Path(__file__).parent.parent))
# ここでmurmurnetモジュールをインポート
from murmurnet.distributed_slm import DistributedSLM

# ログ設定
parser = argparse.ArgumentParser(description="MurmurNet Console App")
parser.add_argument('--debug', action='store_true', help='デバッグ情報を表示')
parser.add_argument('--log', action='store_true', help='ログをファイルに保存')
args, _ = parser.parse_known_args()
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    filename='console_app.log' if args.log else None,
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# コンソールにもログ出力（ファイル出力時のみ追加）
if args.log:
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console)


def print_debug(slm):
    """デバッグモード時の詳細情報表示"""
    print("\n[DEBUG] 黒板の内容:")
    # 簡略化されたビューを使用
    debug_view = slm.blackboard.get_debug_view()
    for k, v in debug_view.items():
        print(f"  {k}: {v}")
    
    print("\n[DEBUG] RAG結果:")
    print(f"  {slm.blackboard.read('rag')}")
    
    print("\n[DEBUG] エージェント出力:")
    for i in range(slm.config.get('num_agents', 2)):
        output = slm.blackboard.read(f'agent_{i}_output')
        if output:
            # 長い出力は省略表示
            if len(output) > 100:
                output = output[:100] + "..."
            print(f"  エージェント{i}: {output}")
    print()


async def main():
    """コンソールアプリメイン関数"""
    # 再度引数パース（debug, log フラグを利用）
    args = parser.parse_args()

    # モデルと関連ファイルのパス設定
    DESKTOP_PATH = r"C:\\Users\\園木優陽\\OneDrive\\デスクトップ"
    MODELS_PATH = os.path.join(DESKTOP_PATH, "models")
    
    # 設定を構成
    config = {
        "num_agents": 2,
        "rag_mode": "zim",
        "rag_score_threshold": 0.0,
        "rag_top_k": 1,
        "debug": args.debug,
        "zim_path": r"C:\\Users\\園木優陽\\AppData\\Roaming\\kiwix-desktop\\wikipedia_en_top_nopic_2025-03.zim",
        "model_path": os.path.join(MODELS_PATH, "gemma-3-1b-it-q4_0.gguf"),
        "chat_template": os.path.join(MODELS_PATH, "gemma3_template.txt"),
        "params": os.path.join(MODELS_PATH, "gemma3_params.json")
    }

    # DistributedSLMインスタンスを生成
    slm = DistributedSLM(config)
    print(f"MurmurNet Console (type 'exit' to quit, 多言語対応)")
    
    # メインループ
    while True:
        try:
            user_input = input("あなた> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("終了します。")
                break
                
            # 非同期で応答生成
            response = await slm.generate(user_input)
            print(f"AI> {response}")
            
            # デバッグモード時は詳細表示
            if args.debug:
                print_debug(slm)
                
            # ログオプション時はログ出力
            if args.log:
                logging.info(f"User Input: {user_input}")
                logging.info(f"AI Response: {response}")
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            logging.error("エラー発生", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
