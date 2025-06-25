#!/usr/bin/env python3
"""
修正されたコンソールアプリケーション
統合レスポンス表示問題を解決
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MurmurNet.distributed_slm import DistributedSLM

def main():
    """メイン実行関数"""
    print("MurmurNet - 分散創発型言語モデルシステム")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print()
    
    # MurmurNet初期化
    config = {
        "num_agents": 2,
        "iterations": 1,
        "use_summary": True,
        "rag_mode": "dummy"
    }
    
    murmur = DistributedSLM(config)
    
    # メインループ
    while True:
        try:
            user_input = input("あなた> ").strip()
            
            if user_input.lower() in ("quit", "exit", "終了"):
                print("システムを終了します...")
                break
            
            if not user_input:
                continue
            
            # 応答生成
            print("処理中...")
            response = murmur.run(user_input)
            
            # 結果表示 - これが重要！
            if response:
                print(f"\nAI> {response}\n")
            else:
                print("\nAI> 申し訳ありません。応答を生成できませんでした。\n")
                
        except KeyboardInterrupt:
            print("\n\nシステムを終了します...")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}\n")

if __name__ == "__main__":
    main()
