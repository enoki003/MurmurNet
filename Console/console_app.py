#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet コンソールアプリケーション
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分散SLMシステムのコマンドラインインターフェース
対話的に質問応答を行うためのコンソールUI

作者: Yuhi Sonoki
"""
# C:\Users\園木優陽\AppData\Roaming\kiwix-desktop\wikipedia_en_top_nopic_2025-03.zim
import sys
import os
import argparse
import asyncio
from pathlib import Path
import logging

# murmurnetパスを追加
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
murmurnet_dir = project_root / "MurmurNet"
sys.path.append(str(project_root))  # プロジェクトルートをパスに追加

# ここでモジュールをインポート
from MurmurNet.distributed_slm import DistributedSLM

# ログ設定
parser = argparse.ArgumentParser(description="MurmurNet Console App")
parser.add_argument('--debug', action='store_true', help='デバッグ情報を表示')
parser.add_argument('--log', action='store_true', help='ログをファイルに保存')
parser.add_argument('--iter', type=int, default=1, help='反復回数（デフォルト: 1）')
parser.add_argument('--agents', type=int, default=2, help='エージェント数（デフォルト: 2）')
parser.add_argument('--no-summary', action='store_true', help='要約機能を無効化')
parser.add_argument('--parallel', action='store_true', help='並列処理を有効化')
# RAGモードのオプションを追加
parser.add_argument('--rag-mode', choices=['zim', 'embedding'], default='zim', 
                    help='RAGモード（zim: ZIMファイル使用、embedding: 埋め込みベース検索）')
parser.add_argument('--zim-path', type=str, 
                    default=r"C:\Users\admin\Desktop\課題研究\KNOWAGE_DATABASE\wikipedia_en_top_nopic_2025-03.zim",
                    help='ZIMファイルのパス（RAGモードがzimの場合に使用）')
# 並列処理の安全性に関するオプション
parser.add_argument('--safe-parallel', action='store_true', 
                    help='安全な並列処理モード（GGMLエラー回避のためのグローバルロックを使用）')
parser.add_argument('--max-workers', type=int, default=0, 
                    help='並列処理時の最大ワーカー数（0: 自動決定）')
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
    logging.getLogger('').addHandler(console)

# libzimがインストールされているか確認
try:
    from libzim.reader import Archive
    HAS_LIBZIM = True
    print("libzimライブラリが利用可能です")
except ImportError:
    HAS_LIBZIM = False
    print("警告: libzimライブラリがインストールされていません")
    print("ZIMモードを使用するには、以下のコマンドを実行してください:")
    print("pip install libzim")
    if args.rag_mode == "zim":
        print("ZIMモードが指定されましたが、libzimがないため検索機能が制限されます")

def print_debug(slm):
    """デバッグモード時の詳細情報表示"""
    try:
        # システム統計情報を表示
        stats = slm.get_stats()
        print(f"\n[DEBUG] システム統計: {stats}")
        
        # 通信統計情報を表示
        comm_stats = slm.get_communication_stats()
        print(f"[DEBUG] 通信統計: {comm_stats}")
        
        # 黒板が利用可能な場合の情報表示
        if slm.blackboard:
            print("\n[DEBUG] 黒板の内容:")
            # 簡略化されたビューを使用
            try:
                debug_view = slm.blackboard.get_debug_view()
                for k, v in debug_view.items():
                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"  黒板ビュー取得エラー: {e}")
            
            print("\n[DEBUG] RAG結果:")
            try:
                rag_result = slm.blackboard.read('rag')
                print(f"  {rag_result}")
            except Exception:
                print("  RAG結果の取得に失敗")
            
            # 要約結果を表示
            if slm.use_summary:
                print("\n[DEBUG] 要約結果:")
                for i in range(slm.iterations):
                    try:
                        summary = slm.blackboard.read(f'summary_{i}')
                        if summary:
                            print(f"  反復{i+1}の要約: {summary[:100]}...")
                    except Exception:
                        print(f"  反復{i+1}の要約取得に失敗")
            
            print("\n[DEBUG] エージェント出力:")
            for i in range(slm.num_agents):
                try:
                    output = slm.blackboard.read(f'agent_{i}_output')
                    if output:
                        # 長い出力は省略表示
                        if len(output) > 100:
                            output = output[:100] + "..."
                        print(f"  エージェント{i+1}: {output}")
                except Exception:
                    print(f"  エージェント{i+1}の出力取得に失敗")
        else:
            print("\n[DEBUG] 黒板は利用できません（新システムモード）")
    except Exception as e:
        print(f"\n[DEBUG] デバッグ情報の表示中にエラー: {e}")
    print()

async def chat_loop():
    """会話ループのメイン関数"""
    # 設定
    config = {
        # "model_path": r"C:\Users\園木優陽\OneDrive\デスクトップ\models\gemma-3-1b-it-q4_0.gguf",
        # "chat_template": r"C:\Users\園木優陽\OneDrive\デスクトップ\models\gemma3_template.txt",
        "model_path": r"C:\Users\admin\Desktop\課題研究\models\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"C:\Users\admin\Desktop\課題研究\models\gemma3_template.txt",        "num_agents": args.agents,
        "iterations": args.iter,
        "use_summary": not args.no_summary,
        "use_parallel": args.parallel,
        "debug": args.debug,
        # RAG設定
        "rag_mode": args.rag_mode,  # コマンドライン引数から設定
        "rag_score_threshold": 0.5,
        "rag_top_k": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        # 並列処理の安全性向上オプション
        "safe_parallel": args.safe_parallel,
        "max_workers": args.max_workers if args.max_workers > 0 else None,
        "use_global_lock": True,  # GGMLエラー回避のためのグローバルロック
    }

    # ZIMモードの場合、パスを追加
    if args.rag_mode == "zim":
        config["zim_path"] = args.zim_path
        # ZIMファイルの存在確認
        if not os.path.exists(args.zim_path):
            print(f"警告: 指定されたZIMファイルが見つかりません: {args.zim_path}")
            print("ファイルパスが正しいか確認してください")
        else:
            print(f"ZIMファイルを確認: {args.zim_path} (ファイルサイズ: {os.path.getsize(args.zim_path) / (1024*1024):.1f} MB)")
      # SLMインスタンス作成
    try:
        slm = DistributedSLM(config)
        print("分散SLMシステムの初期化が完了しました")
        
        # システム状態の確認
        system_status = slm.get_system_status()
        print(f"[システム情報] 互換性モード: {system_status['compatibility_mode']}")
        print(f"[システム情報] パフォーマンス監視: {system_status['performance_monitoring']}")
        
    except Exception as e:
        print(f"分散SLMシステムの初期化中にエラーが発生しました: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return      # RAGモードのチェック
    try:
        from MurmurNet.modules.rag_retriever import RAGRetriever
        rag = RAGRetriever(config)
        if args.rag_mode == "zim" and rag.mode != "zim":
            print("警告: ZIMモードを指定しましたが、ZIMモードになっていません")
            print("以下の理由が考えられます:")
            print("- libzimライブラリがインストールされていない")
            print("- ZIMファイルのパスが間違っている")
            print("- ZIMファイルが壊れている")
    except Exception as e:
        print(f"RAGシステムの確認中にエラー: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    print(f"MurmurNet Console ({args.agents}エージェント, {args.iter}反復)")
    print("終了するには 'quit' または 'exit' を入力してください")
    
    if args.parallel:
        print("[設定] 並列処理: 有効")
        if args.safe_parallel:
            print("[設定] 安全な並列処理モード: 有効（GGMLエラー回避用）")
        if args.max_workers > 0:
            print(f"[設定] 最大並列ワーカー数: {args.max_workers}")
    if not args.no_summary:
        print("[設定] 要約機能: 有効")
    print(f"[設定] RAGモード: {rag.mode} (指定: {args.rag_mode})")
    if args.rag_mode == "zim":
        print(f"[設定] ZIMファイル: {args.zim_path}")
    
    history = []
      # 対話的環境かどうかをチェック
    is_interactive = sys.stdin.isatty()
    if not is_interactive:
        print("警告: 非対話的環境で実行されています。標準入力から質問を読み取ります。")
    
    while True:
        try:
            # ユーザー入力（EOFError対応）
            try:
                if is_interactive:
                    user_input = input("\nあなた> ")
                else:
                    # 非対話的環境では改行なしプロンプト
                    user_input = input()
            except EOFError:
                # EOF (Ctrl+D/Ctrl+Z) が押された場合の処理
                print("\n\n入力が終了しました。プログラムを終了します。")
                break
            except KeyboardInterrupt:
                # Ctrl+C が押された場合
                print("\n中断されました。終了するには 'exit' を入力するか、再度 Ctrl+C を押してください。")
                continue
            
            # 終了コマンドチェック
            if user_input.lower() in ["quit", "exit", "終了", "q"]:
                break
            
            # 空入力はスキップ
            if not user_input.strip():
                continue
            
            # 履歴に追加
            history.append({"role": "user", "content": user_input})
            
            # 生成開始
            print("AI> ", end="", flush=True)
            
            start_time = asyncio.get_event_loop().time()
            response = await slm.generate(user_input)
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # 応答表示
            print(f"{response}")
            
            # デバッグ情報表示
            if args.debug:
                print_debug(slm)
                print(f"[DEBUG] 実行時間: {elapsed:.2f}秒")
            
            # 履歴に追加
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\nプログラムを強制終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    print("MurmurNet Console を終了しました。")
    
    # システムリソースのクリーンアップ
    try:
        if hasattr(slm, 'cleanup'):
            slm.cleanup()
        print("リソースのクリーンアップが完了しました。")
    except Exception as e:
        if args.debug:
            print(f"クリーンアップ中にエラー: {e}")

if __name__ == "__main__":
    asyncio.run(chat_loop())
