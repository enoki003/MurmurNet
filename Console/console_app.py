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

# MurmurNetのルートディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ここでモジュールをインポート
try:
    from MurmurNet.distributed_slm import DistributedSLM
    from Database.zim_reader import ZIMReader
    print("必要なモジュールが正常にインポートされました")
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")
    print("必要なファイルが不足している可能性があります")
    sys.exit(1)

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
    import libzim
    HAS_LIBZIM = True
    print("libzimライブラリが利用可能です")
except ImportError:
    HAS_LIBZIM = False
    print("警告: libzimライブラリがインストールされていません")
    print("ZIMモードを使用するには、以下のコマンドを実行してください:")
    print("pip install libzim")
    if args.rag_mode == "zim":
        print("ZIMモードが指定されましたが、libzimがないため検索機能が制限されます")

def print_debug(distributed_slm):
    """デバッグモード時の詳細情報表示"""
    try:
        # システム統計情報を表示
        if hasattr(distributed_slm, 'get_stats'):
            stats = distributed_slm.get_stats()
            print(f"\n[DEBUG] システム統計: {stats}")
        
        # 通信統計情報を表示
        if hasattr(distributed_slm, 'get_communication_stats'):
            comm_stats = distributed_slm.get_communication_stats()
            print(f"[DEBUG] 通信統計: {comm_stats}")
        
        # システム状態を表示
        system_status = distributed_slm.get_system_status()
        print(f"\n[DEBUG] システム状態: {system_status}")
        
    except Exception as e:
        print(f"\n[DEBUG] デバッグ情報の表示中にエラー: {e}")
    print()

async def chat_loop():
    """会話ループのメイン関数"""
    # 設定ファイルを読み込み
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    # 設定を読み込み
    config = None
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                print("設定ファイルを読み込みました")
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
    
    if config is None:
        print("デフォルト設定を使用します")
        config = {
            "model_path": r"C:\Users\admin\Desktop\課題研究\models\gemma-3-1b-it-q4_0.gguf",
            "chat_template": r"C:\Users\admin\Desktop\課題研究\models\gemma3_template.txt",
            "num_agents": args.agents,
            "iterations": args.iter,
            "use_summary": not args.no_summary,
            "use_parallel": args.parallel,
            "debug": args.debug,
            "rag_mode": args.rag_mode,
            "rag_score_threshold": 0.5,
            "rag_top_k": 3,
            "embedding_model": "all-MiniLM-L6-v2",
            "safe_parallel": args.safe_parallel,
            "max_workers": args.max_workers if args.max_workers > 0 else None,
            "use_global_lock": True,
        }

    # ZIMモードの場合、パスを追加
    if args.rag_mode == "zim":
        config["zim_path"] = args.zim_path
        # ZIMファイルの存在確認
        if not os.path.exists(args.zim_path):
            print(f"警告: 指定されたZIMファイルが見つかりません: {args.zim_path}")
            print("ファイルパスが正しいか確認してください")
        else:
            print(f"ZIMファイルを確認: {args.zim_path} (ファイルサイズ: {os.path.getsize(args.zim_path) / (1024*1024):.1f} MB)")      # SLMインスタンス作成
    print("分散SLMシステムを初期化中...")
    try:
        # 設定を一時ファイルに保存
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            temp_config_path = f.name
        
        try:
            distributed_slm = DistributedSLM(temp_config_path)
            print("分散SLMシステムが正常に初期化されました")
        finally:
            # 一時ファイルを削除
            os.unlink(temp_config_path)
            
        print("分散SLMシステムの初期化が完了しました")
        
        # システム状態の確認
        system_status = distributed_slm.get_system_status()
        print(f"[システム情報] 互換性モード: {system_status.get('compatibility_mode', 'N/A')}")
        print(f"[システム情報] パフォーマンス監視: {system_status.get('performance_monitoring', 'N/A')}")
        
    except Exception as e:
        print(f"分散SLMシステムの初期化中にエラーが発生しました: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

    # RAGモードのチェック
    try:
        # 簡単なRAGテスト
        if args.rag_mode == "zim" and not HAS_LIBZIM:
            print("警告: ZIMモードを指定しましたが、libzimライブラリがありません")
        print(f"[設定] RAGモード: {args.rag_mode}")
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
                    user_input = input()
            except EOFError:
                print("\n\n入力が終了しました。プログラムを終了します。")
                break
            except KeyboardInterrupt:
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
            response = await distributed_slm.generate(user_input)
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # 応答表示
            print(f"{response}")
            
            # デバッグ情報表示
            if args.debug:
                print_debug(distributed_slm)
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
        if hasattr(distributed_slm, 'cleanup'):
            distributed_slm.cleanup()
        print("リソースのクリーンアップが完了しました。")
    except Exception as e:
        if args.debug:
            print(f"クリーンアップ中にエラー: {e}")

def main():
    """メイン関数"""
    try:
        asyncio.run(chat_loop())
    except Exception as e:
        print(f"アプリケーション実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
