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
parser.add_argument('--performance', action='store_true', help='詳細なパフォーマンス情報を表示')
parser.add_argument('--rag', choices=['dummy', 'zim', 'none'], default='dummy', 
                   help='RAGモード選択: dummy(基本知識), zim(Wikipedia), none(RAG無効)')
parser.add_argument('--threads', type=int, help='推論スレッド数の上書き')
parser.add_argument('--ctx', type=int, help='コンテキスト長の上書き')
parser.add_argument('--log', action='store_true', help='ログをファイルに保存')
parser.add_argument('--iter', type=int, default=1, help='反復回数（デフォルト: 1）')
parser.add_argument('--agents', type=int, default=2, help='エージェント数（デフォルト: 2）')
parser.add_argument('--no-summary', action='store_true', help='要約機能を完全無効化')
parser.add_argument('--summary', choices=['on', 'off', 'smart'], default='smart', 
                   help='要約設定: on(常時), off(無効), smart(自動判定)')
parser.add_argument('--summary-threshold', type=int, default=1000, 
                   help='smart要約の閾値（トークン数、デフォルト1000）')
parser.add_argument('--parallel', action='store_true', help='並列処理を有効化')
# RAGモードのオプションを追加
parser.add_argument('--rag-mode', choices=['dummy', 'zim'], default='dummy', 
                    help='RAGモード（dummy: ダミーモード、zim: ZIMファイル使用）')
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

# パフォーマンス情報をコンソールに表示するためのハンドラー
if args.performance or args.debug:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[PERF] %(message)s'))  # パフォーマンス情報用
    
    # MurmurNetのロガーにコンソールハンドラーを追加
    murmur_logger = logging.getLogger('MurmurNet')
    murmur_logger.addHandler(console)
    murmur_logger.setLevel(logging.INFO)

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
        print("ZIMモードが指定されましたが、libzimがないためdummyモードにフォールバックします")

def print_debug(slm):
    """デバッグモード時の詳細情報表示"""
    print("\n[DEBUG] 黒板の内容:")
    # 簡略化されたビューを使用
    debug_view = slm.blackboard.get_debug_view()
    for k, v in debug_view.items():
        print(f"  {k}: {v}")
    
    print("\n[DEBUG] RAG結果:")
    print(f"  {slm.blackboard.read('rag')}")
    
    # 要約結果を表示
    if slm.use_summary:
        print("\n[DEBUG] 要約結果:")
        for i in range(slm.iterations):
            summary = slm.blackboard.read(f'summary_{i}')
            if summary:
                print(f"  反復{i+1}の要約: {summary[:100]}...")
    
    print("\n[DEBUG] エージェント出力:")
    for i in range(slm.num_agents):
        output = slm.blackboard.read(f'agent_{i}_output')
        if output:
            # 長い出力は省略表示
            if len(output) > 100:
                output = output[:100] + "..."
            print(f"  エージェント{i+1}: {output}")
    print()

async def chat_loop():
    """会話ループのメイン関数"""
    
    # config.yamlからベース設定を読み込み
    config_path = "config.yaml"
    base_config = {}
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f) or {}
            print(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            base_config = {}
      # 設定（config.yamlをベースにコマンドライン引数で上書き）
    config = base_config.copy()  # ベース設定から開始
    
    # 基本設定（常に上書き）
    config.update({
        "model_path": r"C:\Users\admin\Desktop\課題研究\models\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"C:\Users\admin\Desktop\課題研究\models\gemma3_template.txt",
        "num_agents": args.agents,
        "iterations": args.iter,
        "use_summary": not args.no_summary and args.summary != 'off',
        "summary_mode": args.summary,  # on, off, smart
        "summary_threshold": args.summary_threshold,  # smart要約の閾値
        "use_parallel": args.parallel,
        "debug": args.debug,
        "performance_monitoring": True,  # パフォーマンス測定を有効化
        "show_performance": args.performance or args.debug,  # パフォーマンス表示設定
        
        # RAG設定（コマンドライン引数から上書き）
        "rag_mode": args.rag,  # dummy, zim, none
        "rag_enabled": args.rag != 'none',  # RAG無効化対応
        "rag_score_threshold": 0.5,
        "rag_top_k": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        
        # 並列処理の安全性向上オプション
        "safe_parallel": args.safe_parallel,
        "max_workers": args.max_workers if args.max_workers > 0 else None,
        "use_global_lock": True,  # GGMLエラー回避のためのグローバルロック
    })
    
    # パフォーマンス設定（コマンドライン引数で上書き、なければconfig.yamlの値を使用）
    if args.threads:
        config["n_threads"] = args.threads
    if args.ctx:
        config["n_ctx"] = args.ctx

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
    slm = DistributedSLM(config)
      # RAGモードのチェック
    from MurmurNet.modules.rag_retriever import RAGRetriever
    rag = RAGRetriever(config)
    if args.rag_mode == "zim" and rag.mode == "dummy":
        print("警告: ZIMモードを指定しましたが、dummyモードになっています")
        print("以下の理由が考えられます:")
        print("- libzimライブラリがインストールされていない")
        print("- ZIMファイルのパスが間違っている")
        print("- ZIMファイルが壊れている")
    
    print(f"MurmurNet Console ({args.agents}エージェント, {args.iter}反復)")
    print("終了するには 'quit' または 'exit' を入力してください")
    
    # 最適化設定の表示
    print("\n[最適化設定]")
    print(f"  コンテキスト長: {config.get('n_ctx', 'N/A')}")
    print(f"  バッチサイズ: {config.get('n_batch', 'N/A')}")
    print(f"  推論スレッド数: {config.get('n_threads', 'N/A')}")
    if args.performance:
        print("  パフォーマンス測定: 有効 ⚡")
    
    if args.parallel:
        print("[設定] 並列処理: 有効")
        if args.safe_parallel:
            print("[設定] 安全な並列処理モード: 有効（GGMLエラー回避用）")
        if args.max_workers > 0:
            print(f"[設定] 最大並列ワーカー数: {args.max_workers}")
    if not args.no_summary:
        print(f"[設定] 要約機能: 有効 (モード: {args.summary})")
    print(f"[設定] RAGモード: {rag.mode} (指定: {args.rag_mode})")
    if args.rag_mode == "zim":
        print(f"[設定] ZIMファイル: {args.zim_path}")
    
    history = []
    
    while True:
        try:
            # ユーザー入力
            user_input = input("\nあなた> ")
            if user_input.lower() in ["quit", "exit", "終了"]:
                break
            
            # 空入力はスキップ
            if not user_input.strip():
                continue
              # 履歴に追加
            history.append({"role": "user", "content": user_input})
            
            # 生成開始
            print("AI> ", end="", flush=True)
            
            import time
            start_time = time.time()
            response = await slm.generate(user_input)
            elapsed = time.time() - start_time
              # 応答表示（応答時間を常に表示）
            print(f"{response}")
            print(f"[応答時間: {elapsed:.2f}秒]")
            
            # パフォーマンス最適化効果を表示
            if elapsed <= 3.0:
                print("⚡ 高速応答!")
            elif elapsed <= 6.0:
                print("✅ 良好な応答速度")
            elif elapsed <= 10.0:
                print("⚠️  やや低速")
            else:
                print("🐌 最適化が必要")
            
            # 詳細なパフォーマンス情報を表示（--performance または --debug フラグ）
            if args.performance or args.debug:
                print(f"[詳細] n_ctx: {config.get('n_ctx', 'N/A')}, n_batch: {config.get('n_batch', 'N/A')}, スレッド: {config.get('n_threads', 'N/A')}")
                print(f"[詳細] RAGモード: {config.get('rag_mode', 'N/A')}, 要約: {config.get('summary_mode', 'N/A')}")
                
                # メモリ使用量を表示（可能であれば）
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"[詳細] メモリ使用量: {memory_mb:.1f} MB")
                except ImportError:
                    pass
            
            # デバッグ情報表示
            if args.debug:
                print_debug(slm)
            
            # 履歴に追加
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n中断されました。終了するには 'exit' を入力してください。")
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(chat_loop())
