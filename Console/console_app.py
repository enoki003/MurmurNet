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
import signal
from pathlib import Path
import logging
import atexit

# MurmurNetパッケージへのパスを追加
current_dir = Path(__file__).parent
murmur_net_dir = current_dir.parent / "MurmurNet"
if str(murmur_net_dir) not in sys.path:
    sys.path.insert(0, str(murmur_net_dir))

# murmurnetパスを追加
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
murmurnet_dir = project_root / "MurmurNet"
sys.path.append(str(project_root))  # プロジェクトルートをパスに追加

# ここでモジュールをインポート
from MurmurNet.distributed_slm import DistributedSLM

# グローバルなSLMインスタンス（シャットダウン用）
_global_slm = None

# ログ設定
parser = argparse.ArgumentParser(description="MurmurNet Console App - 分散SLMシステム")
parser.add_argument('--debug', action='store_true', help='デバッグ情報を表示')
parser.add_argument('--log', action='store_true', help='ログをファイルに保存')
parser.add_argument('--iterations', '--iter', '-iter', type=int, default=1, 
                    help='反復回数（デフォルト: 1） ※--iter, -iter でも指定可能')
parser.add_argument('--agents', type=int, default=2, help='エージェント数（デフォルト: 2）')
parser.add_argument('--no-summary', action='store_true', help='要約機能を無効化')
parser.add_argument('--parallel', action='store_true', help='並列処理を有効化')
parser.add_argument('--model-type', choices=['llama', 'huggingface'], required=True,
                    help='使用するモデルタイプ（必須）: llama=Gemmaモデル, huggingface=HuggingFaceモデル')
parser.add_argument('--model-name', '--huggingface-model', type=str, default=None,
                    help='モデル名（model-type=huggingface時は必須、150Mへの自動フォールバック禁止）')
parser.add_argument('--model-path', type=str, default=None,
                    help='モデルファイルパス（model-type=llama時は必須）')

# パフォーマンス最適化オプション
parser.add_argument('--no-local-files', action='store_true', 
                    help='ローカルファイルモードを無効化（HuggingFaceへのHTTPアクセスを許可）')
parser.add_argument('--cache-folder', type=str, 
                    default=r"C:\Users\admin\Desktop\課題研究\ワークスペース\MurmurNet\cache\models",
                    help='モデルキャッシュフォルダのパス')

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

async def safe_shutdown(slm):
    """安全なシャットダウン処理"""
    global _global_slm
    
    try:
        # 1. SLMのシャットダウン
        if hasattr(slm, 'shutdown'):
            print("SLMシステムをシャットダウン中...")
            await slm.shutdown()
        
        # 2. InputReceptionの強制終了
        try:
            from MurmurNet.modules.input_reception import InputReception
            print("InputReceptionを終了中...")
            InputReception.force_exit_all()
        except Exception as e:
            print(f"InputReception終了エラー: {e}")
          # 3. DistributedSLMのシャットダウン（タイムアウト付き）
        if _global_slm:
            try:
                print("DistributedSLMをシャットダウン中...")
                # 15秒のタイムアウト付きでシャットダウン
                await asyncio.wait_for(_global_slm.shutdown(), timeout=15.0)
                print("DistributedSLMシャットダウン完了")
            except asyncio.TimeoutError:
                print("DistributedSLMシャットダウンがタイムアウトしました - 強制終了します")
            except Exception as e:
                print(f"DistributedSLMシャットダウンエラー: {e}")
        
        # 4. 最終クリーンアップ
        _global_slm = None
        
        print("全てのシャットダウン処理が完了しました")
        
    except Exception as e:
        print(f"safe_shutdown内でエラーが発生: {e}")
        raise

async def chat_loop(args):
    """会話ループのメイン関数"""
    # 設定
    config = {
        # "model_path": r"C:\Users\園木優陽\OneDrive\デスクトップ\models\gemma-3-1b-it-q4_0.gguf",
        # "chat_template": r"C:\Users\園木優陽\OneDrive\デスクトップ\models\gemma3_template.txt",
        "model_path": r"C:\Users\admin\Desktop\課題研究\models\gemma-3-1b-it-q4_0.gguf",
        "chat_template": r"C:\Users\admin\Desktop\課題研究\models\gemma3_template.txt",

        "num_agents": args.agents,
        "iterations": args.iterations,
        "use_summary": not args.no_summary,
        "use_parallel": args.parallel,
        "debug": args.debug,
        
        # モデル設定のオーバーライド
        "model_type": args.model_type if args.model_type else "llama",  # デフォルトはllama
        "huggingface_model_name": args.model_name,  # --model-name に変更
        "model_path": args.model_path,  # モデルパス追加
        "device": "cpu",  # CPUを使用
        "torch_dtype": "auto",
        
        # 基本設定
        "local_files_only": not args.no_local_files,  # --no-local-filesフラグに基づく
        "cache_folder": args.cache_folder,  # コマンドライン引数から取得
        
        # RAG設定（基本設定）
        "rag_mode": "local",  # デフォルトRAGモード
        "rag_score_threshold": 0.5,
        "rag_top_k": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_cache_folder": args.cache_folder,  # 埋め込みモデル用キャッシュフォルダ
        
        # 並列処理設定
        "use_global_lock": True,  # GGMLエラー回避のためのグローバルロック
    }

    # 基本設定完了
    print(f"設定完了: モデルタイプ={config['model_type']}")
    if config['model_type'] == 'huggingface':
        print(f"HuggingFaceモデル: {config['huggingface_model_name']}")
    elif config['model_type'] == 'llama':
        print(f"Llamaモデルパス: {config.get('model_path', 'デフォルト')}")
    
    # ローカルファイル設定の表示
    print(f"ローカルファイルモード: {config['local_files_only']}")
    print(f"キャッシュフォルダ: {config['cache_folder']}")
    
    # キャッシュフォルダの存在確認
    import os
    if not os.path.exists(config['cache_folder']):
        print(f"警告: キャッシュフォルダが存在しません: {config['cache_folder']}")
        try:
            os.makedirs(config['cache_folder'], exist_ok=True)
            print(f"キャッシュフォルダを作成しました: {config['cache_folder']}")
        except Exception as e:
            print(f"キャッシュフォルダの作成に失敗: {e}")
            
    # HuggingFaceモデルのローカルキャッシュ確認
    if config['model_type'] == 'huggingface' and config['local_files_only']:
        model_cache_path = os.path.join(config['cache_folder'], f"models--{config['huggingface_model_name'].replace('/', '--')}")
        if os.path.exists(model_cache_path):
            print(f"ローカルモデルキャッシュが見つかりました: {model_cache_path}")
        else:
            print(f"警告: ローカルモデルキャッシュが見つかりません: {model_cache_path}")
            print("モデルを事前にダウンロードするか、--no-local-files オプションを使用してください")
        
    # SLMインスタンス作成
    global _global_slm
    slm = DistributedSLM(config)
    _global_slm = slm  # グローバル参照を設定
    
    print(f"MurmurNet Console ({args.agents}エージェント, {args.iterations}反復)")
    print("終了するには 'quit' または 'exit' を入力してください")
    
    if args.parallel:
        print("[設定] 並列処理: 有効")
    if not args.no_summary:
        print("[設定] 要約機能: 有効")
    print(f"[設定] RAGモード: {config['rag_mode']}")
    print(f"[設定] ローカルファイルモード: {config['local_files_only']}")
    if args.debug:
        print(f"[DEBUG] 使用キャッシュフォルダ: {config['cache_folder']}")
        print(f"[DEBUG] 埋め込みモデルキャッシュフォルダ: {config['embedding_cache_folder']}")
        if config['model_type'] == 'huggingface':
            model_cache_path = os.path.join(config['cache_folder'], f"models--{config['huggingface_model_name'].replace('/', '--')}")
            print(f"[DEBUG] 期待されるモデルキャッシュパス: {model_cache_path}")
            print(f"[DEBUG] モデルキャッシュ存在: {os.path.exists(model_cache_path)}")
            # 埋め込みモデルキャッシュ確認
            embed_cache_path = os.path.join(config['embedding_cache_folder'], f"models--sentence-transformers--{config['embedding_model']}")
            print(f"[DEBUG] 期待される埋め込みキャッシュパス: {embed_cache_path}")
            print(f"[DEBUG] 埋め込みキャッシュ存在: {os.path.exists(embed_cache_path)}")
    
    history = []
    while True:
        try:            
            # ユーザー入力
            user_input = input("\nあなた> ")
            if user_input.lower() in ["quit", "exit", "終了"]:
                print("システムを終了しています...")
                # 適切なシャットダウン処理
                try:
                    print("完全シャットダウンを開始...")
                    await safe_shutdown(slm)
                    print("全てのシャットダウン処理が完了しました")
                except Exception as e:
                    print(f"シャットダウンエラー: {e}")
                    # フォールバック：強制終了
                    try:
                        cleanup_system()
                    except:
                        pass
                finally:
                    # 確実にプログラムを終了
                    print("シャットダウン完了")
                    import os
                    os._exit(0)
            
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
                print(f"[DEBUG] 実行時間: {elapsed:.2f}秒")            # 履歴に追加
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n中断されました。システムをシャットダウンします...")
            try:
                await safe_shutdown(slm)
            except:
                cleanup_system()
            finally:
                import os
                os._exit(0)
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

def cleanup_system():
    """システムの緊急クリーンアップ"""
    global _global_slm
    if _global_slm:
        try:
            print("システムを緊急シャットダウン中...")
            # 非同期関数を同期で実行
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 既にイベントループが動いている場合
                    asyncio.create_task(_global_slm.shutdown())
                else:
                    loop.run_until_complete(_global_slm.shutdown())
            except RuntimeError:
                # イベントループがない場合は新しく作成
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_global_slm.shutdown())
                loop.close()
        except Exception as e:
            print(f"緊急シャットダウンエラー: {e}")
        finally:
            _global_slm = None

def signal_handler(signum, frame):
    """シグナルハンドラ"""
    print(f"\nシグナル {signum} を受信しました。システムを終了します...")
    cleanup_system()
    sys.exit(0)

# シグナルハンドラとatexit登録
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_system)

def validate_args(args):
    """CLI引数の検証（150Mフォールバック禁止）"""
    errors = []
    
    # model-type必須チェック
    if not args.model_type:
        errors.append("--model-type は必須です。'llama' または 'huggingface' を指定してください。")
    
    # model-type別の必須引数チェック
    if args.model_type == 'llama':
        if not args.model_path:
            errors.append("--model-type=llama の場合、--model-path は必須です。")
        elif not os.path.exists(args.model_path):
            errors.append(f"指定されたモデルファイルが見つかりません: {args.model_path}")
    
    elif args.model_type == 'huggingface':
        if not args.model_name:
            errors.append("--model-type=huggingface の場合、--model-name は必須です。")
            errors.append("150Mモデルへの自動フォールバックは禁止されています。")
        
        # ローカルファイルモード時のキャッシュ存在チェック
        if not args.no_local_files:
            model_cache_path = os.path.join(args.cache_folder, f"models--{args.model_name.replace('/', '--')}")
            if not os.path.exists(model_cache_path):
                errors.append(f"ローカルファイルモードですが、モデルキャッシュが見つかりません: {model_cache_path}")
                errors.append("モデルを事前にダウンロードするか、--no-local-files オプションを使用してください。")
    
    # iterations範囲チェック
    if args.iterations < 1 or args.iterations > 10:
        errors.append("--iterations は1-10の範囲で指定してください。")
    
    # agents範囲チェック
    if args.agents < 1 or args.agents > 20:
        errors.append("--agents は1-20の範囲で指定してください。")
    
    # エラーがある場合は表示して終了
    if errors:
        print("=== 引数エラー ===")
        for error in errors:
            print(f"エラー: {error}")
        print("\n使用例:")
        print("  Llamaモデル使用:")
        print("    python console_app.py --model-type llama --model-path ./models/model.gguf")
        print("  HuggingFaceモデル使用:")
        print("    python console_app.py --model-type huggingface --model-name rinna/japanese-gpt2-medium")
        print("  反復回数指定:")
        print("    python console_app.py --model-type llama --model-path ./model.gguf --iterations 3")
        sys.exit(1)

if __name__ == "__main__":
    # 引数をパース
    args = parser.parse_args()
    
    # 引数を検証
    validate_args(args)
    
    print("=== MurmurNet Console App ===")
    print(f"モデルタイプ: {args.model_type}")
    if args.model_type == 'llama':
        print(f"モデルパス: {args.model_path}")
    elif args.model_type == 'huggingface':
        print(f"モデル名: {args.model_name}")
    print(f"反復回数: {args.iterations}")
    print(f"エージェント数: {args.agents}")
    print("=" * 30)
    
    # メインループ実行
    asyncio.run(chat_loop(args))
