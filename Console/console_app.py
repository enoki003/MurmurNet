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

# murmurnetパスを追加
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))  # プロジェクトルートをパスに追加

# ここでモジュールをインポート
from murmurnet.distributed_slm import DistributedSLM

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
# エージェント別モデル設定のオプションを追加
parser.add_argument('--internal-model', type=str, default=None,
                    help='内部エージェント用のモデル名（HuggingFace形式）')
parser.add_argument('--output-model', type=str, default=None,
                    help='出力エージェント用のモデル名（HuggingFace形式）')
parser.add_argument('--summary-model', type=str, default=None,
                    help='要約エンジン用のモデル名（HuggingFace形式）')

# エージェント別モデルタイプとパス指定オプションを追加
parser.add_argument('--internal-model-type', choices=['llama', 'huggingface'], default=None,
                    help='内部エージェント用のモデルタイプ')
parser.add_argument('--output-model-type', choices=['llama', 'huggingface'], default=None,
                    help='出力エージェント用のモデルタイプ')
parser.add_argument('--summary-model-type', choices=['llama', 'huggingface'], default=None,
                    help='要約エンジン用のモデルタイプ')

parser.add_argument('--internal-model-path', type=str, default=None,
                    help='内部エージェント用のモデルファイルパス（GGUF形式）')
parser.add_argument('--output-model-path', type=str, default=None,
                    help='出力エージェント用のモデルファイルパス（GGUF形式）')
parser.add_argument('--summary-model-path', type=str, default=None,
                    help='要約エンジン用のモデルファイルパス（GGUF形式）')

# エージェント別生成パラメータ
parser.add_argument('--internal-temp', type=float, default=None,
                    help='内部エージェントの温度パラメータ（デフォルト: 0.7）')
parser.add_argument('--output-temp', type=float, default=None,
                    help='出力エージェントの温度パラメータ（デフォルト: 0.7）')
parser.add_argument('--summary-temp', type=float, default=None,
                    help='要約エンジンの温度パラメータ（デフォルト: 0.6）')

parser.add_argument('--internal-tokens', type=int, default=None,
                    help='内部エージェントの最大トークン数（デフォルト: 200）')
parser.add_argument('--output-tokens', type=int, default=None,
                    help='出力エージェントの最大トークン数（デフォルト: 250）')
parser.add_argument('--summary-tokens', type=int, default=None,
                    help='要約エンジンの最大トークン数（デフォルト: 150）')

# 全体的な生成パラメータ（空レス対策）
parser.add_argument('--max-new-tokens', type=int, default=128,
                    help='最大新規トークン数（デフォルト: 128、空レス対策）')

# 並列処理の安全性に関するオプション
parser.add_argument('--safe-parallel', action='store_true', 
                    help='安全な並列処理モード（GGMLエラー回避のためのグローバルロックを使用）')
parser.add_argument('--max-workers', type=int, default=0, 
                    help='並列処理時の最大ワーカー数（0: 自動決定）')

# Slotアーキテクチャオプション
parser.add_argument('--slots', action='store_true', 
                    help='Slotアーキテクチャモードを有効化（多角的議論システム）')
parser.add_argument('--parallel-only', action='store_true',
                    help='並列実行のみ（議論・相互参照を無効化）')
# Slot関連の詳細設定
parser.add_argument('--slot-temperature', type=float, default=0.7,
                    help='Slotの温度パラメータ（デフォルト: 0.7）')
parser.add_argument('--slot-max-length', type=int, default=300,
                    help='Slotの最大出力長（デフォルト: 300）')
parser.add_argument('--slot-similarity-threshold', type=float, default=0.7,
                    help='Slot間の類似度閾値（デフォルト: 0.7）')
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
    
    # Slotモードの場合はSlot情報を表示
    if slm.use_slots and hasattr(slm.blackboard, 'get_slot_debug_view'):
        print("\n[DEBUG] Slot情報:")
        slot_debug = slm.blackboard.get_slot_debug_view()
        slot_info = slot_debug.get('slot_info', {})
        print(f"  登録Slot数: {len(slot_info.get('registered_slots', []))}")
        print(f"  総エントリ数: {slot_info.get('total_entries', 0)}")
        
        for slot_name, summary in slot_info.get('slot_summaries', {}).items():
            latest_text = summary.get('latest_text', '')
            if latest_text:
                preview = latest_text[:50] + "..." if len(latest_text) > 50 else latest_text
                print(f"    {slot_name}: {preview}")
    else:
        # 要約結果を表示（従来モード）
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
            from murmurnet.modules.input_reception import InputReception
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

def _build_agent_models_config(args):
    """エージェント別モデル設定を構築（混在使用対応）"""
    agent_models = {}
    
    # 内部エージェント設定
    internal_config = {}
    if args.internal_model or args.internal_model_path:
        # モデルタイプを決定（明示的指定 or 推論）
        if args.internal_model_type:
            internal_config['model_type'] = args.internal_model_type
        elif args.internal_model_path:
            internal_config['model_type'] = 'llama'
        elif args.internal_model:
            internal_config['model_type'] = 'huggingface'
        else:
            internal_config['model_type'] = args.model_type
        
        # モデル指定
        if internal_config['model_type'] == 'huggingface':
            internal_config['model_name'] = args.internal_model
        elif internal_config['model_type'] == 'llama':
            internal_config['model_path'] = args.internal_model_path
    
    if args.internal_temp is not None:
        internal_config['temperature'] = args.internal_temp
    if args.internal_tokens is not None:
        internal_config['max_tokens'] = args.internal_tokens
    
    if internal_config:
        agent_models['internal_agents'] = internal_config
    
    # 出力エージェント設定
    output_config = {}
    if args.output_model or args.output_model_path:
        # モデルタイプを決定
        if args.output_model_type:
            output_config['model_type'] = args.output_model_type
        elif args.output_model_path:
            output_config['model_type'] = 'llama'
        elif args.output_model:
            output_config['model_type'] = 'huggingface'
        else:
            output_config['model_type'] = args.model_type
        
        # モデル指定
        if output_config['model_type'] == 'huggingface':
            output_config['model_name'] = args.output_model
        elif output_config['model_type'] == 'llama':
            output_config['model_path'] = args.output_model_path
    
    if args.output_temp is not None:
        output_config['temperature'] = args.output_temp
    if args.output_tokens is not None:
        output_config['max_tokens'] = args.output_tokens
    
    if output_config:
        agent_models['output_agent'] = output_config
    
    # 要約エンジン設定
    summary_config = {}
    if args.summary_model or args.summary_model_path:
        # モデルタイプを決定
        if args.summary_model_type:
            summary_config['model_type'] = args.summary_model_type
        elif args.summary_model_path:
            summary_config['model_type'] = 'llama'
        elif args.summary_model:
            summary_config['model_type'] = 'huggingface'
        else:
            summary_config['model_type'] = args.model_type
        
        # モデル指定
        if summary_config['model_type'] == 'huggingface':
            summary_config['model_name'] = args.summary_model
        elif summary_config['model_type'] == 'llama':
            summary_config['model_path'] = args.summary_model_path
    
    if args.summary_temp is not None:
        summary_config['temperature'] = args.summary_temp
    if args.summary_tokens is not None:
        summary_config['max_tokens'] = args.summary_tokens
    
    if summary_config:
        agent_models['summary_engine'] = summary_config
    
    return agent_models

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
        
        # 生成パラメータ（空レス対策）
        "max_new_tokens": args.max_new_tokens,  # CLI引数から取得
        
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
        
        # エージェント別モデル設定
        "agent_models": _build_agent_models_config(args)
    }

    # CLI引数を最後に上書き（CLI引数の優先順位を確保）
    config.update(vars(args))

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
        
    # エージェント別モデル設定の表示
    if config.get('agent_models'):
        print("\n=== エージェント別モデル設定 ===")
        agent_models = config['agent_models']
        
        if 'internal_agents' in agent_models:
            internal = agent_models['internal_agents']
            model_info = internal.get('model_name') or internal.get('model_path', '共通設定')
            model_type = internal.get('model_type', 'unknown')
            print(f"内部エージェント: {model_info} ({model_type}) "
                  f"(温度: {internal.get('temperature', '共通設定')}, "
                  f"トークン: {internal.get('max_tokens', '共通設定')})")
        
        if 'output_agent' in agent_models:
            output = agent_models['output_agent']
            model_info = output.get('model_name') or output.get('model_path', '共通設定')
            model_type = output.get('model_type', 'unknown')
            print(f"出力エージェント: {model_info} ({model_type}) "
                  f"(温度: {output.get('temperature', '共通設定')}, "
                  f"トークン: {output.get('max_tokens', '共通設定')})")
        
        if 'summary_engine' in agent_models:
            summary = agent_models['summary_engine']
            model_info = summary.get('model_name') or summary.get('model_path', '共通設定')
            model_type = summary.get('model_type', 'unknown')
            print(f"要約エンジン: {model_info} ({model_type}) "
                  f"(温度: {summary.get('temperature', '共通設定')}, "
                  f"トークン: {summary.get('max_tokens', '共通設定')})")
        print("=" * 35)
    else:
        print("\n全エージェントで共通モデルを使用")
    
    # Slotアーキテクチャ設定
    if args.slots:
        print(f"\n=== Slotアーキテクチャ設定 ===")
        config['use_slots'] = True
        config['slot_temperature'] = args.slot_temperature
        config['slot_max_output_length'] = args.slot_max_length
        config['slot_similarity_threshold'] = args.slot_similarity_threshold
        
        # 実行モード設定（デフォルトは協調）
        if args.parallel_only:
            config['use_collaboration'] = False
            print("📋 並列実行モード: Slotが独立実行")
        else:
            config['use_collaboration'] = True
            print("🤝 協調モード: Slotが議論・相互参照")
        
        # Slotモード時は従来のAgentは無効化
        config['iterations'] = 1  # Slotは単一実行
        config['use_summary'] = False  # 従来の要約は無効
        
        print(f"Slot温度: {config['slot_temperature']}")
        print(f"Slot最大長: {config['slot_max_output_length']}")
        print(f"類似度閾値: {config['slot_similarity_threshold']}")
        print("=" * 30)
    else:
        config['use_slots'] = False
        config['use_collaboration'] = False
        
    # SLMインスタンス作成
    global _global_slm
    slm = DistributedSLM(config)
    _global_slm = slm  # グローバル参照を設定
    
    mode_info = "Slotアーキテクチャ" if args.slots else f"{args.agents}エージェント, {args.iterations}反復"
    if args.slots and args.parallel_only:
        mode_info += " (並列実行)"
    elif args.slots:
        mode_info += " (協調モード)"
    print(f"MurmurNet Console ({mode_info})")
    print("終了するには 'quit' または 'exit' を入力してください")
    
    if args.parallel and not args.slots:
        print("[設定] 並列処理: 有効")
    if not args.no_summary and not args.slots:
        print("[設定] 要約機能: 有効")
    if args.slots:
        collaboration_mode = "並列実行" if args.parallel_only else "協調モード"
        print(f"[設定] Slotアーキテクチャ: 有効 ({collaboration_mode})")
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
