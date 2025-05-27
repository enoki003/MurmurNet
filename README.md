# MurmurNet - 分散創発型言語モデルシステム

MurmurNetは、複数の小規模言語モデル（SLM）が協調して動作する分散型アーキテクチャを採用した言語モデルシステムです。「黒板」と呼ばれる共有メモリを介して通信し、より洗練された応答を生成します。

**Author: Yuhi Sonoki**

## 🎯 更新情報 (2025年5月26日)

### ✅ 完了した改良点
- **モック実装の完全除去**: すべてのダミー・モック実装を削除し、実機能のみで動作
- **Gemma3モデル対応**: 最新のGemma-3-1B-ITモデルを標準搭載
- **ZIMベースRAG**: Wikipediaなどの実データに基づく検索拡張生成
- **埋め込みベースRAG**: 高速で効率的な意味検索機能
- **構文エラー修正**: すべてのモジュールの構文問題を解決
- **設定管理強化**: 型安全な設定管理とバリデーション機能

### 🔧 技術的改善
- ConfigManagerクラスの完全リファクタリング
- RAGRetrieverのlibzim API互換性向上
- Blackboardクラスのメモリ管理機能追加
- モデルファクトリーからのモック排除
- Console_app.pyのRAGモード選択肢更新

## 🔄 最新アップデート (2025年5月26日)

✅ **モック実装の完全除去**
- すべてのダミー/モック実装を削除
- テスト用の仮実装を本格実装に移行
- RAGモードから「dummy」を除去し、「zim」と「embedding」のみをサポート

✅ **Gemma3モデルへの移行**
- デフォルトモデルをLlamaからGemma3に変更
- 実運用準備完了

✅ **システム最適化**
- ZIMベースRAG検索の安定化
- 埋め込みベースRAG検索の追加
- エラーハンドリングの改善
- パフォーマンス向上

## 主な特長

- **分散創発型アーキテクチャ**: 複数の小規模言語モデルが協調動作
- **黒板（共有メモリ）アーキテクチャ**: エージェント間の効率的な情報共有
- **RAG (Retrieval-Augmented Generation)**: ZIMファイルを使った知識検索
- **非同期処理**: 複数エージェントの並列実行によるパフォーマンス向上
- **要約機能**: 会話コンテキストの効率的な管理
- **モジュール化設計**: 拡張性と保守性の高いコードベース

## システム要件

- Python 3.10以上
- 8GB以上のRAM
- 必須: Gemma 3 1Bモデル

## インストール方法

1. リポジトリをクローンする：
   ```
   git clone https://github.com/yuhi-sonoki/murmurnet.git
   cd murmurnet
   ```

2. 依存パッケージをインストールする：
   ```
   pip install -r requirements.txt
   ```

3. Gemmaモデルをダウンロードして配置する：
   - [Gemma-3-1B-IT-Q4_0.gguf](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)をダウンロード
   - `models/gemma-3-1b-it-q4_0.gguf`に配置（または設定ファイルでパスを指定）
   - チャットテンプレートファイル（`gemma3_template.txt`）も用意

## 基本的な使い方

### コンソールアプリケーション（CLI）

コンソールから直接MurmurNetを使用できます：

```bash
# 基本的な使用方法
python Console/console_app.py

# デバッグ情報付き
python Console/console_app.py --debug

# RAGモードをZIMファイルに設定
python Console/console_app.py --rag-mode zim --zim-path "パス/to/wikipedia.zim"

# 並列処理と要約機能を調整
python Console/console_app.py --agents 3 --iter 2 --parallel
```

#### コマンドラインオプション
- `--debug`: デバッグ情報を表示
- `--log`: ログをファイルに保存
- `--iter N`: 反復回数（デフォルト: 1）
- `--agents N`: エージェント数（デフォルト: 2）
- `--no-summary`: 要約機能を無効化
- `--parallel`: 並列処理を有効化
- `--rag-mode {zim|embedding}`: RAGモードの選択
- `--zim-path PATH`: ZIMファイルのパス

### プログラムからの使用

Pythonプログラムから直接インポートして使用できます：

```python
from MurmurNet.distributed_slm import DistributedSLM
import asyncio

# 基本設定（gemma3モデル使用）
config = {
    "num_agents": 2,
    "iterations": 1,
    "use_summary": True,
    "model_type": "gemma3",  # Gemma-3-1B-ITモデル
    "rag_mode": "zim",       # ZIMファイルベースRAG
    "model_path": "models/gemma-3-1b-it-q4_0.gguf",
    "chat_template": "models/gemma3_template.txt"
}

# インスタンス化と実行
slm = DistributedSLM(config)
response = await slm.process_question("AIは教育をどのように変えますか？")
print(response)
```

## 🏗️ システムアーキテクチャ

MurmurNetは以下のコンポーネントで構成されています：

```
DistributedSLM (メインシステム)
 ├─ ConfigManager - 設定管理と型安全性
 ├─ InputReception - 入力処理・前処理
 ├─ Blackboard - 共有メモリ・エージェント間通信
 ├─ AgentPoolManager - エージェント管理・協調処理
 ├─ RAGRetriever - 知識検索（ZIM/埋め込みベース）
 ├─ ModelFactory - Gemma3モデル管理
 ├─ SummaryEngine - コンテキスト要約
 ├─ OutputAgent - 回答生成・統合
 └─ ConversationMemory - 会話履歴管理
 ├─ AgentPoolManager - エージェント管理
 │    ├─ Agent 1
 │    ├─ Agent 2
 │    └─ ... ├─ RAGRetriever - 知識検索
 │    ├─ ZIM Mode - Wikipedia検索
 │    └─ Embedding Mode - 埋め込みベース検索
 └─ OutputAgent - 最終応答生成
```

### RAG (検索拡張生成)

MurmurNetには二種類のRAGモードがあります：

1. **ZIMモード**: WikipediaのZIMファイルを使用した検索拡張モード
   - 必要なもの：Kiwix ZIMファイル（[ダウンロード](https://wiki.kiwix.org/wiki/Content)）
   - 特徴：実際のWikipediaデータを使った知識検索

2. **埋め込みモード**: 埋め込みベクトルベースの検索拡張モード
   - 特徴：事前学習された埋め込みモデルを使った意味検索
   - 高速で効率的な検索が可能

## プロジェクト構造

```
murmurnet/
├── distributed_slm.py      # メインモジュール
├── prompt_config.yaml      # プロンプト設定
├── test_script.py          # テストスクリプト
├── modules/                # 各モジュール
│   ├── agent_pool.py       # エージェント管理
│   ├── blackboard.py       # 共有メモリ
│   ├── input_reception.py  # 入力処理
│   ├── output_agent.py     # 出力生成
│   ├── rag_retriever.py    # 知識検索
│   └── summary_engine.py   # 要約処理
└── __pycache__/            # コンパイル済みファイル
Console/
├── console_app.py          # CLIアプリケーション
└── console_app.log         # ログファイル
```

## 今後の開発予定

- GPU対応の強化
- モデルの追加（Llama 3, Mistral AI等）
- WebUI実装
- 多言語対応の強化
- エージェント役割の拡張

## トラブルシューティング

**Q: モデルの読み込みでエラーが発生する**
A: モデルファイルのパスが正しく設定されているか確認してください。

**Q: ZIMモードが動作しない**
A: ZIMファイルのパスが正しいか、sentence-transformersがインストールされているか確認してください。

**Q: メモリエラーが発生する**
A: エージェント数を減らすか、n_ctxパラメータを小さくしてください。

## ライセンス

MITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 謝辞

- [Llama.cpp](https://github.com/ggerganov/llama.cpp)プロジェクト
- [Gemma](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)モデル
- [Sentence Transformers](https://www.sbert.net/)
- [libzim](https://github.com/openzim/libzim)および[Kiwix](https://www.kiwix.org/)プロジェクト


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/enoki003/MurmurNet)