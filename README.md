# MurmurNet - 分散創発型言語モデルシステム

MurmurNetは、複数の小規模言語モデル（SLM）が協調して動作する分散型アーキテクチャを採用した言語モデルシステムです。「黒板」と呼ばれる共有メモリを介して通信し、より洗練された応答を生成します。

**Author: Yuhi Sonoki**

## 主な特長

- **分散創発型アーキテクチャ**: 複数の小規模言語モデルが協調動作
- **黒板（共有メモリ）アーキテクチャ**: エージェント間の効率的な情報共有
- **RAG (Retrieval-Augmented Generation)**: ZIMファイルを使った知識検索
- **非同期処理**: 複数エージェントの並列実行によるパフォーマンス向上
- **要約機能**: 会話コンテキストの効率的な管理
- **パフォーマンス最適化**: 事前リパック、短文スキップ、テンプレート最適化による高速化
- **ベンチマーク機能**: 定量的なパフォーマンス測定とボトルネック分析
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

4. (オプション) パフォーマンス最適化のため事前リパック済みモデルを作成：
   ```
   python create_model.py
   ```

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

# 事前リパック済みモデルを作成
python create_model.py

# パフォーマンス最適化を統合実行
python optimize.py

# ベンチマーク測定
python benchmark.py
```

#### コマンドラインオプション
- `--debug`: デバッグ情報を表示
- `--log`: ログをファイルに保存
- `--iter N`: 反復回数（デフォルト: 1）
- `--agents N`: エージェント数（デフォルト: 2）
- `--no-summary`: 要約機能を無効化
- `--parallel`: 並列処理を有効化
- `--rag-mode {dummy|zim}`: RAGモードの選択
- `--zim-path PATH`: ZIMファイルのパス

### プログラムからの使用

Pythonプログラムから直接インポートして使用できます：

```python
from murmurnet.distributed_slm import DistributedSLM
import asyncio

# 基本設定
config = {
    "num_agents": 2,
    "iterations": 1,
    "use_summary": True,
    "rag_mode": "zim",
    "zim_path": "path/to/wikipedia.zim"
}

# インスタンス化と実行
slm = DistributedSLM(config)
response = asyncio.run(slm.generate("AIは教育をどのように変えますか？"))
print(response)
```

## アーキテクチャ

MurmurNetは以下のコンポーネントで構成されています：

```
DistributedSLM
 ├─ InputReception - 入力処理
 ├─ Blackboard - 共有メモリ
 ├─ SummaryEngine - 要約処理
 ├─ AgentPoolManager - エージェント管理
 │    ├─ Agent 1
 │    ├─ Agent 2
 │    └─ ...
 ├─ RAGRetriever - 知識検索
 │    ├─ Dummy Mode - シンプル検索
 │    └─ ZIM Mode - Wikipedia検索
 └─ OutputAgent - 最終応答生成
```

### RAG (検索拡張生成)

MurmurNetには二種類のRAGモードがあります：

1. **ダミーモード**: 外部知識なしの基本動作モード
2. **ZIMモード**: WikipediaのZIMファイルを使用した検索拡張モード
   - 必要なもの：Kiwix ZIMファイル（[ダウンロード](https://wiki.kiwix.org/wiki/Content)）
   - 特徴：埋め込みベクトルを使った意味検索

## パフォーマンス最適化

MurmurNetには複数の最適化機能が含まれています：

### 事前リパック処理
```bash
python create_model.py
```
- 毎回のリパック処理（約2.5s）を削減
- 起動時間を大幅短縮（2.9s → 1.2s）

### 要約エンジン最適化
```python
# 短文（64文字未満）の場合、LLM呼び出しをスキップ
if len(input_text) < 64:
    return None  # 要約処理をバイパス
```

### テンプレート最適化
- system role警告を解消（300ms削減）
- チャットテンプレートの動的修正

### 推論速度向上
- スレッド数最適化（--threads 6）
- バッチサイズ拡大（--n_batch 1024）
- KVキャッシュ共有（n_seq_max=2）

### ベンチマーク機能
```bash
python benchmark.py
```
- 処理時間の詳細測定
- メモリ使用量監視
- 改善効果の定量化

**最適化効果（実測）：**
- 総所要時間: 31.28s → 12s以下（▼60%）
- メモリ使用量: 1328MB → 1078MB（▼250MB）
- トークン生成速度: 13 t/s → 20 t/s（目標）

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
Performance/
├── create_repacked_model.py    # 事前リパック処理
├── summary_optimizer.py       # 要約処理最適化
├── template_optimizer.py      # テンプレート最適化
├── output_agent_optimizer.py  # 推論速度最適化
├── performance_benchmark.py   # ベンチマーク測定ツール
└── run_optimization.py        # 統合最適化実行スクリプト
```

## 今後の開発予定

- GPU対応の強化
- モデルの追加（Llama 3, Mistral AI等）
- WebUI実装
- 多言語対応の強化
- エージェント役割の拡張
- さらなるパフォーマンス最適化
  - 動的スレッド調整
  - インテリジェントキャッシング
  - 分散処理対応

## トラブルシューティング

**Q: モデルの読み込みでエラーが発生する**
A: モデルファイルのパスが正しく設定されているか確認してください。

**Q: ZIMモードが動作しない**
A: ZIMファイルのパスが正しいか、sentence-transformersがインストールされているか確認してください。

**Q: メモリエラーが発生する**
A: エージェント数を減らすか、n_ctxパラメータを小さくしてください。

**Q: パフォーマンスが向上しない**
A: 以下を確認してください：
- 事前リパック処理が完了しているか
- 最適化スクリプトが正しく実行されているか
- ベンチマークで効果を測定してください

## ライセンス

MITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 謝辞

- [Llama.cpp](https://github.com/ggerganov/llama.cpp)プロジェクト
- [Gemma](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)モデル
- [Sentence Transformers](https://www.sbert.net/)
- [libzim](https://github.com/openzim/libzim)および[Kiwix](https://www.kiwix.org/)プロジェクト


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/enoki003/MurmurNet)