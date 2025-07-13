# MurmurNet - 分散SLMシステム

分散創発型言語モデルアーキテクチャによる高度な対話生成システム

## 概要

MurmurNetは、複数の小型言語モデル（Small Language Model）を協調させることで、より高度な対話応答を生成する分散型システムです。エージェント別にモデルを指定することで、各役割に特化した最適化が可能です。

## 🚀 Phase 5-8: 分散システム拡張機能

### 新機能概要
最新のアップデート（Phase 5-8）では、エンタープライズグレードの分散システム機能を追加しました：

#### Phase 5: 分散協調メカニズム
- **分散合意アルゴリズム**: Raft、PBFT、Simple Majorityによる分散合意
- **負荷分散**: ハイブリッド負荷分散戦略（ラウンドロビン、最少負荷、能力ベース）
- **障害検出・フェイルオーバー**: 自動的な障害検出と復旧機能

#### Phase 6: 監視・メトリクス
- **Prometheusメトリクス統合**: リアルタイムメトリクス収集・配信
- **アラート管理**: 閾値ベースのアラート発報とSlack/Webhook通知
- **Grafanaダッシュボード**: 分散システム可視化ダッシュボード

#### Phase 7: オートスケーリング
- **動的スケーリング**: CPU/メモリ/キュー長ベースの自動スケーリング
- **予測スケーリング**: 機械学習による予測的スケーリング（実験的）
- **Kubernetes統合**: K8sクラスターでの自動スケーリング

#### Phase 8: パフォーマンス最適化
- **レイテンシ最適化**: リアルタイムレイテンシ追跡・ボトルネック特定
- **メモリ効率向上**: インテリジェントGC管理・オブジェクトプール
- **ネットワーク最適化**: LZ4/Zstd圧縮・バッチング・接続プーリング

## 主な特徴

- **分散協調アーキテクチャ**: 複数のエージェントが連携して知識を創発
- **エージェント別モデル設定**: 内部エージェント、出力エージェント、要約エンジンに個別のモデルを指定可能
- **役割ベース処理**: 研究者、批判者、作家など、質問に応じた動的役割割り当て
- **反復処理**: 複数ラウンドの思考を通じた応答品質向上
- **RAG統合**: 外部知識ベースとの連携
- **並列処理サポート**: 高速化のための並列実行
- **柔軟なプロンプト管理**: モデル固有のプロンプトテンプレート対応
- **🆕 エンタープライズ機能**: 分散協調、監視、オートスケーリング、パフォーマンス最適化

## インストール

### 必要条件

- Python 3.8+
- PyTorch
- transformers
- その他依存関係

### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/your-username/MurmurNet.git
cd MurmurNet

# 依存関係のインストール
pip install -r requirements.txt

# モデルキャッシュディレクトリの作成
mkdir -p cache/models

# 🆕 分散システム拡張機能のテスト
python test_distributed_extensions.py
```

### 🆕 分散システム拡張機能の設定

分散システム機能を有効にするには、設定ファイルで以下の項目を設定してください：

```yaml
# config_slots.yaml または config.yaml に追加

# Phase 5: 分散協調メカニズム
distributed_coordination:
  enable: true
  consensus_algorithm: "simple_majority"  # または "raft", "pbft"
  load_balance_strategy: "hybrid"

# Phase 6: 監視・メトリクス
monitoring:
  enable_prometheus: true
  prometheus_port: 8000
  alert_check_interval: 30.0
  notification_channels:
    - type: "webhook"
      url: "http://localhost:8080/alerts"

# Phase 7: オートスケーリング
autoscaling:
  enable: true
  scaling_strategy: "hybrid"
  min_workers: 1
  max_workers: 10
  target_cpu_utilization: 0.7

# Phase 8: パフォーマンス最適化
performance_optimization:
  enable: true
  enable_compression: true
  compression_algorithm: "lz4"
  enable_auto_optimization: true
```

## 使用方法

### 基本的な使用例

#### 1. HuggingFaceモデルを使用した基本実行

```bash
python Console/console_app.py \
  --model-type huggingface \
  --model-name llm-jp/llm-jp-3-150m-instruct3 \
  --agents 2 \
  --iterations 1
```

#### 2. Llamaモデル（GGUF形式）を使用した実行

```bash
python Console/console_app.py \
  --model-type llama \
  --model-path ./models/gemma-3-1b-it-q4_0.gguf \
  --agents 3 \
  --iterations 2
```

### エージェント別モデル設定（推奨）

#### 3. 内部エージェント用LLM-jp、出力・要約用Gemma3での並列実行

```bash
# HuggingFaceモデルでの統一設定（推奨）
python Console/console_app.py \
  --model-type huggingface \
  --model-name llm-jp/llm-jp-3-150m-instruct3 \
  --internal-model llm-jp/llm-jp-3-150m-instruct3 \
  --output-model google/gemma-3-1b-it \
  --summary-model google/gemma-3-1b-it \
  --internal-temp 0.8 \
  --output-temp 0.6 \
  --summary-temp 0.5 \
  --internal-tokens 200 \
  --output-tokens 300 \
  --summary-tokens 150 \
  --agents 3 \
  --iterations 2 \
  --parallel \
  --debug
```

#### 4. 高速なLLM-jp-3での軽量実行

```bash
python Console/console_app.py \
  --model-type huggingface \
  --model-name llm-jp/llm-jp-3-150m-instruct3 \
  --internal-temp 0.7 \
  --output-temp 0.6 \
  --summary-temp 0.5 \
  --agents 2 \
  --iterations 1 \
  --parallel
```

**注意**: 
- HuggingFaceモデル間では異なるモデルの混在使用が可能です
- Llamaモデル（GGUF）とHuggingFaceモデルの混在使用は現在サポートされていません
- メモリが限られている場合は、同一の小型モデル（LLM-jp-3-150m）の使用を推奨します

#### 5. 高度な設定例（デバッグ有効）

```bash
python Console/console_app.py \
  --model-type huggingface \
  --model-name llm-jp/llm-jp-3-150m-instruct3 \
  --internal-model llm-jp/llm-jp-3-150m-instruct3 \
  --output-model google/gemma-3-1b-it \
  --summary-model google/gemma-3-1b-it \
  --internal-temp 0.9 \
  --output-temp 0.5 \
  --summary-temp 0.4 \
  --agents 4 \
  --iterations 3 \
  --parallel \
  --no-local-files \
  --debug \
  --log
``` \
  --output-temp 0.5 \
  --summary-temp 0.4 \
  --agents 4 \
  --iterations 3 \
  --parallel \
  --no-local-files \
  --debug \
  --log
```

## コマンドラインオプション

### 基本オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model-type` | モデルタイプ (`huggingface` or `llama`) | **必須** |
| `--model-name` | HuggingFaceモデル名 | HuggingFace使用時**必須** |
| `--model-path` | Llamaモデルファイルパス | Llama使用時**必須** |
| `--agents` | エージェント数 | 2 |
| `--iterations` | 反復回数 | 1 |

### エージェント別モデル設定

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--internal-model` | 内部エージェント用モデル名 | 共通設定を使用 |
| `--output-model` | 出力エージェント用モデル名 | 共通設定を使用 |
| `--summary-model` | 要約エンジン用モデル名 | 共通設定を使用 |

### 生成パラメータ

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--internal-temp` | 内部エージェントの温度パラメータ | 0.7 |
| `--output-temp` | 出力エージェントの温度パラメータ | 0.7 |
| `--summary-temp` | 要約エンジンの温度パラメータ | 0.6 |
| `--internal-tokens` | 内部エージェントの最大トークン数 | 200 |
| `--output-tokens` | 出力エージェントの最大トークン数 | 250 |
| `--summary-tokens` | 要約エンジンの最大トークン数 | 150 |

### その他オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--parallel` | 並列処理を有効化 | 無効 |
| `--no-summary` | 要約機能を無効化 | 有効 |
| `--debug` | デバッグ情報を表示 | 無効 |
| `--log` | ログをファイルに保存 | 無効 |
| `--no-local-files` | オンラインからモデルをダウンロード | ローカルファイル優先 |
| `--cache-folder` | モデルキャッシュフォルダ | `./cache/models` |

## エージェント別モデル設定

### 設定の利点

MurmurNetでは、各エージェントタイプに異なるモデルを割り当てることで、処理効率と品質を最適化できます：

- **内部エージェント**: 高速な小型モデル（LLM-jp-3-150m-instruct3）で多角的思考を並列実行
- **出力エージェント**: 高品質モデル（Gemma-3-1b-it）で最終的な文章生成
- **要約エンジン**: 中型モデルで効率的な情報集約

### 推奨構成パターン

#### パターン1: 軽量高速構成
```
内部: llm-jp/llm-jp-3-150m-instruct3 (温度0.8)
出力: llm-jp/llm-jp-3-150m-instruct3 (温度0.6)  
要約: llm-jp/llm-jp-3-150m-instruct3 (温度0.5)
```

#### パターン2: バランス構成（推奨）
```
内部: llm-jp/llm-jp-3-150m-instruct3 (温度0.8)
出力: google/gemma-3-1b-it (温度0.6)
要約: google/gemma-3-1b-it (温度0.5)
```

#### パターン3: 高品質構成
```
内部: google/gemma-3-1b-it (温度0.8)
出力: より大きなモデル (温度0.6)
要約: より大きなモデル (温度0.5)
```

## アーキテクチャ

### システム構成

```
ユーザー入力
    ↓
[入力受信] → [RAG検索] → [黒板への書き込み]
    ↓
[内部エージェント群] ← → [黒板システム]
    ↓
[要約エンジン] → [最終統合]
    ↓
[出力エージェント] → 最終応答
```

### エージェントの役割

1. **内部エージェント群**
   - 研究者、批判者、作家、専門家など多角的な視点
   - 質問タイプに応じた動的役割割り当て
   - 他エージェントの意見を参考にした協調的思考
   - **コンテキスト統合**: ユーザー質問 + RAG情報 + 会話履歴 + 他エージェント出力を全て含むプロンプト

2. **要約エンジン**
   - 複数エージェントの出力を簡潔に要約
   - 情報の重複排除と重要ポイント抽出

3. **出力エージェント**
   - 全ての情報を統合した最終応答生成
   - 一貫性のある自然な文章作成

### 黒板システム

- **共有メモリ**: 全エージェントが情報を読み書き
- **情報統合**: RAG、会話履歴、エージェント出力の集約
- **状態管理**: 反復処理の状態追跡
- **コンテキスト伝播**: 内部エージェントのプロンプトに全ての利用可能情報を自動統合

## パフォーマンス最適化

### 推奨設定

- **CPU環境**: `--agents 2 --iterations 1`で高速動作
- **GPU環境**: `--agents 4 --iterations 2 --parallel`で高品質
- **メモリ制約**: `--no-summary`で軽量化

### モデル選択の指針

- **小型高速**: `llm-jp-3-150m-instruct3` (150M) - 内部エージェント向け
- **バランス**: `google/gemma-3-1b-it` (1B) - 出力・要約向け
- **高品質**: より大きなモデル（要GPU）

### エージェント別モデル組み合わせ例

1. **軽量構成**: 全エージェントでLLM-jp-3-150m-instruct3
2. **バランス構成**: 内部=LLM-jp-3-150m-instruct3、出力/要約=Gemma-3-1b-it  
3. **高品質構成**: より大きなモデルの組み合わせ

## トラブルシューティング

### よくある問題

1. **モデルキャッシュが見つからない**
   ```bash
   # オンラインダウンロードを許可
   --no-local-files
   ```

2. **メモリ不足**
   ```bash
   # エージェント数とトークン数を削減
   --agents 2 --internal-tokens 100 --output-tokens 150
   ```

3. **応答が空または短い**
   ```bash
   # 温度パラメータを調整
   --internal-temp 0.8 --output-temp 0.7
   ```

### デバッグ方法

```bash
# 詳細ログとデバッグ情報を有効化
python Console/console_app.py \
  --model-type huggingface \
  --model-name llm-jp/llm-jp-3-150m-instruct3 \
  --debug \
  --log
```

## 開発

### テスト実行

```bash
# エージェント別モデル設定のテスト
python test_agent_models.py

# システムプロンプト調整のテスト
python test_system_prompt_tuning.py

# LLM-jp固有のテスト
python test_llm_jp_fixed_v2.py
```

### カスタマイズ

- **新しい役割の追加**: `modules/agent_pool.py`の`role_templates`を編集
- **プロンプトテンプレート**: `modules/prompt_manager.py`でモデル固有テンプレートを追加
- **新しいモデル対応**: `modules/model_factory.py`でモデルローダーを拡張

## 作者

Yuhi Sonoki

---

