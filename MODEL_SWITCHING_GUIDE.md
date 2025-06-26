# MurmurNet モデル切り替え機能

MurmurNetでは、GemmaモデルとHuggingFace llm-jp-3-150mモデルを簡単に切り替えて使用できます。

## サポートされているモデル

### 1. Llamaモデル (Gemma)
- **モデルタイプ**: `llama`
- **形式**: GGUF形式のローカルファイル
- **特徴**: 量子化済みで高速、CPUで効率的に動作

### 2. HuggingFaceモデル (llm-jp-3-150m)
- **モデルタイプ**: `huggingface`
- **モデル名**: `llm-jp/llm-jp-3-150m`
- **特徴**: 150Mパラメータの軽量日本語モデル、自動ダウンロード

## 使用方法

### 1. コマンドライン引数での切り替え

```bash
# Gemmaモデルを使用（デフォルト）
python Console/console_app.py --model-type llama

# llm-jp-3-150mを使用
python Console/console_app.py --model-type huggingface

# 別のHuggingFaceモデルを指定
python Console/console_app.py --model-type huggingface --huggingface-model "rinna/japanese-gpt2-medium"
```

### 2. 設定ファイルでの切り替え

`config.yaml`を編集してモデルを切り替えます：

```yaml
# Gemmaモデルを使用する場合
model_type: llama
model_path: "C:\\Users\\admin\\Desktop\\課題研究\\models\\gemma-3-1b-it-q4_0.gguf"
chat_template: "C:\\Users\\admin\\Desktop\\課題研究\\models\\gemma3_template.txt"

# llm-jp-3-150mを使用する場合
model_type: huggingface
huggingface_model_name: "llm-jp/llm-jp-3-150m"
device: "cpu"
torch_dtype: "auto"
```

### 3. プログラム内での切り替え

```python
from MurmurNet.distributed_slm import DistributedSLM

# Gemmaモデル設定
gemma_config = {
    "model_type": "llama",
    "model_path": "path/to/gemma-model.gguf",
    "n_ctx": 2048,
    "temperature": 0.7,
    "max_tokens": 256
}

# llm-jp-3-150mモデル設定
llm_jp_config = {
    "model_type": "huggingface",
    "huggingface_model_name": "llm-jp/llm-jp-3-150m",
    "device": "cpu",
    "temperature": 0.7,
    "max_tokens": 256
}

# 使用例
slm_gemma = DistributedSLM(gemma_config)
slm_llm_jp = DistributedSLM(llm_jp_config)
```

## モデル別の特徴と推奨用途

### Gemmaモデル
- **メリット**: 量子化済みで高速、メモリ使用量が少ない
- **推奨用途**: リアルタイム応答が必要な場合、リソースが限られた環境
- **設定のポイント**: `n_threads`でCPUスレッド数を調整

### llm-jp-3-150m
- **メリット**: 日本語に特化、自動ダウンロード、比較的軽量
- **推奨用途**: 日本語の処理品質を重視する場合、実験的用途
- **設定のポイント**: `device`でCPU/GPU選択、`torch_dtype`で精度調整

## パフォーマンス比較

| 項目 | Gemma (llama) | llm-jp-3-150m (huggingface) |
|------|---------------|------------------------------|
| 初期化時間 | 中程度 | 長い（初回ダウンロード） |
| 推論速度 | 高速 | 中程度 |
| メモリ使用量 | 少ない | 中程度 |
| 日本語品質 | 良好 | 優秀 |
| セットアップ | 手動配置必要 | 自動ダウンロード |

## トラブルシューティング

### HuggingFaceモデルが動作しない場合

1. **依存関係の確認**:
```bash
pip install transformers torch
```

2. **ネットワーク接続の確認**: 初回実行時はモデルのダウンロードが必要

3. **キャッシュディレクトリの確認**: `cache/models`ディレクトリが作成可能か確認

### Gemmaモデルが動作しない場合

1. **ファイルパスの確認**: GGUFファイルが正しいパスに存在するか確認
2. **llama-cpp-pythonの確認**:
```bash
pip install llama-cpp-python
```

## テスト方法

モデル切り替え機能のテストを実行できます：

```bash
python test_model_switching.py
```

このスクリプトでは以下をテストします：
- 両モデルの利用可能性チェック
- 基本的な推論テスト
- 設定ファイル読み込みテスト
- 依存関係チェック

## 高度な設定

### 並列処理での使用

```yaml
# config.yaml
use_parallel: true
num_agents: 4
model_type: huggingface  # 軽量なllm-jp-3-150mで並列処理
```

### メモリ最適化

```yaml
# HuggingFaceモデル用
torch_dtype: "float16"  # メモリ使用量を削減
device: "cpu"           # GPU使用を避ける場合
```

## 今後の拡張予定

- より多くのHuggingFaceモデルのサポート
- GPU使用時の最適化
- モデル間でのアンサンブル推論
- 動的モデル切り替え機能
