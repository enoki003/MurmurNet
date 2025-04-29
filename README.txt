# MurmurNet 起動方法

## 必要条件
- Python 3.12 以上
- 依存パッケージ（必要に応じてrequirements.txtを参照、またはpipでインストール）
- モデルファイル（models/gemma-3-1b-it-q4_0.gguf）はgit管理外です。別途配置してください。

## 起動方法
1. コマンドプロンプトまたはPowerShellで、プロジェクトのルートディレクトリ（課題研究フォルダ）に移動します。
2. 以下のコマンドでテスト用スクリプトを実行できます：

```
python murmurnet/test_script.py
```

または、分散SLMのメインを実行する場合：

```
python murmurnet/distributed_slm.py
```

## 注意
- モデルファイルは大容量のため、別途ダウンロード・配置してください。
- 詳細な使い方や設定は、計画.txtや各モジュールのdocstringを参照してください。
