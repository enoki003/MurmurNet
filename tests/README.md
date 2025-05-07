# MurmurNet テストファイル

このディレクトリには、MurmurNetシステムのテストに関連するファイルが含まれています。

## 含まれるファイル

- **zim_test.py**: ZIMモードのRAGリトリーバーをテストするスクリプト
- **blackbox_test_program.py**: ブラックボックステスト用のプログラム
- **output_agent_test.py**: 出力エージェントのテスト用スクリプト
- **test_log.txt**: テスト実行時のログファイル

## 使用方法

### ZIMモードのテスト

ZIMファイルを使ったRAG検索機能をテストするには、以下のコマンドを実行します：

```bash
python tests/zim_test.py
```

ZIMファイルのパスは、スクリプト内で設定されています。必要に応じて変更してください。

### ブラックボックステスト

システム全体のブラックボックステストを実行するには、以下のコマンドを実行します：

```bash
python tests/blackbox_test_program.py
```

### 出力エージェントテスト

出力エージェントの単体テストを実行するには、以下のコマンドを実行します：

```bash
python tests/output_agent_test.py
```

## 注意事項

- テストを実行する前に、必要なモデルファイルが正しく配置されていることを確認してください。
- ZIMモードのテストには、別途ZIMファイルが必要です。
- テスト結果はログファイルに出力されます。

## Author

Yuhi Sonoki 