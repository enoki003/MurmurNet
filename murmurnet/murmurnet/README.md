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

## 注意## モジュール詳細設計書: 分散型SLMアーキテクチャ（モジュール版）

**目的**：

- 本システムを「単一のLLMのように呼び出せるモジュール」として提供
- 内部で複数SLMエージェント＋黒板＋RAG＋メタ適応を実行
- 後段でTauri+React GUIと組み合わせ可能なAPI設計

---

### 1. 全体構成図

```
Client (Python API or Tauri/React)
    │  call .generate(request)
    ▼
DistributedSLM Module
 ├─ Input Reception (Input Preprocessor)
 ├─ Blackboard (Shared Memory)
 ├─ Summary Engine
 ├─ Agent Pool Manager
 │    ├─ Model Agent 1
 │    ├─ Agent 1
 │    ├─ Model Agent 2
 │    ├─ Agent 2
 │    └─ Agent N
 ├─ RAG Knowledge Base (RAG Retriever)
 ├─ Output Agent (Output Synthesizer)
```

---

### 2. モジュール一覧と責務

| モジュール                          | 主な責務                                                      |
| ------------------------------ | --------------------------------------------------------- |
| Input Reception                | ・ユーザー入力の受け取り・正規化・トークナイズ・embedding取得                       |
| Blackboard (Shared Memory)     | ・共有メモリAPI（write/read）提供・同期・状態管理                           |
| Summary Engine                 | ・黒板上の情報を要約し簡潔化                                            |
| Agent Pool Manager             | ・各エージェント（Model Agent / Functional Agent）のライフサイクル管理・並列実行制御 |
| Model Agent                    | ・主モデル推論（Gemma-3 1B のみ）                                    |
| Functional Agent               | ・役割別処理（要約Agent, 分析Agent, 批判Agentなど）                       |
| RAG Knowledge Base (Retriever) | ・オンデバイス検索（埋め込み近傍／SQLiteインデックス）・外部知識取得                     |
| Output Agent                   | ・最終応答生成（黒板＋要約結果の統合）                                       |
| Public API                     | ・`generate(request: str) -> response: str` インターフェース提供     |

---

### 3. API仕様（Public API）

```python
class DistributedSLM:
    def __init__(self, config: dict = None):
        """コンストラクタ：各モジュール初期化"""
    
    async def generate(self, input_text: str) -> str:
        """
        入力文字列から最終応答を生成
        ・エンドツーエンド非同期呼び出し
        """
```

- **config**: モジュールごとのパラメータ
  - `num_agents`: int
  - `rag_top_k`: int

---

### 4. 設計の注意点

- **KISS原則**（Keep It Simple, Stupid）を遵守: モジュールはできるだけ単純に保ち、不要な複雑化を避ける
- **プラグイン／拡張性**: 各エージェントやリトリーバーはインターフェースを実装するプラグイン形式とし、管理者が新しいAgentクラスを追加可能
- **設定ファイル駆動**: 全モジュールのパラメータ（エージェント数、RAG設定など）は外部YAML/JSONファイルで定義し、コードを変更せずに調整可能
- **フックポイント提供**: 入出力前後や黒板読書時点にフックを用意し、追加処理やログ出力を容易に差し込める設計
- **バージョン管理**: モジュールAPIにはバージョン番号を付与し、後方互換性を保ちながら段階的に機能追加できるようにする
- **ドキュメント自動生成**: config定義やプラグイン仕様からAPIドキュメントを自動生成し、管理者が変更点を即座に把握できるようにする

---

### 5. 動作フロー詳細

`DistributedSLM.generate(input_text)` 呼び出し時のシーケンス：

1. **Input Reception**

   - ユーザー入力文字列を受け取り、黒板に初期エントリとして書き込む
   - 必要に応じて会話履歴を取得し、黒板に追加

2. **Summary Engine**

   - 黒板上の全エントリを読み取り要約を生成
   - 要約テキストを黒板に追記

3. **Agent Pool Execution**

   - 設定された `num_agents` に従い、Model Agent および Functional Agent を並列起動
   - 各エージェントは最新の黒板コンテキストを読み込み、役割に応じた処理（分析・批判・補足など）を実行
   - 生成されたエントリを黒板に書き込む

4. **Iteration**

   - 手順2〜3 を指定人数分（例：2人）繰り返し、黒板上で協調的な議論を展開

5. **Final Synthesis**

   - 黒板上のエントリを再度要約・整理し、最終プロンプトと共に Model Agent に渡して応答を生成
   - 生成結果を取得し黒板に書き込む

6. **Output Return**

   - 最終応答テキストを呼び出し元に返却

このフローにより、内部で複数SLMエージェントが協調的に動作し、あたかも単一の高度なLLMのような一貫性ある応答を提供します。


- モデルファイルは大容量のため、別途ダウンロード・配置してください。
- 詳細な使い方や設定は、計画.txtや各モジュールのdocstringを参照してください。

推奨モデル
https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf

---

## 使い方（モジュールとして呼び出す例・引数と設定）

Pythonプログラムから直接importして利用できます。

```python
from murmurnet.distributed_slm import DistributedSLM
import asyncio

# 設定例
config = {
    "num_agents": 2,           # 並列エージェント数（デフォルト2）
    "rag_top_k": 3,             # RAGで検索する知識数（デフォルト3）
    # その他、各モジュールの詳細設定もdictで渡せます
}

slm = DistributedSLM(config)
result = asyncio.run(slm.generate("AIは教育をどう変えると思う？"))
print(result)
```

### DistributedSLMの主な引数・設定項目
- `num_agents` : 並列で動作させるエージェント数（int）
- `rag_top_k`  : RAG Knowledge Baseから取得する知識数（int）
- その他、各モジュール（AgentPool, RAGRetriever, OutputAgent等）の詳細設定もconfigで渡せます

#### 例: カスタム設定
```python
config = {
    "num_agents": 4,
    "rag_top_k": 5,
    "output": {"max_tokens": 256},
    # ...他の詳細設定も可
}
slm = DistributedSLM(config)
```

- `generate`は非同期関数です。Jupyterや他のasync環境では`await slm.generate(...)`で呼び出してください。
- configを省略した場合はデフォルト設定で動作します。

## （補足）コマンドラインでの動作確認

テストスクリプトを使ってコマンドラインから動作確認も可能です。

```
python murmurnet/test_script.py > output.txt
```

output.txtにAIの応答や黒板の内容が記録されます。

---

### CUI版の説明

#### 概要
CUI（キャラクターユーザーインターフェース）版のアプリケーションは、`Console/console_app.py` に実装されています。このアプリケーションを使用することで、コマンドラインから直接MurmurNetを操作できます。

#### 起動方法
1. コマンドプロンプトまたはPowerShellを開きます。
2. プロジェクトのルートディレクトリに移動します。
3. 以下のコマンドを実行してCUIアプリケーションを起動します：
   ```
   python Console/console_app.py
   ```

#### 主な機能
- **対話型モード**: ユーザーが入力した質問に対して、分散SLMが応答を生成します。
- **ログ記録**: 応答内容やエラー情報が `Console/log/` フォルダに記録されます。

#### 使用例
以下は、CUIアプリケーションの実行例です：
```
> python Console/console_app.py
MurmurNet CUI版へようこそ！
質問を入力してください（終了するには "exit" と入力）:
> AIは教育をどう変えると思う？
[応答]: AIは教育において個別化学習を促進し、教師の負担を軽減する可能性があります。
> exit
終了します。ご利用ありがとうございました！
```

#### 注意点
- ログファイルは `Console/log/` フォルダに日付ごとに保存されます。
- モデルファイルが正しく配置されていない場合、エラーが発生する可能性があります。

---

何か不明点があれば、計画.txtや各モジュールのdocstringも参照してください。


