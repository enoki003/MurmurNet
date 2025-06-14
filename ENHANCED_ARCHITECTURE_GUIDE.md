# Enhanced MurmurNet Architecture Documentation

## 概要

新しいMurmurNetアーキテクチャは、従来のBLACKBOARDベースの密結合設計から、通信インターフェースベースの疎結合設計への移行を実現します。これにより、モジュール間の依存関係が明確になり、システムの保守性と拡張性が大幅に向上します。

## アーキテクチャの改善点

### 1. 疎結合設計の実現

**従来**:
- 直接的なBLACKBOARD依存
- 複雑な初期化チェーン
- モジュール間の循環依存

**新システム**:
- 通信インターフェースによる疎結合
- 依存性注入パターン
- 明確なプロトコル定義

### 2. 通信インターフェースの導入

```python
from MurmurNet.modules.communication_interface import (
    create_communication_system,
    create_message,
    MessageType
)

# 通信システムの初期化
comm_manager = create_communication_system()

# メッセージの送信
message = create_message(MessageType.USER_INPUT, {'text': 'ユーザー入力'})
comm_manager.publish(message)

# データの取得
data = comm_manager.get_data('key')
```

### 3. モジュラー設計

各モジュールは明確に定義されたインターフェースを通じて通信し、内部実装の詳細を隠蔽します。

## 主要コンポーネント

### 1. 通信インターフェース (`communication_interface.py`)

- **ModuleCommunicationManager**: 中央通信管理器
- **MessageType**: メッセージタイプの列挙
- **Message**: 標準化されたメッセージフォーマット
- **CommunicationAdapter**: BLACKBOARD互換性レイヤー

### 2. モジュールシステム調整器 (`module_system_coordinator.py`)

- **ModuleSystemCoordinator**: 新しい通信インターフェースを使用する改良版システム調整器
- **AgentPoolProtocol**: エージェントプールの抽象インターフェース
- **SummaryEngineProtocol**: 要約エンジンの抽象インターフェース

### 3. モジュールアダプター (`module_adapters.py`)

- **AgentPoolAdapter**: 既存AgentPoolManagerのアダプター
- **SummaryEngineAdapter**: 既存SummaryEngineのアダプター
- **BlackboardBridgeAdapter**: BLACKBOARD互換性ブリッジ
- **ConversationMemoryAdapter**: 会話メモリのアダプター

### 4. 改良版分散SLM (`enhanced_distributed_slm.py`)

- **EnhancedDistributedSLM**: 新しいアーキテクチャを使用する改良版SLMシステム
- 互換性モードサポート
- 段階的移行機能

## メッセージタイプ

```python
class MessageType(Enum):
    USER_INPUT = "user_input"           # ユーザー入力
    AGENT_RESPONSE = "agent_response"   # エージェント応答
    RAG_RESULTS = "rag_results"         # RAG検索結果
    SUMMARY = "summary"                 # 要約
    ERROR = "error"                     # エラー
    AGENT_ERROR = "agent_error"         # エージェントエラー
    DATA_STORE = "data_store"          # データ格納
    SYSTEM_STATUS = "system_status"     # システム状態
    INITIAL_SUMMARY = "initial_summary" # 初期要約
    FINAL_RESPONSE = "final_response"   # 最終応答
```

## 使用方法

### 1. 基本的な使用例

```python
from MurmurNet.modules.enhanced_distributed_slm import EnhancedDistributedSLM

# 新モードで初期化
slm = EnhancedDistributedSLM(compatibility_mode=False)

# テキスト生成
response = await slm.generate("機械学習について教えてください")
print(response)
```

### 2. 互換性モードでの使用

```python
# 既存システムとの互換性を保持
slm = EnhancedDistributedSLM(compatibility_mode=True)

# 従来と同じインターフェースで使用可能
response = await slm.generate("質問テキスト")
```

### 3. カスタム通信システムの使用

```python
from MurmurNet.modules.communication_interface import create_communication_system

# カスタム通信システム
custom_comm = create_communication_system()

# 改良版SLMで使用
slm = EnhancedDistributedSLM(comm_manager=custom_comm)
```

## 移行ガイド

### フェーズ1: 新システムの並行運用

1. 既存システムを維持
2. 新しい通信インターフェースを導入
3. アダプターを使用して既存モジュールを統合

```python
# 既存システムとの並行運用
slm_legacy = DistributedSLM()  # 既存システム
slm_enhanced = EnhancedDistributedSLM(compatibility_mode=True)  # 新システム
```

### フェーズ2: 段階的移行

1. モジュールごとに新しいインターフェースに移行
2. テストを実行して互換性を確認
3. パフォーマンスを監視

```python
# モジュールの段階的移行
from MurmurNet.modules.module_adapters import create_module_adapters

adapters = create_module_adapters(
    agent_pool=existing_agent_pool,
    summary_engine=existing_summary_engine
)
```

### フェーズ3: 完全移行

1. 互換性モードを無効化
2. レガシーコードの除去
3. 新しいアーキテクチャでの最適化

```python
# 完全移行後
slm = EnhancedDistributedSLM(compatibility_mode=False)
```

## テストの実行

### 1. 統合テストの実行

```bash
python -m pytest MurmurNet/modules/test_enhanced_architecture.py -v
```

### 2. デモンストレーションの実行

```bash
python MurmurNet/modules/enhanced_architecture_demo.py
```

## パフォーマンス監視

新しいアーキテクチャには包括的なパフォーマンス監視機能が含まれています：

```python
# システム状態の取得
status = slm.get_system_status()
print(status)

# 通信統計の取得
comm_stats = slm.get_communication_stats()
print(comm_stats)

# 実行統計の取得
exec_stats = slm._system_coordinator.get_execution_stats()
print(exec_stats)
```

## エラーハンドリング

新しいシステムでは構造化されたエラーハンドリングを提供します：

```python
from MurmurNet.modules.communication_interface import create_message, MessageType

# エラーメッセージの送信
error_message = create_message(MessageType.ERROR, {
    'error': 'エラーの詳細',
    'component': 'module_name',
    'timestamp': time.time()
})
comm_manager.publish(error_message)
```

## ベストプラクティス

### 1. 依存性注入の使用

```python
# 依存関係を明示的に注入
coordinator = ModuleSystemCoordinator(
    comm_manager=comm_manager,
    agent_pool=agent_pool,
    summary_engine=summary_engine
)
```

### 2. プロトコルの実装

```python
class CustomAgentPool(AgentPoolProtocol):
    async def run_agent(self, agent_id: int, prompt: str) -> str:
        # カスタム実装
        pass
    
    def run_agent_sync(self, agent_id: int, prompt: str) -> str:
        # カスタム実装
        pass
```

### 3. メッセージの構造化

```python
# 構造化されたメッセージの作成
message = create_message(MessageType.AGENT_RESPONSE, {
    'agent_id': agent_id,
    'response': response_text,
    'metadata': {
        'execution_time': elapsed_time,
        'model_info': model_info
    }
})
```

## トラブルシューティング

### 一般的な問題と解決方法

1. **モジュール初期化エラー**
   - 依存関係の確認
   - 設定ファイルの検証
   - ログの確認

2. **通信エラー**
   - メッセージフォーマットの確認
   - 通信管理器の状態確認
   - ネットワーク接続の確認

3. **パフォーマンス問題**
   - メモリ使用量の監視
   - プロファイリングの実行
   - 設定の最適化

### ログの確認

```python
import logging

# ログレベルの設定
logging.getLogger('MurmurNet').setLevel(logging.DEBUG)

# ログファイルの確認
tail -f enhanced_murmurnet_demo.log
```

## 今後の拡張計画

1. **分散処理サポート**
   - マルチプロセス/マルチマシン対応
   - 負荷分散機能

2. **プラグインアーキテクチャ**
   - 動的モジュール読み込み
   - サードパーティ拡張サポート

3. **高度な監視機能**
   - リアルタイムダッシュボード
   - アラート機能

4. **API統合**
   - RESTful API
   - GraphQL サポート

## サポート

質問や問題が発生した場合は、以下を参照してください：

- ドキュメント: `README.md`
- テストコード: `test_enhanced_architecture.py`
- デモコード: `enhanced_architecture_demo.py`
- ログファイル: `enhanced_murmurnet_demo.log`
