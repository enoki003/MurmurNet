#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blackboard モジュール
~~~~~~~~~~~~~~~~~~~
エージェント間の共有メモリとして機能する黒板パターン実装
データの読み書き、監視、履歴管理を提供

作者: Yuhi Sonoki
"""

# 黒板（共有メモリ）モジュール
from typing import Dict, Any, List, Optional
import time
import numpy as np
import re

class Blackboard:
    """
    分散エージェント間の共有メモリ実装
    - シンプルなkey-value保存
    - 時系列的な記録と管理
    - コンテキスト管理と自動サイズ制限
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.debug = config.get('debug', False)
        # 保持するキーを制限（会話コンテキストは保持するが、生テキスト制限を別途設定）
        self.persistent_keys = config.get('persistent_keys', ['conversation_context'])
        
        # コンテキスト管理設定
        self.max_context_length = config.get('max_context_length', 2000)  # 会話コンテキストの最大文字数
        self.context_compression_ratio = config.get('context_compression_ratio', 0.7)  # 圧縮時に残す割合
        
        # コンテキスト管理の統計情報
        self.context_stats = {
            'compressions': 0,
            'original_chars': 0,
            'compressed_chars': 0,
            'last_compression_time': None
        }
        
        # 応答失敗ログを除外するためのフラグ（デフォルトで有効）
        self.filter_failed_responses = config.get('filter_failed_responses', True)
        
        # 失敗応答を検出するためのキーワード（拡充版）
        self.failure_keywords = config.get('failure_keywords', [
            # 一般的なエラーメッセージ
            '応答できませんでした', 'エラー', '失敗', '考え中', 
            'すみません', '申し訳ありません', '回答を生成できません',
            '再帰エラー', '再試行', 'リトライ', '対応できかねます',
            # エージェント間メタ発言
            'エージェント', 'は応答できません', 'が応答でき', 'が回答でき',
            'は回答できません', 'さんは応答', 'さんが回答',
            'から回答がありませんでした', 'から応答がありませんでした',
            # エラー通知
            '接続エラー', '処理中にエラー', 'タイムアウト', '接続が切断',
            'エラーが発生しました', '問題が発生しました', '実行エラー',
            # 処理中表現
            '処理中', '計算中', '考えています', '分析中', '準備中', 
            '実行中', '思考中', 'プロセス中',
            # AIアイデンティティ
            'AIアシスタント', 'AIとして', 'モデルとして', 'LLMとして',
            'アシスタントとして', 'AIモデル', '言語モデル',
            # その他失敗表現
            '時間切れ', '応答生成失敗', '回答できませ', '回答失敗',
            '応答できませ', '応答失敗', '応答なし', '回答なし',
            '生成できませんでした', '出力できませんでした'
        ])
        
        # 正規表現パターン（エージェントの応答失敗パターン）
        self.failure_patterns = [
            r'エージェント\s*\d+\s*は.*応答.*できませんでした',
            r'エージェント\s*\d+\s*は.*回答.*できませんでした',
            r'.*さんは.*応答.*できませんでした',
            r'.*さんは.*回答.*できませんでした',
            r'応答を生成できませ.*でした',
            r'回答を生成できませ.*でした',
            r'エラーが発生しました',
            r'申し訳.*応答.*できません',
            r'申し訳.*回答.*できません'
        ]

    def has(self, key: str) -> bool:
        """
        指定したキーが黒板に存在するかどうかを確認
        
        引数:
            key: 確認するキー
            
        戻り値:
            bool: キーが存在する場合はTrue、そうでない場合はFalse
        """
        return key in self.memory

    def write(self, key: str, value: Any) -> Dict[str, Any]:
        """
        黒板に情報を書き込む
        埋め込みベクトルは表示用に簡略化
        特定キーの場合はサイズ制限を適用
        応答失敗の書き込みをフィルタリング
        
        引数:
            key: 保存するキー
            value: 保存する値
            
        戻り値:
            Dict[str, Any]: 保存したエントリ情報
        """
        # 応答失敗をフィルタリング
        if self.filter_failed_responses and (key.endswith('_output') or key.endswith('_response') 
                                           or key == 'agent_message' or '_message' in key):
            # エージェントメッセージか出力の場合は、エラーフラグと応答の内容を確認
            if isinstance(value, dict):
                # エラーフラグがある場合
                if value.get('error', False):
                    if self.debug:
                        print(f"応答失敗をフィルタリング: エラーフラグあり - {key}")
                    return {'skipped': True, 'key': key, 'timestamp': time.time()}
                
                # 応答テキストから失敗を示す文言をチェック
                response_text = str(value.get('text', ''))
                if self._is_failed_response(response_text):
                    if self.debug:
                        print(f"応答失敗をフィルタリング: 失敗キーワード検出 - {key}: {response_text[:30]}...")
                    return {'skipped': True, 'key': key, 'timestamp': time.time()}
            
            elif isinstance(value, str):
                # 文字列値の場合も同様にチェック
                if self._is_failed_response(value):
                    if self.debug:
                        print(f"応答失敗をフィルタリング: 失敗キーワード検出 - {key}: {value[:30]}...")
                    return {'skipped': True, 'key': key, 'timestamp': time.time()}
        
        # 埋め込みベクトルの場合は表示用に圧縮
        stored_value = self._process_value_for_storage(value)
        
        # 会話コンテキストの場合はサイズ制限を適用
        if key == 'conversation_context' and isinstance(stored_value, str):
            stored_value = self._limit_context_size(stored_value)
            
        self.memory[key] = stored_value
        
        entry = {
            'timestamp': time.time(),
            'key': key,
            'value': stored_value
        }
        self.history.append(entry)
        
        # 履歴のサイズ制限
        max_history_entries = self.config.get('max_history_entries', 1000)
        if len(self.history) > max_history_entries:
            self.history = self.history[-max_history_entries:]
            
        return entry
    
    def _is_failed_response(self, text: str) -> bool:
        """
        応答が失敗したものかどうかを判定
        
        引数:
            text: チェック対象のテキスト
            
        戻り値:
            bool: 失敗応答の場合はTrue
        """
        if not text or not isinstance(text, str):
            return False
            
        # 小文字に変換して比較
        text_lower = text.lower()
        
        # キーワードチェック
        if any(keyword in text_lower for keyword in self.failure_keywords):
            return True
            
        # 正規表現パターンチェック
        for pattern in self.failure_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        # 極度に短いテキスト (30文字未満) で特定のパターンをチェック
        if len(text) < 30 and any(x in text_lower for x in ['エラー', 'エージェント', '応答なし', '回答なし', 'できません']):
            return True
            
        return False

    def _limit_context_size(self, context: str) -> str:
        """
        会話コンテキストのサイズを制限する
        最大サイズを超える場合は古い情報を要約・削減する
        応答失敗を示すメッセージもフィルタリングする
        
        引数:
            context: 制限対象のコンテキスト文字列
            
        戻り値:
            str: 制限されたコンテキスト文字列
        """
        if not context:
            return ""
            
        # 失敗ログを除外する処理を追加（フィルタリング機能が有効な場合のみ）
        if self.filter_failed_responses:
            # 各行をチェックし、エラーメッセージや応答失敗を示す行を除外
            lines = context.split('\n')
            filtered_lines = []
            for line in lines:
                # 空行はスキップしない
                if not line.strip():
                    filtered_lines.append(line)
                    continue
                
                # エラーメッセージや応答失敗を含む行をスキップ
                if self._is_failed_response(line):
                    continue
                
                # それ以外の行は保持
                filtered_lines.append(line)
            
            # フィルタリング後のコンテキスト
            context = '\n'.join(filtered_lines)
        
        # サイズチェック
        if len(context) <= self.max_context_length:
            return context
            
        # 統計情報を更新
        self.context_stats['compressions'] += 1
        self.context_stats['original_chars'] += len(context)
        self.context_stats['last_compression_time'] = time.time()
        
        # 圧縮（単純な方法： 最新部分を優先的に残す）
        prefix = "...(過去の会話履歴は省略されました)...\n"
        
        # 最大長を超えている場合のみプレフィックスを付加
        if len(context) > self.max_context_length:
            available_length = self.max_context_length - len(prefix)
            compressed = prefix + context[-available_length:]
        else:
            compressed = context
        
        # 統計情報を更新
        self.context_stats['compressed_chars'] += len(compressed)
        
        if self.debug:
            print(f"会話コンテキスト圧縮: {len(context)}文字 → {len(compressed)}文字")
            
        return compressed

    def _process_value_for_storage(self, value: Any) -> Any:
        """表示用に値を加工"""
        # 埋め込みベクトルを含む辞書の場合
        if isinstance(value, dict) and 'embedding' in value:
            # 元の値をコピー
            processed = value.copy()
            
            # 埋め込みベクトルの場合は形状情報のみ保持
            if isinstance(value['embedding'], np.ndarray):
                vector = value['embedding']
                # 形状情報とベクトルの先頭と末尾の一部だけ保持
                processed['embedding'] = f"<Vector shape={vector.shape}, mean={vector.mean():.4f}>"
            return processed
        
        return value

    def read(self, key: str) -> Any:
        """
        黒板から情報を読み込む
        """
        return self.memory.get(key)
        
    def read_all(self) -> Dict[str, Any]:
        """
        黒板のすべての現在値を取得
        """
        return self.memory.copy()

    def clear_current_turn(self) -> None:
        """
        新しいターンのために黒板をクリアするが、特定のキーの値は保持する
        persistent_keysに指定されたキーの値は保持される
        """
        # 保持するべき値を一時的に保存
        preserved_values = {}
        for key in self.persistent_keys:
            if key in self.memory:
                preserved_values[key] = self.memory[key]
        
        # 黒板をクリア
        self.memory = {}
        
        # 保持するべき値を復元
        for key, value in preserved_values.items():
            self.memory[key] = value
            
        if self.debug:
            print(f"黒板クリア: 保持キー {len(preserved_values)}個")
        
    def get_history(self, key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        特定キーまたは全体の履歴を取得
        """
        if key is None:
            return self.history
        return [entry for entry in self.history if entry['key'] == key]
        
    def get_debug_view(self) -> Dict[str, Any]:
        """
        デバッグ表示用に簡略化した黒板データを取得
        """
        result = {}
        for key, value in self.memory.items():
            # 入力データは正規化テキストだけ表示
            if key == 'input' and isinstance(value, dict):
                result[key] = {'normalized': value.get('normalized', '')}
            elif key == 'conversation_context' and isinstance(value, str):
                # 会話コンテキストは長さと先頭/末尾の一部のみ表示
                if len(value) > 200:
                    chars = min(100, len(value) // 2)
                    result[key] = f"{value[:chars]}...（{len(value)}文字）...{value[-chars:]}"
                else:
                    result[key] = value
            else:
                result[key] = value
        
        # コンテキスト管理の統計情報を追加
        if self.context_stats['compressions'] > 0:
            result['_context_stats'] = self.context_stats
            
        return result
        
    def get_context_stats(self) -> Dict[str, Any]:
        """
        コンテキスト管理の統計情報を取得
        """
        return self.context_stats.copy()
