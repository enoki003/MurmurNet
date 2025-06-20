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

class Blackboard:
    """
    分散エージェント間の共有メモリ実装
    - シンプルなkey-value保存
    - 時系列的な記録と管理
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.debug = config.get('debug', False)
        self.persistent_keys = ['conversation_context']  # 保持するキー

    def write(self, key: str, value: Any) -> Dict[str, Any]:
        """
        黒板に情報を書き込む
        埋め込みベクトルは表示用に簡略化
        """
        # 埋め込みベクトルの場合は表示用に圧縮
        stored_value = self._process_value_for_storage(value)
        self.memory[key] = stored_value
        
        entry = {
            'timestamp': time.time(),
            'key': key,
            'value': stored_value
        }
        self.history.append(entry)
        return entry

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
            else:
                result[key] = value
        return result
