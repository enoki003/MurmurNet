#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Blackboard Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Redis の代替として軽量な Python dict + Lock を使用
優先度★のローカル最適化策を実装

作者: Yuhi Sonoki
"""
import multiprocessing as mp
import threading
from typing import Any, Dict, Optional


class SimpleBlackboard:
    """
    シンプルな黒板実装
    - Redis 依存を削減
    - multiprocessing.Manager().dict() を使用
    - 10キー程度の軽量データに最適化
    """
    
    def __init__(self):
        """初期化"""
        self.manager = mp.Manager()
        self.data: Dict[str, Any] = self.manager.dict()
        self.lock = mp.Lock()
    
    def write(self, key: str, value: Any) -> None:
        """データ書き込み"""
        with self.lock:
            self.data[key] = value
    
    def read(self, key: str) -> Optional[Any]:
        """データ読み込み"""
        with self.lock:
            return self.data.get(key)
    
    def delete(self, key: str) -> None:
        """データ削除"""
        with self.lock:
            if key in self.data:
                del self.data[key]
    
    def exists(self, key: str) -> bool:
        """キー存在確認"""
        with self.lock:
            return key in self.data
    
    def keys(self) -> list:
        """全キー取得"""
        with self.lock:
            return list(self.data.keys())
    
    def clear(self) -> None:
        """全データクリア"""
        with self.lock:
            self.data.clear()
    
    def get_debug_view(self) -> Dict[str, Any]:
        """デバッグ用ビュー取得"""
        with self.lock:
            return dict(self.data)


# 使用例とテスト
if __name__ == "__main__":
    # 基本テスト
    bb = SimpleBlackboard()
    
    # 書き込み・読み込みテスト
    bb.write("test_key", "test_value")
    print(f"読み込み結果: {bb.read('test_key')}")
    
    # 複数キーテスト
    bb.write("agent_1_output", "エージェント1の出力")
    bb.write("agent_2_output", "エージェント2の出力")
    bb.write("rag_result", "RAG検索結果")
    
    # デバッグビュー
    print("デバッグビュー:")
    debug_view = bb.get_debug_view()
    for k, v in debug_view.items():
        print(f"  {k}: {v}")
    
    # キー一覧
    print(f"全キー: {bb.keys()}")
    
    # 存在確認
    print(f"'test_key' 存在: {bb.exists('test_key')}")
    print(f"'nonexistent' 存在: {bb.exists('nonexistent')}")
    
    # 削除テスト
    bb.delete("test_key")
    print(f"削除後の 'test_key': {bb.read('test_key')}")
    
    print("SimpleBlackboard テスト完了")
