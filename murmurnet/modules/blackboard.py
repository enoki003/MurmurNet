#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blackboard モジュール
~~~~~~~~~~~~~~~~~~~
エージェント間の共有メモリとして機能する黒板パターン実装
データの読み書き、監視、履歴管理を提供
スレッドセーフな実装でメモリ管理も改善

作者: Yuhi Sonoki
"""

import time
import threading
import weakref
from collections import deque
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from .common import BlackboardError, ResourceLimitError
from .config_manager import get_config


class BlackboardEntry:
    """黒板エントリ（不変オブジェクト）"""
    
    def __init__(self, key: str, value: Any, timestamp: float = None):
        self.key = key
        self.value = self._process_value_for_storage(value)
        self.timestamp = timestamp or time.time()
        
    def _process_value_for_storage(self, value: Any) -> Any:
        """表示用に値を加工"""
        if isinstance(value, dict) and 'embedding' in value:
            processed = value.copy()
            if isinstance(value['embedding'], np.ndarray):
                vector = value['embedding']
                processed['embedding'] = f"<Vector shape={vector.shape}, mean={vector.mean():.4f}>"
            return processed
        return value


class Blackboard:
    """
    スレッドセーフな分散エージェント間の共有メモリ実装
    
    改善点:
    - 真のスレッドセーフティ（読み書きロック）
    - メモリ制限機能
    - 効率的な履歴管理
    - 循環参照の回避
    """
    def __init__(self, config: Dict[str, Any] = None):
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
        self._memory: Dict[str, Any] = {}
        
        # スレッドセーフティのための読み書きロック
        self._lock = threading.RLock()
        
        # ConfigManagerから履歴制限を取得
        history_limit = self.config_manager.memory.blackboard_history_limit
        self._history: deque = deque(maxlen=history_limit)
        
        # ConfigManagerからデバッグ設定を取得
        self.debug = self.config_manager.logging.debug
        
        # 保持するキー（ターンをまたいで保持）
        self.persistent_keys = {'conversation_context', 'key_facts', 'history_summary'}
        
        # 観察者パターン（弱参照で循環参照を回避）
        self._observers: weakref.WeakSet = weakref.WeakSet()
        
        # 統計情報
        self._stats = {
            'read_count': 0,
            'write_count': 0,
            'clear_count': 0
        }
    
    @property
    def memory(self) -> Dict[str, Any]:
        """
        現在のメモリの読み取り専用ビューを提供
        
        戻り値:
            メモリの辞書ビュー
        """
        with self._lock:
            return dict(self._memory)
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """
        履歴の読み取り専用ビューを提供
        
        戻り値:
            履歴のリスト
        """
        with self._lock:
            return list(self._history)

    def write(self, key: str, value: Any) -> BlackboardEntry:
        """
        黒板に情報を書き込む（スレッドセーフ）
        
        引数:
            key: キー
            value: 値
            
        戻り値:
            作成されたエントリ
            
        例外:
            BlackboardError: 書き込みエラー
        """
        if not isinstance(key, str) or not key.strip():
            raise BlackboardError("キーは空でない文字列である必要があります")
        
        try:
            with self._lock:
                entry = BlackboardEntry(key, value)
                self._memory[key] = entry.value
                self._history.append({
                    'timestamp': entry.timestamp,
                    'key': key,
                    'value': entry.value,
                    'operation': 'write'
                })
                self._stats['write_count'] += 1
                
                # 観察者に通知
                self._notify_observers('write', key, entry.value)
                
                if self.debug:
                    print(f"[Blackboard] Write: {key} = {str(entry.value)[:100]}")
                
                return entry
                
        except Exception as e:
            raise BlackboardError(f"書き込みエラー (key={key}): {e}")
    
    def read(self, key: str) -> Any:
        """
        黒板から情報を読み込む（スレッドセーフ）
        
        引数:
            key: キー
            
        戻り値:
            値（存在しない場合はNone）
        """
        try:
            with self._lock:
                self._stats['read_count'] += 1
                value = self._memory.get(key)
                
                if self.debug and value is not None:
                    print(f"[Blackboard] Read: {key} = {str(value)[:100]}")
                
                return value
                
        except Exception as e:
            raise BlackboardError(f"読み込みエラー (key={key}): {e}")
    
    def read_all(self) -> Dict[str, Any]:
        """
        黒板のすべての現在値を取得（スレッドセーフ）
        
        戻り値:
            現在の全メモリのコピー
        """
        try:
            with self._lock:
                return self._memory.copy()
        except Exception as e:
            raise BlackboardError(f"全読み込みエラー: {e}")
    
    def clear_current_turn(self) -> None:
        """
        新しいターンのために黒板をクリアするが、特定のキーの値は保持する
        """
        try:
            with self._lock:
                # 保持するべき値を一時的に保存
                preserved_values = {}
                for key in self.persistent_keys:
                    if key in self._memory:
                        preserved_values[key] = self._memory[key]
                
                # 黒板をクリア
                self._memory.clear()
                
                # 保持するべき値を復元
                self._memory.update(preserved_values)
                
                # 履歴に記録
                self._history.append({
                    'timestamp': time.time(),
                    'key': '<CLEAR>',
                    'value': f'Preserved {len(preserved_values)} keys',
                    'operation': 'clear'
                })
                self._stats['clear_count'] += 1
                
                # 観察者に通知
                self._notify_observers('clear', None, None)
                
                if self.debug:
                    print(f"[Blackboard] Clear: 保持キー {len(preserved_values)}個")
                    
        except Exception as e:
            raise BlackboardError(f"クリアエラー: {e}")
    
    def get_history(self, key: Optional[str] = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        特定キーまたは全体の履歴を取得
        
        引数:
            key: 特定のキー（Noneの場合は全履歴）
            limit: 取得する履歴の上限
            
        戻り値:
            履歴のリスト
        """
        try:
            with self._lock:
                if key is None:
                    history = list(self._history)
                else:
                    history = [entry for entry in self._history if entry['key'] == key]
                
                if limit:
                    history = history[-limit:]
                
                return history
                
        except Exception as e:
            raise BlackboardError(f"履歴取得エラー: {e}")
    
    def get_debug_view(self) -> Dict[str, str]:
        """
        デバッグ用の簡略化されたビューを取得
        
        戻り値:
            デバッグ用の辞書
        """
        try:
            with self._lock:
                debug_view = {}
                for key, value in self._memory.items():
                    if isinstance(value, str):
                        display_value = value[:100] + "..." if len(value) > 100 else value
                    else:
                        display_value = str(value)[:100]
                    debug_view[key] = display_value
                return debug_view
                
        except Exception as e:
            raise BlackboardError(f"デバッグビュー取得エラー: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self._lock:
            return {
                **self._stats.copy(),
                'memory_keys': len(self._memory),
                'history_entries': len(self._history),
                'observers': len(self._observers)
            }
    
    def add_observer(self, observer: Callable[[str, str, Any], None]) -> None:
        """観察者を追加（弱参照で循環参照を回避）"""
        self._observers.add(observer)
    
    def remove_observer(self, observer: Callable[[str, str, Any], None]) -> None:
        """観察者を削除"""
        self._observers.discard(observer)
    
    def _notify_observers(self, operation: str, key: Optional[str], value: Any) -> None:
        """観察者に通知"""
        try:
            for observer in self._observers.copy():  # コピーして安全に反復
                try:
                    observer(operation, key, value)
                except Exception as e:
                    if self.debug:
                        print(f"[Blackboard] Observer error: {e}")
        except Exception:
            pass  # 観察者の通知エラーは無視
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        try:
            with self._lock:
                self._memory.clear()
                self._history.clear()
                self._observers.clear()
                self._stats = {k: 0 for k in self._stats}
        except Exception as e:
            if self.debug:
                print(f"[Blackboard] Cleanup error: {e}")
    def __del__(self):
        """デストラクタでリソースをクリーンアップ"""
        try:
            self.cleanup()
        except Exception:
            pass  # デストラクタでは例外を無視
        
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
