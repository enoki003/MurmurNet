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
