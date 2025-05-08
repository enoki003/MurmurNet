#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input Reception モジュール
~~~~~~~~~~~~~~~~~~~~~~~
ユーザー入力の前処理を担当
テキストの正規化、トークン化、埋め込み生成など

作者: Yuhi Sonoki
"""

import re
import numpy as np
from typing import Dict, Any

class InputReception:
    """
    入力テキストの前処理を行うクラス
    - テキスト正規化
    - トークン化
    - 埋め込み生成（オプション）
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug = config.get('debug', False)
        self.use_embeddings = config.get('use_embeddings', True)
        self._transformer = None  # 遅延ロード
        
    def _load_transformer(self):
        """埋め込みモデルを必要なときだけロード（内部メソッド）"""
        if self._transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                if self.debug:
                    print(f"埋め込みモデルをロード: {model_name}")
                self._transformer = SentenceTransformer(model_name)
            except ImportError:
                if self.debug:
                    print("SentenceTransformersがインストールされていません。ダミー埋め込みを使用します。")
                self._transformer = None
            except Exception as e:
                if self.debug:
                    print(f"埋め込みモデルロードエラー: {e}")
                self._transformer = None

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        入力テキストを前処理する
        
        引数:
            input_text: 処理する入力テキスト
            
        戻り値:
            処理結果を含む辞書（正規化テキスト、トークン、埋め込み）
        """
        try:
            # 入力の検証
            if not input_text or not isinstance(input_text, str):
                return {'normalized': "", 'tokens': [], 'embedding': np.array([])}
                
            # テキスト長の制限
            input_text = input_text[:1000]  # 最大1000文字に制限
            
            # 正規化
            normalized = input_text
            # トークン分割（スペース区切り）
            tokens = normalized.split()[:100]  # 最大100トークンに制限
            
            result = {'normalized': normalized, 'tokens': tokens}
            
            # 埋め込み生成（オプション）
            if self.use_embeddings:
                try:
                    self._load_transformer()
                    if self._transformer:
                        embedding = self._transformer.encode(normalized)
                        result['embedding'] = embedding
                    else:
                        # ダミー埋め込み
                        result['embedding'] = np.zeros(384)  # MiniLM-L6の次元数
                except Exception as e:
                    if self.debug:
                        print(f"埋め込み生成エラー: {e}")
                    result['embedding'] = np.zeros(384)  # ダミー埋め込み
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"入力処理エラー: {e}")
                import traceback
                traceback.print_exc()
            # エラー時のフォールバック
            return {'normalized': input_text[:100], 'tokens': input_text.split()[:10], 'embedding': np.zeros(384)}
