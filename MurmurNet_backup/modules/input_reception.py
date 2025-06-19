#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input Reception モジュール
~~~~~~~~~~~~~~~~~~~~~~~
ユーザー入力の前処理を担当
テキストの正規化、トークン化、埋め込み生成など

作者: Yuhi Sonoki
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger('MurmurNet.InputReception')

class InputReception:
    """
    入力テキストの前処理を行うクラス
    
    責務:
    - テキスト正規化
    - トークン化
    - 埋め込み生成（オプション）
    
    属性:
        config: 設定辞書
        use_embeddings: 埋め込みを使用するかどうか
    """
    def __init__(self, config: Dict[str, Any]):
        """
        入力処理モジュールの初期化
        
        引数:
            config: 設定辞書
        """
        self.config = config
        self.debug = config.get('debug', False)
        self.use_embeddings = config.get('use_embeddings', True)
        self._transformer = None  # 遅延ロード
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
    def _load_transformer(self) -> None:
        """
        埋め込みモデルを必要なときだけロード（内部メソッド）
        
        モデルロードはリソース消費が大きいため、実際に必要になるまで
        遅延ロードする
        """
        if self._transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                
                if self.debug:
                    logger.debug(f"埋め込みモデルをロード: {model_name}")
                    
                self._transformer = SentenceTransformer(model_name)
                
            except ImportError:
                logger.warning("SentenceTransformersがインストールされていません。ダミー埋め込みを使用します。")
                self._transformer = None
                
            except Exception as e:
                logger.error(f"埋め込みモデルロードエラー: {e}")
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
                logger.warning("無効な入力テキスト")
                return {'normalized': "", 'tokens': [], 'embedding': np.array([])}
                
            # テキスト長の制限
            if len(input_text) > 1000:
                logger.debug(f"入力テキストを切り詰め: {len(input_text)} → 1000文字")
                input_text = input_text[:1000]  # 最大1000文字に制限
            
            # 正規化
            normalized = input_text.strip()
            
            # トークン分割（スペース区切り）
            tokens = normalized.split()
            if len(tokens) > 100:
                logger.debug(f"トークン数を制限: {len(tokens)} → 100")
                tokens = tokens[:100]  # 最大100トークンに制限
            
            result = {'normalized': normalized, 'tokens': tokens}
            
            # 埋め込み生成（オプション）
            if self.use_embeddings:
                try:
                    self._load_transformer()
                    if self._transformer:
                        embedding = self._transformer.encode(normalized)
                        result['embedding'] = embedding
                        
                        if self.debug:
                            logger.debug(f"埋め込み生成: shape={embedding.shape}")
                    else:
                        # ダミー埋め込み
                        result['embedding'] = np.zeros(384)  # MiniLM-L6の次元数
                        logger.warning("ダミー埋め込みを使用")
                except Exception as e:
                    logger.error(f"埋め込み生成エラー: {e}")
                    result['embedding'] = np.zeros(384)  # ダミー埋め込み
            
            return result
            
        except Exception as e:
            logger.error(f"入力処理エラー: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
                
            # エラー時のフォールバック
            return {'normalized': input_text[:100], 'tokens': input_text.split()[:10], 'embedding': np.zeros(384)}
