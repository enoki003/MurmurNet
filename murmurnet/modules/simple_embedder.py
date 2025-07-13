#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Embedder モジュール
~~~~~~~~~~~~~~~~~~~~~~~~
SentenceTransformer依存なしの軽量埋め込み実装
単語ベース・TF-IDF・ハッシュベースの埋め込み手法

機能:
- TF-IDF ベクトル化
- 単語頻度ベース埋め込み
- ハッシュベース高速埋め込み
- オフライン完全動作

作者: Yuhi Sonoki
"""

import logging
import re
import numpy as np
from typing import List, Union, Optional, Dict, Any
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

class SimpleEmbedder:
    """
    シンプルな埋め込み実装
    
    SentenceTransformer不要の軽量埋め込み手法
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        SimpleEmbedder初期化
        
        Args:
            config: 設定辞書
                - embedding_dim: 埋め込み次元数 (default: 384)
                - method: 埋め込み手法 ('tfidf', 'bow', 'hash') (default: 'hash')
                - vocab_size: 語彙サイズ (default: 10000)
        """
        self.config = config or {}
        self.embedding_dim = self.config.get('embedding_dim', 384)
        self.method = self.config.get('embedding_method', 'hash')
        self.vocab_size = self.config.get('vocab_size', 10000)
        
        # 語彙辞書（TF-IDF用）
        self.vocab = {}
        self.idf_scores = {}
        self.is_fitted = False
        
        logger.info(f"SimpleEmbedder初期化: method={self.method}, dim={self.embedding_dim}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークンリスト
        """
        if not text:
            return []
        
        # 簡単なトークン化（単語分割）
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # 句読点除去
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]  # 1文字の単語除去
    
    def _hash_embedding(self, text: str) -> np.ndarray:
        """
        ハッシュベース高速埋め込み
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        # テキストのハッシュを計算
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # ハッシュを数値ベクトルに変換
        hash_values = []
        for i in range(0, len(text_hash), 2):
            hash_values.append(int(text_hash[i:i+2], 16))
        
        # 次元数に合わせて調整
        embedding = np.zeros(self.embedding_dim)
        for i, val in enumerate(hash_values):
            if i < self.embedding_dim:
                embedding[i] = (val - 128) / 128.0  # -1 to 1 に正規化
        
        # 残りの次元は文字列長やトークン数で埋める
        tokens = self._tokenize(text)
        for i in range(len(hash_values), self.embedding_dim):
            if i % 3 == 0:
                embedding[i] = len(text) / 1000.0  # 文字列長
            elif i % 3 == 1:
                embedding[i] = len(tokens) / 100.0  # トークン数
            else:
                embedding[i] = np.random.normal(0, 0.1)  # ノイズ
        
        # L2正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _bow_embedding(self, text: str) -> np.ndarray:
        """
        Bag of Words 埋め込み
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # トークン頻度計算
        token_counts = Counter(tokens)
        
        # ハッシュベースでトークンを次元にマッピング
        embedding = np.zeros(self.embedding_dim)
        for token, count in token_counts.items():
            token_hash = hash(token) % self.embedding_dim
            embedding[token_hash] += count
        
        # 正規化
        total_tokens = len(tokens)
        if total_tokens > 0:
            embedding = embedding / total_tokens
        
        # L2正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _tfidf_embedding(self, text: str) -> np.ndarray:
        """
        TF-IDF ベース埋め込み（簡易版）
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # TF計算
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        embedding = np.zeros(self.embedding_dim)
        for token, count in token_counts.items():
            # TF
            tf = count / total_tokens
            
            # IDF（仮の値、実際のコーパス統計なし）
            idf = self.idf_scores.get(token, 1.0)
            
            # TF-IDF
            tfidf = tf * idf
            
            # ハッシュベースで次元にマッピング
            token_hash = hash(token) % self.embedding_dim
            embedding[token_hash] += tfidf
        
        # L2正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            texts: 単一テキストまたはテキストリスト
            **kwargs: 追加引数（互換性のため）
            
        Returns:
            埋め込みベクトル配列
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            if self.method == 'hash':
                embedding = self._hash_embedding(text)
            elif self.method == 'bow':
                embedding = self._bow_embedding(text)
            elif self.method == 'tfidf':
                embedding = self._tfidf_embedding(text)
            else:
                # デフォルトはハッシュベース
                embedding = self._hash_embedding(text)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def is_available(self) -> bool:
        """埋め込み機能が利用可能かチェック"""
        return True  # 常に利用可能


def get_sentence_transformer(model_name: str = 'simple', 
                           cache_folder: Optional[str] = None,
                           local_files_only: bool = True):
    """
    SimpleEmbedder を SentenceTransformer互換で取得
    
    Args:
        model_name: モデル名（無視される）
        cache_folder: キャッシュフォルダ（無視される）
        local_files_only: ローカルファイルのみ（無視される）
        
    Returns:
        SimpleEmbedderインスタンス
    """
    logger.info(f"🚀 SimpleEmbedder初期化開始（SentenceTransformer互換モード）")
    
    embedder = SimpleEmbedder({
        'embedding_dim': 384,
        'embedding_method': 'hash',
        'vocab_size': 10000
    })
    
    logger.info(f"✅ SimpleEmbedder初期化完了（軽量埋め込み）")
    return embedder


class SharedEmbedder:
    """
    SimpleEmbedder用の共有ラッパー
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        共有Embedderの初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.debug = config.get('debug', False)
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # SimpleEmbedderを作成
        self._embedder = SimpleEmbedder(config)
    
    @property
    def transformer(self):
        """
        SimpleEmbedderインスタンスを取得
        
        Returns:
            SimpleEmbedderインスタンス
        """
        return self._embedder
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Optional[np.ndarray]:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            texts: 単一テキストまたはテキストリスト
            **kwargs: 追加引数
            
        Returns:
            埋め込みベクトル配列 or None
        """
        try:
            if self.debug:
                logger.debug(f"🔍 埋め込み生成: {type(texts)} (len={len(texts) if isinstance(texts, list) else 1})")
            
            embeddings = self._embedder.encode(texts, **kwargs)
            
            if self.debug:
                logger.debug(f"✅ 埋め込み生成完了: shape={embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ 埋め込み生成エラー: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Embedderが利用可能かチェック
        
        Returns:
            利用可能性
        """
        return True  # SimpleEmbedderは常に利用可能
