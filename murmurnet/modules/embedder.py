#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 共有Embedderモジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SentenceTransformerのSingleton実装
プロセス起動時に一度だけロードし、全エージェントで共有

作者: Yuhi Sonoki
"""

import logging
import os
import time
import threading
from functools import lru_cache
from typing import Dict, Any, Optional, List, Union
import numpy as np

logger = logging.getLogger('MurmurNet.Embedder')

# グローバル変数（プロセス全体で共有）
_GLOBAL_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def get_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2', 
                           cache_folder: Optional[str] = None,
                           local_files_only: bool = True):
    """
    SentenceTransformerのSingleton取得
    
    @lru_cache により、同じ引数で何度呼び出されても
    初回のみインスタンス化され、以降は同じインスタンスを返す
    
    引数:
        model_name: モデル名
        cache_folder: キャッシュフォルダ
        local_files_only: ローカルファイルのみ使用
        
    戻り値:
        SentenceTransformerインスタンス
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"🚀 SentenceTransformer初期化開始: {model_name}")
        start_time = time.time()
        
        # オフライン実行に最適化されたキャッシュ設定
        if cache_folder is None:
            cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
        
        os.makedirs(cache_folder, exist_ok=True)
        
        # モデル名の正規化（sentence-transformers/ プレフィックスを除去）
        if model_name.startswith('sentence-transformers/'):
            model_name = model_name[len('sentence-transformers/'):]
        
        transformer = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
            cache_folder=cache_folder,
            trust_remote_code=False  # セキュリティ強化
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ SentenceTransformer初期化完了: {model_name} ({load_time:.2f}s)")
        return transformer
        
    except ImportError:
        logger.error("❌ SentenceTransformersがインストールされていません")
        # SimpleEmbedderにフォールバック
        from .simple_embedder import get_sentence_transformer as get_simple_embedder
        return get_simple_embedder()
    except Exception as e:
        logger.error(f"❌ SentenceTransformer初期化エラー: {e}")
        # SimpleEmbedderにフォールバック
        from .simple_embedder import get_sentence_transformer as get_simple_embedder
        return get_simple_embedder()

class SharedEmbedder:
    """
    共有Embedderクラス
    
    全てのInputReceptionインスタンスで同じSentenceTransformerを共有
    メモリ使用量とロード時間を大幅に削減
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        共有Embedderの初期化
        
        引数:
            config: 設定辞書
        """
        self.config = config
        self.model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.cache_folder = config.get('cache_folder', None)
        self.local_files_only = config.get('local_files_only', True)
        self.debug = config.get('debug', False)
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
    
    @property
    def transformer(self):
        """
        SentenceTransformerインスタンスを取得
        
        初回アクセス時のみロードされ、以降は同じインスタンスを返す
        
        戻り値:
            SentenceTransformerインスタンス or None
        """
        return get_sentence_transformer(
            self.model_name,
            self.cache_folder,
            self.local_files_only
        )
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Optional[np.ndarray]:
        """
        テキストを埋め込みベクトルに変換
        
        引数:
            texts: 単一テキストまたはテキストリスト
            **kwargs: SentenceTransformer.encodeの追加引数
            
        戻り値:
            埋め込みベクトル配列 or None
        """
        transformer = self.transformer
        if transformer is None:
            logger.warning("⚠️ SentenceTransformerが利用できません")
            return None
        
        try:
            if self.debug:
                logger.debug(f"🔍 埋め込み生成: {type(texts)} (len={len(texts) if isinstance(texts, list) else 1})")
            
            # デフォルト設定
            encode_kwargs = {
                'show_progress_bar': False,
                'convert_to_numpy': True,
                'normalize_embeddings': True,
                **kwargs
            }
            
            embeddings = transformer.encode(texts, **encode_kwargs)
            
            if self.debug:
                shape = embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'
                logger.debug(f"✅ 埋め込み生成完了: shape={shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ 埋め込み生成エラー: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Embedderが利用可能かチェック
        
        戻り値:
            True: 利用可能, False: 利用不可
        """
        return self.transformer is not None
    
    def get_embedding_dim(self) -> int:
        """
        埋め込み次元数を取得
        
        戻り値:
            埋め込み次元数
        """
        transformer = self.transformer
        if transformer is None:
            return 384  # all-MiniLM-L6-v2のデフォルト次元数
        
        try:
            # ダミーテキストで次元数を確認
            dummy_embedding = transformer.encode(["test"], show_progress_bar=False)
            return dummy_embedding.shape[1]
        except Exception:
            return 384

# 便利関数
def get_shared_embedder(config: Dict[str, Any]) -> SharedEmbedder:
    """
    共有Embedderインスタンスを取得
    
    引数:
        config: 設定辞書
        
    戻り値:
        SharedEmbedderインスタンス
    """
    return SharedEmbedder(config)

# 後方互換性のためのエイリアス
def get_global_embedder(config: Dict[str, Any]) -> SharedEmbedder:
    """後方互換性のためのエイリアス"""
    return get_shared_embedder(config)

# Embedderクラスのエイリアス（後方互換性）
class Embedder(SharedEmbedder):
    """
    Embedderクラス（SharedEmbedderのエイリアス）
    後方互換性のために提供
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Embedderの初期化
        SharedEmbedderは自動初期化のため、実際の処理は不要
        
        戻り値:
            True: 初期化成功, False: 初期化失敗
        """
        try:
            # SentenceTransformerの可用性をチェック
            available = self.is_available()
            self._initialized = available
            
            if available and self.debug:
                logger.info("✅ Embedder初期化完了")
            elif not available:
                logger.warning("⚠️ Embedder初期化失敗 - SentenceTransformerが利用不可")
            
            return available
            
        except Exception as e:
            logger.error(f"❌ Embedder初期化エラー: {e}")
            self._initialized = False
            return False
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        単一テキストの埋め込み生成
        
        引数:
            text: 埋め込み対象のテキスト
            
        戻り値:
            埋め込みベクトル or None
        """
        return self.encode(text)
    
    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        複数テキストの埋め込み生成
        
        引数:
            texts: 埋め込み対象のテキストリスト
            
        戻り値:
            埋め込みベクトル配列 or None
        """
        return self.encode(texts)
    
    @property
    def is_initialized(self) -> bool:
        """初期化状態を取得"""
        return getattr(self, '_initialized', False)

if __name__ == "__main__":
    # テスト用コード
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'local_files_only': True,
        'debug': True
    }
    
    print("=== SharedEmbedder テスト ===")
    
    # 1回目のロード
    embedder1 = get_shared_embedder(config)
    start_time = time.time()
    embedding1 = embedder1.encode("Hello, world!")
    time1 = time.time() - start_time
    print(f"1回目: {time1:.3f}s, shape: {embedding1.shape if embedding1 is not None else 'None'}")
    
    # 2回目のロード（キャッシュから）
    embedder2 = get_shared_embedder(config)
    start_time = time.time()
    embedding2 = embedder2.encode("Hello, world!")
    time2 = time.time() - start_time
    print(f"2回目: {time2:.3f}s, shape: {embedding2.shape if embedding2 is not None else 'None'}")
    
    # 同じインスタンスかチェック
    print(f"同じtransformerインスタンス: {embedder1.transformer is embedder2.transformer}")
    print(f"速度向上: {time1/time2:.1f}倍")
