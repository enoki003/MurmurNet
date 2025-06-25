#!/usr/bin/env python3
"""
モデル・エンベッディングキャッシュ最適化
毎回の重いロード（6s以上）を削減するシングルトンパターン実装
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class ModelCache:
    """モデルキャッシュのシングルトンクラス"""
    _embedder = None
    _llm_instance = None
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_embedder(cls, model_name="all-MiniLM-L6-v2"):
        """SentenceTransformerのシングルトンインスタンス取得"""
        if cls._embedder is None:
            logger.info(f"Loading SentenceTransformer: {model_name}")
            cls._embedder = SentenceTransformer(
                model_name, 
                device="cpu", 
                local_files_only=True,
                cache_folder="./models/st_cache"
            )
            logger.info("SentenceTransformer loaded and cached")
        return cls._embedder
    
    @classmethod
    def get_llm_instance(cls):
        """LLMインスタンスのキャッシュ取得"""
        return cls._llm_instance
    
    @classmethod
    def set_llm_instance(cls, llm):
        """LLMインスタンスをキャッシュに保存"""
        cls._llm_instance = llm
        logger.info("LLM instance cached for reuse")
    
    @classmethod
    def clear_cache(cls):
        """キャッシュをクリア"""
        cls._embedder = None
        cls._llm_instance = None
        logger.info("Model cache cleared")

def apply_embedder_cache_patch():
    """InputReceptionにエンベッダーキャッシュパッチを適用"""
    try:
        from MurmurNet.modules.input_reception import InputReception
        
        # 元のprocessメソッドをバックアップ
        if not hasattr(InputReception, '_original_process'):
            InputReception._original_process = InputReception.process
        
        def cached_process(self, user_input):
            """キャッシュされたエンベッダーを使用"""
            # エンベッダーをキャッシュから取得
            if not hasattr(self, 'embedder') or self.embedder is None:
                self.embedder = ModelCache.get_embedder()
            return InputReception._original_process(self, user_input)
        
        # メソッドを置き換え
        InputReception.process = cached_process
        logger.info("Embedder cache patch applied to InputReception")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to apply embedder cache patch: {e}")
        return False

def apply_llm_cache_patch():
    """LLMインスタンスキャッシュパッチを適用"""
    try:
        from MurmurNet.modules.model_factory import ModelFactory
        
        # 元のcreate_modelメソッドをバックアップ
        original_create_model = ModelFactory.create_model
        
        def cached_create_model(self, model_config):
            """キャッシュされたLLMインスタンスを使用"""
            cached_llm = ModelCache.get_llm_instance()
            if cached_llm is not None:
                logger.info("Using cached LLM instance")
                return cached_llm
            
            # 新規作成時はキャッシュに保存
            llm = original_create_model(self, model_config)
            ModelCache.set_llm_instance(llm)
            return llm
        
        # メソッドを置き換え
        ModelFactory.create_model = cached_create_model
        logger.info("LLM cache patch applied to ModelFactory")
        
    except ImportError as e:
        logger.error(f"Failed to apply LLM cache patch: {e}")

def optimize_model_loading():
    """モデルロード最適化を実行"""
    logger.info("Applying model cache optimizations...")
    
    # エンベッダーキャッシュパッチ適用
    apply_embedder_cache_patch()
    
    # LLMキャッシュパッチ適用
    apply_llm_cache_patch()
    
    # 環境変数でキャッシュを有効化
    os.environ['TRANSFORMERS_CACHE'] = './models/cache'
    os.environ['HF_HOME'] = './models/cache'
    
    logger.info("Model cache optimization applied")
    return True

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("Testing model cache optimization...")
    
    # 最適化適用
    optimize_model_loading()
    
    # テスト：エンベッダーの初回ロード
    import time
    start_time = time.time()
    embedder = ModelCache.get_embedder()
    load_time = time.time() - start_time
    print(f"Embedder load time: {load_time:.2f}s")
    
    # 2回目アクセス（キャッシュヒット）
    start_time = time.time()
    embedder2 = ModelCache.get_embedder()
    cache_time = time.time() - start_time
    print(f"Embedder cache access time: {cache_time:.4f}s")
    
    # 同一インスタンスかチェック
    print(f"Same instance: {embedder is embedder2}")
    
    print("Model cache optimization test completed")
