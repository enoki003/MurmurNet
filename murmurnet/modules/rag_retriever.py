#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Knowledge Retriever（CPU最適化版）
~~~~~~~~~~~~~~~~~~~~~~~
Retrieval Augmented Generation (RAG) モジュール
CPU効率・並列処理・ベクトル化を最適化

機能:
- マルチスレッド埋め込み計算
- バッチ処理によるベクトル化
- 効率的なLRUキャッシュ
- 非同期I/O処理
- CPU最適化された類似度計算

作者: Yuhi Sonoki
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import psutil
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# scipy.spatial.distanceが利用できない場合のフォールバック
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # フォールバック関数
    def scipy_cosine(v1, v2):
        """Simple cosine distance fallback"""
        return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

logger = logging.getLogger("MurmurNet.RAGRetriever")

# ──────────────────────────────────────────────────────────────
# Optional dependencies & Global Cache
# ──────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not found – dummy modeへフォールバック")

# グローバルなSentenceTransformerキャッシュ（スレッドセーフ）
_GLOBAL_TRANSFORMER_CACHE = {}
_TRANSFORMER_CACHE_LOCK = threading.RLock()

try:
    from libzim.reader import Archive
    from libzim.search import Searcher, Query
    HAS_LIBZIM = True
except ImportError:
    HAS_LIBZIM = False
    logger.warning("libzim not found – dummy modeへフォールバック")

# ──────────────────────────────────────────────────────────────
# Optimized Utility Functions
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1000)
def cached_html_to_text(html: str) -> str:
    """最適化されたHTML→テキスト変換（キャッシュ付き）"""
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", html)
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def vectorized_cosine_similarity(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    """
    ベクトル化されたコサイン類似度計算
    
    Args:
        vectors1: (N, D) の形状の配列
        vectors2: (M, D) の形状の配列
    
    Returns:
        (N, M) の類似度行列
    """
    # L2正規化
    norm1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
    
    vectors1_normalized = vectors1 / (norm1 + 1e-8)
    vectors2_normalized = vectors2 / (norm2 + 1e-8)
    
    # ドット積による類似度計算
    return np.dot(vectors1_normalized, vectors2_normalized.T)


def batch_cosine_similarity(query_vector: np.ndarray, candidate_vectors: List[np.ndarray]) -> List[float]:
    """
    バッチ処理による効率的なコサイン類似度計算
    """
    if not candidate_vectors:
        return []
    
    # NumPy配列に変換
    candidates_array = np.vstack(candidate_vectors)
    query_array = query_vector.reshape(1, -1)
    
    # ベクトル化された類似度計算
    similarities = vectorized_cosine_similarity(query_array, candidates_array)
    
    return similarities[0].tolist()


class OptimizedEmbeddingCache:
    """最適化された埋め込みキャッシュ"""
    
    def __init__(self, max_size: int = 2000, enable_persistence: bool = False):
        self.max_size = max_size
        self.enable_persistence = enable_persistence
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    def _evict_lru(self):
        """LRUエビクション"""
        if len(self.cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """キャッシュから取得"""
        with self.lock:
            if key in self.cache:
                # アクセス順序を更新
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                self.stats['hits'] += 1
                return self.cache[key].copy()
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: np.ndarray):
        """キャッシュに保存"""
        with self.lock:
            self._evict_lru()
            self.cache[key] = value.copy()
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_size': self.max_size
            }


class CPUOptimizedRetriever:
    """CPU最適化されたRAGリトリーバー実装"""
    
    def __init__(self, config):
        self.config = config
        self.mode = config.get("rag_mode", "dummy")
        self.score_threshold = float(config.get("rag_score_threshold", 0.5))
        self.top_k = int(config.get("rag_top_k", 5))
        self.debug = bool(config.get("debug", False))
        self.knowledge_base = config.get("knowledge_base", [])
          # CPU最適化設定
        self.cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
        max_workers_config = config.get("max_workers", 4) or 4
        self.max_workers = min(self.cpu_cores, max_workers_config)
        self.batch_size = config.get("batch_size", 32)
        self.enable_vectorization = config.get("enable_vectorization", True)
        
        # 最適化されたキャッシュ
        cache_size = config.get("embedding_cache_size", min(2000, self.cpu_cores * 200))
        self.embedding_cache = OptimizedEmbeddingCache(max_size=cache_size)
        
        # ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="RAGRetriever"
        )
        
        # 統計情報
        self.stats = {
            'queries': 0,
            'batch_operations': 0,
            'parallel_operations': 0,
            'vectorized_operations': 0,
            'average_response_time': 0.0
        }
        self.stats_lock = threading.Lock()
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # ZIM・埋め込みの初期化
        if self.mode == "zim":
            self._init_zim()
            self._init_embedding()
        
        logger.info("CPU最適化RAGリトリーバー初期化完了: CPU%dコア, ワーカー%d, キャッシュ%d", 
                   self.cpu_cores, self.max_workers, cache_size)
    
    def _init_zim(self):
        """ZIM アーカイブのロード"""
        if not HAS_LIBZIM:
            logger.warning("libzim not available - falling back to dummy mode")
            self.mode = "dummy"
            return

        path = self.config.get("zim_path")
        if not path or not os.path.exists(path):
            logger.warning("ZIM パス無効 – dummy modeに移行")
            self.mode = "dummy"
            return
            
        try:
            self.zim_archive = Archive(path)
            logger.info("Loaded ZIM: %s", path)
        except Exception as e:
            logger.error("ZIM 読み込み失敗: %s", e)
            self.mode = "dummy"
    
    def _init_embedding(self):
        """SentenceTransformer のロード（グローバルキャッシュ対応）"""
        logger.info("埋め込みモデル初期化開始...")
        
        if self.mode != "zim":
            logger.info("ZIMモードではないため、埋め込み初期化をスキップ")
            return
            
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("SentenceTransformerが利用不可のため、dummyモードに変更")
            self.mode = "dummy"
            return

        try:
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            cache_dir = (
                self.config.get("model_cache_dir")
                or os.path.join(
                    os.path.dirname(__file__), "..", "..", "cache", "sentence_transformers"
                )
            )
            
            # グローバルキャッシュから取得を試行
            cache_key = f"{model_name}:{cache_dir}"
            with _TRANSFORMER_CACHE_LOCK:
                if cache_key in _GLOBAL_TRANSFORMER_CACHE:
                    self.transformer = _GLOBAL_TRANSFORMER_CACHE[cache_key]
                    logger.info(f"埋め込みモデルをキャッシュから取得: {model_name}")
                    return
            
            logger.info(f"埋め込みモデル準備: {model_name}, キャッシュ: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)

            # オフライン優先環境変数
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
            os.environ.setdefault("HF_HUB_OFFLINE", "0")

            logger.info("SentenceTransformer読み込み開始...")
            start_time = time.time()
            
            self.transformer = SentenceTransformer(
                model_name, 
                cache_folder=cache_dir, 
                device="cpu"
            )
            
            # CPU最適化設定
            if hasattr(self.transformer, 'encode'):
                # バッチサイズを設定
                self.transformer.batch_size = self.batch_size
            
            # グローバルキャッシュに保存
            with _TRANSFORMER_CACHE_LOCK:
                _GLOBAL_TRANSFORMER_CACHE[cache_key] = self.transformer
            
            load_time = time.time() - start_time
            logger.info("埋め込みモデル読み込み完了: %s (時間: %.2f秒)", model_name, load_time)
            
        except Exception as e:
            logger.error("埋め込みモデル初期化失敗: %s", e)
            self.mode = "dummy"
    
    def retrieve(self, query: str) -> str:
        """入力クエリに関連する情報を効率的に検索"""
        start_time = time.time()
        
        try:
            # 統計更新
            with self.stats_lock:
                self.stats['queries'] += 1
            
            if not isinstance(query, str) or len(query.strip()) < 2:
                return "検索クエリが無効です。"

            query = query.strip()
            
            # モード別検索
            if self.mode == "zim":
                result = self._retrieve_zim(query)
            elif self.mode == "knowledge_base":
                result = self._retrieve_knowledge_base(query)
            else:
                result = self._retrieve_dummy(query)
            
            # レスポンス時間統計更新
            response_time = time.time() - start_time
            with self.stats_lock:
                current_avg = self.stats['average_response_time']
                total_queries = self.stats['queries']
                self.stats['average_response_time'] = (
                    (current_avg * (total_queries - 1) + response_time) / total_queries
                )
            
            return result
            
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return f"検索中にエラーが発生しました: {e}"
    
    def _retrieve_zim(self, query: str) -> str:
        """ZIM検索（最適化版）"""
        if not hasattr(self, "zim_archive"):
            return "ZIM アーカイブが利用できません。"

        try:
            searcher = Searcher(self.zim_archive)
            results = searcher.search(Query().set_query(query))
            
            if results.get_matches_estimated() == 0:
                return "検索結果が見つかりませんでした。"

            # 並列処理でコンテンツ抽出
            snippets = []
            futures = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for _ in range(min(self.top_k, results.get_matches_estimated())):
                    r = results.get_next()
                    if r:
                        future = executor.submit(self._extract_zim_content, r)
                        futures.append(future)
                
                # 結果収集
                for future in as_completed(futures):
                    try:
                        snippet = future.result(timeout=5.0)
                        if snippet:
                            snippets.append(snippet)
                    except Exception as e:
                        if self.debug:
                            logger.debug(f"ZIM コンテンツ抽出エラー: {e}")
            
            with self.stats_lock:
                self.stats['parallel_operations'] += 1
            
            return "\n\n".join(snippets) if snippets else "コンテンツの抽出に失敗しました。"

        except Exception as e:
            if self.debug:
                logger.exception("ZIM検索エラー")
            return f"検索中にエラーが発生しました: {e}"
    
    def _extract_zim_content(self, result) -> Optional[str]:
        """ZIMコンテンツ抽出（並列処理用）"""
        try:
            entry = self.zim_archive.get_entry_by_path(result.get_path())
            if not entry:
                return None
                
            content = entry.get_item().content.tobytes().decode("utf-8", errors="ignore")
            
            if entry.get_mime_type() == "text/html":
                content = cached_html_to_text(content)
            
            return f"【{entry.title}】 {content[:500]}…"
            
        except Exception as e:
            if self.debug:
                logger.debug(f"コンテンツ抽出エラー: {e}")
            return None
    
    def _retrieve_knowledge_base(self, query: str) -> str:
        """知識ベース検索（ベクトル化最適化版）"""
        if not self.knowledge_base:
            return "知識ベースが空です。"

        try:
            if hasattr(self, "transformer"):
                return self._vectorized_knowledge_search(query)
            else:
                return self._keyword_knowledge_search(query)
                
        except Exception as e:
            if self.debug:
                logger.exception("知識ベース検索エラー")
            return f"検索中にエラーが発生しました: {e}"
    
    def _vectorized_knowledge_search(self, query: str) -> str:
        """ベクトル化された知識ベース検索"""
        # クエリ埋め込み取得
        query_embedding = self._get_embedding(query)
        
        # 候補埋め込みを並列で取得・計算
        candidates = []
        embeddings = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 埋め込み取得タスクを並列実行
            future_to_item = {
                executor.submit(self._get_item_embedding, item): item 
                for item in self.knowledge_base
            }
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    embedding = future.result(timeout=3.0)
                    if embedding is not None:
                        candidates.append(item)
                        embeddings.append(embedding)
                except Exception as e:
                    if self.debug:
                        logger.debug(f"埋め込み取得エラー: {e}")
        
        if not embeddings:
            return "検索結果が見つかりませんでした。"
        
        # バッチ類似度計算
        similarities = batch_cosine_similarity(query_embedding, embeddings)
        
        # スコアでフィルタリング・ソート
        scored_items = [
            (sim, item) for sim, item in zip(similarities, candidates)
            if sim >= self.score_threshold
        ]
        scored_items.sort(reverse=True)
        
        with self.stats_lock:
            self.stats['vectorized_operations'] += 1
            self.stats['parallel_operations'] += 1
        
        if not scored_items:
            return "検索結果が見つかりませんでした。"

        return "\n\n".join(
            f"【{item['title']}】 {item['content'][:500]}…" 
            for _, item in scored_items[:self.top_k]
        )
    
    def _keyword_knowledge_search(self, query: str) -> str:
        """キーワードベース知識検索"""
        keywords = query.lower().split()
        scored = []
        
        for item in self.knowledge_base:
            content = (item.get("title", "") + " " + item.get("content", "")).lower()
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                scored.append((score, item))
        
        scored.sort(reverse=True)
        
        if not scored:
            return "検索結果が見つかりませんでした。"

        return "\n\n".join(
            f"【{item['title']}】 {item['content'][:500]}…" 
            for _, item in scored[:self.top_k]
        )
    
    def _get_item_embedding(self, item) -> Optional[np.ndarray]:
        """アイテムの埋め込みを取得（キャッシュ対応）"""
        content = item.get("content", "")
        if not content:
            return None
        
        # 既存の埋め込みがあるか確認
        if "embedding" in item and isinstance(item["embedding"], np.ndarray):
            return item["embedding"]
        
        # 新規生成
        return self._get_embedding(content)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """テキストの埋め込みベクトルを取得（最適化版）"""
        if not text.strip():
            return np.zeros(384)  # デフォルト次元
        
        # キャッシュから検索
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        cached_embedding = self.embedding_cache.get(cache_key)
        
        if cached_embedding is not None:
            return cached_embedding
        
        # 新規生成
        if hasattr(self, 'transformer') and self.transformer:
            try:
                embedding = self.transformer.encode(text, convert_to_numpy=True)
                self.embedding_cache.put(cache_key, embedding)
                return embedding
            except Exception as e:
                if self.debug:
                    logger.debug(f"埋め込み生成エラー: {e}")
        
        # フォールバック
        return np.zeros(384)
    
    def _retrieve_dummy(self, query: str) -> str:
        """最小限のキーワード辞書応答（最適化版）"""
        basic_knowledge = {
            "AI": "人工知能（AI）は、機械が人間に似た知的行動を示す技術分野です。",
            "機械学習": "データからパターンを学び、予測や分類を行う AI 技術です。",
            "深層学習": "多層ニューラルネットワークを活用し、画像・音声・言語で成果を上げています。",
            "自然言語処理": "コンピュータが人間の言語を理解・生成する技術です。",
            "CPU最適化": "CPUの性能を最大限活用するための並列処理とベクトル化技術です。",
            "並列処理": "複数のタスクを同時に実行してパフォーマンスを向上させる手法です。"
        }
        
        query_lower = query.lower()
        hits = [
            f"【{k}】 {v}" 
            for k, v in basic_knowledge.items() 
            if k.lower() in query_lower
        ]
        
        return "\n\n".join(hits) if hits else "より具体的な質問をお聞かせください。"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        cache_stats = self.embedding_cache.get_stats()
        
        return {
            **stats,
            'cache_stats': cache_stats,
            'cpu_cores': self.cpu_cores,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }
    
    def optimize_cache(self):
        """キャッシュ最適化を実行"""
        # メモリ使用量チェック
        memory = psutil.virtual_memory()
        
        if memory.percent > 80:  # メモリ使用率80%超過時
            # キャッシュサイズを半分に削減
            current_size = len(self.embedding_cache.cache)
            target_size = current_size // 2
            
            with self.embedding_cache.lock:
                # 古いエントリから削除
                while len(self.embedding_cache.cache) > target_size and self.embedding_cache.access_order:
                    oldest_key = self.embedding_cache.access_order.pop(0)
                    if oldest_key in self.embedding_cache.cache:
                        del self.embedding_cache.cache[oldest_key]
            
            logger.info(f"キャッシュ最適化完了: {current_size} -> {len(self.embedding_cache.cache)}")
    
    def shutdown(self):
        """適切なシャットダウン"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if self.debug:
            stats = self.get_performance_stats()
            logger.info(f"RAGRetriever shutdown完了. 最終統計: {stats}")


# 後方互換性のためのエイリアス
RAGRetriever = CPUOptimizedRetriever
