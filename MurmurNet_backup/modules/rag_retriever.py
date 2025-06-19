#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Knowledge Retriever
~~~~~~~~~~~~~~~~~~~~~~~
Retrieval Augmented Generation (RAG) モジュール
埋め込み・ZIM・ローカル知識ベース検索に対応

作者: Yuhi Sonoki
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("MurmurNet.RAGRetriever")

# ──────────────────────────────────────────────────────────────
# Optional dependencies
# ──────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not found – dummy modeへフォールバック")

try:
    from libzim.reader import Archive
    from libzim.search import Searcher, Query
    HAS_LIBZIM = True
except ImportError:
    HAS_LIBZIM = False
    logger.warning("libzim not found – dummy modeへフォールバック")

# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────
def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """2 ベクトル間のコサイン類似度 (0–1)。"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _html_to_text(html: str) -> str:
    """最小限の正規表現で HTML→プレーンテキスト変換。"""
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", html)
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────
class RAGRetriever:
    """
    検索拡張生成 (RAG) を提供するリトリーバー（埋め込みキャッシュ対応）
    
    特徴:
    - 埋め込みベクトルのハッシュベースキャッシュ
    - スレッドセーフなキャッシュアクセス
    - LRUキャッシュでメモリ効率向上
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode: str = config.get("rag_mode", "dummy")
        self.score_threshold: float = float(config.get("rag_score_threshold", 0.5))
        self.top_k: int = int(config.get("rag_top_k", 5))
        self.debug: bool = bool(config.get("debug", False))
        self.knowledge_base: List[Dict[str, Any]] = config.get("knowledge_base", [])

        # 埋め込みキャッシュの設定
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._cache_max_size = config.get("embedding_cache_size", 1000)  # 最大1000エントリ
        self._cache_access_order: List[str] = []  # LRU用
        
        if self.debug:
            logger.setLevel(logging.DEBUG)

        if self.mode == "zim":
            self._init_zim()
            self._init_embedding()

        logger.info("RAG リトリーバー初期化 (モード: %s, キャッシュサイズ: %d)", self.mode, self._cache_max_size)

    # ───────────────────── init helpers ─────────────────────
    def _init_zim(self) -> None:
        """ZIM アーカイブのロード。"""
        if not HAS_LIBZIM:
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

    def _init_embedding(self) -> None:
        """SentenceTransformer のロード。"""
        logger.info("埋め込みモデル初期化開始...")
        
        if self.mode != "zim":
            logger.info("ZIMモードではないため、埋め込み初期化をスキップ")
            return
            
        if SentenceTransformer is None:
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
            
            logger.info(f"埋め込みモデル準備: {model_name}, キャッシュ: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)

            # オフライン優先環境変数
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
            os.environ.setdefault("HF_HUB_OFFLINE", "0")

            logger.info("SentenceTransformer読み込み開始...")
            import time
            start_time = time.time()
            
            self.transformer = SentenceTransformer(
                model_name, cache_folder=cache_dir, device="cpu"
            )
            
            load_time = time.time() - start_time
            logger.info("埋め込みモデル読み込み完了: %s (時間: %.2f秒, キャッシュ: %s)", model_name, load_time, cache_dir)
            
        except Exception as e:
            logger.error("埋め込みモデル初期化失敗: %s", e)
            import traceback
            logger.debug("埋め込み初期化エラー詳細:\n%s", traceback.format_exc())
            self.mode = "dummy"

    # ───────────────────── retrieval public ─────────────────────
    def retrieve(self, query: str) -> str:
        """入力クエリに関連する情報を検索して返す。"""
        if not isinstance(query, str) or len(query.strip()) < 2:
            return "検索クエリが無効です。"

        query = query.strip()
        if self.mode == "zim":
            return self._retrieve_zim(query)
        if self.mode == "knowledge_base":
            return self._retrieve_knowledge_base(query)
        return self._retrieve_dummy(query)

    # ───────────────────── internal search ─────────────────────
    def _retrieve_zim(self, query: str) -> str:
        if not hasattr(self, "zim_archive"):
            return "ZIM アーカイブが利用できません。"

        try:
            searcher = Searcher(self.zim_archive)
            results = searcher.search(Query().set_query(query))
            if results.get_matches_estimated() == 0:
                return "検索結果が見つかりませんでした。"

            snippets = []
            for _ in range(min(self.top_k, results.get_matches_estimated())):
                r = results.get_next()
                if not r:
                    continue
                entry = self.zim_archive.get_entry_by_path(r.get_path())
                if not entry:
                    continue
                content = entry.get_item().content.tobytes().decode("utf-8", errors="ignore")
                if entry.get_mime_type() == "text/html":
                    content = _html_to_text(content)
                snippets.append(f"【{entry.title}】 {content[:500]}…")

            return "\n\n".join(snippets)

        except Exception as e:
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return f"検索中にエラーが発生しました: {e}"

    def _retrieve_knowledge_base(self, query: str) -> str:
        if not self.knowledge_base:
            return "知識ベースが空です。"

        try:
            # 埋め込み検索を優先
            if hasattr(self, "transformer"):
                q_emb = self.transformer.encode(query)
                scored = []
                for item in self.knowledge_base:
                    emb = item.get("embedding")
                    if emb is None:
                        emb = self.transformer.encode(item.get("content", ""))
                    sim = cosine(q_emb, emb)
                    if sim >= self.score_threshold:
                        scored.append((sim, item))
                scored.sort(reverse=True)
            else:
                # キーワード一致のみ
                scored = [
                    (1.0, it)
                    for it in self.knowledge_base
                    if any(
                        w.lower()
                        in (it.get("title", "") + it.get("content", "")).lower()
                        for w in query.split()
                    )
                ]

            if not scored:
                return "検索結果が見つかりませんでした。"

            return "\n\n".join(
                f"【{it['title']}】 {it['content'][:500]}…" for _, it in scored[: self.top_k]
            )

        except Exception as e:
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return f"検索中にエラーが発生しました: {e}"

    def _retrieve_dummy(self, query: str) -> str:
        """最小限のキーワード辞書応答。"""
        basic = {
            "AI": "人工知能（AI）は、機械が人間に似た知的行動を示す技術分野です。",
            "機械学習": "データからパターンを学び、予測や分類を行う AI 技術です。",
            "深層学習": "多層ニューラルネットワークを活用し、画像・音声・言語で成果を上げています。",
            "自然言語処理": "コンピュータが人間の言語を理解・生成する技術です。",
        }
        hits = [f"【{k}】 {v}" for k, v in basic.items() if k.lower() in query.lower()]
        return "\n\n".join(hits) if hits else "より具体的な質問をお聞かせください。"

    # ───────────────────── embedding cache helpers ─────────────────────
    def _get_embedding_hash(self, text: str) -> str:
        """テキストのハッシュ値を生成（キャッシュキー用）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """キャッシュから埋め込みベクトルを取得"""
        cache_key = self._get_embedding_hash(text)
        
        with self._cache_lock:
            if cache_key in self._embedding_cache:
                # LRUアクセス順序を更新
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                
                if self.debug:
                    logger.debug(f"埋め込みキャッシュヒット: {text[:50]}...")
                return self._embedding_cache[cache_key]
        
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """埋め込みベクトルをキャッシュに保存"""
        cache_key = self._get_embedding_hash(text)
        
        with self._cache_lock:
            # キャッシュサイズ管理（LRU）
            while len(self._embedding_cache) >= self._cache_max_size:
                if self._cache_access_order:
                    oldest_key = self._cache_access_order.pop(0)
                    if oldest_key in self._embedding_cache:
                        del self._embedding_cache[oldest_key]
                        if self.debug:
                            logger.debug(f"LRU: 古い埋め込みをキャッシュから削除")
                else:
                    break
            
            # 新しい埋め込みを追加
            self._embedding_cache[cache_key] = embedding
            self._cache_access_order.append(cache_key)
            
            if self.debug:
                logger.debug(f"埋め込みをキャッシュに保存: {text[:50]}... (キャッシュサイズ: {len(self._embedding_cache)})")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """テキストの埋め込みベクトルを取得（キャッシュ対応）"""
        # キャッシュから検索
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # キャッシュにない場合は新規生成
        if hasattr(self, 'transformer') and self.transformer:
            embedding = self.transformer.encode(text)
            self._cache_embedding(text, embedding)
            return embedding
        
        # SentenceTransformerが利用できない場合はゼロベクトル
        return np.zeros(384)  # all-MiniLM-L6-v2の次元数
