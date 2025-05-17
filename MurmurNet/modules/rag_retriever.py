#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Knowledge Retriever
~~~~~~~~~~~~~~~~~~~~~
Retrieval Augmented Generation モジュール
様々な検索方法に対応（埋め込み、ZIM、データベース）

Author: Yuhi Sonoki
"""
from __future__ import annotations

import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Union, Optional, ClassVar

import numpy as np

# ───────────────────────────────────────────────────────────────────────────
# Optional deps
# ───────────────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as e:
    SentenceTransformer = None  # type: ignore
    logging.warning("sentence‑transformers not found: %s", e)

try:
    from libzim.reader import Archive  # type: ignore
    from libzim.search import Searcher, Query  # type: ignore
    HAS_LIBZIM = True
except ImportError:
    HAS_LIBZIM = False

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────────────────

def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _html_to_text(html: str) -> str:
    """Very lightweight HTML→plain text."""
    # drop style & script
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", html)
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    # drop tags
    text = re.sub(r"<[^>]+>", " ", html)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ───────────────────────────────────────────────────────────────────────────
# 共有リソース管理
# ───────────────────────────────────────────────────────────────────────────

class SharedModels:
    """モデルやリソースを共有するためのクラス変数ホルダー"""
    # 共有ZIMアーカイブとパス
    _zim_archives: Dict[str, Archive] = {}
    # 共有埋め込みモデル
    _embedding_models: Dict[str, Any] = {}
    # ロック用
    _is_loading: bool = False
    
    @classmethod
    def get_zim_archive(cls, path: str) -> Optional[Archive]:
        """ZIMアーカイブを取得（なければロード）"""
        if path in cls._zim_archives:
            return cls._zim_archives[path]
        
        # ロードを試みる
        if not HAS_LIBZIM or not os.path.exists(path):
            return None
            
        # ダブルチェックロック
        if cls._is_loading:
            return None
            
        cls._is_loading = True
        try:
            # 新しいアーカイブを開く
            archive = Archive(path)
            cls._zim_archives[path] = archive
            logger.info(f"ZIM共有アーカイブをロードしました: {path}")
            return archive
        except Exception as e:
            logger.error(f"ZIMアーカイブ開封エラー: {e}")
            return None
        finally:
            cls._is_loading = False
    
    @classmethod
    def get_embedding_model(cls, model_name: str) -> Optional[Any]:
        """埋め込みモデルを取得（なければロード）"""
        if model_name in cls._embedding_models:
            return cls._embedding_models[model_name]
            
        # ロードを試みる
        if SentenceTransformer is None:
            return None
            
        # ダブルチェックロック
        if cls._is_loading:
            return None
            
        cls._is_loading = True
        try:
            # 新しいモデルをロード
            model = SentenceTransformer(model_name)
            cls._embedding_models[model_name] = model
            logger.info(f"埋め込みモデルを共有ロードしました: {model_name}")
            return model
        except Exception as e:
            logger.error(f"埋め込みモデルロードエラー: {e}")
            return None
        finally:
            cls._is_loading = False


# ───────────────────────────────────────────────────────────────────────────
# Main class
# ───────────────────────────────────────────────────────────────────────────

class RAGRetriever:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode: str = config.get("rag_mode", "dummy")
        self.score_threshold: float = float(config.get("rag_score_threshold", 0.5))
        self.top_k: int = int(config.get("rag_top_k", 5))
        self.debug: bool = bool(config.get("debug", False))
        self.knowledge_base: List[Dict[str, Any]] = config.get("knowledge_base", [])

        # ZIMと埋め込みモデルを初期化
        if self.mode == "zim":
            self._init_zim()
            self._init_embedding()

        if self.debug:
            logger.setLevel(logging.DEBUG)
            if self.mode == "zim":
                logger.debug("RAGモード: ZIM, スコア閾値: %f, 上位K: %d", 
                             self.score_threshold, self.top_k)

    # ───────────────────── init helpers ─────────────────────
    def _init_zim(self) -> None:
        """共有ZIMアーカイブを初期化"""
        if not HAS_LIBZIM:
            logger.warning("libzim absent → dummy mode")
            self.mode = "dummy"; return
            
        path = self.config.get("zim_path")
        if not path or not os.path.exists(path):
            logger.warning("ZIM not found: %s → dummy", path)
            self.mode = "dummy"; return
        
        # 共有ZIMアーカイブを取得
        start_time = None
        if self.debug:
            import time
            start_time = time.time()
            
        self.zim_archive = SharedModels.get_zim_archive(path)
        
        if self.zim_archive is None:
            logger.warning("ZIMアーカイブの取得に失敗しました: %s → dummy", path)
            self.mode = "dummy"; return
            
        if self.debug and start_time:
            import time
            logger.debug("ZIMアーカイブ取得時間: %.3f秒", time.time() - start_time)

    def _init_embedding(self) -> None:
        """共有埋め込みモデルを初期化"""
        if self.mode != "zim":
            return
            
        if SentenceTransformer is None:
            logger.warning("embedding lib absent → no embeddings")
            self.embedding_model = None; return
            
        model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        
        # 共有埋め込みモデルを取得
        start_time = None
        if self.debug:
            import time
            start_time = time.time()
            
        self.embedding_model = SharedModels.get_embedding_model(model_name)
        
        if self.embedding_model is None:
            logger.warning("埋め込みモデルの取得に失敗しました: %s", model_name)
            
        if self.debug and start_time:
            import time
            logger.debug("埋め込みモデル取得時間: %.3f秒", time.time() - start_time)

    # ───────────────────── public ─────────────────────
    def retrieve(self, query: Union[str, Dict[str, Any]]) -> str:
        if self.mode == "zim":
            return self._retrieve_zim(query)
        return self._retrieve_dummy(query)

    # ───────────────────── dummy ─────────────────────
    def _retrieve_dummy(self, query: Union[str, Dict[str, Any]]) -> str:
        if not self.knowledge_base:
            return "ダミーモード: KBが空です"
        emb = query.get("embedding") if isinstance(query, dict) else None
        if emb is None:
            emb = np.random.default_rng(abs(hash(str(query)))%(2**32)).random(384)
        scored = [
            {"text": kb["text"], "s": cosine(emb, kb["embedding"])}
            for kb in self.knowledge_base
        ]
        scored = [s for s in scored if s["s"] >= self.score_threshold]
        scored.sort(key=lambda x: x["s"], reverse=True)
        if not scored:
            return "関連情報が見つかりませんでした"
        return "\n".join(s["text"] for s in scored[: self.top_k])


    # ───────────────────── ZIM ─────────────────────
    def _retrieve_zim(self, query: Union[str, Dict[str, Any]]) -> str:
        qtext = query.get("normalized", "") if isinstance(query, dict) else str(query)
        if not qtext:
            return "検索クエリが空です"
        try:
            srch = Searcher(self.zim_archive)
            paths = srch.search(Query().set_query(qtext)).getResults(0, self.top_k*4)
        except Exception as e:
            logger.exception("search fail: %s", e)
            return f"検索エラー: {e}"
        q_emb = self.embedding_model.encode(qtext) if getattr(self, "embedding_model", None) else None
        hits: List[Dict[str, Any]] = []
        for p in paths:
            try:
                entry = self.zim_archive.get_entry_by_path(p)
                data = entry.get_item().content
                if isinstance(data, memoryview):
                    data = data.tobytes()
                plain = _html_to_text(data.decode("utf-8", "ignore"))
                snippet = textwrap.shorten(plain, 1000, placeholder="…")
                sim = cosine(q_emb, self.embedding_model.encode(snippet)) if q_emb is not None else 1.0
                if sim >= self.score_threshold:
                    hits.append({"title": entry.title, "content": snippet, "s": sim})
            except Exception as e:
                logger.debug("skip: %s", e)
        hits.sort(key=lambda x: x["s"], reverse=True)
        if not hits:
            return "関連情報が見つかりませんでした"
        return "\n\n".join(f"【{h['title']}】\n{h['content']}" for h in hits[: self.top_k])
