#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Knowledge Retriever
~~~~~~~~~~~~~~~~~~~~~
Retrieval Augmented Generation モジュール
ZIMファイルと埋め込みベースの検索をサポート

作者: Yuhi Sonoki
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.RAGRetriever')

# 条件付きインポート
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not found")

try:
    from libzim.reader import Archive
    from libzim.search import Searcher, Query
    HAS_LIBZIM = True
except ImportError:
    HAS_LIBZIM = False
    logger.warning("libzim not found")


def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """コサイン類似度を計算"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _html_to_text(html: str) -> str:
    """HTMLからプレーンテキストを抽出"""
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", html)
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RAGRetriever:
    """
    検索拡張生成（RAG）を提供するリトリーバー
    ZIMファイルまたは埋め込みベースの検索をサポート
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """RAGリトリーバーの初期化"""
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()
        
        self.mode: str = self.config_manager.rag.rag_mode
        self.score_threshold: float = self.config_manager.rag.rag_score_threshold
        self.top_k: int = self.config_manager.rag.rag_top_k
        self.debug: bool = self.config_manager.logging.debug
        self.knowledge_base: List[Dict[str, Any]] = []

        if self.debug:
            logger.setLevel(logging.DEBUG)

        if self.mode == "zim":
            self._init_zim()
            self._init_embedding()
        elif self.mode == "embedding":
            self._init_embedding()
        
        logger.info(f"RAGリトリーバーを初期化しました (モード: {self.mode})")

    def _init_zim(self) -> None:
        """ZIMファイルを初期化"""
        if not HAS_LIBZIM:
            logger.error("libzim is required for ZIM mode")
            return
            
        path = self.config_manager.rag.zim_path
        if not path or not os.path.exists(path):
            logger.error(f"ZIM file not found: {path}")
            return
            
        try:
            self.zim_archive = Archive(path)
            logger.info(f"Loaded ZIM: {path}")
        except Exception as e:
            logger.error(f"Failed to load ZIM archive: {e}")

    def _init_embedding(self) -> None:
        """埋め込みモデルを初期化"""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.error("sentence-transformers is required for embedding mode")
            return
            
        try:
            model_name = self.config_manager.rag.embedding_model
            self.transformer = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

    def retrieve(self, query: str) -> str:
        """クエリに関連する情報を検索"""
        if not query or not isinstance(query, str):
            return "検索クエリが無効です。"
            
        query = query.strip()
        if len(query) < 2:
            return "検索クエリが短すぎます。"
            
        if self.mode == "zim":
            return self._retrieve_zim(query)
        elif self.mode == "embedding":
            return self._retrieve_knowledge_base(query)
        else:
            return "サポートされていない検索モードです。"    def _retrieve_zim(self, query: str) -> str:
        """ZIMファイルから検索"""
        try:
            if not hasattr(self, 'zim_archive'):
                return "ZIMアーカイブが利用できません。"
                
            searcher = Searcher(self.zim_archive)
            search_query = Query().set_query(query)
            results = searcher.search(search_query)
            
            # 結果をリストに変換して数を確認
            result_list = []
            try:
                for i in range(self.top_k):
                    result = results.get_next()
                    if result:
                        result_list.append(result)
                    else:
                        break
            except:
                # 結果がない場合
                pass
            
            if self.debug:
                logger.debug(f"ZIM検索: クエリ='{query}', ヒット={len(result_list)}")
                
            if len(result_list) == 0:
                return "検索結果が見つかりませんでした。"
                  content_results = []
            for result in result_list:
                try:
                    entry = self.zim_archive.get_entry_by_path(result.get_path())
                    if not entry:
                        continue
                        
                    content = entry.get_item().content.tobytes().decode('utf-8', errors='ignore')
                    if entry.get_mime_type() == "text/html":
                        content = _html_to_text(content)
                        
                    if len(content) > 500:
                        content = content[:500] + "..."
                        
                    title = entry.title
                    content_results.append(f"【{title}】 {content}")
                except Exception as e:
                    logger.warning(f"エントリ処理エラー: {e}")
                    continue
                
            return "\n\n".join(content_results) if content_results else "検索結果が見つかりませんでした。"
            
        except Exception as e:
            logger.error(f"ZIM検索エラー: {e}")
            return "検索中にエラーが発生しました。"

    def _retrieve_knowledge_base(self, query: str) -> str:
        """ローカル知識ベースから検索"""
        if not self.knowledge_base:
            return "知識ベースが空です。"
            
        try:
            if not hasattr(self, 'transformer'):
                # 単純なキーワードマッチング
                results = []
                for item in self.knowledge_base:
                    title = item.get('title', '')
                    content = item.get('content', '')
                    score = 0
                    for word in query.split():
                        if word.lower() in title.lower() or word.lower() in content.lower():
                            score += 1
                    if score > 0:
                        results.append((item, score))
                        
                results.sort(key=lambda x: x[1], reverse=True)
                
                content_results = []
                for item, _ in results[:self.top_k]:
                    title = item.get('title', 'No Title')
                    content = item.get('content', '')
                    if len(content) > 500:
                        content = content[:500] + "..."
                    content_results.append(f"【{title}】 {content}")
                    
                return "\n\n".join(content_results) if content_results else "検索結果が見つかりませんでした。"
                
            else:
                # 埋め込みベースの検索
                query_embedding = self.transformer.encode(query)
                
                results = []
                for item in self.knowledge_base:
                    if 'embedding' in item and isinstance(item['embedding'], np.ndarray):
                        item_embedding = item['embedding']
                    else:
                        content = item.get('content', '')
                        item_embedding = self.transformer.encode(content)
                        
                    score = cosine(query_embedding, item_embedding)
                    if score >= self.score_threshold:
                        results.append((item, score))
                        
                results.sort(key=lambda x: x[1], reverse=True)
                
                content_results = []
                for item, score in results[:self.top_k]:
                    title = item.get('title', 'No Title')
                    content = item.get('content', '')
                    if len(content) > 500:
                        content = content[:500] + "..."
                    content_results.append(f"【{title}】 {content}")
                    
                return "\n\n".join(content_results) if content_results else "検索結果が見つかりませんでした。"
                
        except Exception as e:
            logger.error(f"知識ベース検索エラー: {e}")
            return "検索中にエラーが発生しました。"
