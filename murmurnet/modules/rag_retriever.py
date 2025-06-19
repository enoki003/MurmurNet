#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Knowledge Retriever
~~~~~~~~~~~~~~~~~~~~~
Retrieval Augmented Generation モジュール
様々な検索方法に対応（埋め込み、ZIM、データベース）

作者: Yuhi Sonoki
"""
from __future__ import annotations

import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Union, Optional

import numpy as np

logger = logging.getLogger('MurmurNet.RAGRetriever')

# ───────────────────────────────────────────────────────────────────────────
# Optional deps
# ───────────────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────────────────

def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    2つのベクトル間のコサイン類似度を計算
    
    引数:
        v1: 1つ目のベクトル
        v2: 2つ目のベクトル
        
    戻り値:
        コサイン類似度（0〜1の値）
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _html_to_text(html: str) -> str:
    """
    HTMLからプレーンテキストを抽出する軽量変換
    
    引数:
        html: HTML文字列
        
    戻り値:
        変換されたプレーンテキスト
    """
    # スタイルとスクリプトを削除
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", html)
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    # タグを削除
    text = re.sub(r"<[^>]+>", " ", html)
    # 空白を統一
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ───────────────────────────────────────────────────────────────────────────
# Main class
# ───────────────────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    検索拡張生成（RAG）を提供するリトリーバー
    
    責務:
    - 入力クエリに関連する知識の検索
    - 様々な検索方法のサポート（ZIM、埋め込み、知識ベース）
    - 検索結果のフォーマット
    
    属性:
        config: 設定辞書
        mode: 検索モード（'zim', 'dummy'等）
        score_threshold: マッチと見なすスコアの閾値
    """
    def __init__(self, config: Dict[str, Any]):
        """
        RAGリトリーバーの初期化
        
        引数:
            config: 設定辞書
        """
        self.config = config
        self.mode: str = config.get("rag_mode", "dummy")
        self.score_threshold: float = float(config.get("rag_score_threshold", 0.5))
        self.top_k: int = int(config.get("rag_top_k", 5))
        self.debug: bool = bool(config.get("debug", False))
        self.knowledge_base: List[Dict[str, Any]] = config.get("knowledge_base", [])

        if self.debug:
            logger.setLevel(logging.DEBUG)

        if self.mode == "zim":
            self._init_zim()
            self._init_embedding()
        
        logger.info(f"RAGリトリーバーを初期化しました (モード: {self.mode})")

    # ───────────────────── init helpers ─────────────────────    def _init_zim(self) -> None:
        """ZIMファイルを初期化（内部メソッド）"""
        if not HAS_LIBZIM:
            logger.warning("libzim absent → dummy mode")
            self.mode = "dummy"
            return
            
        path = self.config.get("zim_path")
        if not path or not os.path.exists(path):
            logger.warning(f"ZIM not found: {path} → dummy")
            self.mode = "dummy"
            return
            
        try:
            self.zim_archive = Archive(path)
            logger.info(f"Loaded ZIM: {path}")
        except Exception as e:
            logger.error(f"Archive open failed: {e}")
            self.mode = "dummy"

    def _init_embedding(self) -> None:
        """埋め込みモデルを初期化（内部メソッド）"""
        if self.mode != "zim":
            return
            
        if SentenceTransformer is None:
            logger.warning("SentenceTransformer not found → dummy embedding")
            self.mode = "dummy"
            return
            
        try:
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            
            # ローカルキャッシュディレクトリを設定（起動時間短縮のため）
            cache_dir = self.config.get("model_cache_dir")
            if not cache_dir:
                # デフォルトのキャッシュディレクトリを作成
                cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "sentence_transformers")
                os.makedirs(cache_dir, exist_ok=True)
            
            # キャッシュディレクトリを指定してモデルを初期化
            self.transformer = SentenceTransformer(model_name, cache_folder=cache_dir)
            logger.info(f"Loaded embedding model: {model_name} (cache: {cache_dir})")
        except Exception as e:
            logger.error(f"Embedding model init failed: {e}")
            self.mode = "dummy"

    # ───────────────────── retrieval ─────────────────────
    def retrieve(self, query: str) -> str:
        """
        クエリに関連する情報を検索
        
        引数:
            query: 検索クエリ
            
        戻り値:
            検索結果テキスト
        """
        if not query or not isinstance(query, str):
            return "検索クエリが無効です。"
            
        # クエリの前処理
        query = query.strip()
        if len(query) < 2:
            return "検索クエリが短すぎます。"
            
        # モードに応じた検索
        if self.mode == "zim":
            return self._retrieve_zim(query)
        elif self.mode == "knowledge_base":
            return self._retrieve_knowledge_base(query)
        else:
            return self._retrieve_dummy(query)

    def _retrieve_zim(self, query: str) -> str:
        """ZIMファイルから検索（内部メソッド）"""
        try:
            if not hasattr(self, 'zim_archive'):
                return "ZIMアーカイブが利用できません。"
                
            # キーワード検索
            searcher = Searcher(self.zim_archive)
            search_query = Query().set_query(query)
            results = searcher.search(search_query)
            
            if self.debug:
                logger.debug(f"ZIM検索: クエリ='{query}', ヒット={results.get_matches_estimated()}")
                
            if results.get_matches_estimated() == 0:
                return "検索結果が見つかりませんでした。"
                
            # 上位5件を取得
            content_results = []
            for i in range(min(5, results.get_matches_estimated())):
                result = results.get_next()
                if not result:
                    continue
                    
                entry = self.zim_archive.get_entry_by_path(result.get_path())
                if not entry:
                    continue
                    
                # HTMLの場合はテキスト抽出
                content = entry.get_item().content.tobytes().decode('utf-8', errors='ignore')
                if entry.get_mime_type() == "text/html":
                    content = _html_to_text(content)
                    
                # 長すぎる場合は切り詰め
                if len(content) > 500:
                    content = content[:500] + "..."
                    
                title = entry.title
                content_results.append(f"【{title}】 {content}")
                
            return "\n\n".join(content_results)
            
        except Exception as e:
            logger.error(f"ZIM検索エラー: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return f"検索中にエラーが発生しました。"

    def _retrieve_knowledge_base(self, query: str) -> str:
        """ローカル知識ベースから検索（内部メソッド）"""
        if not self.knowledge_base:
            return "知識ベースが空です。"
            
        try:
            if not hasattr(self, 'transformer'):
                # SentenceTransformerがなければ単純なキーワードマッチング
                results = []
                for item in self.knowledge_base:
                    title = item.get('title', '')
                    content = item.get('content', '')
                    # クエリに含まれるキーワードとのマッチングスコア
                    score = 0
                    for word in query.split():
                        if word.lower() in title.lower() or word.lower() in content.lower():
                            score += 1
                    if score > 0:
                        results.append((item, score))
                        
                # スコア順にソート
                results.sort(key=lambda x: x[1], reverse=True)
                
                # 上位5件を取得
                content_results = []
                for item, _ in results[:5]:
                    title = item.get('title', 'No Title')
                    content = item.get('content', '')
                    if len(content) > 500:
                        content = content[:500] + "..."
                    content_results.append(f"【{title}】 {content}")
                    
                if not content_results:
                    return "検索結果が見つかりませんでした。"
                    
                return "\n\n".join(content_results)
                
            else:
                # 埋め込みを使った意味検索
                query_embedding = self.transformer.encode(query)
                
                results = []
                for item in self.knowledge_base:
                    # 埋め込みがあればそれを使用、なければ新たに生成
                    if 'embedding' in item and isinstance(item['embedding'], np.ndarray):
                        item_embedding = item['embedding']
                    else:
                        content = item.get('content', '')
                        item_embedding = self.transformer.encode(content)
                        
                    score = cosine(query_embedding, item_embedding)
                    if score >= self.score_threshold:
                        results.append((item, score))
                        
                # スコア順にソート
                results.sort(key=lambda x: x[1], reverse=True)
                
                # 上位5件を取得
                content_results = []
                for item, score in results[:5]:
                    title = item.get('title', 'No Title')
                    content = item.get('content', '')
                    if len(content) > 500:
                        content = content[:500] + "..."
                    content_results.append(f"【{title}】 {content}")
                    
                if not content_results:
                    return "検索結果が見つかりませんでした。"
                    
                return "\n\n".join(content_results)
                
        except Exception as e:
            logger.error(f"知識ベース検索エラー: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return f"検索中にエラーが発生しました。"
    
    def _retrieve_dummy(self, query: str) -> str:
        """
        基本知識ベース検索（旧ダミー）
        
        外部のZIMファイルが利用できない場合に、内蔵知識ベースを使用して検索を行う
        """
        # 内蔵知識ベースがある場合はそれを使用
        if self.knowledge_base:
            return self._retrieve_knowledge_base(query)
        
        # 内蔵知識ベースもない場合は、基本的な応答を生成
        try:
            # 基本的な知識を含む応答辞書
            basic_knowledge = {
                "AI": "人工知能（AI）は、機械が人間のような知的な行動を示す技術分野です。機械学習、深層学習、自然言語処理などの技術を含みます。",
                "機械学習": "機械学習は、データからパターンを学習し、予測や分類を行うAI技術です。教師あり学習、教師なし学習、強化学習の3つの主要なタイプがあります。",
                "深層学習": "深層学習は、多層のニューラルネットワークを使用した機械学習手法です。画像認識、音声認識、自然言語処理などで大きな成功を収めています。",
                "自然言語処理": "自然言語処理（NLP）は、コンピュータが人間の言語を理解、解釈、生成する技術です。翻訳、要約、質問応答などに応用されます。",
                "ニューラルネットワーク": "ニューラルネットワークは、人間の脳の神経細胞を模倣した計算モデルです。入力層、隠れ層、出力層から構成されます。",
                "データサイエンス": "データサイエンスは、データから価値ある洞察を抽出する学際的な分野です。統計学、機械学習、プログラミングを組み合わせます。",
                "クラウドコンピューティング": "クラウドコンピューティングは、インターネット経由でコンピューティングリソースを提供するサービスです。スケーラビリティとコスト効率が特徴です。",
                "IoT": "IoT（Internet of Things）は、様々な物理的なデバイスがインターネットに接続され、データを収集・交換する技術です。",
                "ブロックチェーン": "ブロックチェーンは、分散型台帳技術で、データの改ざんを防ぎ、透明性と信頼性を提供します。暗号通貨の基盤技術として知られています。",
                "量子コンピューティング": "量子コンピューティングは、量子力学の原理を利用した計算技術です。特定の問題において従来のコンピュータを大幅に上回る性能を発揮する可能性があります。"
            }
            
            # クエリに関連するキーワードを検索
            query_lower = query.lower()
            best_matches = []
            
            for keyword, description in basic_knowledge.items():
                keyword_lower = keyword.lower()
                # キーワードマッチング
                if keyword_lower in query_lower or any(word in query_lower for word in keyword_lower.split()):
                    best_matches.append(f"【{keyword}】 {description}")
            
            # 部分マッチも検索
            if not best_matches:
                for keyword, description in basic_knowledge.items():
                    # より緩い条件での検索
                    query_words = query_lower.split()
                    keyword_words = keyword.lower().split()
                    if any(qword in keyword.lower() or any(kword in qword for kword in keyword_words) for qword in query_words):
                        best_matches.append(f"【{keyword}】 {description}")
            
            if best_matches:
                return "\n\n".join(best_matches[:3])  # 最大3件まで
            else:
                return f"「{query}」に関する直接的な情報は見つかりませんでしたが、AI、機械学習、データサイエンスなどの関連技術について詳しくお答えできます。より具体的な質問をお聞かせください。"
                
        except Exception as e:
            logger.error(f"基本知識検索エラー: {e}")
            return f"申し訳ありませんが、「{query}」に関する情報の検索中にエラーが発生しました。"
