#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conversation Memory モジュール (CPU/並列/メモリ最適化版)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
会話履歴を管理し、関連する記憶を要約・保持する
過去の対話から文脈情報を活用できるようにする
黒板アーキテクチャと統合して分散エージェント間で記憶を共有

最適化機能:
- ベクトル化検索による高速記憶検索
- 並列情報抽出とキャッシュシステム
- LRU履歴管理とメモリ最適化
- パフォーマンス統計とスレッドセーフ処理

作者: Yuhi Sonoki
"""

import logging
import time
import json
import re
import threading
import hashlib
import gc
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from MurmurNet.modules.model_factory import ModelFactory

logger = logging.getLogger('MurmurNet.ConversationMemory')

@dataclass
class MemoryStats:
    """記憶管理統計情報"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_searches: int = 0
    total_extractions: int = 0
    embedding_cache_hits: int = 0
    parallel_processes: int = 0
    total_memory_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    def get_cache_hit_rate(self) -> float:
        """キャッシュヒット率を取得"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

class VectorizedMemoryCache:
    """ベクトル化対応の記憶キャッシュシステム"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.embedding_cache = OrderedDict()
        self.lock = threading.RLock()
        
    def _generate_key(self, text: str) -> str:
        """テキストからキャッシュキーを生成"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_memory(self, key: str) -> Optional[Any]:
        """記憶キャッシュから取得"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put_memory(self, key: str, value: Any):
        """記憶キャッシュに保存"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """埋め込みキャッシュから取得"""
        with self.lock:
            if key in self.embedding_cache:
                self.embedding_cache.move_to_end(key)
                return self.embedding_cache[key]
            return None
    
    def put_embedding(self, key: str, embedding: np.ndarray):
        """埋め込みキャッシュに保存"""
        with self.lock:
            if key in self.embedding_cache:
                self.embedding_cache.move_to_end(key)
            else:
                if len(self.embedding_cache) >= self.max_size // 2:
                    self.embedding_cache.popitem(last=False)
            self.embedding_cache[key] = embedding
    
    def size(self) -> int:
        """キャッシュのサイズを返す"""
        with self.lock:
            return len(self.cache) + len(self.embedding_cache)
    
    def clear(self):
        """キャッシュをクリア"""
        with self.lock:
            self.cache.clear()
            self.embedding_cache.clear()

class OptimizedConversationMemory:
    """
    最適化された会話記憶管理クラス
    
    責務:
    - 過去の会話を効率的に保存・検索
    - ベクトル化による高速類似検索
    - 並列情報抽出と要約
    - キャッシュによる高速アクセス
    
    属性:
        config: 設定辞書
        conversation_history: 会話履歴のリスト
        key_facts: 抽出された重要な情報を保持する辞書
    """
    
    def __init__(self, config: Dict[str, Any] = None, blackboard=None):
        """
        最適化された会話記憶モジュールの初期化
        """
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        
        # 最適化設定
        self.enable_vectorization = self.config.get('enable_vectorization', True)
        self.enable_parallel_processing = self.config.get('enable_parallel_processing', True)
        self.batch_size = self.config.get('memory_batch_size', 10)
        self.cache_size = self.config.get('memory_cache_size', 1000)
        self.max_workers = self.config.get('memory_max_workers', 3)
        
        # キャッシュと統計
        self.memory_cache = VectorizedMemoryCache(self.cache_size)
        self.stats = MemoryStats()
        
        # 並列処理用スレッドプール
        if self.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.thread_pool = None
        
        # 埋め込みモデル（遅延ロード）
        self._transformer = None
        self.embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # 黒板インスタンス
        self.blackboard = blackboard
        
        # 会話履歴の保存
        self.conversation_history: List[Dict[str, Any]] = []
        self.history_summary: Optional[str] = None
        
        # 重要な情報の保存（汎用的な構造）
        self.key_facts: Dict[str, Any] = {
            "entities": [],     # 抽出された重要な実体
            "topics": [],       # 話題や分野
            "context": {}       # 文脈情報（キーバリューペア）
        }
        
        # 履歴管理設定
        self.max_history_entries = self.config.get('max_history_entries', 10)
        self.max_summary_tokens = self.config.get('max_summary_tokens', 256)
        
        # ModelFactoryからモデルを取得
        self.llm = ModelFactory.create_model(self.config)
        
        # 黒板から会話履歴を読み込む（存在する場合）
        if self.blackboard:
            self._load_from_blackboard()
            
        logger.info(f"最適化された会話記憶モジュール初期化: cache_size={self.cache_size}, workers={self.max_workers}")

    def __del__(self):
        """リソースクリーンアップ"""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    def _load_transformer(self) -> None:
        """埋め込みモデルを遅延ロード"""
        if self._transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                if self.debug:
                    logger.debug(f"埋め込みモデルをロード: {self.embedding_model_name}")
                self._transformer = SentenceTransformer(self.embedding_model_name)
                logger.info(f"埋め込みモデルロード完了: {self.embedding_model_name}")
            except ImportError:
                logger.warning("SentenceTransformersがインストールされていません。ベクトル検索は無効化されます。")
                self._transformer = None
            except Exception as e:
                logger.error(f"埋め込みモデルロードエラー: {e}")
                self._transformer = None

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """テキストの埋め込みを取得（キャッシュ付き）"""
        if not self.enable_vectorization:
            return None
        
        cache_key = self.memory_cache._generate_key(text)
        cached_embedding = self.memory_cache.get_embedding(cache_key)
        if cached_embedding is not None:
            self.stats.embedding_cache_hits += 1
            return cached_embedding
        
        self._load_transformer()
        if self._transformer is None:
            return None
        
        try:
            embedding = self._transformer.encode(text, convert_to_numpy=True)
            self.memory_cache.put_embedding(cache_key, embedding)
            return embedding
        except Exception as e:
            logger.error(f"埋め込み生成エラー: {e}")
            return None

    def _vectorized_similarity_search(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Tuple[Dict, float]]:
        """ベクトル化による類似検索"""
        if not self.enable_vectorization or not candidates:
            return []
        
        query_embedding = self._get_text_embedding(query)
        if query_embedding is None:
            return []
        
        similarities = []
        
        # 並列で埋め込み計算
        if self.enable_parallel_processing and self.thread_pool and len(candidates) > 5:
            def compute_similarity(candidate):
                text = candidate.get('user_input', '') + ' ' + candidate.get('system_response', '')
                candidate_embedding = self._get_text_embedding(text)
                if candidate_embedding is not None:
                    similarity = np.dot(query_embedding, candidate_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                    )
                    return (candidate, float(similarity))
                return None
            
            futures = [self.thread_pool.submit(compute_similarity, candidate) for candidate in candidates]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        similarities.append(result)
                except Exception as e:
                    logger.error(f"並列類似度計算エラー: {e}")
        else:
            # 逐次処理
            for candidate in candidates:
                text = candidate.get('user_input', '') + ' ' + candidate.get('system_response', '')
                candidate_embedding = self._get_text_embedding(text)
                if candidate_embedding is not None:
                    similarity = np.dot(query_embedding, candidate_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                    )
                    similarities.append((candidate, float(similarity)))
          # 類似度でソートして上位を返す
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _parallel_extract_info(self, user_input: str, system_response: str) -> Dict[str, Any]:
        """並列で情報抽出（一時的に無効化、逐次処理を使用）"""
        # 構文エラーのため一時的に並列処理を無効化
        return self._extract_info_sequential(user_input, system_response)

    def _extract_info_sequential(self, user_input: str, system_response: str) -> Dict[str, Any]:
        """逐次情報抽出（フォールバック）"""
        extracted = {'entities': [], 'topics': [], 'context': {}}
        
        # 簡単な正規表現ベースの抽出
        # 名前抽出
        name_patterns = [r'私は([^。、\s]+)です', r'僕は([^。、\s]+)です']
        for pattern in name_patterns:
            matches = re.findall(pattern, user_input)
            extracted['entities'].extend([match.strip() for match in matches])
        
        # 話題抽出
        topic_keywords = ['好き', '趣味', '仕事', '勉強']
        for keyword in topic_keywords:
            if keyword in user_input:
                extracted['topics'].append(keyword)
        
        return extracted

    # 既存のメソッドを引き続き使用
    def _load_from_blackboard(self) -> None:
        """黒板から会話履歴と要約を読み込む（内部メソッド）"""
        if not self.blackboard:
            return
            
        # 会話履歴の読み込み
        conversation_history = self.blackboard.read('conversation_history')
        if conversation_history:
            self.conversation_history = conversation_history
            logger.debug(f"黒板から会話履歴を読み込みました ({len(conversation_history)}件)")
            
        # 要約の読み込み
        history_summary = self.blackboard.read('history_summary')
        if history_summary:
            self.history_summary = history_summary
            logger.debug("黒板から履歴要約を読み込みました")

    def add_conversation(self, user_input: str, system_response: str) -> None:
        """
        新しい会話を追加（最適化版）
        """
        start_time = time.time()
        
        try:
            # 会話エントリを作成
            conversation_entry = {
                'timestamp': time.time(),
                'user_input': user_input,
                'system_response': system_response
            }
            
            # 履歴に追加
            self.conversation_history.append(conversation_entry)
            
            # 履歴制限の管理
            if len(self.conversation_history) > self.max_history_entries:
                self.conversation_history = self.conversation_history[-self.max_history_entries:]
            
            # 並列で重要情報を抽出
            self.stats.total_extractions += 1
            extracted_info = self._parallel_extract_info(user_input, system_response)
            
            # 抽出した情報をkey_factsに統合
            self._merge_extracted_info(extracted_info)
            
            # 要約の更新（バッチ処理）
            if len(self.conversation_history) % 3 == 0:  # 3回に1回更新
                self._update_summary_optimized()
            
            # 黒板に保存
            if self.blackboard:
                self.blackboard.write('conversation_history', self.conversation_history)
                self.blackboard.write('key_facts', self.key_facts)
                if self.history_summary:
                    self.blackboard.write('history_summary', self.history_summary)
            
            process_time = time.time() - start_time
            self.stats.total_memory_time += process_time
            
            logger.debug(f"会話追加完了: {process_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"会話追加エラー: {e}")

    def add_conversation_entry(self, user_input: str, system_response: str) -> None:
        """
        会話エントリを追加（下位互換性のためのエイリアス）
        
        引数:
            user_input: ユーザーの入力
            system_response: システムの応答
        """
        self.add_conversation(user_input, system_response)

    def _merge_extracted_info(self, extracted_info: Dict[str, Any]) -> None:
        """抽出された情報をkey_factsに統合"""
        # エンティティの統合
        for entity in extracted_info.get('entities', []):
            if entity not in self.key_facts['entities']:
                self.key_facts['entities'].append(entity)
        
        # トピックの統合
        for topic in extracted_info.get('topics', []):
            if topic not in self.key_facts['topics']:
                self.key_facts['topics'].append(topic)
          # コンテキストの統合（安全性チェック付き）
        context_data = extracted_info.get('context', {})
        if isinstance(context_data, dict):
            for key, values in context_data.items():
                if key not in self.key_facts['context']:
                    self.key_facts['context'][key] = []
                # valuesがリストの場合の処理
                if isinstance(values, list):
                    for value in values:
                        if value not in self.key_facts['context'][key]:
                            self.key_facts['context'][key].append(value)
                elif values not in self.key_facts['context'][key]:
                    # 単一値の場合
                    self.key_facts['context'][key].append(values)
        else:
            # contextが辞書でない場合のログ出力
            logger.warning(f"context データが辞書ではありません: {type(context_data)}, {context_data}")

    def _update_summary_optimized(self) -> None:
        """最適化された会話履歴要約更新"""
        if not self.conversation_history:
            self.history_summary = None
            return
            
        try:
            # キャッシュチェック
            history_hash = hashlib.md5(
                str([entry['user_input'] + entry['system_response'] 
                     for entry in self.conversation_history[-5:]]).encode()
            ).hexdigest()
            
            cached_summary = self.memory_cache.get_memory(f"summary_{history_hash}")
            if cached_summary:
                self.stats.cache_hits += 1
                self.history_summary = cached_summary
                return
            
            self.stats.cache_misses += 1
            
            # 直近5つの会話を使用（並列処理対応）
            recent_history = self.conversation_history[-5:]
            
            if self.enable_parallel_processing and self.thread_pool and len(recent_history) > 2:
                # 並列でテキスト構築
                def build_history_text():
                    history_text = ""
                    for i, entry in enumerate(recent_history):
                        user = entry.get('user_input', '')[:100]
                        system = entry.get('system_response', '')[:200]
                        history_text += f"会話{i+1}:\nユーザー: {user}\nシステム: {system}\n\n"
                    return history_text
                
                future = self.thread_pool.submit(build_history_text)
                history_text = future.result(timeout=1.0)
            else:
                # 逐次処理
                history_text = ""
                for i, entry in enumerate(recent_history):
                    user = entry.get('user_input', '')[:100]
                    system = entry.get('system_response', '')[:200]
                    history_text += f"会話{i+1}:\nユーザー: {user}\nシステム: {system}\n\n"
            
            # 要約プロンプト
            prompt = (
                "以下の会話履歴を要約してください。ユーザーの名前、興味、趣味、個人情報などの重要な詳細を含めてください。"
                "200文字以内で簡潔にまとめてください。\n\n" + history_text
            )
            
            # LLM呼び出し
            resp = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_tokens,
                temperature=0.3,
                top_p=0.9
            )
            
            if isinstance(resp, dict):
                summary = resp['choices'][0]['message']['content'].strip()
            else:
                summary = resp.choices[0].message.content.strip()
                
            # 要約を更新・キャッシュ
            self.history_summary = summary
            self.memory_cache.put_memory(f"summary_{history_hash}", summary)
            
            logger.debug(f"会話履歴要約を更新しました ({len(summary)}文字)")
            
        except Exception as e:
            logger.error(f"履歴要約エラー: {e}")

    def search_relevant_memory(self, query: str) -> Optional[str]:
        """
        関連する記憶を検索（最適化版 - ベクトル検索対応）
        """
        start_time = time.time()
        self.stats.total_searches += 1
        
        try:
            if not query.strip():
                return None
            
            # キャッシュチェック
            cache_key = self.memory_cache._generate_key(query)
            cached_result = self.memory_cache.get_memory(f"search_{cache_key}")
            if cached_result:
                self.stats.cache_hits += 1
                return cached_result
            
            self.stats.cache_misses += 1
            
            # ベクトル化による類似検索
            if self.enable_vectorization and self.conversation_history:
                similar_conversations = self._vectorized_similarity_search(
                    query, self.conversation_history, top_k=3
                )
                
                if similar_conversations:
                    # 最も類似度の高い会話を返す
                    best_match, similarity = similar_conversations[0]
                    if similarity > 0.7:  # 類似度閾値
                        result = f"以前、あなたが「{best_match['user_input'][:50]}...」と言ったとき、私は「{best_match['system_response'][:100]}...」とお答えしました。"
                        self.memory_cache.put_memory(f"search_{cache_key}", result)
                        
                        search_time = time.time() - start_time
                        logger.debug(f"ベクトル検索完了: {search_time:.3f}秒, 類似度={similarity:.3f}")
                        
                        return result
            
            # フォールバック: キーワードベース検索
            result = self._keyword_based_search(query)
            if result:
                self.memory_cache.put_memory(f"search_{cache_key}", result)
            
            search_time = time.time() - start_time
            self.stats.total_memory_time += search_time
            
            return result
            
        except Exception as e:
            logger.error(f"記憶検索エラー: {e}")
            return None

    def _keyword_based_search(self, query: str) -> Optional[str]:
        """キーワードベースの記憶検索（フォールバック）"""
        query_lower = query.lower()
        keywords = [word for word in query_lower.split() if len(word) > 1]
        
        if not keywords:
            return None
        
        # 特定パターンの検索
        name_patterns = ["名前", "なまえ"]
        hobby_patterns = ["好き", "趣味", "しゅみ"]
        
        # 名前関連の質問
        for pattern in name_patterns:
            if pattern in query:
                if self.key_facts["entities"]:
                    names = "、".join(self.key_facts["entities"][:2])
                    return f"あなたのお名前は{names}でしたね。"
        
        # 趣味・好み関連の質問
        for pattern in hobby_patterns:
            if pattern in query:
                if "好きなもの" in self.key_facts["context"] and self.key_facts["context"]["好きなもの"]:
                    likes = "、".join(self.key_facts["context"]["好きなもの"][:3])
                    return f"あなたは{likes}が好きだとおっしゃっていましたね。"
        
        # 履歴要約の活用
        if "覚え" in query or "おぼえ" in query:
            if self.history_summary:
                return f"はい、覚えています。{self.history_summary}"
        
        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        try:
            import psutil
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        return {
            'cache_hit_rate': self.stats.get_cache_hit_rate(),
            'total_searches': self.stats.total_searches,
            'total_extractions': self.stats.total_extractions,
            'embedding_cache_hits': self.stats.embedding_cache_hits,
            'parallel_processes': self.stats.parallel_processes,
            'memory_usage_mb': self.stats.memory_usage_mb,
            'avg_memory_time': self.stats.total_memory_time / max(1, self.stats.total_searches),
            'conversation_count': len(self.conversation_history),
            'key_facts_count': len(self.key_facts['entities']) + len(self.key_facts['topics'])
        }

    def clear_memory(self) -> None:
        """会話履歴をリセット"""
        self.conversation_history = []
        self.history_summary = None
        self.key_facts = {
            "entities": [],
            "topics": [],
            "context": {}
        }
        
        # 黒板があれば情報をクリア
        if self.blackboard:
            self.blackboard.write('conversation_history', [])
            self.blackboard.write('history_summary', None)
            self.blackboard.write('key_facts', self.key_facts)
        
        # キャッシュクリア
        self.memory_cache.cache.clear()
        self.memory_cache.embedding_cache.clear()
        
        # 統計リセット
        self.stats = MemoryStats()
        
        logger.info("会話記憶をクリアしました")

    def get_conversation_context(self) -> Dict[str, Any]:
        """
        会話コンテキストを取得（最適化版）
        
        戻り値:
            会話履歴の要約と重要な情報を含む辞書
        """
        try:
            context = {
                'history_summary': self.history_summary or "",
                'key_facts': self.key_facts.copy(),
                'recent_conversations': [],
                'conversation_count': len(self.conversation_history)
            }
            
            # 直近の会話（最大3件）を追加
            if self.conversation_history:
                recent_count = min(3, len(self.conversation_history))
                context['recent_conversations'] = [
                    {
                        'user_input': conv.get('user_input', '')[:100],
                        'system_response': conv.get('system_response', '')[:150],
                        'timestamp': conv.get('timestamp', 0)
                    }
                    for conv in self.conversation_history[-recent_count:]
                ]
            
            # パフォーマンス統計も含める
            context['performance_stats'] = self.get_performance_stats()
            
            return context
            
        except Exception as e:
            logger.error(f"会話コンテキスト取得エラー: {e}")
            return {
                'history_summary': "",
                'key_facts': {'entities': [], 'topics': [], 'context': {}},
                'recent_conversations': [],
                'conversation_count': 0,
                'performance_stats': {}
            }

    def shutdown(self):
        """
        ConversationMemoryの完全なシャットダウン処理
        
        全てのリソースを適切に終了し、会話データを保存する
        """
        logger.info("ConversationMemoryシャットダウン開始")
        
        try:
            # 1. 実行プールのシャットダウン
            if hasattr(self, 'executor') and self.executor:
                logger.debug("ThreadPoolExecutorをシャットダウン中...")
                try:
                    # 進行中のタスクの完了を待つ
                    self.executor.shutdown(wait=True)
                    logger.debug("ThreadPoolExecutorシャットダウン完了")
                except Exception as e:
                    logger.warning(f"ThreadPoolExecutor強制終了: {e}")
                    # 強制終了
                    try:
                        self.executor.shutdown(wait=False)
                    except:
                        pass
                finally:
                    self.executor = None
            
            # 2. 会話データの保存（オプション）
            if hasattr(self, 'conversation_history') and self.conversation_history:
                try:
                    # 会話履歴の統計情報を記録
                    conv_count = len(self.conversation_history)
                    logger.info(f"会話履歴を保持: {conv_count}件")
                    
                    # 必要に応じてファイル保存（実装例）
                    # self._save_conversation_history()
                    
                except Exception as e:
                    logger.warning(f"会話データ保存エラー: {e}")
            
            # 3. 最終統計の記録
            if hasattr(self, 'stats'):
                final_stats = self.get_performance_stats()
                logger.info(f"ConversationMemory最終統計: {final_stats}")
            
            # 4. キャッシュのクリア
            if hasattr(self, 'memory_cache'):
                try:
                    cache_size = self.memory_cache.size()
                    self.memory_cache.clear()
                    logger.debug(f"記憶キャッシュをクリア: {cache_size}エントリ")
                except Exception as e:
                    logger.warning(f"記憶キャッシュクリアエラー: {e}")
            
            if hasattr(self, 'embedding_cache'):
                try:
                    embedding_count = len(self.embedding_cache)
                    self.embedding_cache.clear()
                    logger.debug(f"埋め込みキャッシュをクリア: {embedding_count}エントリ")
                except Exception as e:
                    logger.warning(f"埋め込みキャッシュクリアエラー: {e}")
            
            # 5. モデル参照のクリア
            if hasattr(self, 'llm'):
                self.llm = None
            
            # 6. 会話データのクリア（完全シャットダウンの場合）
            try:
                self.conversation_history.clear()
                self.history_summary = None
                if hasattr(self, 'key_facts'):
                    self.key_facts = {
                        "entities": [],
                        "topics": [],
                        "context": {}
                    }
                logger.debug("会話記憶データをクリア")
            except Exception as e:
                logger.warning(f"会話データクリアエラー: {e}")
            
            # 7. 黒板参照のクリア
            if hasattr(self, 'blackboard'):
                self.blackboard = None
            
            # 8. メモリクリーンアップ
            try:
                import gc
                collected = gc.collect()
                logger.debug(f"ConversationMemory: ガベージコレクション完了 ({collected}個)")
            except Exception as e:
                logger.warning(f"ガベージコレクションエラー: {e}")
            
            logger.info("ConversationMemoryシャットダウン完了")
            
        except Exception as e:
            logger.error(f"ConversationMemoryシャットダウンエラー: {e}")
            # エラーが発生してもシャットダウンを継続
            import traceback
            logger.debug(traceback.format_exc())

# 下位互換性のためのエイリアス
ConversationMemory = OptimizedConversationMemory
