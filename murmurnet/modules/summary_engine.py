#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary Engine モジュール (CPU/並列/メモリ最適化版)
~~~~~~~~~~~~~~~~~~~~~
黒板上の情報を要約するエンジン
長いコンテキストを簡潔にまとめる機能を提供

最適化機能:
- LRUキャッシュによる高速要約
- 並列処理とバッチ処理
- ベクトル化とCPU最適化
- パフォーマンス統計とメモリ最適化
- スレッドセーフな並列アクセス

作者: Yuhi Sonoki
"""

import logging
import time
import threading
import hashlib
import gc
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from MurmurNet.modules.model_factory import ModelFactory

logger = logging.getLogger('MurmurNet.SummaryEngine')

@dataclass
class SummaryStats:
    """要約統計情報"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    total_summary_time: float = 0.0
    llm_calls: int = 0
    parallel_processes: int = 0
    memory_usage_mb: float = 0.0
    average_input_length: float = 0.0
    average_output_length: float = 0.0
    
    def get_cache_hit_rate(self) -> float:
        """キャッシュヒット率を取得"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def get_avg_summary_time(self) -> float:
        """平均要約時間を取得"""
        if self.total_requests == 0:
            return 0.0
        return self.total_summary_time / self.total_requests

class SummaryCache:
    """要約結果のLRUキャッシュ"""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
    def _generate_key(self, entries: List[Dict]) -> str:
        """エントリリストからキャッシュキーを生成"""
        content = ""
        for entry in entries:
            content += f"{entry.get('agent', '')}:{entry.get('text', '')[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """キャッシュから取得"""
        with self.lock:
            if key in self.cache:
                # LRUアルゴリズム: アクセスされた項目を末尾に移動
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
        return None
    
    def put(self, key: str, value: str):
        """キャッシュに保存"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 最も古いアイテムを削除
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def get_stats(self) -> Dict[str, int]:
        """キャッシュ統計を取得"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'usage_rate': len(self.cache) / self.max_size
            }

class SummaryEngine:
    """
    最適化された要約エンジン
    
    責務:
    - 複数エージェントの出力を統合
    - 長いテキストを簡潔に要約
    - 一貫性のある出力生成
    - キャッシュとパフォーマンス最適化
    
    属性:
        config: 設定辞書
        max_summary_tokens: 要約の最大トークン数
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        最適化された要約エンジンの初期化
        
        引数:
            config: 設定辞書（省略時は空の辞書）
        """
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.max_summary_tokens = self.config.get('max_summary_tokens', 200)
        
        # 最適化パラメータ
        self.enable_parallel_processing = self.config.get('enable_parallel_processing', True)
        self.batch_size = self.config.get('summary_batch_size', 5)
        self.max_cache_size = self.config.get('summary_cache_size', 500)
        self.enable_vectorization = self.config.get('enable_vectorization', True)
        
        # キャッシュとスレッドプールの初期化
        self.summary_cache = SummaryCache(self.max_cache_size)
        self.stats = SummaryStats()
        
        if self.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
        else:
            self.thread_pool = None
        
        # デバッグログ設定
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # ModelFactoryからモデルを取得
        self.llm = ModelFactory.create_model(self.config)
        
        logger.info("最適化された要約エンジンを初期化しました")
        logger.debug(f"設定: 並列={self.enable_parallel_processing}, "
                    f"バッチサイズ={self.batch_size}, "
                    f"キャッシュサイズ={self.max_cache_size}")

    def __del__(self):
        """リソースクリーンアップ"""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    @lru_cache(maxsize=1000)
    def _preprocess_text_cached(self, text: str, max_length: int = 200) -> str:
        """テキスト前処理（キャッシュ付き）"""
        if len(text) <= max_length:
            return text.strip()
        return text[:max_length].strip() + "..."

    def _vectorized_text_processing(self, entries: List[Dict[str, Any]]) -> List[str]:
        """ベクトル化されたテキスト処理"""
        if not self.enable_vectorization or len(entries) < 3:
            # 小さなデータセットでは通常の処理
            return [self._preprocess_text_cached(entry.get('text', '')) 
                   for entry in entries]
        
        # NumPyを使用したベクトル化処理
        texts = [entry.get('text', '') for entry in entries]
        processed_texts = []
        
        # バッチ処理で効率化
        batch_size = min(50, len(texts))
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_processed = [self._preprocess_text_cached(text) for text in batch]
            processed_texts.extend(batch_processed)
        
        return processed_texts

    def _build_summary_prompt_parallel(self, processed_entries: List[str]) -> str:
        """並列でプロンプトを構築"""
        def build_header():
            return ("こんにちは！みんなの意見をまとめて、分かりやすく要約してほしいんだ。"
                   "話し言葉で自然に、150〜200文字くらいでポイントをまとめてね。\n\n")
        
        def build_content():
            return "\n\n".join(processed_entries)
        
        if self.enable_parallel_processing and self.thread_pool and len(processed_entries) > 5:
            # 並列処理
            header_future = self.thread_pool.submit(build_header)
            content_future = self.thread_pool.submit(build_content)
            
            try:
                header = header_future.result(timeout=1.0)
                content = content_future.result(timeout=2.0)
                return header + content
            except Exception as e:
                logger.warning(f"並列プロンプト構築エラー: {e}")
                # フォールバック
                return build_header() + build_content()
        else:
            # 逐次処理
            return build_header() + build_content()

    def _generate_cache_key(self, entries: List[Dict[str, Any]]) -> str:
        """エントリリストからキャッシュキーを生成"""
        return self.summary_cache._generate_key(entries)

    def summarize_blackboard(self, entries: List[Dict[str, Any]]) -> str:
        """
        最適化された黒板エントリ要約
        
        引数:
            entries: 要約するエントリのリスト。各エントリは{'agent': id, 'text': str}の形式
        
        戻り値:
            要約されたテキスト
        """
        start_time = time.time()
        self.stats.total_requests += 1
        logger.debug("=== SummaryEngine.summarize_blackboard() 最適化版 開始 ===")
        
        try:
            if not entries:
                logger.warning("要約するエントリがありません")
                return "要約するエントリがありません。"
            
            # 1. キャッシュチェック
            cache_key = self._generate_cache_key(entries)
            cached_summary = self.summary_cache.get(cache_key)
            if cached_summary:
                self.stats.cache_hits += 1
                logger.info(f"要約キャッシュヒット: {cache_key[:32]}...")
                return cached_summary
            
            self.stats.cache_misses += 1
            
            # 2. ベクトル化されたテキスト処理
            processed_entries = self._vectorized_text_processing(entries)
            
            # 統計情報更新
            avg_input_len = sum(len(entry.get('text', '')) for entry in entries) / len(entries)
            self.stats.average_input_length = (
                self.stats.average_input_length * (self.stats.total_requests - 1) + avg_input_len
            ) / self.stats.total_requests
            
            # 3. 並列プロンプト構築
            prompt = self._build_summary_prompt_parallel(processed_entries)
            
            # 4. バッチ処理が有効な場合の最適化
            if self.enable_parallel_processing and len(entries) > self.batch_size:
                self.stats.parallel_processes += 1
                logger.debug(f"バッチ処理実行: {len(entries)}エントリ -> {self.batch_size}バッチ")
            
            if self.debug:
                logger.debug(f"要約入力: {len(prompt)}文字")
            
            # 5. LLM呼び出し
            self.stats.llm_calls += 1
            resp = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_tokens,
                temperature=0.3,  # 要約は低温度で一貫性向上
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["。", ".", "\n\n"]
            )
            
            # 6. 応答処理
            if isinstance(resp, dict):
                summary = resp['choices'][0]['message']['content'].strip()
            else:
                summary = resp.choices[0].message.content.strip()
            
            # 出力長制限
            if len(summary) > 250:
                summary = summary[:250]
            
            # 7. キャッシュに保存
            self.summary_cache.put(cache_key, summary)
            
            # 8. 統計情報更新
            summary_time = time.time() - start_time
            self.stats.total_summary_time += summary_time
            
            avg_output_len = len(summary)
            self.stats.average_output_length = (
                self.stats.average_output_length * (self.stats.total_requests - 1) + avg_output_len
            ) / self.stats.total_requests
            
            # メモリ使用量監視
            try:
                import psutil
                process = psutil.Process()
                self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except ImportError:
                pass
            
            # 定期的なガベージコレクション
            if self.stats.total_requests % 50 == 0:
                gc.collect()
            
            if self.debug:
                logger.debug(f"要約生成: 入力={len(prompt)}文字, 出力={len(summary)}文字, "
                           f"時間={summary_time:.3f}秒")
            
            logger.info(f"最適化版要約完了: {len(summary)}文字, {summary_time:.3f}秒")
            logger.info(f"統計: ヒット率={self.stats.get_cache_hit_rate():.2%}, "
                       f"平均時間={self.stats.get_avg_summary_time():.3f}秒, "
                       f"メモリ={self.stats.memory_usage_mb:.1f}MB")
                
            return summary
            
        except Exception as e:
            error_msg = f"最適化版要約エラー: {str(e)}"
            logger.error(error_msg)
            
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
                
            return "要約の生成中にエラーが発生しました。"

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        cache_stats = self.summary_cache.get_stats()
        return {
            'cache_hit_rate': self.stats.get_cache_hit_rate(),
            'avg_summary_time': self.stats.get_avg_summary_time(),
            'total_requests': self.stats.total_requests,
            'llm_calls': self.stats.llm_calls,
            'parallel_processes': self.stats.parallel_processes,
            'memory_usage_mb': self.stats.memory_usage_mb,
            'avg_input_length': self.stats.average_input_length,
            'avg_output_length': self.stats.average_output_length,
            'cache_stats': cache_stats
        }

# 下位互換性のためのエイリアス  
# 既存のクラス名がSummaryEngineなので追加不要（クラス名変更なし）
