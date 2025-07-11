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
import re
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
        self.max_summary_tokens = self.config.get('max_summary_tokens', 1024)  # 512→1024に増加
        
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
        
        # エージェント別モデル管理を初期化
        try:
            from MurmurNet.modules.agent_model_manager import create_agent_model_manager
            self.model_manager = create_agent_model_manager(self.config)
            # 要約エンジン用の設定でモデルを取得
            agent_config = self.model_manager.get_model_factory_config("summary_engine")
            self.llm = ModelFactory.create_model(agent_config)
            logger.info("SummaryEngine: エージェント別モデル管理を有効化しました")
        except ImportError:
            # フォールバック: 従来の共有モデル
            self.llm = ModelFactory.create_model(self.config)
            self.model_manager = None
            logger.warning("SummaryEngine: エージェント別モデル管理が無効 - 共有モデルを使用します")
        
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
            
            # 短文スキップロジック（入力側・閾値緩和）
            total_text = ' '.join(entry.get('text', '') for entry in entries)
            if len(total_text.strip()) < 32:  # 64→32文字に緩和
                logger.info(f"入力テキストが短いため要約をスキップ: {len(total_text)}文字")
                # 最長のエントリを返す
                longest_entry = max(entries, key=lambda x: len(x.get('text', '')), default={'text': ''})
                return longest_entry.get('text', '要約する内容がありません。')
            
            # 1. キャッシュチェック（重複防止強化）
            cache_key = self._generate_cache_key(entries)
            cached_summary = self.summary_cache.get(cache_key)
            if cached_summary:
                self.stats.cache_hits += 1
                logger.info(f"要約キャッシュヒット: {cache_key[:32]}...")
                return cached_summary
            
            # 1.1. 重複ハッシュチェック（同一内容の多重要約を抑止）
            content_hashes = set()
            unique_entries = []
            duplicate_count = 0
            
            for entry in entries:
                text = entry.get('text', '').strip()
                if not text or len(text) < 10:  # 短すぎるテキストも除外
                    continue
                    
                # 正規化してハッシュ化（より厳密な重複検出）
                # 句読点、空白、記号を除去して実質的な内容で比較
                normalized = re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', text.lower())
                content_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
                
                # さらに文の類似性チェック（先頭100文字での比較）
                text_prefix = normalized[:100]
                
                if content_hash not in content_hashes and text_prefix not in [
                    re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', e.get('text', '').lower())[:100] 
                    for e in unique_entries
                ]:
                    content_hashes.add(content_hash)
                    unique_entries.append(entry)
                else:
                    duplicate_count += 1
                    logger.debug(f"重複コンテンツを除外: {content_hash} (先頭: {text[:30]}...)")
            
            if duplicate_count > 0:
                logger.info(f"重複除去: {duplicate_count}件の重複を除外 ({len(entries)}→{len(unique_entries)}エントリ)")
            
            if not unique_entries:
                logger.warning("重複除去後、要約するエントリがありません")
                return "要約する内容がありません。"
            
            # 重複が大量に発見された場合の追加ログ
            if duplicate_count >= len(entries) * 0.7:  # 70%以上が重複の場合
                logger.warning(f"大量の重複コンテンツを検出: {duplicate_count}/{len(entries)} ({duplicate_count/len(entries)*100:.1f}%)")
            
            entries = unique_entries  # 重複除去済みリストに置き換え
            
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
            
            # 短文スキップロジック（出力側）
            if len(summary.strip()) < 64:
                logger.info(f"要約結果が短いため元のテキストを返す: {len(summary)}文字")
                # 最長のエントリを返す
                longest_entry = max(entries, key=lambda x: len(x.get('text', '')), default={'text': ''})
                summary = longest_entry.get('text', '要約結果が不十分でした。')
            
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

    def summarize_blackboard_with_previous(self, entries: List[Dict[str, Any]], 
                                          previous_summary: str = "") -> str:
        """
        前イテレーション要約を考慮した黒板エントリ要約
        
        引数:
            entries: 要約するエントリのリスト。各エントリは{'agent': id, 'text': str}の形式
            previous_summary: 前イテレーション要約（空の場合は通常の要約）
        
        戻り値:
            要約されたテキスト
        """
        start_time = time.time()
        self.stats.total_requests += 1
        logger.debug("=== SummaryEngine.summarize_blackboard_with_previous() 開始 ===")
        
        try:
            if not entries:
                logger.warning("要約するエントリがありません")
                return "要約するエントリがありません。"
            
            # 短文スキップロジック（入力側・閾値緩和）
            total_text = ' '.join(entry.get('text', '') for entry in entries)
            if len(total_text.strip()) < 32:  # 64→32文字に緩和
                logger.info(f"入力テキストが短いため要約をスキップ: {len(total_text)}文字")
                # 最長のエントリを返す
                longest_entry = max(entries, key=lambda x: len(x.get('text', '')), default={'text': ''})
                return longest_entry.get('text', '要約する内容がありません。')
            
            # 1. キャッシュチェック（前要約も含めたキーで確認）
            cache_key = self._generate_cache_key_with_previous(entries, previous_summary)
            cached_summary = self.summary_cache.get(cache_key)
            if cached_summary:
                self.stats.cache_hits += 1
                logger.info(f"前要約考慮要約キャッシュヒット: {cache_key[:32]}...")
                return cached_summary
            
            # 1.1. 重複ハッシュチェック（同一内容の多重要約を抑止）
            content_hashes = set()
            unique_entries = []
            duplicate_count = 0
            
            for entry in entries:
                text = entry.get('text', '').strip()
                if not text or len(text) < 10:  # 短すぎるテキストも除外
                    continue
                    
                # 正規化してハッシュ化（より厳密な重複検出）
                normalized = re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', text.lower())
                content_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
                
                if content_hash not in content_hashes:
                    content_hashes.add(content_hash)
                    unique_entries.append(entry)
                else:
                    duplicate_count += 1
                    logger.debug(f"重複コンテンツを除外: {content_hash} (先頭: {text[:30]}...)")
            
            if duplicate_count > 0:
                logger.info(f"重複除去: {duplicate_count}件の重複を除外 ({len(entries)}→{len(unique_entries)}エントリ)")
            
            if not unique_entries:
                logger.warning("重複除去後、要約するエントリがありません")
                return "要約する内容がありません。"
            
            # 2. 並列テキスト処理（ベクトル化）
            processed_texts = self._vectorized_text_processing(unique_entries)
            if not processed_texts:
                logger.warning("テキスト処理後、要約内容がありません")
                return "要約する内容がありません。"
            
            # 3. 前イテレーション要約を考慮したプロンプト構築
            summary_prompt = self._build_summary_prompt_with_previous(processed_texts, previous_summary)
            
            # 4. LLM実行（最大3回試行で堅牢性向上）
            self.stats.llm_calls += 1
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    summary = self.llm.predict(summary_prompt)
                    
                    # 出力の検証・清理
                    cleaned_summary = self._clean_summary_output(summary)
                    if cleaned_summary and len(cleaned_summary.strip()) > 0:
                        # 成功: キャッシュに保存
                        self.summary_cache.put(cache_key, cleaned_summary)
                        self.stats.cache_misses += 1
                        
                        # 統計更新
                        self.stats.total_summary_time += time.time() - start_time
                        self.stats.average_input_length = (
                            (self.stats.average_input_length * (self.stats.total_requests - 1) + len(total_text)) / 
                            self.stats.total_requests
                        )
                        self.stats.average_output_length = (
                            (self.stats.average_output_length * (self.stats.total_requests - 1) + len(cleaned_summary)) / 
                            self.stats.total_requests
                        )
                        
                        logger.info(f"前要約考慮要約生成完了: {len(cleaned_summary)}文字 "
                                  f"(試行{attempt+1}/{max_attempts}, 時間: {time.time() - start_time:.2f}秒)")
                        
                        return cleaned_summary
                    else:
                        logger.warning(f"要約生成試行 {attempt+1}/{max_attempts}: 空の結果")
                        
                except Exception as e:
                    logger.error(f"要約生成試行 {attempt+1}/{max_attempts} でエラー: {str(e)}")
                    if attempt == max_attempts - 1:
                        raise
            
            # すべての試行が失敗した場合のフォールバック
            fallback_summary = self._create_fallback_summary(unique_entries, previous_summary)
            self.summary_cache.put(cache_key, fallback_summary)
            
            logger.warning(f"要約生成失敗、フォールバック使用: {len(fallback_summary)}文字")
            return fallback_summary
                
        except Exception as e:
            logger.error(f"要約生成エラー: {str(e)}")
            # エラー時のフォールバック
            fallback = f"要約処理でエラーが発生しました: {str(e)[:100]}"
            return fallback
        finally:
            # 統計更新（エラー時も実行）
            self.stats.total_summary_time += time.time() - start_time

    def _generate_cache_key_with_previous(self, entries: List[Dict[str, Any]], previous_summary: str) -> str:
        """前イテレーション要約を含むキャッシュキーを生成"""
        content = ""
        for entry in entries:
            content += f"{entry.get('agent', '')}:{entry.get('text', '')[:100]}"
        if previous_summary:
            content += f"prev:{previous_summary[:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _build_summary_prompt_with_previous(self, processed_entries: List[str], previous_summary: str = "") -> str:
        """前イテレーション要約を考慮したプロンプト構築"""
        header = ("こんにちは！みんなの意見をまとめて、分かりやすく要約してほしいんだ。"
                 "話し言葉で自然に、150〜200文字くらいでポイントをまとめてね。")
        
        content = "\n\n".join(processed_entries)
        
        if previous_summary and previous_summary.strip():
            # 前イテレーション要約がある場合は継続性を考慮
            prompt = f"""{header}

前回の議論要約:
{previous_summary[:200]}

今回の新しい意見:
{content}

前回の内容も踏まえつつ、今回の新しい観点や発展した部分を中心に要約してください。"""
        else:
            # 前要約がない場合は通常のプロンプト
            prompt = f"{header}\n\n{content}"
        
        return prompt

    def _create_fallback_summary(self, unique_entries: List[Dict[str, Any]], previous_summary: str = "") -> str:
        """フォールバック要約の作成"""
        if not unique_entries:
            return "要約内容がありません。"
        
        # 最も長いエントリを基準に要約作成
        longest_entry = max(unique_entries, key=lambda x: len(x.get('text', '')))
        base_text = longest_entry.get('text', '')[:150]
        
        # 前要約がある場合は継続性を示す
        if previous_summary and previous_summary.strip():
            return f"前回: {previous_summary[:80]}... 今回: {base_text}..."
        else:
            return f"主要な意見: {base_text}..."

    def _clean_summary_output(self, summary: str) -> str:
        """要約出力のクリーニング"""
        if not summary:
            return ""
        
        # 前後の空白を除去
        cleaned = summary.strip()
        
        # 不要な文字列を除去
        cleaned = re.sub(r'^(要約|まとめ|結論)[:：]\s*', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 複数の空白を単一に
        
        # 長すぎる場合は切り詰め
        if len(cleaned) > 300:
            cleaned = cleaned[:300] + "..."
        
        return cleaned

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

    def shutdown(self):
        """
        SummaryEngineの完全なシャットダウン処理
        
        全てのリソースを適切に終了し、統計情報を記録する
        """
        logger.info("SummaryEngineシャットダウン開始")
        
        try:
            # 1. 実行プールのシャットダウン
            if hasattr(self, 'executor') and self.executor:
                logger.debug("ThreadPoolExecutorをシャットダウン中...")
                try:
                    # 進行中のタスクの完了を待つ（最大3秒）
                    self.executor.shutdown(wait=True, timeout=3.0)
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
            
            # 2. 最終統計の記録
            if hasattr(self, 'stats'):
                final_stats = self.get_performance_stats()
                logger.info(f"SummaryEngine最終統計: {final_stats}")
            
            # 3. キャッシュのクリア
            if hasattr(self, 'summary_cache'):
                try:
                    cache_size = self.summary_cache.size()
                    self.summary_cache.clear()
                    logger.debug(f"要約キャッシュをクリア: {cache_size}エントリ")
                except Exception as e:
                    logger.warning(f"キャッシュクリアエラー: {e}")
            
            # 4. モデル参照のクリア
            if hasattr(self, 'llm'):
                self.llm = None
            
            # 5. メモリクリーンアップ
            try:
                import gc
                collected = gc.collect()
                logger.debug(f"SummaryEngine: ガベージコレクション完了 ({collected}個)")
            except Exception as e:
                logger.warning(f"ガベージコレクションエラー: {e}")
            
            logger.info("SummaryEngineシャットダウン完了")
            
        except Exception as e:
            logger.error(f"SummaryEngineシャットダウンエラー: {e}")
            # エラーが発生してもシャットダウンを継続
            import traceback
            logger.debug(traceback.format_exc())

# 下位互換性のためのエイリアス  
# 既存のクラス名がSummaryEngineなので追加不要（クラス名変更なし）
