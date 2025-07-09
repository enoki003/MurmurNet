#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input Reception モジュール（CPU最適化版）
~~~~~~~~~~~~~~~~~~~~~~~
ユーザー入力の前処理を担当
テキストの正規化、トークン化、埋め込み生成など
CPU最適化: バッチ処理、並列処理、キャッシュシステム

作者: Yuhi Sonoki
"""

import logging
import numpy as np
import time
import hashlib
import signal
import sys
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger('MurmurNet.InputReception')

class InputReception:
    """
    入力テキストの前処理を行うクラス（CPU最適化版）
    
    責務:
    - テキスト正規化（並列化対応）
    - トークン化（ベクトル化処理）
    - 埋め込み生成（バッチ処理、キャッシュ）
    - CPU最適化: 並列処理、キャッシュ、バッチ処理
    
    属性:
        config: 設定辞書
        use_embeddings: 埋め込みを使用するかどうか
        stats: パフォーマンス統計
        cache: 埋め込みキャッシュ
    """
    def __init__(self, config: Dict[str, Any]):
        """
        入力処理モジュールの初期化（CPU最適化版）
        
        引数:
            config: 設定辞書
        """
        self.config = config
        self.debug = config.get('debug', False)
        self.use_embeddings = config.get('use_embeddings', True)
        self._transformer = None  # 遅延ロード
        
        # CPU最適化設定
        self.enable_vectorization = config.get('enable_vectorization', True)
        self.enable_caching = config.get('cpu_optimization', {}).get('enable_caching', True)
        self.batch_size = config.get('batch_size', 32)
        self.max_workers = max(1, config.get('max_workers', 4))  # 最小値1を保証
        self.embedding_cache_size = config.get('embedding_cache_size', 1000)
        
        # 並列処理用スレッドプール
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # パフォーマンス統計
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_processes': 0,
            'total_time': 0.0,
            'embedding_time': 0.0,
            'tokenization_time': 0.0
        }
          # 埋め込みキャッシュ（LRU）
        if self.enable_caching:
            self.embedding_cache = {}
            self.cache_order = []  # 簡易LRU実装
          # シャットダウン状態の管理
        self._is_shutdown = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_event = threading.Event()
        
        # シグナルハンドラーの設定
        self._setup_signal_handlers()
        
        # atexitハンドラーの登録（プログラム終了時の確実なクリーンアップ）
        atexit.register(self._emergency_shutdown)
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"InputReception初期化完了 - バッチ処理: {self.batch_size}, ワーカー: {self.max_workers}")
    
    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了時にシャットダウン"""
        self.shutdown()
    def shutdown(self) -> None:
        """
        リソースのクリーンアップとシャットダウン処理
        
        - ThreadPoolExecutorの終了
        - キャッシュのクリア
        - モデルリソースの解放
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            
            try:
                if self.debug:
                    logger.debug("InputReceptionのシャットダウンを開始...")
                  # ThreadPoolExecutorの強制終了
                if hasattr(self, 'thread_pool') and self.thread_pool:
                    try:
                        # 実行中のタスクをキャンセル
                        self.thread_pool.shutdown(wait=False)
                        
                        # プライベートメンバーへの直接アクセスで強制終了
                        if hasattr(self.thread_pool, '_shutdown'):
                            self.thread_pool._shutdown = True
                        
                        # 全スレッドを強制終了
                        if hasattr(self.thread_pool, '_threads'):
                            for thread in list(self.thread_pool._threads):
                                if thread.is_alive():
                                    try:
                                        thread.join(timeout=0.1)  # 短いタイムアウト
                                    except:
                                        pass
                        
                        # 完全に削除
                        del self.thread_pool
                        self.thread_pool = None
                        
                        if self.debug:
                            logger.debug("ThreadPoolExecutorを強制終了しました")
                    except Exception as e:
                        logger.warning(f"ThreadPoolExecutor終了エラー: {e}")
                
                # キャッシュのクリア
                if self.enable_caching:
                    if hasattr(self, 'embedding_cache'):
                        self.embedding_cache.clear()
                    if hasattr(self, 'cache_order'):
                        self.cache_order.clear()
                    if self.debug:
                        logger.debug("埋め込みキャッシュをクリアしました")
                
                # モデルリソースの解放
                if self._transformer is not None:
                    # SentenceTransformerモデルのメモリ解放
                    try:
                        if hasattr(self._transformer, 'cpu'):
                            self._transformer.cpu()
                        del self._transformer
                        self._transformer = None
                        if self.debug:
                            logger.debug("埋め込みモデルを解放しました")
                    except Exception as e:
                        logger.warning(f"モデル解放時にエラー: {e}")
                
                # 統計情報の最終出力
                if self.debug:
                    self._log_final_stats()
                
                self._is_shutdown = True
                logger.info("InputReceptionのシャットダウンが完了しました")
            except Exception as e:
                logger.error(f"シャットダウン処理中にエラー: {e}")
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
            finally:
                self._is_shutdown = True
    
    def _log_final_stats(self) -> None:
        """最終統計情報をログ出力"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        logger.debug("=== 最終統計情報 ===")
        logger.debug(f"総処理数: {self.stats['total_processed']}")
        logger.debug(f"バッチ処理数: {self.stats['batch_processes']}")
        logger.debug(f"キャッシュヒット率: {cache_hit_rate:.1f}% ({self.stats['cache_hits']}/{total_requests})")
        logger.debug(f"総処理時間: {self.stats['total_time']:.3f}秒")
        logger.debug(f"埋め込み生成時間: {self.stats['embedding_time']:.3f}秒")
        logger.debug(f"トークン化時間: {self.stats['tokenization_time']:.3f}秒")
    
    def _check_shutdown(self) -> None:
        """シャットダウン状態をチェックし、必要に応じて例外を発生"""
        if self._is_shutdown or self._shutdown_requested:
            raise RuntimeError("InputReceptionは既にシャットダウンされています")
        
    def _get_cache_key(self, text: str) -> str:
        """テキストのキャッシュキーを生成"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _update_cache(self, key: str, value: np.ndarray) -> None:
        """LRUキャッシュの更新"""
        if not self.enable_caching:
            return
            
        if key in self.embedding_cache:
            # 既存のキーを順序リストから削除
            self.cache_order.remove(key)
        elif len(self.embedding_cache) >= self.embedding_cache_size:
            # キャッシュサイズ超過時は最古のエントリを削除
            oldest_key = self.cache_order.pop(0)
            del self.embedding_cache[oldest_key]
        
        # 新しいエントリを追加
        self.embedding_cache[key] = value
        self.cache_order.append(key)
    
    def _get_from_cache(self, key: str) -> Optional[np.ndarray]:
        """キャッシュから埋め込みを取得"""
        if not self.enable_caching or key not in self.embedding_cache:
            self.stats['cache_misses'] += 1
            return None        # LRU順序を更新
        self.cache_order.remove(key)
        self.cache_order.append(key)
        self.stats['cache_hits'] += 1
        return self.embedding_cache[key]
    
    def _load_transformer(self) -> None:
        """
        埋め込みモデルを必要なときだけロード（内部メソッド）
        
        モデルロードはリソース消費が大きいため、実際に必要になるまで
        遅延ロードする
        """
        if self._transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                
                # まずローカルファイルのみで試行（オフライン対応）
                cache_folder = self.config.get('embedding_cache_folder', self.config.get('cache_folder', './models/st_cache'))
                local_files_only = self.config.get('local_files_only', True)
                
                try:
                    if self.debug:
                        logger.debug(f"埋め込みモデルをローカルからロード: {model_name} (local_files_only={local_files_only})")
                    
                    # configの設定に従ってモデルをロード
                    self._transformer = SentenceTransformer(
                        model_name, 
                        local_files_only=local_files_only,
                        cache_folder=cache_folder
                    )
                    logger.info(f"✓ 埋め込みモデルをローカルからロード: {model_name}")
                    
                except Exception as local_error:
                    logger.warning(f"ローカルモデルロード失敗: {local_error}")
                    
                    # オンライン接続を試行
                    if not local_files_only:
                        try:
                            logger.info("オンラインからモデルをダウンロード中...")
                            self._transformer = SentenceTransformer(
                                model_name,
                                cache_folder=cache_folder
                            )
                            logger.info(f"✓ 埋め込みモデルをオンラインからロード: {model_name}")
                        except Exception as online_error:
                            logger.error(f"オンラインモデルロード失敗: {online_error}")
                            raise online_error
                    else:
                        logger.warning("force_offline=True のため、オンライン取得をスキップ")
                        raise local_error
                        
            except ImportError:
                logger.warning("SentenceTransformersがインストールされていません。ダミー埋め込みを使用します。")
                self._transformer = None
                
            except Exception as e:
                logger.error(f"埋め込みモデルロードエラー: {e}")
                logger.info("ダミー埋め込みにフォールバック")
                self._transformer = None

    def _parallel_tokenize(self, texts: List[str]) -> List[List[str]]:
        """並列トークン化処理"""
        if not self.enable_vectorization or len(texts) == 1:
            return [text.split() for text in texts]
        
        def tokenize_batch(text_batch):
            return [text.split() for text in text_batch]
        
        # バッチサイズに基づいて並列処理
        batch_size = max(1, len(texts) // self.max_workers)
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(tokenize_batch, batches))
        
        # 結果を統合
        return [token for batch_result in results for token in batch_result]
    
    def _batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """バッチ埋め込み生成（キャッシュ対応）"""
        if not self.use_embeddings:
            return [np.zeros(384) for _ in texts]
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # キャッシュから取得
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = self._get_from_cache(cache_key)
            
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 未キャッシュのテキストを一括処理
        if uncached_texts:
            try:
                self._load_transformer()
                if self._transformer:
                    # バッチ処理で埋め込み生成
                    batch_embeddings = self._transformer.encode(
                        uncached_texts, 
                        batch_size=self.batch_size,
                        show_progress_bar=False
                    )
                    
                    # キャッシュに保存
                    for text, embedding in zip(uncached_texts, batch_embeddings):
                        cache_key = self._get_cache_key(text)
                        self._update_cache(cache_key, embedding)
                    
                    # 結果に追加
                    for i, embedding in zip(uncached_indices, batch_embeddings):
                        embeddings.append((i, embedding))
                else:
                    # ダミー埋め込み（オフライン環境対応）
                    logger.info("SentenceTransformerが利用できません。ダミー埋め込みを使用")
                    for i in uncached_indices:
                        # テキストベースの擬似埋め込み（ハッシュベース）
                        dummy_embedding = self._create_dummy_embedding(uncached_texts[uncached_indices.index(i)])
                        embeddings.append((i, dummy_embedding))
            except Exception as e:
                logger.error(f"バッチ埋め込み生成エラー: {e}")
                logger.info("ダミー埋め込みにフォールバック")
                for i in uncached_indices:
                    # エラー時もダミー埋め込みを生成
                    dummy_embedding = self._create_dummy_embedding(uncached_texts[uncached_indices.index(i)])
                    embeddings.append((i, dummy_embedding))
                    embeddings.append((i, np.zeros(384)))
        
        # インデックス順に並び替え
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
    
    def process_batch(self, input_texts: List[str]) -> List[Dict[str, Any]]:
        """
        複数の入力テキストをバッチ処理（CPU最適化版）
        
        引数:
            input_texts: 処理する入力テキストのリスト
            
        戻り値:
            処理結果のリスト
        """
        self._check_shutdown()  # シャットダウン状態をチェック
        
        start_time = time.time()
        self.stats['batch_processes'] += 1
        
        try:
            # 入力検証とフィルタリング
            valid_texts = []
            for text in input_texts:
                if text and isinstance(text, str):
                    if len(text) > 1000:
                        text = text[:1000]
                    valid_texts.append(text.strip())
                else:
                    valid_texts.append("")
            
            # 並列トークン化
            tokenize_start = time.time()
            tokens_list = self._parallel_tokenize(valid_texts)
            self.stats['tokenization_time'] += time.time() - tokenize_start
            
            # トークン数制限
            for i, tokens in enumerate(tokens_list):
                if len(tokens) > 100:
                    tokens_list[i] = tokens[:100]
            
            # バッチ埋め込み生成
            embedding_start = time.time()
            embeddings = self._batch_embeddings(valid_texts)
            self.stats['embedding_time'] += time.time() - embedding_start
            
            # 結果構築
            results = []
            for text, tokens, embedding in zip(valid_texts, tokens_list, embeddings):
                results.append({
                    'normalized': text,
                    'tokens': tokens,
                    'embedding': embedding
                })
            
            self.stats['total_processed'] += len(input_texts)
            self.stats['total_time'] += time.time() - start_time
            
            if self.debug:
                logger.debug(f"バッチ処理完了: {len(input_texts)}件, {time.time() - start_time:.3f}秒")
            
            return results
            
        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            
            # エラー時のフォールバック
            return [{'normalized': text[:100], 'tokens': text.split()[:10], 'embedding': np.zeros(384)} 
                   for text in input_texts]

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        入力テキストを前処理する
        
        引数:
            input_text: 処理する入力テキスト
            
        戻り値:
            処理結果を含む辞書（正規化テキスト、トークン、埋め込み）
        """
        self._check_shutdown()  # シャットダウン状態をチェック
        
        try:
            # 入力の検証
            if not input_text or not isinstance(input_text, str):
                logger.warning("無効な入力テキスト")
                return {'normalized': "", 'tokens': [], 'embedding': np.array([])}
                
            # テキスト長の制限
            if len(input_text) > 1000:
                logger.debug(f"入力テキストを切り詰め: {len(input_text)} → 1000文字")
                input_text = input_text[:1000]  # 最大1000文字に制限
            
            # 正規化
            normalized = input_text.strip()
            
            # トークン分割（スペース区切り）
            tokens = normalized.split()
            if len(tokens) > 100:
                logger.debug(f"トークン数を制限: {len(tokens)} → 100")
                tokens = tokens[:100]  # 最大100トークンに制限
            
            result = {'normalized': normalized, 'tokens': tokens}
            
            # 埋め込み生成（オプション）
            if self.use_embeddings:
                try:
                    self._load_transformer()
                    if self._transformer:
                        embedding = self._transformer.encode(normalized)
                        result['embedding'] = embedding
                        
                        if self.debug:
                            logger.debug(f"埋め込み生成: shape={embedding.shape}")
                    else:
                        # ダミー埋め込み
                        result['embedding'] = np.zeros(384)  # MiniLM-L6の次元数
                        logger.warning("ダミー埋め込みを使用")
                except Exception as e:
                    logger.error(f"埋め込み生成エラー: {e}")
                    result['embedding'] = np.zeros(384)  # ダミー埋め込み            
            return result
            
        except Exception as e:
            logger.error(f"入力処理エラー: {e}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
                
            # エラー時のフォールバック
            return {'normalized': input_text[:100], 'tokens': input_text.split()[:10], 'embedding': np.zeros(384)}
    
    def _setup_signal_handlers(self) -> None:
        """
        シグナルハンドラーの設定
        Ctrl+Cやプログラム終了時の適切なクリーンアップを確保
        """
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"シグナル {signal_name} を受信しました。シャットダウンを開始...")
            
            # フリーズを防ぐため、シャットダウンフラグを設定してメインスレッドに処理を委ねる
            self._shutdown_requested = True
            self._shutdown_event.set()
            
            # 別スレッドでシャットダウン処理を実行（デッドロック回避）
            def async_shutdown():
                try:
                    time.sleep(0.1)  # 少し待機してからシャットダウン
                    self.shutdown()
                    # 強制終了（ただし即座ではなく少し待機後）
                    time.sleep(0.5)
                    import os
                    os._exit(0)  # sys.exit()よりも強制的
                except Exception as e:
                    logger.error(f"非同期シャットダウンエラー: {e}")
                    import os
                    os._exit(1)
            
            # バックグラウンドでシャットダウン実行
            shutdown_thread = threading.Thread(target=async_shutdown, daemon=True)
            shutdown_thread.start()
        
        try:
            # SIGINT (Ctrl+C) のハンドラー設定
            signal.signal(signal.SIGINT, signal_handler)
            
            # Windows/Unix両対応のためのSIGTERM設定
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_handler)
            
            if self.debug:
                logger.debug("シグナルハンドラーを設定しました")
                
        except Exception as e:
            logger.warning(f"シグナルハンドラー設定エラー: {e}")
    
    def _emergency_shutdown(self) -> None:
        """
        緊急シャットダウン処理（atexitハンドラー）
        プログラム終了時に確実にリソースをクリーンアップ
        """
        if not self._is_shutdown:
            logger.info("緊急シャットダウンを実行中...")
            self.shutdown()
    
    @classmethod
    def force_exit_all(cls):
        """
        全てのInputReceptionインスタンスを強制終了し、プロセスを終了
        メインプログラムから呼び出し可能な静的メソッド
        """
        logger.info("全InputReceptionインスタンスの強制終了を開始...")
        
        # すべてのThreadPoolExecutorを強制終了
        import concurrent.futures
        import threading
        
        # アクティブなスレッドを強制終了
        for thread in threading.enumerate():
            if thread != threading.current_thread() and thread.is_alive():
                if hasattr(thread, '_stop'):
                    thread._stop()
        
        # プロセス全体を強制終了
        logger.info("プロセスを強制終了します...")
        import os
        os._exit(0)
