#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary Engine モジュール
~~~~~~~~~~~~~~~~~~~~~
黒板上の情報を要約するエンジン
長いコンテキストを簡潔にまとめる機能を提供

作者: Yuhi Sonoki
"""

# summary_engine.py
from llama_cpp import Llama
import os
import time
import threading
import hashlib
import concurrent.futures
import asyncio
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional, Union

# グローバルモデルキャッシュ - シングルトン実装
_MODEL_CACHE = {}
_MODEL_LOCK = threading.RLock()
_SUMMARY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # スレッド数を増加
_CACHE_LOCK = threading.RLock()  # キャッシュ操作用のロック

class SummaryEngine:
    def __init__(self, config: dict = None):
        config = config or {}
        self.model_path = config.get('model_path') or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')
        )
        
        # 親設定から値を取得（もしくはデフォルト値を使用）
        self.llama_kwargs = dict(
            model_path=self.model_path,
            n_ctx=config.get('n_ctx', 2048),  # 親設定から受け取る
            n_threads=config.get('n_threads', 4),  # 親設定から受け取る
            use_mmap=True,
            use_mlock=False,
            n_gpu_layers=0,
            seed=42,
            chat_format="gemma",
            verbose=False  # ログ出力抑制
        )
        
        chat_template = config.get('chat_template')
        if chat_template and os.path.exists(chat_template):
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    self.llama_kwargs['chat_template'] = f.read()
            except Exception as e:
                if config.get('debug'):
                    print(f"テンプレートロードエラー: {e}")
        
        self.debug = config.get('debug', False)
        if self.debug:
            print(f"要約エンジン: モデル設定 n_ctx={self.llama_kwargs['n_ctx']}, n_threads={self.llama_kwargs['n_threads']}")
        
        self.max_summary_tokens = config.get('max_summary_tokens', 256)  # 要約の最大トークン数
        
        # キャッシュの設定 - LRUキャッシュに移行
        self.use_cache = config.get('use_summary_cache', True)  # キャッシュ機能を使うかどうか
        self.max_cache_entries = config.get('max_summary_cache_entries', 100)
        
        # LRUキャッシュをインスタンス単位ではなくクラス単位で管理
        if not hasattr(SummaryEngine, '_summary_cache'):
            SummaryEngine._summary_cache = {}
            SummaryEngine._cache_timestamps = {}  # アクセスタイムスタンプ
        
        # パフォーマンス最適化設定
        self.batch_enabled = config.get('enable_summary_batch', True)  # バッチ処理機能を使うかどうか
        self.batch_size = config.get('summary_batch_size', 5)  # バッチサイズを増加
        
        # バッチ処理キューをシングルトン化
        if not hasattr(SummaryEngine, '_batch_queue'):
            SummaryEngine._batch_queue = []
            SummaryEngine._batch_lock = threading.RLock()
            SummaryEngine._batch_results = {}
            SummaryEngine._batch_events = {}
            SummaryEngine._processing_batch = False
            
        # モデルの初期化は遅延させる
        self.llm = None
        
    def _get_llm(self):
        """モデルのシングルトンインスタンスを取得（メモリ効率改善）"""
        if self.llm is not None:
            return self.llm
        
        # モデルキーを生成（モデルパスとスレッド数でユニークに）
        model_key = f"{self.model_path}_{self.llama_kwargs['n_threads']}_{self.llama_kwargs['n_ctx']}"
        
        # モデルキャッシュをロックして取得または作成
        with _MODEL_LOCK:
            if model_key in _MODEL_CACHE:
                if self.debug:
                    print(f"既存のモデルインスタンスを使用: {model_key}")
                self.llm = _MODEL_CACHE[model_key]
            else:
                if self.debug:
                    print(f"新しいモデルインスタンスを作成: {model_key}")
                try:
                    # リソース効率のため、モデル読み込みオプションを最適化
                    self.llm = Llama(**self.llama_kwargs)
                    _MODEL_CACHE[model_key] = self.llm
                except Exception as e:
                    if self.debug:
                        print(f"モデルロードエラー: {e}")
                    # エラー時はインスタンス毎に生成
                    self.llm = Llama(**self.llama_kwargs)
        
        return self.llm

    @lru_cache(maxsize=64)  # キャッシュサイズを増加
    def _cache_key(self, entries_tuple):
        """エントリーからキャッシュキーを生成（LRUキャッシュ対応、効率改善）"""
        if not entries_tuple:
            return None
            
        try:
            # 各エントリから重要部分を抽出してハッシュ化
            key_parts = []
            for e in entries_tuple:
                if isinstance(e, dict):
                    text = e.get('text', '')
                    if text:
                        # 長いテキストは切り詰める+途中をスキップして特徴的な部分だけ使用
                        if len(text) > 200:
                            key_parts.append(text[:100] + text[-100:])
                        else:
                            key_parts.append(text)
                elif isinstance(e, str):
                    if len(e) > 200:
                        key_parts.append(e[:100] + e[-100:])
                    else:
                        key_parts.append(e)
                    
            if not key_parts:
                return None
                
            # 連結してハッシュ化（より効率的なアルゴリズム）
            combined = "|||".join(key_parts)
            return hashlib.blake2b(combined.encode('utf-8'), digest_size=16).hexdigest()
            
        except Exception as e:
            if self.debug:
                print(f"キャッシュキー生成エラー: {e}")
            return None

    def _process_batch(self):
        """バッチキューに溜まったエントリを処理する（効率改善版）"""
        # 並列処理の再入を防ぐ
        with SummaryEngine._batch_lock:
            if SummaryEngine._processing_batch:
                return  # すでに処理中
            
            if not SummaryEngine._batch_queue:
                return  # キューが空
            
            # 処理中フラグをセット
            SummaryEngine._processing_batch = True
            
            # 現在のバッチを取得
            current_batch = SummaryEngine._batch_queue[:self.batch_size]
            SummaryEngine._batch_queue = SummaryEngine._batch_queue[self.batch_size:]
        
        if not current_batch:
            with SummaryEngine._batch_lock:
                SummaryEngine._processing_batch = False
            return
        
        try:
            # モデルを取得
            llm = self._get_llm()
            
            start_time = time.time()
            
            # 非同期処理用の関数
            def process_entry(entry):
                batch_id = entry['id']
                prompt = entry['prompt']
                
                try:
                    # モデル呼び出し
                    resp = llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_summary_tokens,
                        temperature=0.3,
                        stop=["。", ".", "\n\n"]
                    )
                    
                    # レスポンスの形式によって適切にアクセス
                    if isinstance(resp, dict):
                        summary = resp['choices'][0]['message']['content'].strip()
                    else:
                        summary = resp.choices[0].message.content.strip()
                    
                    # 出力を制限
                    summary = summary[:300]
                    
                    return {
                        'batch_id': batch_id,
                        'success': True, 
                        'summary': summary
                    }
                    
                except Exception as e:
                    # エラー情報を保存
                    return {
                        'batch_id': batch_id,
                        'success': False, 
                        'error': str(e),
                        'summary': "要約の生成中にエラーが発生しました。"
                    }
            
            # 複数のエントリを並列処理
            futures = []
            for entry in current_batch:
                futures.append(_SUMMARY_EXECUTOR.submit(process_entry, entry))
            
            # 結果を処理
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                batch_id = result['batch_id']
                
                # 結果を保存
                with SummaryEngine._batch_lock:
                    SummaryEngine._batch_results[batch_id] = result
                
                # イベントで処理完了を通知
                if batch_id in SummaryEngine._batch_events:
                    SummaryEngine._batch_events[batch_id].set()
            
            process_time = time.time() - start_time
            
            if self.debug:
                print(f"バッチ要約処理完了: {len(current_batch)}件, 処理時間={process_time:.2f}秒")
        
        except Exception as e:
            if self.debug:
                print(f"バッチ処理エラー: {str(e)}")
                import traceback
                traceback.print_exc()
        
        finally:
            # 処理中フラグをリセット
            with SummaryEngine._batch_lock:
                SummaryEngine._processing_batch = False
                
                # 続きのキューがあれば再度処理を開始
                if SummaryEngine._batch_queue:
                    _SUMMARY_EXECUTOR.submit(self._process_batch)

    def summarize_blackboard(self, entries: list) -> str:
        """
        黒板のエントリを要約する（効率化版）
        
        引数:
            entries: 要約するエントリのリスト。各エントリは{'agent': id, 'text': str}の形式
        
        戻り値:
            要約されたテキスト
        """
        # エントリが空の場合
        if not entries:
            return "要約するエントリがありません。"
        
        # エントリが少なすぎる場合はそのまま返す（最適化）
        if len(entries) == 1:
            if isinstance(entries[0], dict) and 'text' in entries[0]:
                text = entries[0]['text'].strip()
                if len(text) < 100:  # 短いテキストはそのまま返す
                    return text
        
        # キャッシュキー生成のためにタプルに変換
        entries_tuple = tuple(
            tuple(sorted(e.items())) if isinstance(e, dict) else e
            for e in entries
        )
        
        # キャッシュから要約を取得
        if self.use_cache:
            cache_key = self._cache_key(entries_tuple)
            if cache_key:
                with _CACHE_LOCK:
                    if cache_key in SummaryEngine._summary_cache:
                        # アクセスタイムスタンプを更新（LRU実装用）
                        SummaryEngine._cache_timestamps[cache_key] = time.time()
                        if self.debug:
                            print(f"キャッシュから要約を取得: {cache_key[:8]}...")
                        return SummaryEngine._summary_cache[cache_key]
        
        try:
            # 入力テキストを効率的に制限
            combined = []
            total_length = 0
            max_total = 1000  # 最大合計文字数
            
            for e in entries:
                if isinstance(e, dict) and 'text' in e:
                    text = e['text'][:200]  # 各エントリは200文字まで
                elif isinstance(e, str):
                    text = e[:200]
                else:
                    continue
                    
                combined.append(text)
                total_length += len(text)
                
                if total_length >= max_total:
                    break
            
            # 要約用のプロンプト - より効果的な指示
            prompt = (
                "以下の複数のテキストの要点を簡潔にまとめてください。\n"
                "・要約は200文字以内で\n"
                "・箇条書きではなく、文章で\n"
                "・重要な情報だけを抽出\n"
                "・余計な説明は不要\n\n" + 
                "\n\n".join(combined)
            )
            
            # バッチ処理が有効な場合はバッチキューに追加
            if self.batch_enabled:
                # ユニークなバッチIDを生成
                import uuid
                batch_id = f"sum_{uuid.uuid4().hex[:8]}"
                completion_event = threading.Event()
                
                # バッチキューに追加
                with SummaryEngine._batch_lock:
                    SummaryEngine._batch_queue.append({
                        'id': batch_id,
                        'prompt': prompt,
                        'timestamp': time.time()
                    })
                    SummaryEngine._batch_events[batch_id] = completion_event
                    
                    # バッチ処理が実行中でなければ開始
                    if not SummaryEngine._processing_batch:
                        _SUMMARY_EXECUTOR.submit(self._process_batch)
                
                # 処理完了を待機（最大8秒 - 速度向上のため）
                completion_event.wait(8.0)
                
                # 結果を取得
                result = None
                with SummaryEngine._batch_lock:
                    result = SummaryEngine._batch_results.get(batch_id)
                
                if result and result.get('success'):
                    summary = result['summary']
                else:
                    # バッチ処理でエラーまたはタイムアウトの場合、通常の処理にフォールバック
                    if self.debug:
                        print(f"バッチ処理失敗、通常処理にフォールバック: {batch_id}")
                    
                    # フォールバック処理
                    llm = self._get_llm()
                    try:
                        resp = llm.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=self.max_summary_tokens,
                            temperature=0.3,
                            stop=["。", ".", "\n\n"]
                        )
                        
                        if isinstance(resp, dict):
                            summary = resp['choices'][0]['message']['content'].strip()
                        else:
                            summary = resp.choices[0].message.content.strip()
                        
                        summary = summary[:300]
                    except Exception as e:
                        if self.debug:
                            print(f"要約フォールバックエラー: {str(e)}")
                        summary = "要約の生成中にエラーが発生しました。"
                
                # リソース解放
                with SummaryEngine._batch_lock:
                    if batch_id in SummaryEngine._batch_events:
                        del SummaryEngine._batch_events[batch_id]
                    if batch_id in SummaryEngine._batch_results:
                        del SummaryEngine._batch_results[batch_id]
                    
            else:
                # 通常の処理（バッチなし）
                llm = self._get_llm()
                resp = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_summary_tokens,
                    temperature=0.3,
                    stop=["。", ".", "\n\n"]
                )
                
                if isinstance(resp, dict):
                    summary = resp['choices'][0]['message']['content'].strip()
                else:
                    summary = resp.choices[0].message.content.strip()
                
                summary = summary[:300]
            
            # キャッシュに結果を保存
            if self.use_cache and cache_key:
                with _CACHE_LOCK:
                    # キャッシュが上限に達した場合、最も古いエントリを削除
                    if len(SummaryEngine._summary_cache) >= self.max_cache_entries:
                        # LRUアルゴリズムで最も古いキャッシュを削除
                        oldest_key = min(
                            SummaryEngine._cache_timestamps, 
                            key=SummaryEngine._cache_timestamps.get
                        )
                        del SummaryEngine._summary_cache[oldest_key]
                        del SummaryEngine._cache_timestamps[oldest_key]
                    
                    # 新しいキャッシュを追加    
                    SummaryEngine._summary_cache[cache_key] = summary
                    SummaryEngine._cache_timestamps[cache_key] = time.time()
                
            return summary
            
        except Exception as e:
            error_msg = f"要約エラー: {str(e)}"
            if self.debug:
                print(error_msg)
                import traceback
                traceback.print_exc()
            return "要約の生成中にエラーが発生しました。"
            
    # バッチキューのクリア（必要に応じて）
    def clear_batch_queue(self):
        """バッチ処理キューをクリアする"""
        with SummaryEngine._batch_lock:
            SummaryEngine._batch_queue = []
            
    @classmethod
    def clear_all_caches(cls):
        """すべてのキャッシュをクリアする（メモリ解放用）"""
        with _CACHE_LOCK:
            if hasattr(cls, '_summary_cache'):
                cls._summary_cache.clear()
                cls._cache_timestamps.clear()
        
        # モデルキャッシュは重いのでオプションで解放
        # with _MODEL_LOCK:
        #     _MODEL_CACHE.clear()
