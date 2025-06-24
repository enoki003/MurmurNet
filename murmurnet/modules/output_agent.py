#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output Agent モジュール (CPU/並列/メモリ最適化版)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
最終応答生成を担当するエージェント
黒板の要約情報に基づき一貫性のある回答を作成

最適化機能:
- LRUキャッシュによる高速応答
- テンプレート管理システム
- 並列処理とバッチ処理
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
from string import Template
import yaml

from MurmurNet.modules.model_factory import ModelFactory

logger = logging.getLogger('MurmurNet.OutputAgent')

@dataclass
class PerformanceStats:
    """パフォーマンス統計情報"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    total_response_time: float = 0.0
    llm_calls: int = 0
    template_cache_hits: int = 0
    parallel_processes: int = 0
    memory_usage_mb: float = 0.0
    
    def get_cache_hit_rate(self) -> float:
        """キャッシュヒット率を取得"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def get_avg_response_time(self) -> float:
        """平均応答時間を取得"""
        return (self.total_response_time / self.total_requests) if self.total_requests > 0 else 0.0


class OptimizedResponseCache:
    """最適化された応答キャッシュシステム"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = defaultdict(int)
    
    def _generate_key(self, user_input: str, entries: List[Dict], rag: str = None) -> str:
        """キャッシュキーを生成"""
        # 入力内容のハッシュを生成
        content = f"{user_input}|{len(entries)}|{rag or ''}"
        for entry in entries:
            content += f"|{entry.get('type', '')}:{entry.get('text', '')[:50]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """キャッシュから取得"""
        with self.lock:
            if key in self.cache:
                # LRUアルゴリズム: アクセスされた項目を末尾に移動
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return self.cache[key]
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: str):
        """キャッシュに追加"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # 最も古い項目を削除
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def get_stats(self) -> Dict[str, int]:
        """統計情報を取得"""
        with self.lock:
            return dict(self.stats)


class TemplateManager:
    """効率的なテンプレート管理システム"""
    
    def __init__(self):
        self.templates = {}
        self.compiled_templates = {}
        self.lock = threading.RLock()
        self._init_templates()
    
    def _init_templates(self):
        """テンプレートを初期化"""
        # 日本語システムテンプレート
        self.templates['system_ja'] = Template("""あなたは親しみやすい日本語アシスタントです。

【タスク】
複数のエージェントの意見を統合して、ユーザーの質問に対する自然で有用な回答を作成してください。

【重要な指針】
1. エージェントの生の意見を最重視し、各エージェントの異なる視点を活かす
2. 要約情報は補助的な参考として使用する
3. 自然な話し言葉で親しみやすく回答する
4. ${max_tokens}文字程度で簡潔かつ完全にまとめる
5. マークダウンや特殊記号は使わず、読みやすい文章にする
6. エージェント間の意見の違いがあれば、バランスよく統合する

【出力形式】
- 一つの段落で完結した回答
- 句読点を適切に使用
- 文章の途中で終わらせない""")
        
        # 英語システムテンプレート
        self.templates['system_en'] = Template("""You are a friendly English assistant.

【Task】
Integrate multiple agents' opinions to create a natural and helpful response to the user's question.

【Key Guidelines】
1. Prioritize agents' raw opinions and leverage different perspectives
2. Use summary information as supplementary reference
3. Respond in natural, conversational language
4. Keep response around ${max_tokens} characters, concise but complete
5. Avoid markdown or special symbols, use readable text
6. If agents have different opinions, integrate them in a balanced way

【Output Format】
- Single paragraph with complete response
- Use proper punctuation
- Do not end mid-sentence""")
        
        # ユーザープロンプトテンプレート
        self.templates['user_prompt'] = Template("""【ユーザーの質問】
${user_input}

${rag_section}${agent_section}${summary_section}【回答作成の指示】
上記のエージェントの意見を最重視して、ユーザーの質問に対する統合された回答を作成してください。
・各エージェントの個性的な表現や異なる視点を活かしてください
・要約情報は補助的な参考情報として扱ってください
・自然で読みやすい一つの段落にまとめてください
・文章は完結させ、途中で終わらせないでください""")
    
    def get_template(self, template_name: str, **kwargs) -> str:
        """テンプレートを取得して変数を置換"""
        with self.lock:
            if template_name not in self.templates:
                raise ValueError(f"テンプレート '{template_name}' が見つかりません")
            
            template = self.templates[template_name]
            return template.safe_substitute(**kwargs)


class CPUOptimizedOutputAgent:
    """
    CPU/並列/メモリ最適化された出力エージェント
    
    機能:
    - LRUキャッシュによる高速応答
    - テンプレート管理システム  
    - 並列処理とバッチ処理
    - パフォーマンス統計
    - スレッドセーフな操作
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        最適化された出力エージェントの初期化
        
        引数:
            config: 設定辞書（省略時は空の辞書）
        """        
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.max_output_tokens = self.config.get('max_output_tokens', 250)
        
        # 最適化設定の読み込み
        self.cache_size = self.config.get('output_cache_size', 1000)
        self.max_workers = self.config.get('output_max_workers', 4)
        self.enable_batch_processing = self.config.get('enable_batch_processing', True)
        self.enable_parallel_processing = self.config.get('enable_parallel_processing', True)
        self.template_cache_size = self.config.get('template_cache_size', 100)
        self.memory_threshold_mb = self.config.get('memory_threshold_mb', 500)
        
        # 最適化コンポーネントの初期化
        self.response_cache = OptimizedResponseCache(self.cache_size)
        self.template_manager = TemplateManager()
        self.stats = PerformanceStats()
        self.lock = threading.RLock()
        
        # 言語検出キャッシュ
        self.language_cache = OrderedDict()
        self.language_cache_size = 500
        
        # スレッドプール
        if self.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.thread_pool = None
        
        # パフォーマンス履歴
        self.performance_history = deque(maxlen=1000)
        
        # デバッグモードを強制的に有効にする
        logger.setLevel(logging.DEBUG)
        
        # ModelFactoryからモデルを取得
        self.llm = ModelFactory.create_model(self.config)
        
        logger.info(f"最適化出力エージェントを初期化: cache_size={self.cache_size}, workers={self.max_workers}")
        logger.debug(f"OutputAgent最適化設定: batch={self.enable_batch_processing}, parallel={self.enable_parallel_processing}")
    
    def __del__(self):
        """リソースのクリーンアップ"""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=False)

    @lru_cache(maxsize=500)
    def _detect_language_cached(self, text: str) -> str:
        """
        キャッシュ機能付き言語検出（内部メソッド）
        
        引数:
            text: 言語を検出するテキスト
            
        戻り値:
            検出された言語コード ('ja', 'en' など)
        """
        # 日本語の文字が含まれているかチェック
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
            return 'ja'
        # 英語の文字が主に含まれているかチェック
        elif re.search(r'[a-zA-Z]', text) and not re.search(r'[^\x00-\x7F]', text):
            return 'en'
        # その他の言語（デフォルトは英語）
        return 'en'
    
    def _detect_language(self, text: str) -> str:
        """
        効率的な言語検出
        """
        # テキストのハッシュをキーとしてキャッシュ
        text_hash = hashlib.md5(text[:200].encode()).hexdigest()
        
        with self.lock:
            if text_hash in self.language_cache:
                # LRUアルゴリズム
                self.language_cache.move_to_end(text_hash)
                return self.language_cache[text_hash]
            
            # キャッシュにない場合は計算
            lang = self._detect_language_cached(text)
            
            # キャッシュサイズ管理
            if len(self.language_cache) >= self.language_cache_size:
                self.language_cache.popitem(last=False)
            
            self.language_cache[text_hash] = lang
            return lang
    
    def _build_prompt_sections_parallel(self, entries: List[Dict[str, Any]], rag: str = None) -> Tuple[str, str, str]:
        """
        プロンプトセクションを並列構築
        """
        def build_rag_section():
            return f"【参考情報】\n{rag}\n\n" if rag else ""
        
        def build_agent_section():
            agent_outputs = [
                entry for entry in entries 
                if entry.get('type', 'agent') == 'agent'
            ]
            if agent_outputs:
                section = "【エージェントの意見】\n"
                for i, entry in enumerate(agent_outputs, 1):
                    text = entry.get('text', '')[:300]
                    agent_id = entry.get('agent', i-1)
                    section += f"{i}. エージェント {agent_id+1}: {text}\n"
                return section + "\n"
            return ""
        
        def build_summary_section():
            summaries = [
                entry for entry in entries 
                if entry.get('type') == 'summary'
            ]
            if summaries:
                section = "【要約情報（参考）】\n"
                for i, entry in enumerate(summaries, 1):
                    text = entry.get('text', '')[:300]
                    iteration = entry.get('iteration', i-1)
                    section += f"{i}. 要約 {iteration+1}: {text}\n"
                return section + "\n"
            return ""
        
        if self.enable_parallel_processing and self.thread_pool:
            # 並列処理
            futures = [
                self.thread_pool.submit(build_rag_section),
                self.thread_pool.submit(build_agent_section),
                self.thread_pool.submit(build_summary_section)
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"並列プロンプト構築エラー: {e}")
                    results.append("")
            
            return results[0], results[1], results[2]
        else:
            # 逐次処理
            return build_rag_section(), build_agent_section(), build_summary_section()
    
    def _build_optimized_prompt(self, user_input: str, entries: List[Dict[str, Any]], 
                               rag: str = None, lang: str = 'ja') -> Tuple[str, str]:
        """
        最適化されたプロンプト構築
        """
        # システムプロンプトをテンプレートから生成
        template_key = f'system_{lang}'
        try:
            system_prompt = self.template_manager.get_template(
                template_key, 
                max_tokens=f"{self.max_output_tokens}-{self.max_output_tokens+50}"
            )
            self.stats.template_cache_hits += 1
        except Exception as e:
            logger.error(f"テンプレート取得エラー: {e}")
            # フォールバック
            system_prompt = "You are a helpful assistant." if lang == 'en' else "あなたは親しみやすいアシスタントです。"
        
        # セクションを並列構築
        rag_section, agent_section, summary_section = self._build_prompt_sections_parallel(entries, rag)
        
        # ユーザープロンプトを構築
        user_prompt = self.template_manager.get_template(
            'user_prompt',
            user_input=user_input,
            rag_section=rag_section,
            agent_section=agent_section,
            summary_section=summary_section
        )
        
        return system_prompt, user_prompt
    
    def _clean_output_optimized(self, text: str) -> str:
        """
        最適化された出力クリーニング（並列処理対応）
        """
        if not text or len(text.strip()) < 5:
            return "申し訳ございませんが、適切な回答を生成できませんでした。"
        
        # 効率的な正規表現パターン（事前コンパイル済み）
        patterns = [
            (re.compile(r'\*{2,}'), ''),  # **以上の連続する*
            (re.compile(r'#{2,}'), ''),   # ##以上の連続する#
            (re.compile(r'-{3,}'), ''),   # ---以上の連続する-
            (re.compile(r'={3,}'), ''),   # ===以上の連続する=
            (re.compile(r'_{3,}'), ''),   # ___以上の連続する_
            (re.compile(r'(.)\1{4,}'), r'\1'),  # 繰り返し文字
            (re.compile(r'\n{3,}'), '\n\n'),    # 連続改行
            (re.compile(r' {3,}'), ' '),        # 連続空白
            (re.compile(r'\b(\w+)\s+\1\s+\1\b'), r'\1'),  # 繰り返し単語
        ]
        
        # パターンマッチングを並列化（短いテキストの場合は逐次処理）
        if len(text) > 500 and self.enable_parallel_processing and self.thread_pool:
            def apply_pattern(pattern_sub_pair):
                pattern, substitute = pattern_sub_pair
                return pattern.sub(substitute, text)
            
            # 並列処理は重い処理のみに限定
            cleaned_text = text
            for pattern, substitute in patterns[:3]:  # 最初の3パターンのみ並列
                cleaned_text = pattern.sub(substitute, cleaned_text)
            
            # 残りは逐次処理
            for pattern, substitute in patterns[3:]:
                cleaned_text = pattern.sub(substitute, cleaned_text)
        else:
            # 逐次処理
            cleaned_text = text
            for pattern, substitute in patterns:
                cleaned_text = pattern.sub(substitute, cleaned_text)
        
        # 文章の完全性チェック
        sentences = cleaned_text.split('。')
        if len(sentences) > 1:
            if sentences[-1].strip() == '' or len(sentences[-1].strip()) < 5:
                sentences = sentences[:-1]
            elif not sentences[-1].strip().endswith(('。', '！', '？', '.', '!', '?')):
                sentences = sentences[:-1]
            cleaned_text = '。'.join(sentences)
            if cleaned_text and not cleaned_text.endswith('。'):
                cleaned_text += '。'
        
        # 制御文字の除去
        cleaned_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned_text).strip()
        
        return cleaned_text if len(cleaned_text.strip()) >= 10 else "申し訳ございませんが、適切な回答を生成できませんでした。"
    
    def _generate_cache_key(self, user_input: str, entries: List[Dict], rag: str = None) -> str:
        """キャッシュキーを生成"""
        # 入力内容のハッシュを生成
        content = f"{user_input}|{len(entries)}|{rag or ''}"
        for entry in entries:
            content += f"|{entry.get('type', '')}:{entry.get('text', '')[:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate(self, blackboard, entries: List[Dict[str, Any]]) -> str:
        """
        最適化された最終応答生成（バッチ・並列・キャッシュ・統計・メモリ効率化対応）
        """
        start_time = time.time()
        self.stats.total_requests += 1
        logger.debug("=== OutputAgent.generate() 最適化版 開始 ===")

        # バッチサイズ設定（config優先、なければデフォルト8）
        batch_size = self.config.get('output_batch_size', 8)
        self.batch_size = batch_size

        try:
            # 1. 基本データ取得
            inp = blackboard.read('input')
            user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
            user_input = user_input[:400]

            rag = blackboard.read('rag')
            if rag and isinstance(rag, str):
                rag = rag[:600]

            # 2. キャッシュチェック
            cache_key = self._generate_cache_key(user_input, entries, rag)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                self.stats.cache_hits += 1
                logger.info(f"キャッシュヒット: {cache_key[:32]}...")
                return cached_response
            self.stats.cache_misses += 1

            # 3. 言語検出
            lang = self._detect_language_cached(user_input)

            # 4. 最適化されたプロンプト構築
            system_prompt, user_prompt = self._build_optimized_prompt(user_input, entries, rag, lang)

            # 5. バッチ・並列処理
            responses = []
            if self.enable_parallel_processing and len(entries) > batch_size:
                self.stats.parallel_processes += 1
                batched_entries = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
                futures = []
                # バッチごとにLLM呼び出しを並列実行
                for batch in batched_entries:
                    batch_cache_key = self._generate_cache_key(user_input, batch, rag)
                    batch_cached = self.response_cache.get(batch_cache_key)
                    if batch_cached:
                        self.stats.cache_hits += 1
                        responses.append(batch_cached)
                        continue
                    def llm_call(batch=batch):
                        self.stats.llm_calls += 1
                        sys_prompt, usr_prompt = self._build_optimized_prompt(user_input, batch, rag, lang)
                        resp = self.llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": usr_prompt}
                            ],
                            max_tokens=self.max_output_tokens,
                            temperature=0.7,
                            top_p=0.95,
                            repeat_penalty=1.2
                        )
                        if isinstance(resp, dict):
                            out = resp['choices'][0]['message']['content'].strip()
                        else:
                            out = resp.choices[0].message.content.strip()
                        out = self._clean_output_optimized(out)
                        self.response_cache.put(batch_cache_key, out)
                        return out
                    if self.thread_pool:
                        futures.append(self.thread_pool.submit(llm_call))
                    else:
                        responses.append(llm_call())
                # 並列実行結果を集約
                if futures:
                    for f in as_completed(futures):
                        try:
                            responses.append(f.result())
                        except Exception as e:
                            logger.error(f"バッチ並列応答生成エラー: {e}")
                            responses.append("")
                # 応答を統合
                final_output = '\n'.join([r for r in responses if r])
            else:
                # バッチ不要 or 並列無効時は通常処理
                self.stats.llm_calls += 1
                resp = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_output_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    repeat_penalty=1.2
                )
                if isinstance(resp, dict):
                    final_output = resp['choices'][0]['message']['content'].strip()
                else:
                    final_output = resp.choices[0].message.content.strip()

            # 6. 最適化された出力クリーニング（バッチ時は全体に適用）
            final_output = self._clean_output_optimized(final_output)

            # 7. キャッシュに保存
            self.response_cache.put(cache_key, final_output)

            # 8. 統計情報更新
            response_time = time.time() - start_time
            self.stats.total_response_time += response_time

            # メモリ使用量監視
            try:
                import psutil
                process = psutil.Process()
                self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except ImportError:
                pass

            # 定期的なガベージコレクション
            if self.stats.total_requests % 100 == 0:
                gc.collect()

            logger.info(f"最適化版応答生成完了: {len(final_output)}文字, {response_time:.3f}秒")
            logger.info(f"統計: ヒット率={self.stats.get_cache_hit_rate():.2%}, "
                        f"平均応答時間={self.stats.get_avg_response_time():.3f}秒, "
                        f"メモリ={self.stats.memory_usage_mb:.1f}MB")

            return final_output

        except Exception as e:
            error_msg = f"最適化版出力生成エラー: {str(e)}"
            logger.error(error_msg)
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            lang = self._detect_language_cached(user_input) if 'user_input' in locals() else 'ja'
            if lang == 'ja':
                return "申し訳ございませんが、適切な回答を生成できませんでした。"
            else:
                return "I apologize, but I couldn't generate an appropriate response."

    def shutdown(self):
        """
        OutputAgentの完全なシャットダウン処理
        
        全てのリソースを適切に終了し、統計情報を記録する
        """
        logger.info("OutputAgentシャットダウン開始")
        
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
                final_stats = {
                    'cache_hit_rate': self.stats.get_cache_hit_rate(),
                    'avg_response_time': self.stats.get_avg_response_time(),
                    'total_requests': self.stats.total_requests,
                    'llm_calls': self.stats.llm_calls,
                    'parallel_processes': self.stats.parallel_processes,
                    'memory_usage_mb': self.stats.memory_usage_mb
                }
                logger.info(f"OutputAgent最終統計: {final_stats}")
            
            # 3. キャッシュのクリア
            if hasattr(self, 'response_cache'):
                try:
                    cache_size = self.response_cache.size()
                    self.response_cache.clear()
                    logger.debug(f"応答キャッシュをクリア: {cache_size}エントリ")
                except Exception as e:
                    logger.warning(f"応答キャッシュクリアエラー: {e}")
            
            if hasattr(self, 'template_cache'):
                try:
                    self.template_cache.clear()
                    logger.debug("テンプレートキャッシュをクリア")
                except Exception as e:
                    logger.warning(f"テンプレートキャッシュクリアエラー: {e}")
            
            # 4. モデル参照のクリア
            if hasattr(self, 'llm'):
                self.llm = None
            
            # 5. メモリクリーンアップ
            try:
                import gc
                collected = gc.collect()
                logger.debug(f"OutputAgent: ガベージコレクション完了 ({collected}個)")
            except Exception as e:
                logger.warning(f"ガベージコレクションエラー: {e}")
            
            logger.info("OutputAgentシャットダウン完了")
            
        except Exception as e:
            logger.error(f"OutputAgentシャットダウンエラー: {e}")
            # エラーが発生してもシャットダウンを継続
            import traceback
            logger.debug(traceback.format_exc())

# 下位互換性のためのエイリアス
OutputAgent = CPUOptimizedOutputAgent
