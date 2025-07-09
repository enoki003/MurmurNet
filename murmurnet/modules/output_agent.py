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


class CPUOptimizedOutputAgent:
    """
    CPU/並列/メモリ最適化された出力エージェント（公式チャットテンプレート対応）
    
    機能:
    - LRUキャッシュによる高速応答
    - テンプレート管理システム  
    - 並列処理とバッチ処理
    - パフォーマンス統計
    - スレッドセーフな操作
    """
    
    def __init__(self, config: Dict[str, Any] = None, shared_llm=None):
        """
        最適化された出力エージェントの初期化
        
        引数:
            config: 設定辞書（省略時は空の辞書）
            shared_llm: 既存のLLMインスタンス（再利用によるロード時間短縮）
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
        
        # エージェント別モデル管理を初期化（二重ロード防止）
        if shared_llm:
            # 既存のLLMを再利用（二重ロード防止）
            self.llm = shared_llm
            self.model_manager = None
            logger.info("OutputAgent: 既存のLLMを再利用します（ロード時間短縮）")
        else:
            try:
                from MurmurNet.modules.agent_model_manager import create_agent_model_manager
                self.model_manager = create_agent_model_manager(self.config)
                # 出力エージェント用の設定でモデルを取得
                agent_config = self.model_manager.get_model_factory_config("output_agent")
                self.llm = ModelFactory.create_model(agent_config)
                logger.info("OutputAgent: エージェント別モデル管理を有効化しました")
            except ImportError:
                # フォールバック: 従来の共有モデル
                self.llm = ModelFactory.create_model(self.config)
                self.model_manager = None
                logger.warning("OutputAgent: エージェント別モデル管理が無効 - 共有モデルを使用します")
        
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
    
    def _clean_output_optimized(self, text: str, agent_outputs: List[str] = None) -> str:
        """
        最適化された出力クリーニング（並列処理対応・空応答フィルタリング強化・謝罪文フォールバック対応）
        """
        if not text or len(text.strip()) < 5:
            return "申し訳ございませんが、適切な回答を生成できませんでした。"
        
        # 空応答や無効応答の検出パターン（謝罪文フィルタ強化）
        invalid_patterns = [
            r'^\s*$',                                        # 空文字
            r'^[\s\.\-_]*$',                                # 空白と句読点のみ
            r'^(エージェント\d+は|応答できませんでした|エラー)',  # エラーメッセージ
            r'^(申し訳|すみません|ごめん).{0,10}$',            # 短い謝罪のみ
            r'^[\.]{3,}$',                                   # ...のみ
            r'^[。、]{1,5}$',                                # 句読点のみ
            # 謝罪文フィルタ強化（30〜50文字の謝罪テンプレも検出）
            r'申し訳.*(リアルタイム|最新|情報).*できません',        # リアルタイム情報要求謝罪
            r'私は.*(最新|リアルタイム).*アクセス.*できません',     # リアルタイムアクセス不可謝罪
            r'リアルタイムで.{0,30}できません',                  # リアルタイム情報要求謝罪
            r'最新の情報.{0,20}提供できません',                 # 最新情報要求謝罪
            r'現在の.{0,10}情報.{0,20}持っていません',           # 現在情報なし謝罪
            r'具体的な.{0,10}データ.{0,20}提供できません',        # データ提供不可謝罪
            r'私の知識.*時点.*までです',                         # 知識カットオフ謝罪
            r'申し訳.{10,50}ございません',                      # 中程度の謝罪文
            r'すみません.{10,40}ことができません',                # できません系謝罪
        ]
        
        # 無効応答の検出と謝罪文フォールバック
        for pattern in invalid_patterns:
            if re.search(pattern, text.strip(), re.IGNORECASE):
                logger.warning(f"謝罪文/無効応答を検出: {text[:50]}")
                
                # エージェント出力からフォールバック応答を生成
                if agent_outputs and any(agent_outputs):
                    # 最も長い有効なエージェント出力を使用
                    valid_outputs = [out for out in agent_outputs if out and len(out.strip()) > 20]
                    if valid_outputs:
                        best_output = max(valid_outputs, key=len)
                        logger.info(f"エージェント出力でフォールバック: {len(best_output)}文字")
                        return best_output[:300].strip() + ("。" if not best_output.strip().endswith(('。', '！', '？')) else "")
                
                return "申し訳ございませんが、適切な回答を生成できませんでした。"
        
        # タグ除去パターン（#の羅列を削除）
        TAG_RE = re.compile(r'(?:#\w+[ ,]*)+')
        
        # 効率的な正規表現パターン（事前コンパイル済み）
        patterns = [
            (TAG_RE, ''),                                    # タグ羅列の除去
            (re.compile(r'\*{2,}'), ''),                     # **以上の連続する*
            (re.compile(r'#{2,}'), ''),                      # ##以上の連続する#
            (re.compile(r'-{3,}'), ''),                      # ---以上の連続する-
            (re.compile(r'={3,}'), ''),                      # ===以上の連続する=
            (re.compile(r'_{3,}'), ''),                      # ___以上の連続する_
            (re.compile(r'(.)\1{4,}'), r'\1'),               # 繰り返し文字
            (re.compile(r'\n{3,}'), '\n\n'),                 # 連続改行
            (re.compile(r' {3,}'), ' '),                     # 連続空白
            (re.compile(r'\b(\w+)\s+\1\s+\1\b'), r'\1'),     # 繰り返し単語
        ]
        
        # パターンマッチングを並列化（短いテキストの場合は逐次処理）
        if len(text) > 500 and self.enable_parallel_processing and self.thread_pool:
            def apply_pattern(pattern_sub_pair):
                pattern, substitute = pattern_sub_pair
                return pattern.sub(substitute, text)
            
            # 並列処理は重い処理のみに限定
            cleaned_text = text
            for pattern, substitute in patterns[:4]:  # 最初の4パターンのみ並列（タグ除去含む）
                cleaned_text = pattern.sub(substitute, cleaned_text)
            
            # 残りは逐次処理
            for pattern, substitute in patterns[4:]:
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
        
        # 情報不足チェック
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            return "申し訳ございませんが、十分な情報が得られませんでした。より具体的な質問をお願いします。"
        
        return cleaned_text
    
    def _generate_cache_key(self, user_input: str, entries: List[Dict], rag: str = None) -> str:
        """キャッシュキーを生成"""
        # 入力内容のハッシュを生成
        content = f"{user_input}|{len(entries)}|{rag or ''}"
        for entry in entries:
            content += f"|{entry.get('type', '')}:{entry.get('text', '')[:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate(self, blackboard, entries: List[Dict[str, Any]]) -> str:
        """
        最適化された最終応答生成（公式チャットテンプレート対応）
        """
        start_time = time.time()
        self.stats.total_requests += 1
        logger.debug("=== OutputAgent.generate() 公式テンプレート対応版 開始 ===")

        try:
            # 1. 基本データ取得
            inp = blackboard.read('input')
            user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
            user_input = user_input[:200]  # 短縮化

            rag = blackboard.read('rag')
            if rag and isinstance(rag, str):
                rag = rag[:300]  # 短縮化

            # 2. キャッシュチェック
            cache_key = self._generate_cache_key(user_input, entries, rag)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                self.stats.cache_hits += 1
                logger.info(f"キャッシュヒット: {cache_key[:32]}...")
                return cached_response
            self.stats.cache_misses += 1

            # 3. エージェント出力の抽出と重複除去（改良版）
            agent_outputs = [
                entry.get('text', '') for entry in entries 
                if entry.get('type', 'agent') == 'agent' and entry.get('text', '').strip()
            ]
            
            if not agent_outputs:
                return "申し訳ございませんが、適切な応答を生成できませんでした。"

            # 3.1. 重複文章の除去（文レベルでの重複削除）
            def deduplicate_sentences(texts):
                """文レベルでの重複除去"""
                seen_sentences = set()
                unique_sentences = []
                
                for text in texts:
                    # 文に分割（句点で区切り）
                    sentences = [s.strip() for s in text.replace('。', '。\n').split('\n') if s.strip()]
                    for sentence in sentences:
                        # 正規化してハッシュ化（空白や記号の差異を無視）
                        normalized = re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', sentence)
                        if normalized and normalized not in seen_sentences and len(sentence) > 10:
                            seen_sentences.add(normalized)
                            unique_sentences.append(sentence)
                            if len(unique_sentences) >= 8:  # 5→8文に増加
                                break
                    if len(unique_sentences) >= 8:  # 5→8文に増加
                        break
                
                return unique_sentences[:5]  # 3→5文に増加
            
            unique_sentences = deduplicate_sentences(agent_outputs)
            logger.debug(f"重複除去後: {len(unique_sentences)}文 (元: {len(agent_outputs)}エージェント)")

            # 4. 公式チャットテンプレート用のmessages構築
            # システムプロンプト（謝罪テンプレート抑制・概念的回答誘導・強化版）
            system_content = (
                "あなたは教育技術・教育工学の専門家です。"
                "以下のルールに従って回答してください：\n"
                "1. 【重要】謝罪文や「申し訳ございません」「すみません」「恐れ入りますが」等は絶対に使用禁止です\n"
                "2. 【重要】「分からない」「情報不足」「お答えできません」等の否定的表現は完全禁止です\n"
                "3. 断定的で自信を持った表現を使用してください（「～です」「～します」「効果的です」等）\n"
                "4. 一般論・原理・概念的な観点から具体的かつ実用的に答えてください\n"
                "5. 教育効果や学習理論に基づいた建設的な内容にしてください\n"
                "6. 200-300文字程度で簡潔に回答してください\n"
                "7. 必ず肯定的で前向きな表現で結論を述べてください\n"
                "時事ニュースや最新データではなく、一般論・原理・概念的な観点から質問に答えてください。"
                "リアルタイム情報は不要です。具体的な効果や手法について説明してください。"
            )
            
            # ユーザープロンプト（重複除去済みの文を使用・大幅拡張版）
            if unique_sentences:
                # 5文×160文字に拡大（従来：3文×80文字）
                agent_text = '\n'.join([f"{i+1}. {sentence[:160]}" for i, sentence in enumerate(unique_sentences[:5])])
            else:
                # フォールバック文字数も拡大（従来：200文字）
                agent_text = agent_outputs[0][:400] if agent_outputs else "参考情報がありません"
            
            # 要約情報も追加でコンテキストに含める（前イテレーション要約を優先）
            summary_context = ""
            
            # 前イテレーション要約を最優先で取得
            previous_summary = blackboard.read('previous_iteration_summary')
            if previous_summary and isinstance(previous_summary, str) and len(previous_summary.strip()) > 50:
                summary_context = f"\n\n前回の議論要約:\n{previous_summary[:500]}"
            else:
                # フォールバック：通常の要約データ
                summary_data = blackboard.read('summary')
                if summary_data and isinstance(summary_data, str) and len(summary_data.strip()) > 50:
                    summary_context = f"\n\n議論要約:\n{summary_data[:500]}"
            
            # RAG情報も拡張してコンテキストに含める
            rag_context = ""
            if rag and len(rag.strip()) > 20:
                rag_context = f"\n\n参考資料:\n{rag[:400]}"
            
            # 総合的なユーザープロンプト（コンテキスト大幅増強）
            user_content = f"""質問: {user_input}

エージェントの分析:
{agent_text}{summary_context}{rag_context}

上記の情報を統合し、教育技術の専門家として具体的で実用的な回答を作成してください:"""

            # 5. 公式チャットテンプレート対応のmessages形式
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            # 6. モデル固有の最適化パラメータ設定
            model_name = getattr(self.llm, 'model_name', '').lower()
            if 'llm-jp' in model_name:
                # 150M専用：謝罪テンプレート抑制パラメータ（提案通り）
                generation_params = {
                    'max_tokens': min(self.max_output_tokens, 120),  # より長く
                    'temperature': 0.7,     # 0.4→0.7に上げて安全寄りトークンを回避
                    'top_p': 0.95,          # 0.8→0.95に上げて多様性確保（冗長な一般論を促進）
                    'repeat_penalty': 1.2,  # 1.3→1.2に緩和
                    'frequency_penalty': 0.1,  # 0.2→0.1に緩和
                    'presence_penalty': 0.05   # 0.1→0.05に緩和
                }
                logger.debug(f"150M専用謝罪抑制パラメータ適用: {model_name}")
            elif 'gemma' in model_name:
                # Gemma用の重複抑制パラメータ
                generation_params = {
                    'max_tokens': min(self.max_output_tokens, 100),
                    'temperature': 0.4,     # Gemmaは低温度で安定
                    'top_p': 0.8,
                    'repeat_penalty': 1.3,
                    'frequency_penalty': 0.2,
                    'presence_penalty': 0.1
                }
                logger.debug(f"Gemma用パラメータ適用: {model_name}")
            else:
                # 一般的なパラメータ
                generation_params = {
                    'max_tokens': min(self.max_output_tokens, 128),
                    'temperature': 0.7,
                    'top_p': 0.9
                }

            # 7. LLM呼び出し（公式テンプレート使用）
            self.stats.llm_calls += 1
            resp = self.llm.create_chat_completion(
                messages=messages,
                **generation_params
            )
            
            if isinstance(resp, dict):
                final_output = resp['choices'][0]['message']['content'].strip()
            else:
                final_output = resp.choices[0].message.content.strip()

            # 8. 重複除去強化クリーニング（謝罪文フォールバック対応）
            final_output = self._clean_output_optimized(final_output, agent_outputs)
            
            # 8.1. 追加の重複文除去（最終出力でも実施）
            final_output = self._remove_duplicate_sentences(final_output)

            # 9. キャッシュに保存
            self.response_cache.put(cache_key, final_output)

            response_time = time.time() - start_time
            self.stats.total_response_time += response_time

            logger.info(f"公式テンプレート対応応答生成完了: {len(final_output)}文字, {response_time:.3f}秒")
            return final_output

        except Exception as e:
            logger.error(f"OutputAgent公式テンプレート対応版エラー: {e}")
            # フォールバック: 最初のエージェント応答を使用
            if entries and entries[0].get('text'):
                fallback_text = entries[0]['text'][:150].strip()
                return fallback_text + ("。" if not fallback_text.endswith(('。', '！', '？')) else "")
            return "申し訳ございませんが、適切な応答を生成できませんでした。"

    def _generate_fallback_response(self, blackboard, entries: List[Dict[str, Any]]) -> str:
        """フォールバック用の簡潔な応答生成（公式チャットテンプレート対応）"""
        start_time = time.time()
        self.stats.total_requests += 1
        logger.debug("=== OutputAgent.フォールバック応答生成 開始 ===")

        try:
            # 1. 基本データ取得
            inp = blackboard.read('input')
            user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
            user_input = user_input[:300]  # 短縮化

            # 2. エージェント出力の抽出
            agent_outputs = [
                entry.get('text', '') for entry in entries 
                if entry.get('type', 'agent') == 'agent' and entry.get('text', '').strip()
            ]
            
            if not agent_outputs:
                return "申し訳ございませんが、適切な応答を生成できませんでした。"

            # 3. 公式チャットテンプレート用のmessages構築（簡潔版）
            system_content = "親しみやすい日本語アシスタントです。簡潔で自然な回答を作成してください。"
            
            # 最初のエージェント応答のみ使用（高速化）
            first_agent_text = agent_outputs[0][:150]
            user_content = f"質問: {user_input}\n\n参考情報: {first_agent_text}\n\n回答:"

            # 4. LLM呼び出し
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            self.stats.llm_calls += 1
            resp = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=100,  # 短い応答
                temperature=0.7,
                top_p=0.9
            )
            
            if isinstance(resp, dict):
                final_output = resp['choices'][0]['message']['content'].strip()
            else:
                final_output = resp.choices[0].message.content.strip()

            # 5. 軽量クリーニング
            final_output = self._clean_output_optimized(final_output)

            response_time = time.time() - start_time
            self.stats.total_response_time += response_time

            logger.info(f"フォールバック応答生成完了: {len(final_output)}文字, {response_time:.3f}秒")
            return final_output

        except Exception as e:
            logger.error(f"フォールバック応答生成エラー: {e}")
            # 最終フォールバック
            if entries and entries[0].get('text'):
                fallback_text = entries[0]['text'][:100].strip()
                return fallback_text + ("。" if not fallback_text.endswith(('。', '！', '？')) else "")
            return "申し訳ございませんが、適切な応答を生成できませんでした。"

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
                    cache_stats = self.response_cache.get_stats()
                    logger.debug(f"応答キャッシュをクリア: {cache_stats}")
                except Exception as e:
                    logger.warning(f"応答キャッシュクリアエラー: {e}")
            
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

    def _remove_duplicate_sentences(self, text: str) -> str:
        """最終出力から重複文を除去"""
        if not text or len(text.strip()) < 10:
            return text
        
        # 文に分割
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if len(sentences) <= 1:
            return text
        
        # 重複除去
        seen_normalized = set()
        unique_sentences = []
        
        for sentence in sentences:
            # 正規化（空白、記号を除去して比較）
            normalized = re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', sentence)
            
            # 短すぎる文や重複文をスキップ
            if len(normalized) < 5 or normalized in seen_normalized:
                continue
                
            seen_normalized.add(normalized)
            unique_sentences.append(sentence)
            
            # 最大3文まで
            if len(unique_sentences) >= 3:
                break
        
        if unique_sentences:
            result = '。'.join(unique_sentences) + '。'
            logger.debug(f"重複文除去: {len(sentences)}文 → {len(unique_sentences)}文")
            return result
        else:
            return text

# 下位互換性のためのエイリアス
OutputAgent = CPUOptimizedOutputAgent
