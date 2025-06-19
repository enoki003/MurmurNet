#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet モデルファクトリモジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
様々な小型言語モデルの統一インターフェースを提供

作者: Yuhi Sonoki
"""

import logging
import os
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod

# 条件付きインポート
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    Llama = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


# モデルキャッシュ用のグローバル変数とLRU機能
_MODEL_CACHE = {}
_CACHE_LOCK = None
_CACHE_MAX_SIZE = 3  # 最大3つのモデルをキャッシュ
_CACHE_ACCESS_ORDER = []  # LRU追跡用

try:
    import threading
    _CACHE_LOCK = threading.RLock()  # 再帰ロックに変更（デッドロック対策）
except ImportError:
    # スレッドロックが利用できない場合は単純な実装
    class DummyLock:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    _CACHE_LOCK = DummyLock()


def clear_model_cache():
    """モデルキャッシュをクリア（RAM節約のため）"""
    global _MODEL_CACHE, _CACHE_ACCESS_ORDER
    with _CACHE_LOCK:
        # メモリ解放のため明示的にdeleteを呼び出し
        for model in _MODEL_CACHE.values():
            if hasattr(model, '_llm') and model._llm:
                del model._llm
        _MODEL_CACHE.clear()
        _CACHE_ACCESS_ORDER.clear()
        logging.info("モデルキャッシュをクリアしました")


def _manage_cache_size():
    """LRUアルゴリズムでキャッシュサイズを管理（内部メソッド）"""
    global _MODEL_CACHE, _CACHE_ACCESS_ORDER
    
    while len(_MODEL_CACHE) > _CACHE_MAX_SIZE:
        # 最も古いエントリを除去
        if _CACHE_ACCESS_ORDER:
            oldest_key = _CACHE_ACCESS_ORDER.pop(0)
            if oldest_key in _MODEL_CACHE:
                old_model = _MODEL_CACHE.pop(oldest_key)
                # メモリ解放
                if hasattr(old_model, '_llm') and old_model._llm:
                    del old_model._llm
                logging.info(f"LRU: 古いモデルをキャッシュから削除しました: {oldest_key}")


def _update_cache_access(cache_key: str):
    """キャッシュアクセス順序を更新（内部メソッド）- ロック不要"""
    global _CACHE_ACCESS_ORDER
    
    if cache_key in _CACHE_ACCESS_ORDER:
        _CACHE_ACCESS_ORDER.remove(cache_key)
    _CACHE_ACCESS_ORDER.append(cache_key)


def _get_model_cache_key(config: Dict[str, Any]) -> str:
    """モデルキャッシュ用のキーを生成"""
    model_path = config.get('model_path', '')
    model_type = config.get('model_type', '')
    n_ctx = config.get('n_ctx', 1024)  # デフォルト1024に変更（高速化）
    cache_key = f"{model_type}:{model_path}:{n_ctx}"
    logging.info(f"モデルキャッシュキー生成: {cache_key}")  # DEBUGからINFOに変更
    return cache_key


class BaseModel(ABC):
    """言語モデルの基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('MurmurNet.Model')
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成の抽象メソッド
        
        引数:
            prompt: 入力プロンプト
            **kwargs: 追加パラメータ
            
        戻り値:
            生成されたテキスト
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        モデルが利用可能かどうかをチェック
        
        戻り値:
            利用可能ならTrue
        """        
        pass


class LlamaModel(BaseModel):
    """
    Llama.cpp を使用するローカルモデル
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get('model_path')
        self.n_ctx = config.get('n_ctx', 1024)  # CLI引数を正しく反映（デフォルト1024）
        self.n_threads = config.get('n_threads', 4)
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 256)
        self.chat_template = config.get('chat_template')
        
        # Llamaモデルの初期化
        self._llm = None
        self._initialization_attempted = False
        self._initialization_error = None
          # 遅延初期化：最初の使用時に初期化する
        # これにより循環参照の問題を回避
    
    def _ensure_initialized(self):
        """モデルの初期化を確実に行う（遅延初期化）"""
        if not self._initialization_attempted:
            self.logger.info("遅延初期化: 最初の使用時にモデルロードを開始します")
            self._initialization_attempted = True
            if self._check_prerequisites():
                self._init_model()
        else:
            if self._llm is None:
                self.logger.warning("モデル初期化が既に試行されましたが失敗しています")
    
    def _check_prerequisites(self) -> bool:
        """初期化の前提条件をチェック（内部メソッド）"""
        if not HAS_LLAMA_CPP:
            self.logger.error("llama-cpp-python がインストールされていません")
            return False
        if not self.model_path:
            self.logger.error("model_pathが設定されていません")
            return False
        if not os.path.exists(self.model_path):
            self.logger.error(f"モデルファイルが見つかりません: {self.model_path}")
            return False
        return True
    
    def _init_model(self):
        """Llamaモデルを初期化（内部メソッド）"""
        import time
        start_time = time.time()        
        try:
            self.logger.info(f"Llamaモデルの実際のロード開始: {self.model_path}")
            llama_kwargs = {
                'model_path': self.model_path,
                'n_ctx': self.n_ctx,
                'n_threads': self.n_threads,
                'n_batch': self.config.get('n_batch', 64),  # バッチサイズ拡大（5600G対応）
                'use_mmap': self.config.get('use_mmap', True),
                'use_mlock': self.config.get('use_mlock', False),
                'logits_all': False,  # メモリ節約（全トークンlogits保持しない）
                'n_gpu_layers': 0,
                'seed': 42,
                'chat_format': "gemma",
                'verbose': False
            }
            
            self.logger.debug(f"Llamaパラメータ: {llama_kwargs}")
            
            # チャットテンプレートの設定
            if self.chat_template and os.path.exists(self.chat_template):
                try:
                    with open(self.chat_template, 'r', encoding='utf-8') as f:
                        llama_kwargs['chat_template'] = f.read()
                        self.logger.debug("チャットテンプレートを読み込みました")
                except Exception as e:
                    self.logger.warning(f"チャットテンプレート読み込みエラー: {e}")
            
            # 実際のモデルロード
            self.logger.info("Llamaライブラリによるモデルロード中...")
            self._llm = Llama(**llama_kwargs)
            
            load_time = time.time() - start_time
            self.logger.info(f"Llamaモデルを初期化しました: {self.model_path} ({load_time:.2f}秒)")
            
        except Exception as e:
            load_time = time.time() - start_time
            self.logger.error(f"Llamaモデル初期化エラー ({load_time:.2f}秒): {e}")
            self._initialization_error = str(e)
            self._llm = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Llamaモデルを使用してテキストを生成
        
        引数:
            prompt: 入力プロンプト
            **kwargs: 追加パラメータ
            
        戻り値:
            生成されたテキスト
        """
        # 遅延初期化を確実に実行
        self._ensure_initialized()
        
        if not self._llm:
            error_msg = "Llamaモデルが利用できません。"
            if self._initialization_error:
                error_msg += f" エラー: {self._initialization_error}"
            return error_msg
        
        try:
            # パラメータの設定
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            top_p = kwargs.get('top_p', 0.9)
            top_k = kwargs.get('top_k', 40)
            
            # テキスト生成
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False,
                stop=["</s>", "<|end|>", "\n\n"]
            )
            
            # 生成されたテキストを取得
            if isinstance(output, dict) and 'choices' in output:
                generated_text = output['choices'][0]['text'].strip()
            else:
                generated_text = str(output).strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"テキスト生成エラー: {e}")
            return "申し訳ありませんが、現在応答を生成できません。"
    
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット完了機能
        
        引数:
            messages: メッセージのリスト
            **kwargs: 追加パラメータ
            
        戻り値:
            チャット完了のレスポンス
        """        # 遅延初期化を確実に実行
        self._ensure_initialized()
        
        if not self._llm:
            error_msg = "Llamaモデルが利用できません。"
            if self._initialization_error:
                error_msg += f" エラー: {self._initialization_error}"
            
            return {
                'choices': [{
                    'message': {
                        'content': error_msg,
                        'role': 'assistant'
                    },
                    'finish_reason': 'error'
                }],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
        
        try:
            # パラメータの設定（研究に基づく最適化）
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            top_p = kwargs.get('top_p', 0.95)  # 研究推奨値
            repeat_penalty = kwargs.get('repeat_penalty', 1.2)  # 繰り返し抑制
            
            # チャット完了を実行（出力崩壊防止パラメータ付き）
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,  # 繰り返しペナルティ追加
                stream=False
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"チャット完了エラー: {e}")
            return {
                'choices': [{
                    'message': {
                        'content': f"エラーが発生しました: {str(e)}",
                        'role': 'assistant'
                    },
                    'finish_reason': 'error'
                }],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
    
    def is_available(self) -> bool:
        """Llamaモデルが利用可能かチェック"""
        # 基本的な前提条件をチェック
        if not HAS_LLAMA_CPP:
            return False
        if not self.model_path:
            return False
        if not os.path.exists(self.model_path):
            return False
        
        # 遅延初期化を試行（まだ試行していない場合）
        self._ensure_initialized()
        
        # 実際のモデルインスタンスが正常に初期化されているかもチェック
        return self._llm is not None


class ModelFactory:
    """
    言語モデルのファクトリクラス
    設定に基づいて適切なモデルインスタンスを作成
    """
      # 利用可能なモデルタイプ
    MODEL_TYPES = {
        'llama': LlamaModel,
        'local': LlamaModel,  # localは LlamaModel にマッピング
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        """
        設定に基づいてモデルインスタンスを作成
        
        引数:
            config: モデル設定辞書
            
        戻り値:
            作成されたモデルインスタンス
        """
        model_type = config.get('model_type', 'llama').lower()
        
        # 未知のモデルタイプのハンドリング
        if model_type not in cls.MODEL_TYPES:
            logging.warning(f"未知のモデルタイプ: {model_type}. llamaモデルを試行します。")
            model_type = 'llama'
        
        model_class = cls.MODEL_TYPES[model_type]
          # LRUキャッシュによるモデルインスタンスの管理
        cache_key = _get_model_cache_key(config)
        
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:
                logging.info(f"新しいモデルインスタンスを作成中: {cache_key}")
                start_time = __import__('time').time()
                
                # キャッシュサイズ管理（新しいモデル追加前に）
                _manage_cache_size()
                
                _MODEL_CACHE[cache_key] = model_class(config)
                load_time = __import__('time').time() - start_time
                logging.info(f"モデルインスタンス作成完了: {load_time:.2f}秒（注：実際のモデルロードは遅延初期化）")
            else:
                logging.info(f"キャッシュされたモデルを再利用: {cache_key}")
        
        # ロックの外でLRUアクセス順序を更新（デッドロック回避）
        _update_cache_access(cache_key)
        
        return _MODEL_CACHE[cache_key]
    
    @classmethod
    def get_available_models(cls, configs: List[Dict[str, Any]]) -> List[BaseModel]:
        """
        利用可能なモデルのリストを取得
        
        引数:
            configs: モデル設定のリスト
            
        戻り値:
            利用可能なモデルインスタンスのリスト
        """
        available_models = []
        
        for config in configs:
            try:
                model = cls.create_model(config)
                if model.is_available():
                    available_models.append(model)
                else:
                    logging.warning(f"モデルが利用できません: {config.get('model_type', 'unknown')}")
            except Exception as e:
                logging.error(f"モデル作成エラー: {e}")
        
        return available_models
    
    @classmethod
    def get_best_available_model(cls, config: Dict[str, Any]) -> BaseModel:
        """
        利用可能な最適なモデルを取得
        
        引数:
            config: 基本設定辞書
            
        戻り値:
            利用可能な最適なモデルインスタンス
        """
        # まず設定されたモデルタイプを試行
        model = cls.create_model(config)
        if model.is_available():
            logging.info(f"設定されたモデルを使用します: {config.get('model_type', 'llama')}")
            return model
        
        # 設定されたモデルが利用できない場合、詳細なエラー情報を提供
        model_type = config.get('model_type', 'llama')
        if model_type == 'llama':
            if not HAS_LLAMA_CPP:
                raise RuntimeError("llama-cpp-pythonがインストールされていません。pip install llama-cpp-pythonを実行してください。")
            if not config.get('model_path'):
                raise RuntimeError("model_pathが設定されていません。config.yamlを確認してください。")
            if not os.path.exists(config.get('model_path', '')):
                raise RuntimeError(f"モデルファイルが見つかりません: {config.get('model_path')}")
            
            # 詳細なエラー情報を取得
            if hasattr(model, '_initialization_error') and model._initialization_error:
                raise RuntimeError(f"Llamaモデルの初期化に失敗しました: {model._initialization_error}")
            else:
                raise RuntimeError("Llamaモデルの初期化に失敗しました。ログを確認してください。")
        
        # 他のエラー
        raise RuntimeError(f"モデルタイプ '{model_type}' は利用できません。")


# デフォルト設定
DEFAULT_MODEL_CONFIGS = [
    {
        'model_type': 'llama',
        'model_path': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                  '../../models/gemma-3-1b-it-q4_0.gguf')),
        'n_ctx': 1024,  # デフォルト1024（高速化）
        'n_threads': 4,
        'temperature': 0.7,
        'max_tokens': 256,
        'chat_template': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                     '../../models/gemma3_template.txt'))
    }
]


def get_default_model() -> BaseModel:
    """
    デフォルトのモデルインスタンスを取得
    
    戻り値:
        利用可能な最適なモデル
    """
    return ModelFactory.get_best_available_model(DEFAULT_MODEL_CONFIGS[0])
