#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet モデルファクトリモジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# モデルキャッシュ用のグローバル変数
_MODEL_CACHE = {}
_CACHE_LOCK = None

try:
    import threading
    _CACHE_LOCK = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    _CACHE_LOCK = DummyLock()

def _get_model_cache_key(config: Dict[str, Any]) -> str:
    model_path = config.get('model_path', '')
    model_type = config.get('model_type', '')
    n_ctx = config.get('n_ctx', 2048)
    return f"{model_type}:{model_path}:{n_ctx}"

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('MurmurNet.Model')

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class LlamaModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get('model_path')
        self.n_ctx = config.get('n_ctx', 2048)
        self.n_threads = config.get('n_threads', 4)
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 256)
        self.chat_template = config.get('chat_template')
        self._llm = None
        self._initialization_attempted = False
        self._initialization_error = None

    def _ensure_initialized(self):
        if not self._initialization_attempted:
            self._initialization_attempted = True
            if self._check_prerequisites():
                self._init_model()
                
    def _check_prerequisites(self) -> bool:
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
        try:
            self.logger.info(f"モデル読み込み開始: {self.model_path}")
            self.logger.info("モデルサイズによっては30秒以上かかる場合があります...")
            
            llama_kwargs = {
                'model_path': self.model_path,
                'n_ctx': self.n_ctx,
                'n_threads': self.n_threads,
                'use_mmap': True,
                'use_mlock': False,
                'n_gpu_layers': 0,
                'seed': 42,
                'chat_format': "gemma",
                'verbose': True  # 進捗表示を有効化
            }
            if self.chat_template and os.path.exists(self.chat_template):
                try:
                    with open(self.chat_template, 'r', encoding='utf-8') as f:
                        llama_kwargs['chat_template'] = f.read()
                except Exception as e:
                    self.logger.warning(f"チャットテンプレート読み込みエラー: {e}")
            
            self.logger.info("Llamaインスタンスを作成中...")
            self._llm = Llama(**llama_kwargs)
            self.logger.info(f"✓ Llamaモデルの初期化が完了しました: {self.model_path}")
        except Exception as e:
            self.logger.error(f"✗ Llamaモデル初期化エラー: {e}")
            self._initialization_error = str(e)
            self._llm = None

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_initialized()
        if not self._llm:
            error_msg = "Llamaモデルが利用できません。"
            if self._initialization_error:
                error_msg += f" エラー: {self._initialization_error}"
            return error_msg
        try:
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            top_p = kwargs.get('top_p', 0.9)
            top_k = kwargs.get('top_k', 40)
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False,
                stop=["</s>", "<|end|>", "\n\n"]
            )
            if isinstance(output, dict) and 'choices' in output:
                generated_text = output['choices'][0]['text'].strip()
            else:
                generated_text = str(output).strip()
            return generated_text
        except Exception as e:
            self.logger.error(f"テキスト生成エラー: {e}")
            return "申し訳ありませんが、現在応答を生成できません。"

    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
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
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            top_p = kwargs.get('top_p', 0.95)
            repeat_penalty = kwargs.get('repeat_penalty', 1.2)
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
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
        if not HAS_LLAMA_CPP:
            return False
        if not self.model_path:
            return False
        if not os.path.exists(self.model_path):
            return False
        self._ensure_initialized()
        return self._llm is not None

class ModelFactory:
    MODEL_TYPES = {
        'llama': LlamaModel,
        'local': LlamaModel,
    }

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        model_type = config.get('model_type', 'llama').lower()
        if model_type not in cls.MODEL_TYPES:
            logging.warning(f"未知のモデルタイプ: {model_type}. llamaモデルを試行します。")
            model_type = 'llama'
        model_class = cls.MODEL_TYPES[model_type]
        cache_key = _get_model_cache_key(config)
        
        # キャッシュから取得を試行
        with _CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                logging.info(f"キャッシュからモデルを取得: {cache_key}")
                return _MODEL_CACHE[cache_key]
        
        # ロックの外で重い処理（モデル読み込み）を実行
        logging.info(f"新しいモデルインスタンスを作成中: {model_type}")
        model = model_class(config)
        
        # 作成完了後にキャッシュに保存
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:  # 二重チェック
                _MODEL_CACHE[cache_key] = model
                logging.info(f"モデルをキャッシュに保存: {cache_key}")
            else:
                # 他のスレッドが先にキャッシュした場合はそちらを返す
                logging.info(f"他のスレッドが先にキャッシュしたモデルを使用: {cache_key}")
                return _MODEL_CACHE[cache_key]
        return model

    @classmethod
    def get_available_models(cls, configs: List[Dict[str, Any]]) -> List[BaseModel]:
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
        model = cls.create_model(config)
        if model.is_available():
            logging.info(f"設定されたモデルを使用します: {config.get('model_type', 'llama')}")
            return model
        model_type = config.get('model_type', 'llama')
        if model_type == 'llama':
            if not HAS_LLAMA_CPP:
                raise RuntimeError("llama-cpp-pythonがインストールされていません。pip install llama-cpp-pythonを実行してください。")
            if not config.get('model_path'):
                raise RuntimeError("model_pathが設定されていません。config.yamlを確認してください。")
            if not os.path.exists(config.get('model_path', '')):
                raise RuntimeError(f"モデルファイルが見つかりません: {config.get('model_path')}")
            if hasattr(model, '_initialization_error') and model._initialization_error:
                raise RuntimeError(f"Llamaモデルの初期化に失敗しました: {model._initialization_error}")
            else:
                raise RuntimeError("Llamaモデルの初期化に失敗しました。ログを確認してください。")
        raise RuntimeError(f"モデルタイプ '{model_type}' は利用できません。")    @staticmethod
    def get_shared_model(config: Dict[str, Any]) -> 'BaseModel':
        cache_key = _get_model_cache_key(config)
        
        # キャッシュから取得を試行
        with _CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                logging.info(f"共有モデルインスタンスをキャッシュから取得: {cache_key}")
                return _MODEL_CACHE[cache_key]
        
        # キャッシュにない場合は新規作成（ロックの外で実行）
        logging.info(f"新しい共有モデルインスタンスを作成中: {cache_key}")
        model = ModelFactory.create_model(config)
        
        # 作成完了後にキャッシュに保存
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:  # 二重チェック
                _MODEL_CACHE[cache_key] = model
                logging.info(f"新しい共有モデルインスタンスをキャッシュに保存: {cache_key}")
            else:
                # 他のスレッドが先にキャッシュした場合はそちらを返す
                logging.info(f"他のスレッドが先にキャッシュした共有モデルを使用: {cache_key}")
                return _MODEL_CACHE[cache_key]
        
        return model

# デフォルト設定
DEFAULT_MODEL_CONFIGS = [
    {
        'model_type': 'llama',
        'model_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')),
        'n_ctx': 2048,
        'n_threads': 4,
        'temperature': 0.7,
        'max_tokens': 256,
        'chat_template': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gemma3_template.txt'))
    }
]

def get_default_model() -> BaseModel:
    return ModelFactory.get_best_available_model(DEFAULT_MODEL_CONFIGS[0])
