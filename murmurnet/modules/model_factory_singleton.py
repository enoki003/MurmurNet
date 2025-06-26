#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet モデルファクトリモジュール（Singleton対応版）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SingletonModelManagerを使用したLlamaモデル統合

作者: Yuhi Sonoki
"""

import logging
import os
import psutil
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
        
        # SingletonModelManagerを使用
        from MurmurNet.modules.model_manager import get_singleton_manager
        self.model_manager = get_singleton_manager(config)
        
        # 後方互換性のための属性
        self.model_path = config.get('model_path')
        self.n_ctx = config.get('n_ctx', 2048)
        self.n_threads = config.get('n_threads', 4)
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 256)
        self.chat_template = config.get('chat_template')
        self._initialization_attempted = True
        self._initialization_error = None

    def _ensure_initialized(self):
        """後方互換性のため"""
        pass
        
    def _check_prerequisites(self) -> bool:
        """後方互換性のため"""
        return self.model_manager.is_available()

    def _init_model(self):
        """後方互換性のため"""
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """SingletonModelManagerを使用してテキスト生成"""
        return self.model_manager.generate(prompt, **kwargs)

    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """チャット形式でのテキスト生成"""
        # メッセージを単一プロンプトに変換
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nassistant:"
        
        # 通常のgenerate()を使用
        response = self.generate(prompt, **kwargs)
        
        # ChatCompletion形式で返却
        return {
            'choices': [{
                'message': {
                    'content': response,
                    'role': 'assistant'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': len(response.split()),
                'total_tokens': len(prompt.split()) + len(response.split())
            }
        }

    def is_available(self) -> bool:
        """モデルが利用可能かチェック"""
        return self.model_manager.is_available()

class TransformersModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = config.get('device', 'cpu')
        self.max_length = config.get('max_length', 256)
        self.temperature = config.get('temperature', 0.7)
        self._tokenizer = None
        self._model = None
        self._initialization_attempted = False
        self._initialization_error = None

    def _ensure_initialized(self):
        if not self._initialization_attempted:
            self._initialization_attempted = True
            if self._check_prerequisites():
                self._init_model()

    def _check_prerequisites(self) -> bool:
        if not HAS_TRANSFORMERS:
            self.logger.error("transformers ライブラリがインストールされていません")
            return False
        return True

    def _init_model(self):
        try:
            self.logger.info(f"Transformersモデル読み込み開始: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self.logger.info(f"✓ Transformersモデルの初期化が完了しました: {self.model_name}")
        except Exception as e:
            self.logger.error(f"✗ Transformersモデル初期化エラー: {e}")
            self._initialization_error = str(e)
            self._tokenizer = None
            self._model = None

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_initialized()
        if not self._model or not self._tokenizer:
            error_msg = "Transformersモデルが利用できません。"
            if self._initialization_error:
                error_msg += f" エラー: {self._initialization_error}"
            return error_msg

        try:
            inputs = self._tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            temperature = kwargs.get('temperature', self.temperature)
            max_length = kwargs.get('max_length', self.max_length)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            return response if response else "申し訳ありませんが、適切な応答を生成できませんでした。"
        except Exception as e:
            self.logger.error(f"テキスト生成エラー: {e}")
            return "申し訳ありませんが、現在応答を生成できません。"

    def is_available(self) -> bool:
        if not HAS_TRANSFORMERS:
            return False
        self._ensure_initialized()
        return self._model is not None and self._tokenizer is not None

class ModelFactory:
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get('model_type', 'llama')
        
        if model_type == 'llama':
            return LlamaModel(config)
        elif model_type == 'transformers':
            return TransformersModel(config)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

    @staticmethod
    def get_best_available_model(config: Dict[str, Any]) -> BaseModel:
        model = ModelFactory.create_model(config)
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
            raise RuntimeError("Llamaモデルの初期化に失敗しました。ログを確認してください。")
        raise RuntimeError(f"モデルタイプ '{model_type}' は利用できません。")

    @staticmethod
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
