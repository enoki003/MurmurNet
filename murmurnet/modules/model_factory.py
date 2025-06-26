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
            
            # CPU最適化設定
            cpu_count_physical = psutil.cpu_count(logical=False) or 4
            cpu_count_logical = psutil.cpu_count(logical=True) or 8
            
            # CPUスレッド数を動的に調整
            optimal_threads = min(self.n_threads, cpu_count_logical)
            self.logger.info(f"CPU最適化: {optimal_threads}スレッド使用 (物理:{cpu_count_physical}, 論理:{cpu_count_logical})")
            
            llama_kwargs = {
                'model_path': self.model_path,
                'n_ctx': self.n_ctx,
                'n_threads': optimal_threads,
                'use_mmap': True,
                'use_mlock': False,
                'n_gpu_layers': 0,  # GPU使用禁止
                'seed': 42,
                'chat_format': "gemma",
                'verbose': True,  # 進捗表示を有効化
                # CPU最適化パラメータ
                'n_batch': 512,  # バッチサイズ最適化
                'last_n_tokens_size': 64,  # コンテキスト最適化
                'rope_scaling_type': 0,  # RoPE最適化
                'rope_freq_base': 10000.0,
                'rope_freq_scale': 1.0,
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

class HuggingFaceModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('huggingface_model_name', 'llm-jp/llm-jp-3-150m')
        self.device = config.get('device', 'cpu')
        self.torch_dtype = config.get('torch_dtype', 'auto')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 256)
        self._model = None
        self._tokenizer = None
        self._initialization_attempted = False
        self._initialization_error = None
          # キャッシュディレクトリの設定
        self.cache_dir = config.get('model_cache_dir', 'cache/models')
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _ensure_initialized(self):
        if not self._initialization_attempted:
            self._initialization_attempted = True
            if not HAS_TRANSFORMERS:
                self.logger.error("transformersライブラリがインストールされていません")
                self._initialization_error = "transformersライブラリが必要です"
                return
                
            self._init_model()
    
    def _init_model(self):
        try:
            self.logger.info(f"HuggingFaceモデル読み込み開始: {self.model_name}")
            
            # トークナイザーの読み込み
            self.logger.info("トークナイザーを読み込み中...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # パッドトークンの設定
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # モデルの読み込み
            self.logger.info("モデルを読み込み中...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != 'cpu' else None,
                trust_remote_code=True
            )
            
            # CPUの場合は明示的にCPUに移動
            if self.device == 'cpu':
                self._model = self._model.to('cpu')
            
            self.logger.info(f"✓ HuggingFaceモデルの初期化が完了しました: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"✗ HuggingFaceモデル初期化エラー: {e}")
            self._initialization_error = str(e)
            self._model = None
            self._tokenizer = None

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_initialized()
        if not self._model or not self._tokenizer:
            error_msg = "HuggingFaceモデルが利用できません。"
            if self._initialization_error:
                error_msg += f" エラー: {self._initialization_error}"
            return error_msg
            
        try:
            # パラメータの設定
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            top_p = kwargs.get('top_p', 0.95)
              # プロンプトのトークン化
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            # token_type_idsを除去（一部のモデルで不要）
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            if self.device == 'cpu':
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成設定
            generation_config = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True if temperature > 0 else False,
                'pad_token_id': self._tokenizer.pad_token_id,
                'eos_token_id': self._tokenizer.eos_token_id,
                'repetition_penalty': kwargs.get('repeat_penalty', 1.1)
            }
            
            # 推論実行
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_config
                )
              # 結果のデコード
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"HuggingFace生成エラー: {e}")
            return f"エラーが発生しました: {str(e)}"

    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self.chat_completion(messages, **kwargs)    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        self._ensure_initialized()
        if not self._model or not self._tokenizer:
            error_msg = "HuggingFaceモデルが利用できません。"
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
            # メッセージをプロンプトに変換（シンプルな形式を使用）
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt += f"ユーザー: {content}\n"
                elif role == 'assistant':
                    prompt += f"アシスタント: {content}\n"
                elif role == 'system':
                    prompt += f"システム: {content}\n"
            
            # 最後にアシスタントの応答を促すプロンプトを追加
            if not prompt.endswith("アシスタント: "):
                prompt += "アシスタント: "
            
            # 生成実行
            response_text = self.generate(prompt, **kwargs)
            
            return {
                'choices': [{
                    'message': {
                        'content': response_text,
                        'role': 'assistant'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': len(response_text.split()),
                    'total_tokens': len(prompt.split()) + len(response_text.split())
                }
            }
            
        except Exception as e:
            self.logger.error(f"HuggingFace chat completion エラー: {e}")
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
        if not HAS_TRANSFORMERS:
            return False
        self._ensure_initialized()
        return self._model is not None and self._tokenizer is not None

class ModelFactory:
    MODEL_TYPES = {
        'llama': LlamaModel,
        'local': LlamaModel,
        'huggingface': HuggingFaceModel,
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
    },
    {
        'model_type': 'huggingface',
        'huggingface_model_name': 'llm-jp/llm-jp-3-150m',
        'device': 'cpu',
        'torch_dtype': 'auto',
        'temperature': 0.7,
        'max_tokens': 256,
        'model_cache_dir': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cache/models'))
    }
]

def get_default_model() -> BaseModel:
    return ModelFactory.get_best_available_model(DEFAULT_MODEL_CONFIGS[0])
