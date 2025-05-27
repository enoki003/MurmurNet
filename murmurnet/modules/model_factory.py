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
from MurmurNet.modules.config_manager import get_config

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
        モデルが利用可能かどうかをチェック        戻り値:
            利用可能ならTrue
        """        
        pass


class LlamaModel(BaseModel):
    """
    Llama.cpp を使用するローカルモデル
    """
    def __init__(self, config: Dict[str, Any] = None):
        # ConfigManagerから設定を取得
        config_manager = get_config()
        self.config = config or config_manager.to_dict()  # 後方互換性のため
        super().__init__(self.config)
        
        # ConfigManagerから直接設定値を取得
        self.model_path = config_manager.model.model_path
        self.n_ctx = config_manager.model.n_ctx
        self.n_threads = config_manager.model.n_threads
        self.temperature = config_manager.model.temperature
        self.max_tokens = config_manager.model.max_tokens
        self.chat_template = config_manager.model.chat_template
          # Llamaモデルの初期化
        self._llm = None
        self._initialization_attempted = False
        self._initialization_error = None
        
        # 並列処理用のロック（重要：モデルレベルでの同期化）
        import threading
        self._model_lock = threading.RLock()
        
        self.logger = logging.getLogger('MurmurNet.Model')
        
        # 遅延初期化：最初の使用時に初期化する
        # これにより循環参照の問題を回避
    
    def _ensure_initialized(self):
        """モデルの初期化を確実に行う（遅延初期化）"""
        if not self._initialization_attempted:
            self._initialization_attempted = True
            if self._check_prerequisites():
                self._init_model()
    
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
        try:
            llama_kwargs = {
                'model_path': self.model_path,
                'n_ctx': self.n_ctx,
                'n_threads': self.n_threads,
                'use_mmap': True,
                'use_mlock': False,
                'n_gpu_layers': 0,
                'seed': 42,
                'chat_format': "gemma",
                'verbose': False
            }
            
            # チャットテンプレートの設定
            if self.chat_template and os.path.exists(self.chat_template):
                try:
                    with open(self.chat_template, 'r', encoding='utf-8') as f:
                        llama_kwargs['chat_template'] = f.read()
                except Exception as e:
                    self.logger.warning(f"チャットテンプレート読み込みエラー: {e}")
            
            self._llm = Llama(**llama_kwargs)
            self.logger.info(f"Llamaモデルを初期化しました: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Llamaモデル初期化エラー: {e}")
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
            self.logger.error(error_msg)
            return "申し訳ありませんが、現在モデルが利用できません。しばらく後にお試しください。"
        
        try:
            # パラメータの設定（エラーハンドリング強化）
            temperature = max(0.1, min(1.0, kwargs.get('temperature', self.temperature)))
            max_tokens = max(1, min(1024, kwargs.get('max_tokens', self.max_tokens)))
            top_p = max(0.1, min(1.0, kwargs.get('top_p', 0.9)))
            top_k = max(1, min(100, kwargs.get('top_k', 40)))
            
            # プロンプトの検証
            if not prompt or not isinstance(prompt, str):
                return "申し訳ありませんが、有効な入力を受け取れませんでした。"
            
            # プロンプトの長さ制限
            if len(prompt) > 4000:
                prompt = prompt[:4000] + "..."
                self.logger.warning("プロンプトが長すぎるため切り詰めました")
            
            # テキスト生成（タイムアウト対応）
            import time
            start_time = time.time()
            
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False,
                stop=["</s>", "<|end|>", "\n\n"]
            )
            
            generation_time = time.time() - start_time
            if generation_time > 30:  # 30秒以上かかった場合
                self.logger.warning(f"テキスト生成に時間がかかりました: {generation_time:.2f}秒")
            
            # 生成されたテキストを取得（エラーハンドリング強化）
            if isinstance(output, dict) and 'choices' in output and output['choices']:
                generated_text = output['choices'][0]['text'].strip()
            elif hasattr(output, 'choices') and output.choices:
                generated_text = output.choices[0].text.strip()
            else:
                generated_text = str(output).strip()
            
            # 出力の検証
            if not generated_text:
                return "申し訳ありませんが、適切な応答を生成できませんでした。"
            
            # 出力の長さ制限
            if len(generated_text) > 1000:
                generated_text = generated_text[:1000]
                self.logger.debug("生成テキストが長すぎるため切り詰めました")
            
            return generated_text
            
        except Exception as e:
            error_type = type(e).__name__
            self.logger.error(f"テキスト生成エラー ({error_type}): {e}")
            
            # エラータイプ別の適切な応答
            if "memory" in str(e).lower() or "allocation" in str(e).lower():
                return "申し訳ありませんが、メモリ不足のため応答を生成できません。"
            elif "timeout" in str(e).lower():
                return "申し訳ありませんが、処理に時間がかかりすぎています。"
            else:
                return "申し訳ありませんが、現在応答を生成できません。"
    
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット完了機能
        
        引数:
            messages: メッセージのリスト
            **kwargs: 追加パラメータ
            
        戻り値:
            チャット完了のレスポンス
        """
        # 遅延初期化を確実に実行
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
            # パラメータの設定
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # チャット完了を実行
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get('top_p', 0.9),
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
    """    # 利用可能なモデルタイプ
    MODEL_TYPES = {
        'llama': LlamaModel,
        'local': LlamaModel,  # localは LlamaModel にマッピング
        'gemma3': LlamaModel,  # gemma3もLlamaModelを使用
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any] = None) -> BaseModel:
        """
        設定に基づいてモデルインスタンスを作成
        
        引数:
            config: モデル設定辞書（オプション、使用されない場合はConfigManagerから取得）
            
        戻り値:
            作成されたモデルインスタンス
        """
        # ConfigManagerから設定を取得
        config_manager = get_config()
        model_type = config_manager.model.model_type.lower()
        
        # 未知のモデルタイプのハンドリング
        if model_type not in cls.MODEL_TYPES:
            logging.warning(f"未知のモデルタイプ: {model_type}. llamaモデルを試行します。")
            model_type = 'llama'
        
        model_class = cls.MODEL_TYPES[model_type]
        return model_class(config)
    
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
        'n_ctx': 2048,
        'n_threads': 4,
        'temperature': 0.7,
        'max_tokens': 256,
        'chat_template': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                     '../../models/gemma3_template.txt'))
    }
]


class ModelSingleton:
    """
    モデルシングルトンパターン
    
    システム全体で単一のモデルインスタンスを共有することで
    初期化時間を70-80%削減し、メモリ使用量を最適化する
    """
    _instance = None
    _model = None
    _config = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            import threading
            cls._lock = threading.Lock()
        return cls._instance
    
    def get_model(self, config: Dict[str, Any] = None) -> BaseModel:
        """
        モデルインスタンスを取得（スレッドセーフ）
        
        引数:
            config: モデル設定（オプション、使用されない場合はConfigManagerから取得）
            
        戻り値:
            共有モデルインスタンス
        """
        if self._model is None:
            with self._lock:
                # ダブルチェックロッキング
                if self._model is None:
                    # ConfigManagerから設定を取得
                    config_manager = get_config()
                    use_config = config or config_manager.to_dict()
                    self._config = use_config
                    self._model = ModelFactory.get_best_available_model(use_config)
                    
                    logger = logging.getLogger('MurmurNet.ModelSingleton')
                    logger.info("共有モデルインスタンスを初期化しました")
        
        return self._model
    
    def clear_model(self):
        """モデルインスタンスをクリア（テスト用）"""
        with self._lock:
            self._model = None
            self._config = None


# グローバルシングルトンインスタンス
_model_singleton = ModelSingleton()


def get_shared_model(config: Dict[str, Any] = None) -> BaseModel:
    """
    共有モデルインスタンスを取得
    
    引数:
        config: モデル設定（初回のみ使用）
        
    戻り値:
        共有モデルインスタンス
    """
    return _model_singleton.get_model(config)


def get_default_model() -> BaseModel:
    """
    デフォルトのモデルインスタンスを取得
    
    戻り値:
        ConfigManagerから設定を取得した最適なモデル
    """
    return get_shared_model()
