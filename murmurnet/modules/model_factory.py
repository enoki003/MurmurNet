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

def sanitize_hf_name(hf_id: str) -> str:
    """HuggingFace IDのスラッシュをアンダースコアに変換"""
    if not hf_id:
        return ""
    return hf_id.replace("/", "_")

def _get_model_cache_key(config: Dict[str, Any]) -> str:
    model_type = config.get('model_type', '')
    n_ctx = config.get('n_ctx', 2048)
    
    if model_type == 'huggingface':
        model_name = config.get('huggingface_model_name', '')
        # HuggingFace IDを正規化
        sanitized_name = sanitize_hf_name(model_name)
        return f"{model_type}:{sanitized_name}:{n_ctx}"
    else:
        model_path = config.get('model_path', '')
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
        self.n_ctx = config.get('n_ctx', 2048)  # コンテキストサイズを設定
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
            
            # local_files_only設定を取得
            local_files_only = self.config.get('local_files_only', True)
            cache_folder = self.config.get('cache_folder', self.cache_dir)
            
            self.logger.info(f"ローカルファイルモード: {local_files_only}, キャッシュ: {cache_folder}")
            
            # トークナイザーの読み込み
            self.logger.info("トークナイザーを読み込み中...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_folder,
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            
            # パッドトークンの設定
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # 最大長の設定（警告回避）
            if not hasattr(self._tokenizer, 'model_max_length') or self._tokenizer.model_max_length > 100000:
                self._tokenizer.model_max_length = self.n_ctx  # 設定されたコンテキストサイズを使用
                self.logger.debug(f"tokenizer.model_max_length を {self.n_ctx} に設定")
            
            # トークナイザーのデフォルト設定
            self._tokenizer.padding_side = 'left'  # 左パディング（生成に適している）
            
            # モデルの読み込み
            self.logger.info("モデルを読み込み中...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=cache_folder,
                local_files_only=local_files_only,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != 'cpu' else None,
                trust_remote_code=True
            )
            
            # CPUの場合は明示的にCPUに移動
            if self.device == 'cpu':
                self._model = self._model.to('cpu')
            
            # HuggingFaceモデル用のn_ctx属性を設定（Gemmaとの統一インターフェース）
            if not hasattr(self, "n_ctx"):
                # モデルの設定から最大コンテキスト長を推測
                self.n_ctx = getattr(
                    self._model.config, 
                    "max_position_embeddings",
                    getattr(self._tokenizer, "model_max_length", 2048)
                )
                self.logger.debug(f"HuggingFaceモデル用n_ctx設定: {self.n_ctx}")
            
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
            # llm-jp-3-150m-instruct3公式テンプレート形式の検出と処理
            if "<|system|>" in prompt and "<|user|>" in prompt:
                # 公式テンプレート形式の場合、messages形式に変換
                system_part = prompt.split("<|system|>")[1].split("<|user|>")[0].strip()
                user_part = prompt.split("<|user|>")[1].split("<|assistant|>")[0].strip()
                
                messages = [
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part}
                ]
                
                # apply_chat_templateを使用して正しいプロンプトを生成
                try:
                    inputs = self._tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                    if self.device == 'cpu':
                        inputs = inputs.to('cpu')
                    else:
                        inputs = inputs.to(self.device)
                    
                    # inputsが既にtensor形式なのでそのまま使用
                    inputs_dict = {'input_ids': inputs}
                except Exception as e:
                    self.logger.warning(f"apply_chat_template失敗、フォールバック: {e}")
                    # フォールバックとして従来の方法
                    inputs_dict = self._tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=min(self.n_ctx, 2048)  # 最大長を明示的に指定
                    )
                    if 'token_type_ids' in inputs_dict:
                        del inputs_dict['token_type_ids']
                    if self.device == 'cpu':
                        inputs_dict = {k: v.to('cpu') for k, v in inputs_dict.items()}
                    else:
                        inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
            else:
                # 従来形式の場合
                inputs_dict = self._tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=min(self.n_ctx, 2048)  # 最大長を明示的に指定
                )
                if 'token_type_ids' in inputs_dict:
                    del inputs_dict['token_type_ids']
                if self.device == 'cpu':
                    inputs_dict = {k: v.to('cpu') for k, v in inputs_dict.items()}
                else:
                    inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
            
            # 生成設定（小型モデル向け重複防止最適化・警告回避）
            # CLI引数の max_new_tokens を優先使用（空レス対策）
            config_max_new_tokens = self.config.get('max_new_tokens', 128)
            effective_max_tokens = config_max_new_tokens  # CLI引数を強制優先
            
            # モデル固有の最適化（重複防止強化）
            model_name = self.model_name.lower()
            if 'gemma' in model_name or 'llm-jp' in model_name:
                # 小型モデル用：重複抑制を強化（警告回避版）
                generation_config = {
                    'max_new_tokens': min(effective_max_tokens, 100),  # より短く
                    'temperature': max(temperature, 0.4),   # 低温度で安定化
                    'top_p': 0.8,                          # 選択肢を絞る
                    'top_k': 20,                           # 選択肢をさらに制限
                    'do_sample': True,                     # サンプリング有効
                    'pad_token_id': self._tokenizer.pad_token_id,
                    'eos_token_id': self._tokenizer.eos_token_id,
                    'repetition_penalty': 1.3,             # 繰り返しペナルティ強化
                    'use_cache': True,                     # キャッシュ有効
                    'no_repeat_ngram_size': 3,             # 3-gram重複を禁止
                    # ビーム探索パラメータを削除（num_beams=1の場合は無効）
                    # 'early_stopping': True,              # ← 削除
                    # 'length_penalty': 0.9,               # ← 削除
                }
                self.logger.debug(f"小型モデル用重複防止設定適用: {model_name}")
            else:
                # 一般的なモデル用
                generation_config = {
                    'max_new_tokens': effective_max_tokens,  # CLI引数を強制適用
                    'temperature': max(temperature, 0.7),   # 創造性を高める
                    'top_p': 0.9,                          # 多様性を確保
                    'top_k': 30,                           # 選択肢を適度に制限
                    'do_sample': True,                     # サンプリング有効
                    'pad_token_id': self._tokenizer.pad_token_id,
                    'eos_token_id': self._tokenizer.eos_token_id,
                    'repetition_penalty': 1.0,             # 繰り返しペナルティを軽減
                    'use_cache': True,                     # キャッシュ有効
                    'no_repeat_ngram_size': 0,             # n-gram制限を無効化
                    # ビーム探索パラメータを削除
                    # 'early_stopping': False,             # ← 削除
                    # 'length_penalty': 1.0,               # ← 削除
                }
            
            # 推論実行
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs_dict,
                    **generation_config
                )
              # 結果のデコード（入力プロンプトを除外）
            generated_tokens = outputs[0][inputs_dict['input_ids'].shape[1]:]
            response = self._tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            # llm-jp-3-150m-instruct3向けの応答後処理
            processed_response = self._post_process_response(response, prompt)
            
            # 必ず文字列を返す（辞書等を返さない）
            return str(processed_response).strip() if processed_response else "申し訳ございませんが、適切な回答を生成できませんでした。"
            
        except Exception as e:
            self.logger.error(f"HuggingFace生成エラー: {e}")
            return f"エラーが発生しました: {str(e)}"
    
    def _post_process_response(self, response: str, original_prompt: str) -> str:
        """
        llm-jp-3-150m-instruct3向けの応答後処理（改良版）
        
        Args:
            response: 生成された応答
            original_prompt: 元のプロンプト
            
        Returns:
            後処理された応答
        """
        # デバッグ用の生ログ出力
        if self.config.get('debug', False):
            self.logger.debug(f"[RAW] 生成された生応答: «{response}» (長さ: {len(response)})")
        
        if not response:
            return ""  # 空の場合は空文字列を返す
        
        # 制御タグの削除
        import re
        response = re.sub(r'<\|(?:system|user|assistant)\|>', '', response)
        response = re.sub(r'</?(?:human|bot|s)>', '', response)
        response = re.sub(r'</s>', '', response)
        
        # 「申し訳ありませんが」で始まる場合の特別処理
        if response.strip().startswith("申し訳ありませんが、そのリクエストにはお応えできません。"):
            # 実際の回答部分を抽出
            lines = response.split('\n')
            filtered_lines = []
            found_content = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 最初の謝罪文をスキップして、実際の内容を探す
                if not found_content:
                    if (line.startswith("教育") or line.startswith("AI") or 
                        line.startswith("1.") or line.startswith("・") or
                        ":" in line or "：" in line or
                        any(keyword in line for keyword in ["影響", "効果", "変化", "学習", "技術"])):
                        found_content = True
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            
            if filtered_lines:
                response = '\n'.join(filtered_lines)
        
        # 先頭の不要な文字を削除
        response = response.lstrip('：:、。\n\r\t ')
        
        # デバッグ用の後処理ログ出力
        if self.config.get('debug', False):
            self.logger.debug(f"[PROCESSED] 後処理後応答: «{response}» (長さ: {len(response)})")
        
        # 2文字未満の場合のみ空を返す（5文字制限を撤廃）
        if len(response.strip()) < 2:
            return ""
        
        return response.strip()

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
            # 公式chat_templateを使用してプロンプト生成
            if messages and len(messages) > 0:
                # messages形式から公式テンプレート適用
                try:
                    # tokenizer.apply_chat_templateを使用
                    prompt = self._tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    self.logger.debug(f"apply_chat_template成功: {prompt[:100]}...")
                except Exception as e:
                    self.logger.warning(f"apply_chat_template失敗、フォールバック: {e}")
                    # フォールバック: 従来の方法
                    if len(messages) == 1:
                        prompt = messages[0]['content']
                    elif len(messages) >= 2:
                        system_content = messages[0]['content'] if messages[0]['role'] == 'system' else ""
                        user_content = messages[-1]['content'] if messages[-1]['role'] == 'user' else messages[0]['content']
                        prompt = f"{system_content}\n\n{user_content}" if system_content else user_content
                    else:
                        prompt = "質問に回答してください。"
            else:
                prompt = "質問に回答してください。"
            
            # 生成実行
            response_text = self.generate(prompt, **kwargs)
            
            # 必ず文字列として返す
            response_content = str(response_text) if response_text else "申し訳ございませんが、適切な回答を生成できませんでした。"
            
            return {
                'choices': [{
                    'message': {
                        'content': response_content,
                        'role': 'assistant'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': len(response_content.split()),
                    'total_tokens': len(prompt.split()) + len(response_content.split())
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
    # 注意: HuggingFaceモデルは明示的に指定された場合のみ使用
    # 自動フォールバックでは150Mモデルを使用しない
]

def get_default_model() -> BaseModel:
    """デフォルトモデルを取得（明示的な設定必須）"""
    if not DEFAULT_MODEL_CONFIGS:
        raise ValueError("モデル設定が不正です。config.yamlでmodel_typeとmodel_pathを明示的に指定してください。")
    return ModelFactory.get_best_available_model(DEFAULT_MODEL_CONFIGS[0])

def create_model_from_args(model_type: str = None, model_name: str = None, model_path: str = None) -> BaseModel:
    """CLI引数からモデルを作成（150Mフォールバック禁止）"""
    if not model_type:
        raise ValueError("--model-type は必須です。'llama' または 'huggingface' を指定してください。")
    
    if model_type == 'llama':
        if not model_path:
            raise ValueError("model_type='llama' の場合、--model-path は必須です。")
        config = {
            'model_type': 'llama',
            'model_path': model_path,
            'n_ctx': 2048,
            'n_threads': 4,
            'temperature': 0.7,
            'max_tokens': 256,
        }
    elif model_type == 'huggingface':
        if not model_name:
            raise ValueError("model_type='huggingface' の場合、--model-name は必須です。150Mモデルへの自動フォールバックは禁止されています。")
        config = {
            'model_type': 'huggingface',
            'huggingface_model_name': model_name,
            'device': 'cpu',
            'torch_dtype': 'auto',
            'temperature': 0.7,
            'max_tokens': 256,
            'model_cache_dir': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cache/models'))
        }
    else:
        raise ValueError(f"未対応のmodel_type: {model_type}. 'llama' または 'huggingface' を指定してください。")
    
    return ModelFactory.create_model(config)
