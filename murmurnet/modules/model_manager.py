#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet Singleton LLMマネージャー
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LlamaモデルのSingleton実装
メモリ使用量とロード時間を大幅に削減

作者: Yuhi Sonoki
"""

import logging
import os
import time
import threading
from functools import lru_cache
from typing import Dict, Any, Optional, Union, List
import psutil

logger = logging.getLogger('MurmurNet.ModelManager')

# 条件付きインポート
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    Llama = None

# グローバル変数
_MODEL_INSTANCES = {}
_MODEL_LOCK = threading.Lock()

@lru_cache(maxsize=4)
def get_llama_model(model_path: str, 
                   n_ctx: int = 2048,
                   n_threads: int = 6,  # 4C/8T CPUに最適化（物理コア+2）
                   n_gpu_layers: int = 0,
                   chat_template: Optional[str] = None) -> Optional[object]:
    """
    LlamaモデルのSingleton取得
    
    @lru_cache により、同じ引数で何度呼び出されても
    初回のみインスタンス化され、以降は同じインスタンスを返す
    
    引数:
        model_path: モデルファイルパス
        n_ctx: コンテキストサイズ
        n_threads: スレッド数
        n_gpu_layers: GPU層数
        chat_template: チャットテンプレートファイルパス
        
    戻り値:
        Llamaインスタンス or None
    """
    if not HAS_LLAMA_CPP:
        logger.error("❌ llama-cpp-python がインストールされていません")
        return None
    
    if not os.path.exists(model_path):
        logger.error(f"❌ モデルファイルが見つかりません: {model_path}")
        return None
    
    try:
        logger.info(f"🚀 Llamaモデル初期化開始: {os.path.basename(model_path)}")
        start_time = time.time()
          # Llamaインスタンス生成
        llama_kwargs = {
            'model_path': model_path,
            'n_ctx': n_ctx,
            'n_threads': n_threads,
            'n_gpu_layers': n_gpu_layers,
            'use_mmap': True,
            'use_mlock': True,
            'verbose': False,  # 初期化時のログを抑制
            # パフォーマンス最適化パラメータ
            'n_batch': 1024,  # 512→1024に増加（バッチ処理効率向上）
            'n_seq_max': 2,   # 並列シーケンス数を制限
            'last_n_tokens_size': 64,
            'rope_scaling_type': 0,
            'rope_freq_base': 10000.0,
            'rope_freq_scale': 1.0,
        }
        
        # チャットテンプレートの読み込み
        if chat_template and os.path.exists(chat_template):
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    llama_kwargs['chat_template'] = f.read()
            except Exception as e:
                logger.warning(f"⚠️ チャットテンプレート読み込みエラー: {e}")
        
        # Llamaインスタンス作成
        llm = Llama(**llama_kwargs)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Llamaモデル初期化完了: {os.path.basename(model_path)} ({load_time:.2f}s)")
        
        return llm
        
    except Exception as e:
        logger.error(f"❌ Llamaモデル初期化エラー: {e}")
        return None

class SingletonModelManager:
    """
    Singleton LLMマネージャー
    
    全てのModelFactoryインスタンスで同じLlamaモデルを共有
    メモリ使用量とロード時間を大幅に削減
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Singleton ModelManagerの初期化
        
        引数:
            config: 設定辞書
        """
        self.config = config
        self.model_path = config.get('model_path')
        self.n_ctx = config.get('n_ctx', 2048)
        self.n_threads = config.get('n_threads', 4)
        self.n_gpu_layers = config.get('n_gpu_layers', 0)
        self.chat_template = config.get('chat_template')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 256)
        self.debug = config.get('debug', False)
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
    
    @property
    def model(self):
        """
        Llamaモデルインスタンスを取得
        
        初回アクセス時のみロードされ、以降は同じインスタンスを返す
        
        戻り値:
            Llamaインスタンス or None
        """
        if not self.model_path:
            logger.error("❌ model_pathが設定されていません")
            return None
        
        return get_llama_model(
            self.model_path,
            self.n_ctx,
            self.n_threads,
            self.n_gpu_layers,
            self.chat_template
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成
        
        引数:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ
            
        戻り値:
            生成されたテキスト
        """
        model = self.model
        if model is None:
            return "❌ Llamaモデルが利用できません"
        
        try:
            # 生成パラメータの設定
            generation_kwargs = {
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'top_p': kwargs.get('top_p', 0.9),
                'top_k': kwargs.get('top_k', 40),
                'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
                'stop': kwargs.get('stop', []),
                'echo': False,
            }
            
            if self.debug:
                logger.debug(f"🤖 テキスト生成開始: {len(prompt)}文字")
            
            # テキスト生成実行
            start_time = time.time()
            response = model(prompt, **generation_kwargs)
            generation_time = time.time() - start_time
            
            # レスポンスの処理
            if isinstance(response, dict) and 'choices' in response:
                if response['choices']:
                    if isinstance(response['choices'][0], dict):
                        text = response['choices'][0].get('text', '').strip()
                    else:
                        text = response['choices'][0].text.strip()
                else:
                    text = ""
            else:
                text = str(response).strip()
            
            if self.debug:
                tokens = len(text.split())
                speed = tokens / generation_time if generation_time > 0 else 0
                logger.debug(f"✅ テキスト生成完了: {tokens}トークン, {speed:.1f}tok/s")
            
            return text
            
        except Exception as e:
            logger.error(f"❌ テキスト生成エラー: {e}")
            return f"❌ 生成エラー: {str(e)}"
    
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[Dict[str, Any], object]:
        """
        チャット完了API（llama-cpp-python互換）
        
        引数:
            messages: メッセージリスト [{"role": "user", "content": "..."}]
            **kwargs: 生成パラメータ
            
        戻り値:
            レスポンス辞書またはオブジェクト        """
        model = self.model
        if model is None:
            return {"error": "❌ Llamaモデルが利用できません"}
        
        try:
            # メッセージをプロンプトに変換 (format_chat_promptは使用しない)
            prompt_parts = []
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
              # チャット完了パラメータの設定
            chat_kwargs = {
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'top_p': kwargs.get('top_p', 0.9),
                'top_k': kwargs.get('top_k', 40),
                'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
                'stop': kwargs.get('stop', ['\n\n', 'User:', 'System:']),
                # 'echo': False  # create_chat_completionでは使用しない
            }
            
            if self.debug:
                logger.debug(f"💬 チャット完了開始: {len(messages)}メッセージ")
            
            # model.create_chat_completionがある場合はそれを使用
            if hasattr(model, 'create_chat_completion'):
                return model.create_chat_completion(messages=messages, **chat_kwargs)
            else:
                # フォールバック: 通常のテキスト生成を使用
                # 通常の生成用のパラメータ（echoを追加）
                generation_kwargs = dict(chat_kwargs)
                generation_kwargs['echo'] = False
                
                start_time = time.time()
                response = model(prompt, **generation_kwargs)
                generation_time = time.time() - start_time
                  # レスポンスの処理 - より堅牢なエラーハンドリング
                text = ""
                try:
                    if isinstance(response, dict):
                        if 'choices' in response and response['choices']:
                            # llama-cpp-python形式のレスポンス
                            choice = response['choices'][0]
                            if isinstance(choice, dict):
                                if 'text' in choice:
                                    text = choice['text'].strip()
                                elif 'message' in choice and 'content' in choice['message']:
                                    text = choice['message']['content'].strip()
                            else:
                                text = str(choice).strip()
                        else:
                            # 他の形式の辞書レスポンス
                            text = str(response).strip()
                    else:
                        # 文字列またはその他のレスポンス
                        text = str(response).strip()
                except Exception as parse_error:
                    logger.warning(f"レスポンス解析エラー: {parse_error}")
                    text = str(response).strip() if response else ""
                
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_path,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(text.split()),
                        "total_tokens": len(prompt.split()) + len(text.split())
                    }
                }
                
                if self.debug:
                    tokens = len(text.split())
                    speed = tokens / generation_time if generation_time > 0 else 0
                    logger.debug(f"✅ チャット完了: {tokens}トークン, {speed:.1f}tok/s")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ チャット完了エラー: {e}")
            return {"error": f"❌ チャット完了エラー: {str(e)}"}

    def is_available(self) -> bool:
        """
        モデルが利用可能かチェック
        
        戻り値:
            True: 利用可能, False: 利用不可
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得
        
        戻り値:
            モデル情報辞書
        """
        return {
            'model_path': self.model_path,
            'n_ctx': self.n_ctx,
            'n_threads': self.n_threads,
            'n_gpu_layers': self.n_gpu_layers,
            'available': self.is_available(),
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """
        メモリ使用量を取得（MB）
        
        戻り値:
            メモリ使用量（MB）
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

# 便利関数
def get_singleton_manager(config: Dict[str, Any]) -> SingletonModelManager:
    """
    Singleton ModelManagerインスタンスを取得
    
    引数:
        config: 設定辞書
        
    戻り値:
        SingletonModelManagerインスタンス
    """
    return SingletonModelManager(config)

# 後方互換性のためのエイリアス
def get_shared_model_manager(config: Dict[str, Any]) -> SingletonModelManager:
    """後方互換性のためのエイリアス"""
    return get_singleton_manager(config)

if __name__ == "__main__":
    # テスト用コード
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    # テスト設定（実際のモデルパスに変更してください）
    config = {
        'model_path': 'path/to/your/model.gguf',
        'n_ctx': 2048,
        'n_threads': 4,
        'debug': True
    }
    
    print("=== SingletonModelManager テスト ===")
    
    # 1回目のロード
    manager1 = get_singleton_manager(config)
    start_time = time.time()
    available1 = manager1.is_available()
    time1 = time.time() - start_time
    print(f"1回目: {time1:.3f}s, 利用可能: {available1}")
    
    # 2回目のロード（キャッシュから）
    manager2 = get_singleton_manager(config)
    start_time = time.time()
    available2 = manager2.is_available()
    time2 = time.time() - start_time
    print(f"2回目: {time2:.3f}s, 利用可能: {available2}")
    
    # 同じインスタンスかチェック
    print(f"同じmodelインスタンス: {manager1.model is manager2.model}")
    if time1 > 0 and time2 > 0:
        print(f"速度向上: {time1/time2:.1f}倍")
