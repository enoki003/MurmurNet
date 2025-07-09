#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""model_factory.py
~~~~~~~~~~~~~~~~~~~
MurmurNet 用モデルファクトリ（Singleton 対応）
Gemma／Llama 系 GGUF（llama‑cpp‑python）と HuggingFace Transformers の
両方を透過的に扱う。重複していた関数・クラス、壊れたインデント、
不完全な例外処理を全面的に整理した。

主要ポイント
-------------
* **BaseModel** ‑ 共通インターフェース。
* **LlamaModel** ‑ llama_cpp.Llama を SingletonModelManager 経由で共有。
* **TransformersModel** ‑ AutoTokenizer + AutoModelForCausalLM を遅延ロード。
* **ModelFactory / ModelFactorySingleton**  ‑        
  * `get_shared_model()` でスレッドセーフなインスタンス共有。  
  * シンプルな優先ロジック：設定 → キャッシュ → fallback 検索。

依存ライブラリが無い環境でもスムーズに degrade するため、
ImportError を捕捉して ``HAS_LLAMA_CPP``/``HAS_TRANSFORMERS`` を立てている。
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

###############################################################################
# Optional imports & feature flags
###############################################################################

try:
    from llama_cpp import Llama  # type: ignore

    HAS_LLAMA_CPP = True
except ImportError:  # pragma: no cover – optional
    Llama = None  # type: ignore
    HAS_LLAMA_CPP = False

try:
    import torch  # noqa: F401 – used only if transformers present
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover – optional
    AutoModelForCausalLM = AutoTokenizer = torch = None  # type: ignore
    HAS_TRANSFORMERS = False

###############################################################################
# Global cache for shared model instances
###############################################################################

_MODEL_CACHE: Dict[str, "BaseModel"] = {}
_CACHE_LOCK = threading.Lock()

###############################################################################
# Helper
###############################################################################

def _cache_key(conf: Dict[str, Any]) -> str:
    return f"{conf.get('model_type')}:{conf.get('model_path') or conf.get('model_name')}:{conf.get('n_ctx', 2048)}"

###############################################################################
# Base class
###############################################################################


class BaseModel(ABC):
    """マルチバックエンドの共通抽象クラス。"""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.log = logging.getLogger("MurmurNet.BaseModel")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, prompt: str, **gen_kwargs) -> str:  # noqa: D401
        pass

    @abstractmethod
    def is_available(self) -> bool:  # noqa: D401
        pass

###############################################################################
# llama‑cpp wrapper
###############################################################################


class LlamaModel(BaseModel):
    """Thread‑safe shared Llama instance via external SingletonModelManager."""

    def __init__(self, cfg: Dict[str, Any]):
        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama‑cpp‑python is not installed")
        super().__init__(cfg)

        from MurmurNet.modules.model_manager import (  # local import to avoid cycle
            get_singleton_manager,
        )

        self.manager = get_singleton_manager(cfg)
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens = cfg.get("max_tokens", 256)
        self.top_p = cfg.get("top_p", 0.9)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **gen_kwargs) -> str:
        params = {
            "max_tokens": gen_kwargs.get("max_tokens", self.max_tokens),
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "top_p": gen_kwargs.get("top_p", self.top_p),
        }
        return self.manager.generate(prompt, **params)

    def create_chat_completion(self, messages: List[Dict[str, str]], **kw) -> Dict[str, Any]:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        completion = self.generate(
            prompt,
            max_tokens=kw.get("max_tokens", self.max_tokens),
            temperature=kw.get("temperature", self.temperature),
            top_p=kw.get("top_p", self.top_p),
        )
        return {
            "choices": [{"message": {"role": "assistant", "content": completion}, "finish_reason": "stop"}],
            "usage": {},
        }

    def is_available(self) -> bool:
        return self.manager.is_available()

###############################################################################
# Transformers wrapper
###############################################################################


class TransformersModel(BaseModel):
    """Lazy‑loaded HF causal‑LM."""

    def __init__(self, cfg: Dict[str, Any]):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")
        super().__init__(cfg)

        self.model_name = cfg.get("model_name") or cfg.get("model_path", "gpt2")
        self.device = cfg.get("device", "cpu")
        self.max_new_tokens = cfg.get("max_tokens", 256)
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.9)

        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _load(self):
        with self._load_lock:
            if self._model is not None:
                return  # already loaded
            self.log.info(f"Loading HF model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **gen_kwargs) -> str:
        self._load()
        assert self._model and self._tokenizer  # for type checker

        inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        g_kw = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "top_p": gen_kwargs.get("top_p", self.top_p),
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        with torch.no_grad():  # type: ignore[attr-defined]
            out = self._model.generate(inputs, **g_kw)
        return self._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:  # pylint: disable=broad-except
            self.log.warning(f"HF model unavailable: {e}")
            return False

###############################################################################
# Factory (static methods only)
###############################################################################


class ModelFactory:  # noqa: D401
    """Stateless helper offering shared model instances."""

    # --------------------------------------------------------------
    # Low‑level helpers
    # --------------------------------------------------------------

    @staticmethod
    def _build(cfg: Dict[str, Any]) -> BaseModel:
        mtype = cfg.get("model_type", "llama").lower()
        if mtype == "llama":
            return LlamaModel(cfg)
        if mtype in {"transformers", "huggingface"}:
            return TransformersModel(cfg)
        raise ValueError(f"Unsupported model_type: {mtype}")

    # --------------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------------

    @staticmethod
    def get_shared_model(cfg: Dict[str, Any]) -> BaseModel:
        key = _cache_key(cfg)
        with _CACHE_LOCK:
            if key in _MODEL_CACHE:
                return _MODEL_CACHE[key]
        model = ModelFactory._build(cfg)
        with _CACHE_LOCK:
            _MODEL_CACHE.setdefault(key, model)
        return model

    @staticmethod
    def get_best_available(cfg: Dict[str, Any]) -> BaseModel:
        model = ModelFactory.get_shared_model(cfg)
        if model.is_available():
            return model
        raise RuntimeError("No available model for given configuration")

###############################################################################
# Singleton façade for backward compatibility
###############################################################################


class ModelFactorySingleton:  # noqa: D401
    _instance: Optional["ModelFactorySingleton"] = None
    _lock = threading.Lock()

    def __new__(cls, cfg: Optional[Dict[str, Any]] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init(cfg or {})
            return cls._instance

    # ------------------------------------------------------------------
    # internal init
    # ------------------------------------------------------------------

    def _init(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.log = logging.getLogger("MurmurNet.ModelFactorySingleton")
        self._models: Dict[str, BaseModel] = {}
        self._models_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public façade
    # ------------------------------------------------------------------

    def get_model(self, key: str = "default") -> BaseModel:
        with self._models_lock:
            if key not in self._models:
                self.log.info(f"Creating model instance ({key})")
                self._models[key] = ModelFactory.get_shared_model(self.cfg)
            return self._models[key]

    def get_any_available_model(self) -> Optional[BaseModel]:
        # default first
        m = self.get_model("default")
        if m.is_available():
            return m
        # others
        with self._models_lock:
            for mdl in self._models.values():
                if mdl.is_available():
                    return mdl
        return None

    def clear_cache(self):
        with self._models_lock:
            self._models.clear()
            self.log.info("Cleared local model cache")

    # ------------------------------------------------------------------
    # Backward compatibility factory method
    # ------------------------------------------------------------------
    
    @classmethod
    def get_factory(cls, cfg: Dict[str, Any]) -> "ModelFactorySingleton":
        """Create or get singleton instance with given config (backward compatibility)."""
        return cls(cfg)
    
    # ------------------------------------------------------------------
    # Additional backward compatibility aliases
    # ------------------------------------------------------------------
    
    def create_model(self, cfg: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create model instance (backward compatibility)."""
        if cfg is None:
            cfg = self.cfg
        return ModelFactory.get_shared_model(cfg)

###############################################################################
# Default configuration helper
###############################################################################

def _default_gguf() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../models/gemma-3-1b-it-q4_0.gguf")
    )

def get_model_factory() -> ModelFactorySingleton:
    """ModelFactorySingletonのインスタンスを取得"""
    return ModelFactorySingleton()

def get_default_model() -> BaseModel:
    """Convenience wrapper returning a ready‑to‑use model."""

    default_cfg = {
        "model_type": "llama",
        "model_path": _default_gguf(),
        "n_ctx": 2048,
        "n_threads": 4,
        "temperature": 0.7,
        "max_tokens": 256,
    }
    return ModelFactory.get_best_available(default_cfg)
