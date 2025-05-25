#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Manager モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
設定ファイルの読み込み、バリデーション、型安全性を提供

作者: Yuhi Sonoki
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger('MurmurNet.ConfigManager')


class ConfigValidationError(Exception):
    """設定値バリデーションエラー"""
    pass


@dataclass
class AgentConfig:
    """エージェント設定"""
    num_agents: int = 2
    iterations: int = 1
    use_parallel: bool = False
    use_summary: bool = True
    use_memory: bool = True
    role_type: str = "default"
    
    def __post_init__(self):
        if self.num_agents < 1 or self.num_agents > 10:
            raise ConfigValidationError("num_agents must be between 1 and 10")
        if self.iterations < 1 or self.iterations > 5:
            raise ConfigValidationError("iterations must be between 1 and 5")
        if self.role_type not in ["default", "discussion", "planning", "informational", "conversational"]:
            raise ConfigValidationError(f"Invalid role_type: {self.role_type}")


@dataclass
class ModelConfig:
    """モデル設定"""
    model_type: str = "gemma3"
    model_path: str = ""
    chat_template: str = ""
    n_ctx: int = 2048
    n_threads: int = 4
    temperature: float = 0.7
    max_tokens: int = 256
    def __post_init__(self):
        if self.n_ctx < 512 or self.n_ctx > 8192:
            raise ConfigValidationError("n_ctx must be between 512 and 8192")
        if self.n_threads < 1 or self.n_threads > 16:
            raise ConfigValidationError("n_threads must be between 1 and 16")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ConfigValidationError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1 or self.max_tokens > 2048:
            raise ConfigValidationError("max_tokens must be between 1 and 2048")
        if self.model_type not in ["llama", "local", "gemma3"]:
            raise ConfigValidationError(f"Invalid model_type: {self.model_type}")


@dataclass
class RAGConfig:
    """RAG設定"""
    rag_enabled: bool = True
    rag_mode: str = "zim"
    zim_path: str = ""
    rag_score_threshold: float = 0.5
    rag_top_k: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"
    
    def __post_init__(self):
        if self.rag_mode not in ["zim", "embedding"]:
            raise ConfigValidationError(f"Invalid rag_mode: {self.rag_mode}")
        if self.rag_score_threshold < 0.0 or self.rag_score_threshold > 1.0:
            raise ConfigValidationError("rag_score_threshold must be between 0.0 and 1.0")
        if self.rag_top_k < 1 or self.rag_top_k > 10:
            raise ConfigValidationError("rag_top_k must be between 1 and 10")


@dataclass
class LoggingConfig:
    """ログ設定"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    debug: bool = False
    performance_monitoring: bool = True
    memory_tracking: bool = True
    
    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ConfigValidationError(f"log_level must be one of: {valid_levels}")


@dataclass
class MemoryConfig:
    """メモリ設定"""
    conversation_memory_limit: int = 10
    summary_max_length: int = 200
    blackboard_history_limit: int = 100
    max_summary_tokens: int = 200
    max_history_entries: int = 10
    
    def __post_init__(self):
        if self.conversation_memory_limit < 1 or self.conversation_memory_limit > 100:
            raise ConfigValidationError("conversation_memory_limit must be between 1 and 100")
        if self.summary_max_length < 50 or self.summary_max_length > 500:
            raise ConfigValidationError("summary_max_length must be between 50 and 500")
        if self.blackboard_history_limit < 10 or self.blackboard_history_limit > 1000:
            raise ConfigValidationError("blackboard_history_limit must be between 10 and 1000")
        if self.max_summary_tokens < 50 or self.max_summary_tokens > 1000:
            raise ConfigValidationError("max_summary_tokens must be between 50 and 1000")
        if self.max_history_entries < 1 or self.max_history_entries > 50:
            raise ConfigValidationError("max_history_entries must be between 1 and 50")


@dataclass
class MurmurNetConfig:
    """MurmurNet全体設定"""
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得（後方互換性用）"""
        return {
            # Agent settings
            'num_agents': self.agent.num_agents,
            'iterations': self.agent.iterations,
            'use_parallel': self.agent.use_parallel,
            'use_summary': self.agent.use_summary,
            'use_memory': self.agent.use_memory,
            'role_type': self.agent.role_type,
            
            # Model settings
            'model_type': self.model.model_type,
            'model_path': self.model.model_path,
            'chat_template': self.model.chat_template,
            'n_ctx': self.model.n_ctx,
            'n_threads': self.model.n_threads,
            'temperature': self.model.temperature,
            'max_tokens': self.model.max_tokens,
            
            # RAG settings
            'rag_mode': self.rag.rag_mode,
            'rag_score_threshold': self.rag.rag_score_threshold,
            'rag_top_k': self.rag.rag_top_k,
            'zim_path': self.rag.zim_path,
            'embedding_model': self.rag.embedding_model,
            
            # Logging settings
            'log_level': self.logging.log_level,
            'log_file': self.logging.log_file,
            'debug': self.logging.debug,
            'performance_monitoring': self.logging.performance_monitoring,
            'memory_tracking': self.logging.memory_tracking,
            
            # Memory settings
            'conversation_memory_limit': self.memory.conversation_memory_limit,
            'summary_max_length': self.memory.summary_max_length,
            'blackboard_history_limit': self.memory.blackboard_history_limit,
            'max_summary_tokens': self.memory.max_summary_tokens,
            'max_history_entries': self.memory.max_history_entries,
        }


class ConfigManager:
    """
    設定管理クラス
    
    責務:
    - 設定ファイルの読み込み
    - 設定値のバリデーション
    - 型安全な設定アクセス
    - デフォルト値の管理
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        設定マネージャーの初期化
        
        引数:
            config_path: 設定ファイルのパス（省略時はデフォルトを検索）
        """
        if self._config is None:
            self.config_path = self._find_config_file(config_path)
            self._config = self._load_and_validate_config()
        
    def _find_config_file(self, config_path: Optional[str] = None) -> Optional[str]:
        """設定ファイルを検索"""
        if config_path and os.path.exists(config_path):
            return config_path
            
        # デフォルトの検索場所
        search_paths = [
            "config.yaml",
            "config.yml",
            "../config.yaml", 
            "../config.yml",
            "../../config.yaml",
            "../../config.yml",
            os.path.join(os.path.dirname(__file__), "../../config.yaml"),
            os.path.join(os.path.dirname(__file__), "../../../config.yaml")
        ]
        
        for path in search_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                logger.info(f"設定ファイルを発見: {abs_path}")
                return abs_path
                
        logger.warning("設定ファイルが見つかりません。デフォルト設定を使用します。")
        return None
    
    def _load_and_validate_config(self) -> MurmurNetConfig:
        """設定ファイルを読み込み、バリデーションを実行"""
        try:
            if self.config_path:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    raw_config = yaml.safe_load(f) or {}
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                raw_config = {}
            
            # デフォルト設定から開始
            config = MurmurNetConfig()
            
            # 設定ファイルの値でオーバーライド
            if raw_config:
                config = self._merge_config(config, raw_config)
            
            logger.info("設定のバリデーションが完了しました")
            return config
            
        except FileNotFoundError:
            logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
            return MurmurNetConfig()
        except yaml.YAMLError as e:
            logger.error(f"YAML解析エラー: {e}")
            raise ConfigValidationError(f"設定ファイルの解析に失敗しました: {e}")
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            raise ConfigValidationError(f"設定の読み込みに失敗しました: {e}")
    
    def _merge_config(self, config: MurmurNetConfig, raw_config: Dict[str, Any]) -> MurmurNetConfig:
        """設定をマージ"""
        
        # Agent設定
        if 'agent' in raw_config or any(key in raw_config for key in ['num_agents', 'iterations', 'use_parallel', 'use_summary', 'use_memory', 'role_type']):
            agent_config = raw_config.get('agent', {})
            # トップレベルの値もagent設定としてマージ
            for key in ['num_agents', 'iterations', 'use_parallel', 'use_summary', 'use_memory', 'role_type']:
                if key in raw_config:
                    agent_config[key] = raw_config[key]
            
            if agent_config:
                config.agent = AgentConfig(**{**config.agent.__dict__, **agent_config})
        
        # Model設定
        if 'model' in raw_config or any(key in raw_config for key in ['model_type', 'model_path', 'chat_template', 'n_ctx', 'n_threads', 'temperature', 'max_tokens']):
            model_config = raw_config.get('model', {})
            # トップレベルの値もmodel設定としてマージ
            for key in ['model_type', 'model_path', 'chat_template', 'n_ctx', 'n_threads', 'temperature', 'max_tokens']:
                if key in raw_config:
                    model_config[key] = raw_config[key]
            
            if model_config:
                config.model = ModelConfig(**{**config.model.__dict__, **model_config})
        
        # RAG設定
        if 'rag' in raw_config or any(key in raw_config for key in ['rag_enabled', 'rag_mode', 'zim_path', 'rag_score_threshold', 'rag_top_k', 'embedding_model']):
            rag_config = raw_config.get('rag', {})
            # トップレベルの値もrag設定としてマージ
            for key in ['rag_enabled', 'rag_mode', 'zim_path', 'rag_score_threshold', 'rag_top_k', 'embedding_model']:
                if key in raw_config:
                    rag_config[key] = raw_config[key]
            
            if rag_config:
                config.rag = RAGConfig(**{**config.rag.__dict__, **rag_config})
        
        # Logging設定
        if 'logging' in raw_config or any(key in raw_config for key in ['log_level', 'log_file', 'debug', 'performance_monitoring', 'memory_tracking']):
            logging_config = raw_config.get('logging', {})
            # トップレベルの値もlogging設定としてマージ
            for key in ['log_level', 'log_file', 'debug', 'performance_monitoring', 'memory_tracking']:
                if key in raw_config:
                    logging_config[key] = raw_config[key]
            
            if logging_config:
                config.logging = LoggingConfig(**{**config.logging.__dict__, **logging_config})
        
        # Memory設定
        if 'memory' in raw_config or any(key in raw_config for key in ['conversation_memory_limit', 'summary_max_length', 'blackboard_history_limit', 'max_summary_tokens', 'max_history_entries']):
            memory_config = raw_config.get('memory', {})
            # トップレベルの値もmemory設定としてマージ
            for key in ['conversation_memory_limit', 'summary_max_length', 'blackboard_history_limit', 'max_summary_tokens', 'max_history_entries']:
                if key in raw_config:
                    memory_config[key] = raw_config[key]
            
            if memory_config:
                config.memory = MemoryConfig(**{**config.memory.__dict__, **memory_config})
        
        return config
    
    @property
    def config(self) -> MurmurNetConfig:
        """設定オブジェクトを取得"""
        return self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得（後方互換性）"""
        config_dict = {}
        
        # Agent設定
        agent_dict = self._config.agent.__dict__.copy()
        config_dict.update(agent_dict)
        config_dict['agent'] = agent_dict
        
        # Model設定
        model_dict = self._config.model.__dict__.copy()
        config_dict.update(model_dict)
        config_dict['model'] = model_dict
        
        # RAG設定
        rag_dict = self._config.rag.__dict__.copy()
        config_dict.update(rag_dict)
        config_dict['rag'] = rag_dict
        
        # Logging設定
        logging_dict = self._config.logging.__dict__.copy()
        config_dict.update(logging_dict)
        config_dict['logging'] = logging_dict
        
        # Memory設定
        memory_dict = self._config.memory.__dict__.copy()
        config_dict.update(memory_dict)
        config_dict['memory'] = memory_dict
        
        return config_dict
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'ConfigManager':
        """シングルトンインスタンスを取得"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """インスタンスをリセット（テスト用）"""
        cls._instance = None
        cls._config = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """グローバル設定マネージャーを取得"""
    return ConfigManager.get_instance(config_path)


def get_config(config_path: Optional[str] = None) -> MurmurNetConfig:
    """設定オブジェクトを取得"""
    manager = get_config_manager(config_path)
    return manager.config


def get_config_dict(config_path: Optional[str] = None) -> Dict[str, Any]:
    """設定辞書を取得（後方互換性のため）"""
    manager = get_config_manager(config_path)
    return manager.to_dict()
