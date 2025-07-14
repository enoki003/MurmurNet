#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet エージェント別モデル設定管理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
各エージェントに異なるモデルを設定する機能

作者: Yuhi Sonoki
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """個別のモデル設定"""
    model_type: str = "huggingface"  # "huggingface", "llama"
    model_name: Optional[str] = None  # HuggingFace用
    model_path: Optional[str] = None  # Llama用
    device: str = "cpu"
    torch_dtype: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.95
    repeat_penalty: float = 1.2
    cache_folder: Optional[str] = None
    local_files_only: bool = False
    # エージェント固有設定
    agent_role: Optional[str] = None
    description: str = ""

@dataclass 
class AgentModelMapping:
    """エージェント別モデルマッピング設定"""
    # 内部エージェント（AgentPoolのエージェント）
    internal_agents: ModelConfig = field(default_factory=ModelConfig)
    
    # 出力エージェント（最終応答生成）  
    output_agent: ModelConfig = field(default_factory=ModelConfig)
    
    # 要約エンジン（要約生成）
    summary_engine: ModelConfig = field(default_factory=ModelConfig)
    
    # 全エージェント共通設定（オーバーライド可能）
    global_config: ModelConfig = field(default_factory=ModelConfig)

class AgentModelManager:
    """エージェント別モデル管理クラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        エージェント別モデル管理の初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config or {}
        self.model_mapping = self._build_model_mapping()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _build_model_mapping(self) -> AgentModelMapping:
        """設定からエージェント別モデルマッピングを構築"""
        
        # グローバル設定（デフォルト）
        global_config = ModelConfig(
            model_type=self.config.get('model_type', 'huggingface'),
            model_name=self.config.get('huggingface_model_name'),
            model_path=self.config.get('model_path'),
            device=self.config.get('device', 'cpu'),
            torch_dtype=self.config.get('torch_dtype', 'auto'),
            temperature=self.config.get('temperature', 0.7),
            max_tokens=self.config.get('max_tokens', 256),
            cache_folder=self.config.get('cache_folder'),
            local_files_only=self.config.get('local_files_only', False)
        )
        
        # 各エージェント別設定の取得または生成
        agent_models = self.config.get('agent_models', {})
        
        # 内部エージェント設定
        internal_config = self._merge_config(
            global_config,
            agent_models.get('internal_agents', {}),
            agent_role="internal",
            description="多角的視点で分析する内部エージェント"
        )
        
        # 出力エージェント設定
        output_config = self._merge_config(
            global_config,
            agent_models.get('output_agent', {}),
            agent_role="output",
            description="最終応答を統合生成するエージェント"
        )
        
        # 要約エンジン設定
        summary_config = self._merge_config(
            global_config,
            agent_models.get('summary_engine', {}),
            agent_role="summary",
            description="情報を要約するエージェント"
        )
        
        return AgentModelMapping(
            internal_agents=internal_config,
            output_agent=output_config,
            summary_engine=summary_config,
            global_config=global_config
        )
    
    def _merge_config(self, base_config: ModelConfig, agent_override: Dict[str, Any], 
                     agent_role: str, description: str) -> ModelConfig:
        """基本設定とエージェント固有設定をマージ"""
        
        merged = ModelConfig(
            model_type=agent_override.get('model_type', base_config.model_type),
            model_name=agent_override.get('model_name', base_config.model_name),
            model_path=agent_override.get('model_path', base_config.model_path),
            device=agent_override.get('device', base_config.device),
            torch_dtype=agent_override.get('torch_dtype', base_config.torch_dtype),
            temperature=agent_override.get('temperature', base_config.temperature),
            max_tokens=agent_override.get('max_tokens', base_config.max_tokens),
            top_p=agent_override.get('top_p', base_config.top_p),
            repeat_penalty=agent_override.get('repeat_penalty', base_config.repeat_penalty),
            cache_folder=agent_override.get('cache_folder', base_config.cache_folder),
            local_files_only=agent_override.get('local_files_only', base_config.local_files_only),
            agent_role=agent_role,
            description=description
        )
        
        return merged
    
    def get_agent_config(self, agent_type: str) -> ModelConfig:
        """指定されたエージェントタイプの設定を取得"""
        
        if agent_type == "internal" or agent_type == "internal_agents":
            return self.model_mapping.internal_agents
        elif agent_type == "output" or agent_type == "output_agent":
            return self.model_mapping.output_agent
        elif agent_type == "summary" or agent_type == "summary_engine":
            return self.model_mapping.summary_engine
        else:
            self.logger.warning(f"未知のエージェントタイプ: {agent_type}, グローバル設定を返します")
            return self.model_mapping.global_config
    
    def get_model_factory_config(self, agent_type: str) -> Dict[str, Any]:
        """ModelFactoryで使用できる形式の設定辞書を取得"""
        
        agent_config = self.get_agent_config(agent_type)
        
        # ModelFactory用の設定辞書に変換
        factory_config = {
            'model_type': agent_config.model_type,
            'device': agent_config.device,
            'torch_dtype': agent_config.torch_dtype,
            'cache_folder': agent_config.cache_folder,
            'local_files_only': agent_config.local_files_only,
            'debug': self.config.get('debug', False)
        }
        
        # モデル固有設定
        if agent_config.model_type == 'huggingface':
            factory_config['huggingface_model_name'] = agent_config.model_name
        elif agent_config.model_type == 'llama':
            factory_config['model_path'] = agent_config.model_path
            factory_config['chat_template'] = self.config.get('chat_template')
        
        return factory_config
    
    def get_generation_params(self, agent_type: str) -> Dict[str, Any]:
        """指定されたエージェントタイプの生成パラメータを取得"""
        
        agent_config = self.get_agent_config(agent_type)
        
        return {
            'max_tokens': agent_config.max_tokens,
            'temperature': agent_config.temperature,
            'top_p': agent_config.top_p,
            'repeat_penalty': agent_config.repeat_penalty
        }
    
    def list_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """全エージェント設定の一覧を取得（デバッグ用）"""
        
        return {
            'internal_agents': {
                'model_type': self.model_mapping.internal_agents.model_type,
                'model_name': self.model_mapping.internal_agents.model_name,
                'model_path': self.model_mapping.internal_agents.model_path,
                'temperature': self.model_mapping.internal_agents.temperature,
                'max_tokens': self.model_mapping.internal_agents.max_tokens,
                'description': self.model_mapping.internal_agents.description
            },
            'output_agent': {
                'model_type': self.model_mapping.output_agent.model_type,
                'model_name': self.model_mapping.output_agent.model_name,
                'model_path': self.model_mapping.output_agent.model_path,
                'temperature': self.model_mapping.output_agent.temperature,
                'max_tokens': self.model_mapping.output_agent.max_tokens,
                'description': self.model_mapping.output_agent.description
            },
            'summary_engine': {
                'model_type': self.model_mapping.summary_engine.model_type,
                'model_name': self.model_mapping.summary_engine.model_name,
                'model_path': self.model_mapping.summary_engine.model_path,
                'temperature': self.model_mapping.summary_engine.temperature,
                'max_tokens': self.model_mapping.summary_engine.max_tokens,
                'description': self.model_mapping.summary_engine.description
            }
        }

def create_agent_model_manager(config: Dict[str, Any]) -> AgentModelManager:
    """AgentModelManagerのファクトリ関数"""
    return AgentModelManager(config)
