#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet プロンプト管理モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~
様々なモデル向けのプロンプトテンプレート管理

修正内容：
- 空ファイルから完全実装
- System/Instructionテンプレートの追加
- 小型モデル向け最適化
- 役割別プロンプト生成

作者: Yuhi Sonoki
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BasePromptTemplate(ABC):
    """プロンプトテンプレートの基底クラス"""
    
    @abstractmethod
    def build_prompt(self, user_text: str, role: str = "assistant", context: str = "") -> str:
        """プロンプトを構築"""
        pass
    
    @abstractmethod
    def get_system_prompt(self, role: str = "assistant") -> str:
        """システムプロンプトを取得"""
        pass

class LlamaInstructTemplate(BasePromptTemplate):
    """Llama/Gemma向けInstructionテンプレート"""
    
    def get_system_prompt(self, role: str = "assistant") -> str:
        """役割別システムプロンプト"""
        system_prompts = {
            "assistant": "あなたは率直で批判的思考を持つ日本語アシスタントです。質問に対して具体的で実用的な回答を提供してください。",
            "researcher": "あなたは詳細な調査と分析を行う研究者です。事実に基づいた正確な情報を提供し、複数の視点から問題を検討してください。",
            "critic": "あなたは建設的な批評を行う専門家です。問題点や改善案を具体的に指摘し、代替案を提示してください。",
            "writer": "あなたは明確で読みやすい文章を書く専門家です。情報を整理し、理解しやすい形で表現してください。",
            "judge": "あなたは複数の意見を総合的に評価する判断者です。各観点のメリット・デメリットを整理し、バランスの取れた結論を導いてください。"
        }
        return system_prompts.get(role, system_prompts["assistant"])
    
    def build_prompt(self, user_text: str, role: str = "assistant", context: str = "") -> str:
        """Llama/Gemma向けプロンプト構築"""
        system_prompt = self.get_system_prompt(role)
        
        # コンテキスト情報の追加
        if context.strip():
            context_section = f"\n\n参考情報:\n{context.strip()}\n"
        else:
            context_section = ""
        
        # Instruction形式のプロンプト
        prompt = f"""<start_of_turn>user
{system_prompt}

{context_section}
質問: {user_text.strip()}

上記の質問について、あなたの役割に基づいて回答してください。
<end_of_turn>
<start_of_turn>model
"""
        
        return prompt

class HuggingFaceTemplate(BasePromptTemplate):
    """HuggingFace汎用モデル向けテンプレート（複数モデル対応）"""
    
    def __init__(self, model_name: str = ""):
        """
        HuggingFaceテンプレートの初期化
        
        Args:
            model_name: モデル名（テンプレート選択に使用）
        """
        self.model_name = model_name.lower()
        self.template_type = self._detect_template_type()
        logger.info(f"HuggingFaceテンプレート初期化: {self.model_name} -> {self.template_type}")
    
    def _detect_template_type(self) -> str:
        """モデル名からテンプレートタイプを検出"""
        if "llm-jp" in self.model_name:
            return "llm_jp"
        elif "elyza" in self.model_name:
            return "elyza"
        elif "swallow" in self.model_name:
            return "swallow"
        elif "calm2" in self.model_name:
            return "calm2"
        elif "japanese-stablelm" in self.model_name:
            return "japanese_stablelm"
        elif "rinna" in self.model_name:
            return "rinna"
        else:
            logger.warning(f"未知のモデル: {self.model_name}, llm_jpテンプレートを使用")
            return "llm_jp"
    
    def get_system_prompt(self, role: str = "assistant") -> str:
        """役割別システムプロンプト（自然で直接的）"""
        system_prompts = {
            "assistant": "あなたは親切で知識豊富なアシスタントです。質問に対して直接的で具体的な回答をしてください。",
            "researcher": "あなたは研究者です。事実に基づいて詳細に分析し、根拠を示しながら説明してください。", 
            "critic": "あなたは専門家です。問題点を指摘し、改善案を具体的に提案してください。",
            "writer": "あなたは文章の専門家です。情報を整理して分かりやすく書いてください。",
            "judge": "あなたは判断者です。各意見を比較検討し、根拠と共に結論を示してください。"
        }
        return system_prompts.get(role, system_prompts["assistant"])
    
    def build_prompt(self, user_text: str, role: str = "assistant", context: str = "") -> str:
        """HuggingFace向けプロンプト構築（モデル別テンプレート対応）"""
        system_prompt = self.get_system_prompt(role)
        
        # コンテキスト情報を含むユーザーメッセージを構築
        user_message = user_text.strip()
        if context.strip():
            user_message = f"{context.strip()[:200]}\n\n質問: {user_message}"
        
        # モデル別テンプレート切り替え
        if self.template_type == "llm_jp":
            return self._build_llm_jp_prompt(system_prompt, user_message)
        elif self.template_type == "elyza":
            return self._build_elyza_prompt(system_prompt, user_message)
        elif self.template_type == "swallow":
            return self._build_swallow_prompt(system_prompt, user_message)
        elif self.template_type == "calm2":
            return self._build_calm2_prompt(system_prompt, user_message)
        elif self.template_type == "japanese_stablelm":
            return self._build_japanese_stablelm_prompt(system_prompt, user_message)
        elif self.template_type == "rinna":
            return self._build_rinna_prompt(system_prompt, user_message)
        else:
            return self._build_generic_prompt(system_prompt, user_message)
    
    def _build_llm_jp_prompt(self, system_prompt: str, user_message: str) -> str:
        """llm-jp向けプロンプト（公式チャットテンプレート）"""
        # llm-jp-3の公式テンプレート: <|system|>...<|user|>...<|assistant|>
        return f"<|system|>{system_prompt}</s><|user|>{user_message}</s><|assistant|>"
    
    def _build_elyza_prompt(self, system_prompt: str, user_message: str) -> str:
        """ELYZA向けプロンプト"""
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""
    
    def _build_swallow_prompt(self, system_prompt: str, user_message: str) -> str:
        """Swallow向けプロンプト"""
        return f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
{system_prompt}

### 入力:
{user_message}

### 応答:
"""
    
    def _build_calm2_prompt(self, system_prompt: str, user_message: str) -> str:
        """CALM2向けプロンプト"""
        return f"""USER: {system_prompt}

{user_message}
ASSISTANT: """
    
    def _build_japanese_stablelm_prompt(self, system_prompt: str, user_message: str) -> str:
        """Japanese StableLM向けプロンプト"""
        return f"""<|system|>
{system_prompt}<|user|>
{user_message}<|assistant|>
"""
    
    def _build_rinna_prompt(self, system_prompt: str, user_message: str) -> str:
        """rinna向けプロンプト"""
        return f"""システム: {system_prompt}

ユーザー: {user_message}

アシスタント: """
    
    def _build_generic_prompt(self, system_prompt: str, user_message: str) -> str:
        """汎用プロンプト（フォールバック）"""
        return f"""<human>{system_prompt}

{user_message}</s><bot>"""

class PromptManager:
    """プロンプト管理クラス"""
    
    def __init__(self, model_type: str = "llama", model_name: str = ""):
        self.model_type = model_type
        self.model_name = model_name
        self.template = self._get_template(model_type, model_name)
        logger.info(f"プロンプトマネージャー初期化: {model_type}用テンプレート (モデル: {model_name})")
    
    def _get_template(self, model_type: str, model_name: str = "") -> BasePromptTemplate:
        """モデルタイプに応じたテンプレートを取得"""
        if model_type == "llama":
            return LlamaInstructTemplate()
        elif model_type == "huggingface":
            return HuggingFaceTemplate(model_name)
        else:
            logger.warning(f"未知のモデルタイプ: {model_type}, llama用テンプレートを使用")
            return LlamaInstructTemplate()
    
    def build_prompt(self, user_text: str, role: str = "assistant", context: str = "") -> str:
        """プロンプトを構築"""
        if not user_text or not user_text.strip():
            logger.warning("空のユーザーテキストが入力されました")
            return "質問を入力してください。"
        
        try:
            prompt = self.template.build_prompt(user_text, role, context)
            logger.debug(f"プロンプト構築完了: 役割={role}, 長さ={len(prompt)}文字")
            return prompt
        except Exception as e:
            logger.error(f"プロンプト構築エラー: {str(e)}")
            # フォールバック: 最小限のプロンプト
            return f"質問: {user_text.strip()}\n\n回答:"
    
    def get_system_prompt(self, role: str = "assistant") -> str:
        """システムプロンプトを取得"""
        return self.template.get_system_prompt(role)
    
    def format_multi_agent_prompt(self, user_text: str, agent_roles: List[str], context: str = "") -> Dict[str, str]:
        """複数エージェント用プロンプトを構築"""
        prompts = {}
        
        for i, role in enumerate(agent_roles):
            # エージェント固有のコンテキスト
            agent_context = context
            if len(agent_roles) > 1:
                agent_context += f"\n\n[エージェント{i+1}として{role}の視点で回答してください]"
            
            prompts[f"agent_{i}"] = self.build_prompt(user_text, role, agent_context)
        
        logger.info(f"マルチエージェント用プロンプト構築完了: {len(agent_roles)}エージェント")
        return prompts

# デフォルトのプロンプトマネージャーインスタンス
_default_prompt_manager = None

def get_prompt_manager(model_type: str = "llama", model_name: str = "") -> PromptManager:
    """デフォルトのプロンプトマネージャーを取得"""
    global _default_prompt_manager
    cache_key = f"{model_type}:{model_name}"
    
    if (_default_prompt_manager is None or 
        _default_prompt_manager.model_type != model_type or 
        _default_prompt_manager.model_name != model_name):
        _default_prompt_manager = PromptManager(model_type, model_name)
    return _default_prompt_manager

def build_prompt(user_text: str, role: str = "assistant", context: str = "", 
                model_type: str = "llama", model_name: str = "") -> str:
    """便利関数: プロンプトを構築"""
    manager = get_prompt_manager(model_type, model_name)
    return manager.build_prompt(user_text, role, context)

# テスト用の関数
def test_prompt_templates():
    """プロンプトテンプレートのテスト"""
    test_question = "AIは教育をどう変える？"
    test_context = "人工知能技術の発展により、個別化学習が可能になっています。"
    
    print("=== プロンプトテンプレートテスト ===")
    
    # Llamaテンプレート
    llama_manager = PromptManager("llama")
    llama_prompt = llama_manager.build_prompt(test_question, "researcher", test_context)
    print(f"Llamaテンプレート:\n{llama_prompt}\n")
    
    # HuggingFace llm-jpテンプレート
    llm_jp_manager = PromptManager("huggingface", "llm-jp/llm-jp-3-150m-instruct3") 
    llm_jp_prompt = llm_jp_manager.build_prompt(test_question, "researcher", test_context)
    print(f"llm-jpテンプレート:\n{llm_jp_prompt}\n")
    
    # HuggingFace ELYZAテンプレート
    elyza_manager = PromptManager("huggingface", "elyza/Llama-3-ELYZA-JP-8B")
    elyza_prompt = elyza_manager.build_prompt(test_question, "researcher", test_context)
    print(f"ELYZAテンプレート:\n{elyza_prompt}\n")
    
    # HuggingFace Swallowテンプレート
    swallow_manager = PromptManager("huggingface", "tokyotech-llm/Swallow-7b-instruct-hf")
    swallow_prompt = swallow_manager.build_prompt(test_question, "researcher", test_context)
    print(f"Swallowテンプレート:\n{swallow_prompt}\n")
    
    # マルチエージェント
    roles = ["researcher", "critic", "writer"]
    multi_prompts = llama_manager.format_multi_agent_prompt(test_question, roles, test_context)
    print(f"マルチエージェント用プロンプト: {len(multi_prompts)}個生成")

if __name__ == "__main__":
    test_prompt_templates()
