#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slot Based Architecture (enhanced for structured blackboard)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分散SLMシステムのSlotベースアーキテクチャの実装

主要機能:
- BaseSlot: 汎用Slot基底クラス（model-template alignment対応）
- Slot variants: Reformulator, Critic, Supporter, Synthesizer
- SlotRunner: Slot実行エンジン（構造化Blackboard対応）
- SlotBlackboard: Slotデータストレージ（新しいBlackboardへの移行）

作者: Yuhi Sonoki
改良: 構造化Blackboard対応、model-template alignment修正
"""

import re
import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# 新しい構造化Blackboard関連のインポート
try:
    from .structured_blackboard import StructuredBlackboard, AgentRole, BlackboardSnapshot
    from .slot_adapter import SlotBlackboardAdapter
    HAS_STRUCTURED_BB = True
except ImportError:
    StructuredBlackboard = AgentRole = BlackboardSnapshot = SlotBlackboardAdapter = None
    HAS_STRUCTURED_BB = False

# 既存のモジュール
from .blackboard import Blackboard
from .slot_blackboard import SlotBlackboard, SlotEntry
from .model_factory import ModelFactory
from .model_factory_singleton import ModelFactorySingleton
from .prompt_manager import PromptManager
from .embedder import Embedder

try:
    from .boids import BoidsController
    HAS_BOIDS = True
except ImportError:
    BoidsController = None
    HAS_BOIDS = False

logger = logging.getLogger(__name__)

###############################################################################
# Utility functions
###############################################################################

def _import_boids():
    """Boids 関連コンポーネントを遅延インポート。失敗しても None を返す。"""
    if HAS_BOIDS:
        try:
            from .boids import BoidsController, VectorSpace  # type: ignore
            return BoidsController, VectorSpace
        except ImportError:
            pass
    return None, None

def _import_enhanced_synthesizer():
    try:
        from .enhanced_synthesizer import BoidsBasedSynthesizer  # type: ignore
        return BoidsBasedSynthesizer
    except ImportError:
        return None

###############################################################################
# Base Slot
###############################################################################

class BaseSlot(ABC):
    """全 Slot の基底クラス（抽象クラス）。"""

    # ---------------------------------------------------------------------
    # Initialisation / configuration
    # ---------------------------------------------------------------------

    def __init__(self, name: str, cfg: Dict[str, Any], model_factory: ModelFactory):
        self.name = name
        self.cfg = cfg
        self.model_factory = model_factory
        self.debug: bool = cfg.get("debug", False)

        # Generation parameters (slot‑local override可)
        self.max_output_len: int = cfg.get("slot_max_output_length", 200)
        self.temperature: float = cfg.get("slot_temperature", 0.8)
        self.top_p: float = cfg.get("slot_top_p", 0.9)

        # Statistics
        self.exec_count = 0
        self.total_exec_time = 0.0
        self.last_exec_time: Optional[float] = None

        if self.debug:
            print(f"Slot '{self.name}' 初期化完了")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_role_description(self) -> str:  # noqa: D401 – JP docstring style
        """Slot の役割を簡潔に返す。"""
        pass

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Slot 固有の system prompt を構築。"""
        pass

    @abstractmethod
    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        """ユーザー向け prompt を構築。"""
        pass

    # ------------------------------------------------------------------
    # Collaborative discussion methods (協調的議論メソッド)
    # ------------------------------------------------------------------

    def build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> Optional[str]:
        """他のSlotの意見を参照・反論するプロンプトを構築（オプション実装）。
        
        Args:
            bb: SlotBlackboard
            user_input: ユーザー入力
            other_opinions: 他のSlotの意見リスト
        
        Returns:
            相互参照用プロンプト（実装しない場合はNone）
        """
        return None  # デフォルトは実装なし

    def build_consensus_prompt(self, bb: SlotBlackboard, user_input: str, all_opinions: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Optional[str]:
        """合意形成・対立解消のプロンプトを構築（オプション実装）。
        
        Args:
            bb: SlotBlackboard
            user_input: ユーザー入力
            all_opinions: 全ての意見
            conflicts: 検出された対立点
        
        Returns:
            合意形成用プロンプト（実装しない場合はNone）
        """
        return None  # デフォルトは実装なし

    def evaluate_opinion_quality(self, opinion_text: str, metadata: Dict[str, Any]) -> float:
        """意見の品質を評価（0.0-1.0）。デフォルト実装。"""
        # 基本的な品質指標：長さ、構造、キーワード
        if not opinion_text or len(opinion_text.strip()) < 10:
            return 0.1
        
        quality_score = 0.5  # ベース品質
        
        # 長さによる評価（適度な長さを好む）
        length = len(opinion_text.strip())
        if 50 <= length <= 500:
            quality_score += 0.2
        elif length > 500:
            quality_score += 0.1
        
        # 構造化指標
        if '。' in opinion_text or '.' in opinion_text:
            quality_score += 0.1
        if '、' in opinion_text or ',' in opinion_text:
            quality_score += 0.1
        
        # メタデータからの品質情報
        if metadata.get('confidence', 0) > 0.7:
            quality_score += 0.1
        
        return min(quality_score, 1.0)

    # ------------------------------------------------------------------
    # Public execution entry‑point
    # ------------------------------------------------------------------

    def execute(
        self,
        bb: SlotBlackboard,
        user_input: str,
        embedder=None,
    ) -> Optional[SlotEntry]:
        """Prompt を組み立て、モデル生成→Blackboard へ書き込む。"""

        t0 = time.time()

        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self.build_user_prompt(bb, user_input)

            # プロンプトのNullチェック
            if sys_prompt is None:
                logger.error(f"{self.name}: システムプロンプトがNoneです")
                return None
            
            if usr_prompt is None:
                logger.error(f"{self.name}: ユーザープロンプトがNoneです")
                return None

            if self.debug:
                print(f"\n--- {self.name} ---")
                print(f"システムプロンプト: {sys_prompt[:120].replace('\n', ' ')}")
                print("─" * 60)

            response = self._generate_response(sys_prompt, usr_prompt)
            
            # 応答のNullチェック
            if not response or response.strip() == "":
                logger.warning(f"{self.name}: モデル応答が空です")
                return None

            # optional embedding
            embedding = None
            if embedder and response:
                try:
                    embedding = embedder.embed_text(response)
                except Exception as e:  # pylint: disable=broad-except
                    if self.debug:
                        print(f"埋め込み生成エラー: {e}")

            # write to blackboard
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
            }
            entry = bb.add_slot_entry(self.name, response, embedding, metadata)

            # stats
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time

            return entry
        except Exception as e:  # pylint: disable=broad-except
            if self.debug:
                import traceback
                traceback.print_exc()
            logger.error(f"Slot '{self.name}' 実行エラー: {e}")
            return None

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """モデル種別を判別し、最適な方法でテキストを生成。"""

        # 利用可能なモデルを取得（150mが無い場合は任意のモデル）
        try:
            model = self.model_factory.get_model("150m")
        except Exception as e:
            if self.debug:
                print(f"150Mモデル取得エラー: {e}")
            model = None
        
        if model is None:
            try:
                model = self.model_factory.get_any_available_model()
            except Exception as e:
                if self.debug:
                    print(f"利用可能モデル取得エラー: {e}")
                return f"モデル取得エラー: {str(e)[:50]}"
        
        if model is None:
            return "モデルが利用できません。"

        # llama‑cpp wrapper が持つ create_chat_completion を使えるなら、それで完結
        if hasattr(model, "model_manager") and hasattr(model.model_manager, "model"):
            llm = model.model_manager.model  # type: ignore[attr-defined]
            if hasattr(llm, "create_chat_completion"):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    result = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=self.max_output_len,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stream=False,
                    )
                    text = result["choices"][0]["message"]["content"]
                    return self._clean_response(text)
                except Exception as e:  # pylint: disable=broad-except
                    if self.debug:
                        print(f"Chat‑completion エラー: {e}; テンプレート fallback")

        # 2) manual prompt‑template fallback
        path_lower = getattr(model, "model_path", "").lower()
        is_gemma = "gemma" in path_lower

        if is_gemma:
            full_prompt = (
                f"<|start_of_turn|>system\n{system_prompt}<|end_of_turn|>\n"
                f"<|start_of_turn|>user\n{user_prompt}<|end_of_turn|>\n"
                f"<|start_of_turn|>assistant\n"
            )
        else:  # llama / その他 instruct
            full_prompt = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            )

        # transformers vs other wrapper detection
        if hasattr(model, "_tokenizer") and hasattr(model, "_model"):
            return self._generate_with_transformers(model, full_prompt)

        # generic generate API
        gen_params = {
            "max_new_tokens": self.max_output_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
            "repetition_penalty": 1.1,
        }
        try:
            raw = model.generate(full_prompt, **gen_params)  # type: ignore[arg-type]
        except Exception as e:  # pylint: disable=broad-except
            return f"生成エラー: {e}"

        return self._clean_response(str(raw))

    # --------------------------------------------------------------
    # Transformers‑only helper
    # --------------------------------------------------------------

    def _generate_with_transformers(self, model, prompt: str) -> str:  # type: ignore[no-self-use]
        import torch

        model._ensure_initialized() if hasattr(model, "_ensure_initialized") else None

        device = getattr(model, "device", "cpu")
        tokenizer = model._tokenizer  # type: ignore[attr-defined]
        hf_model = model._model  # type: ignore[attr-defined]

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        gen_params = {
            "max_new_tokens": self.max_output_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }
        with torch.no_grad():
            outputs = hf_model.generate(inputs, **gen_params)
        generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return self._clean_response(generated)

    # --------------------------------------------------------------
    # Output cleaning
    # --------------------------------------------------------------

    def _clean_response(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()

        # Gemma / Llama special‑token removal
        text = re.sub(r"<\|[^>]+?\|>", "", text)  # Gemma tokens
        text = re.sub(r"<\/?(system|user|assistant)>", "", text)
        text = re.sub(r"\[\/?INST]", "", text)
        text = re.sub(r"<<\/?SYS>>", "", text)
        text = re.sub(r"<s>", "", text)

        text = re.sub(r"</?\w+[^>]*?>", "", text)  # stray HTML
        text = text.strip()

        if len(text) > self.max_output_len:
            text = text[: self.max_output_len].rsplit(" ", 1)[0] + "…"
        return text or "応答を生成しました。"

    # --------------------------------------------------------------
    # Statistics helper
    # --------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        avg = self.total_exec_time / self.exec_count if self.exec_count else 0.0
        return {
            "name": self.name,
            "role": self.get_role_description(),
            "execution_count": self.exec_count,
            "total_execution_time": self.total_exec_time,
            "average_execution_time": avg,
            "last_execution_time": self.last_exec_time,
        }

###############################################################################
#                         ––– Individual Slots –––
###############################################################################

class ReformulatorSlot(BaseSlot):
    def get_role_description(self) -> str:  # noqa: D401
        return "入力の再構成・拡張"

    def build_system_prompt(self) -> str:
        return (
            "あなたは入力再構成の専門家です。以下の観点で情報を再構成してください:\n"
            "1. 別表現への言い換え\n2. 具体例\n3. 関連側面\n4. 詳細化\n"
            "謝罪・否定的表現は禁止。簡潔で実用的に。"
        )

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        ctx = [f"入力: {user_input}"]
        try:
            entries = bb.get_slot_entries()
            recent_entries = entries[-3:] if entries else []
            for entry in recent_entries:
                if entry.slot_name != self.name:
                    ctx.append(f"{entry.slot_name}: {entry.text[:80]}")
        except Exception as e:
            if self.debug:
                print(f"SlotBlackboard取得エラー: {e}")
        context = "\n".join(ctx)
        return f"以下の入力を再構成してください:\n\n{context}\n\nより多角的に再構成せよ。"

class CriticSlot(BaseSlot):
    def get_role_description(self) -> str:
        return "批判的分析・課題指摘"

    def build_system_prompt(self) -> str:
        return (
            "あなたは建設的批評家です。次の観点で分析せよ:\n"
            "1. 潜在課題 2. 改善余地 3. 別角度 4. 注意点\n"
            "否定に偏らず実用的に。謝罪不要。"
        )

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        parts = [f"元の入力: {user_input}"]
        try:
            others = [e for e in bb.get_slot_entries() if e.slot_name != self.name]
            if others:
                parts.append("他の視点:")
                for e in others[-2:]:
                    parts.append(f"・{e.slot_name}: {e.text[:100]}")
        except Exception as e:
            if self.debug:
                print(f"SlotBlackboard取得エラー: {e}")
        return "\n".join(parts)

class SupporterSlot(BaseSlot):
    def get_role_description(self) -> str:
        return "肯定的支援・価値発見"

    def build_system_prompt(self) -> str:
        return (
            "あなたは支援的アドバイザーです。以下を行ってください:\n"
            "1. 良点の指摘 2. 可能性 3. 励まし 4. 次の一手\n"
            "常に前向きに。謝罪不要。"
        )

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        parts = [f"元の入力: {user_input}"]
        try:
            critics = bb.get_slot_entries("CriticSlot")
            if critics:
                parts.append(f"批評: {critics[-1].text[:100]}")
        except Exception as e:
            if self.debug:
                print(f"SlotBlackboard取得エラー: {e}")
        return "\n".join(parts)

class SynthesizerSlot(BaseSlot):
    def get_role_description(self) -> str:
        return "多視点統合・最終応答"

    def build_system_prompt(self) -> str:
        return (
            "あなたは統合専門家です。複数の視点を総合し、有用な最終回答を作成せよ。"
        )

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        ctx = [f"ユーザー入力: {user_input}", "", "各 Slot の視点:"]
        try:
            for e in bb.get_slot_entries():
                if e.slot_name != self.name:
                    prefix = {
                        "ReformulatorSlot": "【再構成】",
                        "CriticSlot": "【批評】",
                        "SupporterSlot": "【支援】",
                    }.get(e.slot_name, "【その他】")
                    ctx.append(f"{prefix} {e.text}")
        except Exception as e:
            if self.debug:
                print(f"SlotBlackboard取得エラー: {e}")
        return "\n".join(ctx)

###############################################################################
# SlotRunner
###############################################################################
class SlotRunner:
    """
    Slot を実行し、結果を統合するクラス（構造化Blackboard対応）
    
    主要な改良点:
    - SlotBlackboardAdapterによる新しい構造化Blackboard対応
    - エージェント役割に基づく書き込み制御
    - バージョン管理による一貫性保証
    - 既存APIとの後方互換性維持
    """

    def __init__(self, config: Dict[str, Any], model_factory: ModelFactory, embedder=None):
        self.config = config
        self.model_factory = model_factory
        self.embedder = embedder
        self.debug = config.get('debug', False)
        
        # 構造化Blackboard設定
        self.use_structured_blackboard = config.get('use_structured_blackboard', True) and HAS_STRUCTURED_BB
        
        # Slot構成設定
        self.slot_config = config.get('slot_configuration', {})
        self.use_boids_synthesizer = config.get('use_boids_synthesizer', False)
        
        # 構造化Blackboardアダプターの初期化
        if self.use_structured_blackboard:
            try:
                self.blackboard_adapter = SlotBlackboardAdapter(config)
                if self.debug:
                    logger.info("構造化Blackboardアダプター初期化完了")
            except Exception as e:
                logger.error(f"構造化Blackboardアダプター初期化失敗: {e}")
                self.blackboard_adapter = None
                self.use_structured_blackboard = False
                if self.debug:
                    logger.warning("レガシーBlackboardモードにフォールバック")
        else:
            self.blackboard_adapter = None
            if self.debug:
                logger.info("レガシーBlackboardモード")
        
        # Slotの初期化
        self.slots = self._initialize_slots()
        
        # 実行順序
        self.execution_order = self._build_execution_order()
        
        # 統計
        self.total_runs = 0
        self.successful_runs = 0
        self.quality_history = []

        if self.debug:
            print(f"SlotRunner初期化完了: {len(self.slots)}個のSlot (構造化BB: {self.use_structured_blackboard})")
    
    def _initialize_slots(self) -> Dict[str, BaseSlot]:
        """Slotの初期化（改良版Slot対応）"""
        
        # 改良版Slotが有効な場合
        if self.config.get('use_enhanced_slots', True):
            try:
                from .enhanced_slots import create_enhanced_slots
                slots = create_enhanced_slots(self.config, self.model_factory)
                if self.debug:
                    logger.info("改良版Slotを初期化しました")
                return slots
            except ImportError as e:
                logger.warning(f"改良版Slotのインポートに失敗、従来版を使用: {e}")
        
        # 従来版Slotの初期化
        slots = {}
        
        # 基本Slotを作成
        base_slots = {
            'ReformulatorSlot': ReformulatorSlot,
            'CriticSlot': CriticSlot,
            'SupporterSlot': SupporterSlot,
            'SynthesizerSlot': SynthesizerSlot
        }
        
        for slot_name, slot_class in base_slots.items():
            # 個別Slot設定があればそれを使用
            slot_specific_config = self.slot_config.get(slot_name, {})
            merged_config = {**self.config, **slot_specific_config}
            
            slots[slot_name] = slot_class(slot_name, merged_config, self.model_factory)
        
        return slots
    
    def _build_execution_order(self) -> List[str]:
        """実行順序の構築"""
        custom_order = self.config.get('slot_execution_order')
        if custom_order:
            return [slot for slot in custom_order if slot in self.slots]
        
        # デフォルト順序
        default_order = ['ReformulatorSlot', 'CriticSlot', 'SupporterSlot', 'SynthesizerSlot']
        return [slot for slot in default_order if slot in self.slots]

    def run_all_slots(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """
        全Slotを順序立てて実行し、詳細な結果を返す（構造化Blackboard対応）
        
        Args:
            bb: Slot Blackboard
            user_input: ユーザー入力
            embedder: 埋め込み生成器
        
        Returns:
            実行結果の辞書（distributed_slm.pyが期待する形式）
        """
        start_time = time.time()
        self.total_runs += 1
        
        try:
            results = {}
            execution_times = {}
            quality_scores = []
            
            if self.debug:
                print(f"SlotRunner実行開始: {len(self.execution_order)}個のSlot (構造化BB: {self.use_structured_blackboard})")
            
            # 構造化Blackboardの場合、ラウンドベース実行
            if self.use_structured_blackboard and self.blackboard_adapter:
                try:
                    results = self._run_all_slots_structured(bb, user_input, embedder)
                except Exception as e:
                    logger.error(f"構造化Blackboard実行エラー: {e}")
                    if self.debug:
                        import traceback
                        logger.debug(traceback.format_exc())
                    # フォールバックとしてレガシー実行
                    logger.info("レガシーBlackboardにフォールバック")
                    results = self._run_all_slots_legacy(bb, user_input, embedder)
            else:
                results = self._run_all_slots_legacy(bb, user_input, embedder)
            
            # 共通の結果処理
            return self._process_slot_results(results, start_time, user_input)
            
        except Exception as e:
            return self._handle_slot_error(e, start_time)

    def _run_all_slots_structured(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """構造化Blackboard版の全Slot実行"""
        results = {}
        execution_times = {}
        quality_scores = []  # 品質スコアリストを初期化
        
        # アダプターにSlotBlackboardを接続
        self.blackboard_adapter.connect_legacy_blackboard(bb)
        
        # 新しいラウンドを開始
        snapshot, version = self.blackboard_adapter.structured_bb.start_round()
        
        if self.debug:
            logger.info(f"構造化Blackboardラウンド開始: version {version}")
        
        # Slotを順序立てて実行
        commits = []
        for slot_name in self.execution_order:
            slot = self.slots[slot_name]
            
            slot_start = time.time()
            entry = slot.execute(bb, user_input, embedder or self.embedder)
            slot_time = time.time() - slot_start
            
            execution_times[slot_name] = slot_time
            
            if entry:
                results[slot_name] = {
                    'entry': entry,
                    'text': entry.text,
                    'execution_time': slot_time,
                    'metadata': entry.metadata
                }
                
                # 構造化Blackboardへの書き込み準備
                agent_role = self.blackboard_adapter._get_agent_role(slot_name)
                
                if self.debug:
                    logger.info(f"Slot実行結果: {slot_name} → 役割: {agent_role.value}")
                
                if agent_role in [AgentRole.REFORMULATOR, AgentRole.CRITIC, AgentRole.SUPPORTER]:
                    commits.append(('add_opinion', slot_name, {
                        'agent_role': agent_role,
                        'content': entry.text,
                        'version_read': version,
                        'metadata': entry.metadata
                    }))
                elif agent_role == AgentRole.SYNTHESIZER:
                    commits.append(('update_summary', slot_name, {
                        'content': entry.text,
                        'version_read': version,
                        'metadata': entry.metadata
                    }))
                
                # 品質スコア記録（メタデータから）
                if 'quality_score' in entry.metadata:
                    quality_scores.append(entry.metadata['quality_score'])
                
                if self.debug:
                    print(f"{slot_name} 完了: {slot_time:.2f}秒 (役割: {agent_role.value})")
            else:
                results[slot_name] = {
                    'entry': None,
                    'text': None,
                    'execution_time': slot_time,
                    'error': True
                }
                if self.debug:
                    print(f"{slot_name} エラー")
        
        # 一括コミット
        if commits:
            try:
                success = self.blackboard_adapter.structured_bb.commit_round(commits)
                if self.debug:
                    logger.info(f"ラウンドコミット: {'成功' if success else '失敗'} ({len(commits)}件)")
            except Exception as e:
                success = False
                logger.error(f"構造化Blackboardコミットエラー: {e}")
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # 構造化Blackboardの統計を追加
            try:
                results['structured_blackboard_stats'] = {
                    'commit_success': success,
                    'version': self.blackboard_adapter.structured_bb.get_current_version(),
                    'opinions_count': len(self.blackboard_adapter.structured_bb.opinions),
                    'knowledge_count': len(self.blackboard_adapter.structured_bb.external_knowledge),
                    'has_summary': self.blackboard_adapter.structured_bb.summary is not None
                }
            except Exception as e:
                logger.warning(f"構造化Blackboard統計取得エラー: {e}")
                results['structured_blackboard_stats'] = {'commit_success': False, 'error': str(e)}
        
        results['execution_times'] = execution_times
        results['quality_scores'] = quality_scores
        return results
    
    def _run_all_slots_legacy(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """従来版の全Slot実行（後方互換性）"""
        results = {}
        execution_times = {}
        quality_scores = []
        
        # Slotを順序立てて実行
        for slot_name in self.execution_order:
            slot = self.slots[slot_name]
            
            slot_start = time.time()
            entry = slot.execute(bb, user_input, embedder or self.embedder)
            slot_time = time.time() - slot_start
            
            execution_times[slot_name] = slot_time
            
            if entry:
                results[slot_name] = {
                    'entry': entry,
                    'text': entry.text,
                    'execution_time': slot_time,
                    'metadata': entry.metadata
                }
                
                # 品質スコア記録（メタデータから）
                if 'quality_score' in entry.metadata:
                    quality_scores.append(entry.metadata['quality_score'])
                
                if self.debug:
                    print(f"{slot_name} 完了: {slot_time:.2f}秒")
            else:
                results[slot_name] = {
                    'entry': None,
                    'text': None,
                    'execution_time': slot_time,
                    'error': True
                }
                if self.debug:
                    print(f"{slot_name} エラー")
        
        results['execution_times'] = execution_times
        results['quality_scores'] = quality_scores
        return results
    
    def _process_slot_results(self, results: Dict[str, Any], start_time: float, user_input: str) -> Dict[str, Any]:
        """Slot実行結果の共通処理"""
        execution_times = results.get('execution_times', {})
        quality_scores = results.get('quality_scores', [])
        
        # execution_timesとquality_scoresを除外してslot_resultsを作成
        slot_results = {}
        for key, value in results.items():
            if key not in ['execution_times', 'quality_scores', 'structured_blackboard_stats']:
                slot_results[key] = value
        
        # 最終統合結果を取得
        final_response = ""
        synthesis_quality = 0.0
        
        if 'SynthesizerSlot' in slot_results and slot_results['SynthesizerSlot'].get('text'):
            final_response = slot_results['SynthesizerSlot']['text']
            
            # 統合品質の取得
            synthesizer_metadata = slot_results['SynthesizerSlot'].get('metadata', {})
            synthesis_quality = synthesizer_metadata.get('quality_score', 0.0)
            
            if self.debug and synthesis_quality > 0:
                print(f"統合品質スコア: {synthesis_quality:.2f}")
        
        total_time = time.time() - start_time
        self.successful_runs += 1
        
        # 品質履歴に記録
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            self.quality_history.append(avg_quality)
            # 履歴サイズ制限
            if len(self.quality_history) > 100:
                self.quality_history = self.quality_history[-50:]
        
        # 構造化Blackboard統計があれば追加
        structured_stats = results.get('structured_blackboard_stats', {})
        
        summary = {
            'success': True,
            'final_response': final_response,
            'slot_results': slot_results,  # 正しく構造化されたslot_results
            'execution_times': execution_times,
            'total_execution_time': total_time,
            'synthesis_quality': synthesis_quality,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'user_input': user_input,
            'boids_enabled': self.use_boids_synthesizer,
            'slot_count': len(self.execution_order),
            'structured_blackboard': self.use_structured_blackboard,
            **structured_stats  # 構造化Blackboard統計をマージ
        }
        
        if self.debug:
            print(f"SlotRunner実行完了: {total_time:.2f}秒")
            print(f"最終応答: {final_response[:100]}...")
            if synthesis_quality > 0:
                print(f"統合品質: {synthesis_quality:.2f}")
            if structured_stats:
                print(f"構造化BB統計: Version {structured_stats.get('version', 'N/A')}, "
                      f"意見 {structured_stats.get('opinions_count', 0)}件")
            print(f"Slot結果構造確認: {list(slot_results.keys())}")  # デバッグ用
        
        return summary
    
    def _handle_slot_error(self, e: Exception, start_time: float) -> Dict[str, Any]:
        """Slot実行エラーの処理"""
        error_summary = {
            'success': False,
            'error': str(e),
            'final_response': "Slotシステムでエラーが発生しました。",
            'execution_time': time.time() - start_time,
            'boids_enabled': self.use_boids_synthesizer,
            'structured_blackboard': self.use_structured_blackboard
        }
        
        if self.debug:
            logger.error(f"SlotRunner実行エラー: {e}")
            import traceback
            traceback.print_exc()
        
        return error_summary

    # ===== 協調的議論システム =====
    
    def run_collaborative_discussion(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """
        真の協調を実現する多段階議論システム
        
        フェーズ1: 初期意見提示
        フェーズ2: 相互参照・反論  
        フェーズ3: 引用付き最終統合
        """
        start_time = time.time()
        self.total_runs += 1
        
        if self.debug:
            print(f"\n🤝 === 協調的議論システム開始 ===")
            print(f"参加Slot: {len(self.execution_order)}個")
            print("=" * 50)
        
        try:
            # フェーズ1: 初期意見収集
            bb.add_discussion_round(1, "初期意見")
            phase1_results = self._run_phase1_initial_opinions(bb, user_input, embedder)
            
            # フェーズ2: 相互参照・議論
            bb.add_discussion_round(2, "相互参照")
            phase2_results = self._run_phase2_cross_reference(bb, user_input, embedder, phase1_results)
            
            # フェーズ3: 最終統合
            bb.add_discussion_round(3, "最終統合")
            final_synthesis = self._run_phase3_synthesis(bb, user_input, embedder, phase2_results)
            
            # 協調度メトリクス計算
            collaboration_metrics = bb.calculate_collaboration_metrics()
            
            # 結果統合
            all_results = {
                'phase1_initial': phase1_results,
                'phase2_discussion': phase2_results,
                'phase3_synthesis': final_synthesis,
                'collaboration_metrics': collaboration_metrics,
                'discussion_history': bb.get_discussion_history()
            }
            
            return self._process_collaborative_results(all_results, start_time, user_input)
            
        except Exception as e:
            return self._handle_slot_error(e, start_time)
    
    def _run_phase1_initial_opinions(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """フェーズ1: 各Slotの初期意見を収集"""
        if self.debug:
            print("📝 フェーズ1: 初期意見提示")
        
        results = {}
        execution_times = {}
        
        # 初期Slotのみ実行（Synthesizerは除外）
        initial_slots = [slot for slot in self.execution_order if slot != 'SynthesizerSlot']
        
        for slot_name in initial_slots:
            slot = self.slots[slot_name]
            
            slot_start = time.time()
            entry = slot.execute(bb, user_input, embedder or self.embedder)
            slot_time = time.time() - slot_start
            
            execution_times[slot_name] = slot_time
            
            if entry:
                # フェーズ情報をメタデータに追加
                entry.metadata['phase'] = 1
                entry.metadata['phase_name'] = '初期意見'
                
                results[slot_name] = {
                    'entry': entry,
                    'text': entry.text,
                    'execution_time': slot_time,
                    'metadata': entry.metadata,
                    'phase': 1
                }
                if self.debug:
                    print(f"  ✅ {slot_name}: {entry.text[:60]}...")
            else:
                results[slot_name] = {'entry': None, 'text': None, 'error': True, 'phase': 1}
                if self.debug:
                    print(f"  ❌ {slot_name}: エラー")
        
        results['execution_times'] = execution_times
        return results
    
    def _run_phase2_cross_reference(self, bb: SlotBlackboard, user_input: str, embedder, phase1_results: Dict[str, Any]) -> Dict[str, Any]:
        """フェーズ2: 他Slotの意見を参照して相互議論"""
        if self.debug:
            print("🔄 フェーズ2: 相互参照・議論")
        
        results = {}
        execution_times = {}
        
        # 相互参照用の特別実行
        for slot_name in [slot for slot in self.execution_order if slot != 'SynthesizerSlot']:
            slot = self.slots[slot_name]
            
            # 他のSlotの意見を取得
            cross_ref_context = bb.get_cross_reference_context(slot_name)
            other_opinions = cross_ref_context.get('other_opinions', [])
            
            if not other_opinions:
                if self.debug:
                    print(f"  ⚠️ {slot_name}: 参照可能な他意見なし、スキップ")
                continue
            
            slot_start = time.time()
            
            # 相互参照プロンプトを構築
            cross_ref_prompt = self._build_cross_reference_prompt(slot_name, user_input, other_opinions)
            
            # 相互参照モードで実行
            entry = self._execute_cross_reference_mode(slot, bb, cross_ref_prompt, embedder)
            
            slot_time = time.time() - slot_start
            execution_times[slot_name] = slot_time
            
            if entry:
                # フェーズ情報を追加
                entry.metadata.update({
                    'phase': 2,
                    'phase_name': '相互参照',
                    'referenced_opinions': len(other_opinions),
                    'cross_reference_mode': True
                })
                
                results[slot_name] = {
                    'entry': entry,
                    'text': entry.text,
                    'execution_time': slot_time,
                    'metadata': entry.metadata,
                    'referenced_opinions': other_opinions,
                    'phase': 2
                }
                if self.debug:
                    print(f"  🔄 {slot_name}: {entry.text[:60]}... (参照{len(other_opinions)}件)")
            else:
                results[slot_name] = {'entry': None, 'text': None, 'error': True, 'phase': 2}
                if self.debug:
                    print(f"  ❌ {slot_name}: 相互参照エラー")
        
        results['execution_times'] = execution_times
        return results
    
    def _run_phase3_synthesis(self, bb: SlotBlackboard, user_input: str, embedder, phase2_results: Dict[str, Any]) -> Dict[str, Any]:
        """フェーズ3: 引用付き最終統合"""
        if self.debug:
            print("📋 フェーズ3: 引用付き最終統合")
        
        synthesizer = self.slots.get('SynthesizerSlot')
        if not synthesizer:
            return {'error': 'SynthesizerSlot not found'}
        
        # 議論履歴を構造化
        discussion_history = bb.get_discussion_history()
        
        slot_start = time.time()
        
        # 引用付き統合プロンプトを構築
        synthesis_prompt = self._build_synthesis_prompt(user_input, discussion_history)
        
        # 統合実行
        entry = self._execute_synthesis_mode(synthesizer, bb, synthesis_prompt, embedder)
        
        slot_time = time.time() - slot_start
        
        if entry:
            entry.metadata.update({
                'phase': 3,
                'phase_name': '最終統合',
                'synthesis_mode': True,
                'discussion_phases': len(discussion_history.get('phases', {}))
            })
            
            result = {
                'entry': entry,
                'text': entry.text,
                'execution_time': slot_time,
                'metadata': entry.metadata,
                'discussion_history': discussion_history,
                'phase': 3
            }
            if self.debug:
                print(f"  📋 最終統合完了: {entry.text[:80]}...")
            return result
        else:
            return {'error': 'Synthesis failed', 'phase': 3}
    
    def _build_cross_reference_prompt(self, slot_name: str, user_input: str, other_opinions: List[Dict[str, Any]]) -> str:
        """相互参照用のプロンプトを構築"""
        opinions_text = ""
        if other_opinions:
            opinions_text = "【他のエージェントの意見】\n"
            for i, op in enumerate(other_opinions, 1):
                opinions_text += f"{i}. {op['role']}: {op['content'][:100]}...\n"
        
        role_name = slot_name.replace('Slot', '')
        
        return f"""他のエージェントの意見を踏まえ、あなたの{role_name}としての視点を深化・発展させてください。

【元の質問】
{user_input}

{opinions_text}

【相互参照の指針】
1. 他の意見のどの部分に同意/反対しますか？
2. 見落とされている重要な側面はありませんか？
3. あなたの専門性から、どのような補強ができますか？
4. 他の意見との統合可能性はありますか？

他の意見を明示的に参照しながら、150文字以内で応答してください。"""
    
    def _build_synthesis_prompt(self, user_input: str, discussion_history: Dict[str, Any]) -> str:
        """最終統合用のプロンプトを構築"""
        timeline = discussion_history.get('timeline', [])
        phases = discussion_history.get('phases', {})
        
        discussion_summary = ""
        for phase_num, phase_entries in phases.items():
            phase_name = phase_entries[0].get('phase_name', f'フェーズ{phase_num}') if phase_entries else f'フェーズ{phase_num}'
            discussion_summary += f"\n【{phase_name}】\n"
            for entry in phase_entries:
                discussion_summary += f"- {entry['role']}: {entry['content'][:80]}...\n"
        
        return f"""これまでの議論を統合し、引用付きで明確な結論を提示してください。

【元の質問】
{user_input}

【議論の流れ】{discussion_summary}

【統合の指針】
1. 各エージェントの主要な論点を整理
2. 合意できる部分と対立する部分を明確化
3. 最も妥当と思われる結論を論理的に導出
4. 必ず具体的な引用を含める（「Reformulatorが指摘したように〜」等）

引用付きで200文字以内で統合結論を述べてください。"""
    
    def _execute_cross_reference_mode(self, slot, bb: SlotBlackboard, prompt: str, embedder) -> Optional[SlotEntry]:
        """相互参照モードでSlotを実行（Enhanced Slots対応）"""
        try:
            # Enhanced Slotsの相互参照機能を使用
            if hasattr(slot, 'execute_cross_reference'):
                # 他の意見を取得
                cross_ref_context = bb.get_cross_reference_context(slot.name)
                other_opinions = cross_ref_context.get('other_opinions', [])
                
                if other_opinions:
                    return slot.execute_cross_reference(bb, prompt, other_opinions, embedder)
            
            # フォールバック: 従来の方式
            sys_prompt = slot.build_system_prompt()
            response = slot._generate_response(sys_prompt, prompt)
            
            if not response or response.strip() == "":
                return None
            
            metadata = {
                "role": slot.get_role_description(),
                "execution_time": 0,  # 後で設定
                "cross_reference_mode": True
            }
            
            return bb.add_slot_entry(slot.name, response, None, metadata)
            
        except Exception as e:
            if self.debug:
                print(f"相互参照実行エラー ({slot.name}): {e}")
            return None
    
    def _execute_synthesis_mode(self, synthesizer, bb: SlotBlackboard, prompt: str, embedder) -> Optional[SlotEntry]:
        """統合モードでSynthesizerを実行（Enhanced Slots対応）"""
        try:
            # Enhanced SynthesizerSlotの引用機能を使用
            if hasattr(synthesizer, 'execute_synthesis_with_citations'):
                discussion_history = bb.get_discussion_history()
                return synthesizer.execute_synthesis_with_citations(bb, prompt, discussion_history, embedder)
            
            # フォールバック: 従来の方式
            sys_prompt = synthesizer.build_system_prompt()
            response = synthesizer._generate_response(sys_prompt, prompt)
            
            if not response or response.strip() == "":
                return None
            
            metadata = {
                "role": synthesizer.get_role_description(),
                "execution_time": 0,  # 後で設定
                "synthesis_mode": True
            }
            
            return bb.add_slot_entry(synthesizer.name, response, None, metadata)
            
        except Exception as e:
            if self.debug:
                print(f"統合実行エラー: {e}")
            return None
    
    def _process_collaborative_results(self, all_results: Dict[str, Any], start_time: float, user_input: str) -> Dict[str, Any]:
        """協調的議論結果の処理"""
        collaboration_metrics = all_results.get('collaboration_metrics', {})
        discussion_history = all_results.get('discussion_history', {})
        final_synthesis = all_results.get('phase3_synthesis', {})
        
        # 最終応答の取得
        final_response = final_synthesis.get('text', "協調的議論が完了しましたが、統合に失敗しました。")
        
        total_time = time.time() - start_time
        self.successful_runs += 1
        
        summary = {
            'success': True,
            'final_response': final_response,
            'collaboration_mode': True,
            'collaboration_metrics': collaboration_metrics,
            'discussion_phases': len(discussion_history.get('phases', {})),
            'total_discussion_entries': discussion_history.get('total_entries', 0),
            'phase_results': {
                'phase1': all_results.get('phase1_initial', {}),
                'phase2': all_results.get('phase2_discussion', {}),
                'phase3': all_results.get('phase3_synthesis', {})
            },
            'total_execution_time': total_time,
            'user_input': user_input,
            'structured_blackboard': self.use_structured_blackboard
        }
        
        if self.debug:
            print(f"\n🎉 === 協調的議論完了 ===")
            print(f"実行時間: {total_time:.2f}秒")
            print(f"協調度スコア: {collaboration_metrics.get('collaboration_score', 0):.2f}")
            print(f"多様性スコア: {collaboration_metrics.get('diversity_score', 0):.2f}")
            print(f"相互参照スコア: {collaboration_metrics.get('reference_score', 0):.2f}")
            print(f"コンセンサススコア: {collaboration_metrics.get('consensus_score', 0):.2f}")
            print(f"最終応答: {final_response[:100]}...")
            print("=" * 50)
        
        return summary
