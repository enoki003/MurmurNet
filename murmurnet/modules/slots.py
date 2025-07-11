#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slot Based Architecture
~~~~~~~~~~~~~~~~~~~~~~~
分散SLMシステムのSlotベースアーキテクチャの実装

主要機能:
- BaseSlot: 汎用Slot基底クラス（model-template alignment対応）
- Slot variants: Reformulator, Critic, Supporter, Synthesizer
- SlotRunner: Slot実行エンジン（構造化Blackboard対応）
- SlotBlackboard: Slotデータストレージ（新しいBlackboardへの移行）

作者: Yuhi Sonoki
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

def _import_synthesizer():
    try:
        from .enhanced_synthesizer import IntelligentSynthesizer  # type: ignore
        return IntelligentSynthesizer
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

    def __init__(self, name: str, cfg: Dict[str, Any], model_factory):
        self.name = name
        self.config = cfg  # configという名前で保存
        self.cfg = cfg
        self.model_factory = model_factory
        self.debug: bool = cfg.get("debug", False)

        # PromptManagerの初期化
        try:
            self.prompt_manager = PromptManager(cfg)
        except Exception as e:
            if self.debug:
                logger.warning(f"PromptManager初期化失敗: {e}")
            self.prompt_manager = None

        # Generation parameters (slot‑local override可)
        self.max_output_len: int = cfg.get("slot_max_output_length", 100)  # Gemma向けに短縮
        self.temperature: float = cfg.get("slot_temperature", 0.3)  # Gemma向けに低い温度
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
    # Collaborative execution methods（協調実行メソッド）
    # ------------------------------------------------------------------
    
    def execute_cross_reference(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """他Slotの意見を参照して相互議論を実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            
            # 相互参照プロンプトがあれば使用、なければ通常プロンプト
            cross_ref_prompt = self.build_cross_reference_prompt(bb, user_input, other_opinions)
            if cross_ref_prompt:
                usr_prompt = cross_ref_prompt
            else:
                usr_prompt = self.build_user_prompt(bb, user_input)
            
            if self.debug:
                print(f"\n{self.name} (相互参照モード) ---")
                print(f"参照意見数: {len(other_opinions)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # 埋め込み生成
            embedding = None
            if embedder and response:
                try:
                    embedding = embedder.embed_text(response)
                except Exception as e:
                    if self.debug:
                        print(f"埋め込み生成エラー: {e}")
            
            # メタデータに相互参照情報を追加
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "cross_reference_mode": True,
                "referenced_opinions": len(other_opinions),
                "phase": 2  # 相互参照フェーズ
            }
            
            entry = bb.add_slot_entry(self.name, response, embedding, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            logger.error(f"Slot '{self.name}' 相互参照実行エラー: {e}")
            return None

    def execute_consensus_building(self, bb: SlotBlackboard, user_input: str, all_opinions: List[Dict[str, Any]], conflicts: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """合意形成・対立解消を実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            
            # 合意形成プロンプトがあれば使用、なければ通常プロンプト
            consensus_prompt = self.build_consensus_prompt(bb, user_input, all_opinions, conflicts)
            if consensus_prompt:
                usr_prompt = consensus_prompt
            else:
                usr_prompt = self.build_user_prompt(bb, user_input)
            
            if self.debug:
                print(f"\n{self.name} (合意形成モード) ---")
                print(f"対象意見数: {len(all_opinions)}, 対立点: {len(conflicts)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # 埋め込み生成
            embedding = None
            if embedder and response:
                try:
                    embedding = embedder.embed_text(response)
                except Exception as e:
                    if self.debug:
                        print(f"埋め込み生成エラー: {e}")
            
            # メタデータに合意形成情報を追加
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "consensus_building_mode": True,
                "total_opinions": len(all_opinions),
                "conflicts_addressed": len(conflicts),
                "phase": 3  # 合意形成フェーズ
            }
            
            entry = bb.add_slot_entry(self.name, response, embedding, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            logger.error(f"Slot '{self.name}' 合意形成実行エラー: {e}")
            return None

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

        # ModelFactoryを使ってモデルを作成
        try:
            # ModelFactoryがクラスかインスタンスかを判別
            if isinstance(self.model_factory, type):
                # クラスとして渡された場合
                model = self.model_factory.create_model(self.config)
            else:
                # インスタンスとして渡された場合
                if hasattr(self.model_factory, 'create_model'):
                    model = self.model_factory.create_model(self.config)
                else:
                    # 他の可能性もチェック
                    model = self.model_factory
        except Exception as e:
            if self.debug:
                print(f"モデル作成エラー: {e}")
            return f"モデル作成エラー: {str(e)[:50]}..."
        
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
        
        # デバッグ：生の出力をログ
        if self.debug and len(text) > 50:
            print(f"[{self.name}] 生出力 (最初50文字): {text[:50]}...")
        
        text = text.strip()

        # Gemma / Llama special-token removal (頑健化版)
        # 1. 完全な特殊トークンを除去（改行・空白込み）
        text = re.sub(r"<\|\s*[\w_]+?\s*\|>", "", text, flags=re.DOTALL)  # 改行・空白込み
        
        # 2. 途中で切れた残骸も除去
        text = re.sub(r"<\|\w*$", "", text)  # 末尾の不完全トークン
        text = re.sub(r"<\|[^>]*$", "", text)  # より広範な不完全トークン
        text = re.sub(r"\|\s*[\w_]+?\s*\|>?", "", text)  # 先頭 < が欠落した残骸
        
        # 単独縦棒が行に残ったら削除
        text = re.sub(r"^[| ]+$", "", text, flags=re.MULTILINE)
        
        # 2. 先頭 '<' が欠落して残った '|>' 断片を除去  ←★追加
        text = re.sub(r"\|>+", "", text)          # "|>" も "||>" もまとめて消す
        text = re.sub(r"\|\s*$", "", text)        # 行末の '|' だけ取り残った場合
        text = re.sub(r">\s*$", "", text)         # 行末に '>' が残った場合
        
        # 3. 従来の除去パターン（強化版）
        text = re.sub(r"<\|[^>]+?\|>", "", text)  # 従来パターン
        text = re.sub(r"<\/?(system|user|assistant)>", "", text)
        text = re.sub(r"\[\/?INST]", "", text)
        text = re.sub(r"<<\/?SYS>>", "", text)
        text = re.sub(r"<s>", "", text)
        text = re.sub(r"</s>", "", text)
        
        # 4. 新しい頑健化パターン
        text = re.sub(r"<\|end_of_turn\|>", "", text)  # 明示的除去
        text = re.sub(r"<\|start_of_turn\|>", "", text)  # 明示的除去
        text = re.sub(r"<\|assistant\|>", "", text)
        text = re.sub(r"<\|user\|>", "", text)
        text = re.sub(r"<\|system\|>", "", text)
        
        # 5. 不完全なトークンの断片を除去
        text = re.sub(r"of_turn>", "", text)  # 不完全な end_of_turn
        text = re.sub(r"start_of_", "", text)  # 不完全な start_of_turn
        text = re.sub(r"end_of_", "", text)   # 不完全な end_of_turn
        
        # 6. 危険な文字の全角化（保険）
        text = text.replace("<", "＜").replace(">", "＞")
        
        # Gemmaの問題パターンをより積極的にクリーンアップ
        # 1. "of_of_of|" や "and|endend|end" のような繰り返し（強化版）
        text = re.sub(r"(\b\w+\b)(?:[_\|]\1){2,}[_\|]?", r"\1", text)  # 同じ単語が3回以上連続したら1回に圧縮
        text = re.sub(r"(\w+)(_\1)+(_|\|)?", r"\1", text)  # "word_word_word" -> "word"
        text = re.sub(r"(\w+)\|\1+\|?", r"\1", text)  # "word|wordword|" -> "word"
        text = re.sub(r"(\w+)(\|\w+)?\2{2,}", r"\1\2", text)  # 3回以上の繰り返し
        
        # 2. 連続する縦線やアンダースコアの削除（強化版）
        text = re.sub(r"\|{2,}", "|", text)  # "|||" -> "|"
        text = re.sub(r"_{2,}", "_", text)  # "___" -> "_"
        text = re.sub(r"\|+$", "", text)  # 末尾の縦線削除
        text = re.sub(r"_+$", "", text)  # 末尾のアンダースコア削除
        
        # 単独縦棒を行から完全除去
        text = re.sub(r"^\|+$", "", text, flags=re.MULTILINE)  # 行全体が縦棒のみ
        text = re.sub(r"^\|\s*$", "", text, flags=re.MULTILINE)  # 縦棒と空白のみの行
        text = re.sub(r"\|+\s*\n", "\n", text)  # 行末の縦棒
        text = re.sub(r"\n\s*\|+", "\n", text)  # 行頭の縦棒
        
        # 3. 意味のない短い断片を削除
        if len(text) < 5 and re.match(r"^[\w\|_＜＞]+$", text):
            text = ""  # 短すぎて意味不明な場合は空にする
        
        # 4. 改行が多すぎる場合の正規化
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        text = re.sub(r"</?\w+[^>]*?>", "", text)  # stray HTML
        text = text.strip()

        # 空または無意味な出力の場合のフォールバック
        if not text or len(text.strip()) < 5:
            result = f"{self.name.replace('Slot', '')}は適切な応答を生成できませんでした。"
        else:
            result = text

        # 文字数制限は最後に適用（完全クリーン後）
        if len(result) > self.max_output_len:
            result = result[: self.max_output_len].rsplit(" ", 1)[0] + "…"
        
        # デバッグ：クリーニング後の出力をログ + 危険文字チェック
        if self.debug:
            print(f"[{self.name}] クリーニング後: {result[:100]}...")
            if '＜' in result or '＞' in result or '|' in result:
                print(f"[{self.name}] 注意: 特殊文字が残存しています: {result}")
        
        return result

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
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            return self.prompt_manager.get_slot_system_prompt('reformulator')
        return """あなたは厳密な分析専門家です。以下の責任を果たしてください：

1. 曖昧な概念を明確に定義する
2. 問題の範囲と制約を明示する
3. 隠れた前提条件を暴露する
4. 議論すべき具体的な論点を提示する

必ず具体的で明確な分析をし、他の専門家が反論できる明確な論点を提供してください。"""

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            template = self.prompt_manager.get_slot_user_prompt_template('reformulator')
            return template.format(user_input=user_input)
        return f"""以下の質問について、厳密な分析を行ってください：

質問: {user_input}

分析要求:
1. 「{user_input}」の曖昧な部分を明確に定義する
2. 変化の対象は誰か（学習者・教員・学校・社会）
3. 時間軸は何か（短期・中期・長期）
4. 成功の指標は何か

150文字以内で、他の専門家が具体的に反論できる明確な分析を述べてください："""

    def build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> Optional[str]:
        """他の意見を参照した再構成プロンプト"""
        if not other_opinions:
            return None
        
        opinions_text = ""
        for op in other_opinions[:2]:  # 最大2つの意見
            role = op.get('role', 'Unknown')
            content = op.get('content', '')[:40]  # 40文字に制限
            opinions_text += f"- {role}: {content}\n"
        
        return f"質問: {user_input}\n\n他の意見:\n{opinions_text}\n上記を踏まえた新しい分析（60文字以内）:"

class CriticSlot(BaseSlot):
    def get_role_description(self) -> str:
        return "批判的分析・課題指摘"

    def build_system_prompt(self) -> str:
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            return self.prompt_manager.get_slot_system_prompt('critic')
        return """あなたは厳密な批判専門家です。以下の責任を果たしてください：

1. 楽観的な仮定を徹底的に疑う
2. 実現可能性の具体的な障害を指摘する
3. 見落とされているリスクを明確に示す
4. 既存の失敗事例や制約を引用する

必ず具体的な根拠を示し、他の専門家が反論したくなる鋭い批判を提供してください。"""

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            template = self.prompt_manager.get_slot_user_prompt_template('critic')
            return template.format(user_input=user_input)
        return f"""以下の質問について、厳しい批判的分析を行ってください：

質問: {user_input}

批判要求:
1. この変化が失敗する可能性が高い理由は何か
2. 既存の制度や利害関係者の抵抗はどうか
3. 技術的・経済的・社会的な制約は何か
4. 過去の類似事例で失敗したものはあるか

150文字以内で、具体的な根拠を示して反論を誘う批判を述べてください："""

    def build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> Optional[str]:
        """他の意見を参照した批判プロンプト"""
        if not other_opinions:
            return None
        
        opinions_text = ""
        for op in other_opinions[:2]:  # 最大2つの意見
            role = op.get('role', 'Unknown')
            content = op.get('content', '')[:40]  # 40文字に制限
            opinions_text += f"- {role}: {content}\n"
        
        return f"質問: {user_input}\n\n他の意見:\n{opinions_text}\n上記の課題・問題点（60文字以内）:"

class SupporterSlot(BaseSlot):
    def get_role_description(self) -> str:
        return "肯定的支援・価値発見"

    def build_system_prompt(self) -> str:
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            return self.prompt_manager.get_slot_system_prompt('supporter')
        return """あなたは創造的な革新推進者です。以下の責任を果たしてください：

1. 批判論の欠点を具体的に指摘する
2. 実現可能な具体的な方法論を提示する
3. 成功事例や新しい技術的解決策を示す
4. 長期的な社会的価値を明確に論証する

必ず具体的な解決策と根拠を示し、批判論に対して説得力のある反論を提供してください。"""

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            template = self.prompt_manager.get_slot_user_prompt_template('supporter')
            return template.format(user_input=user_input)
        return f"""以下の質問について、創造的で実現可能な解決策を提示してください：

質問: {user_input}

支援要求:
1. この変化がもたらす具体的な社会的価値は何か
2. 技術的制約を克服する具体的な方法は何か
3. 成功している類似事例はあるか
4. 段階的に実現する具体的なロードマップは何か

150文字以内で、批判論に対する具体的な反論と解決策を述べてください："""

    def build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> Optional[str]:
        """他の意見を参照した支援プロンプト"""
        if not other_opinions:
            return None
        
        opinions_text = ""
        for op in other_opinions[:2]:  # 最大2つの意見
            role = op.get('role', 'Unknown')
            content = op.get('content', '')[:40]  # 40文字に制限
            opinions_text += f"- {role}: {content}\n"
        
        return f"質問: {user_input}\n\n他の意見:\n{opinions_text}\n新しい可能性・解決策（60文字以内）:"

class SynthesizerSlot(BaseSlot):
    def get_role_description(self) -> str:
        return "多視点統合・最終応答"

    def build_system_prompt(self) -> str:
        # PromptManagerから取得、なければデフォルト
        if hasattr(self, 'prompt_manager') and self.prompt_manager:
            return self.prompt_manager.get_slot_system_prompt('synthesizer')
        return """あなたは責任ある統合判断者です。以下の責任を果たしてください：

1. 各専門家の意見の妥当性を厳密に評価する
2. 対立する意見の根拠を比較検討する
3. 現実的で実行可能な統合案を提示する
4. 判断の責任と根拠を明確に示す

必ず「○○は正しいが、△△の懸念もあり、□□すべき」の形で明確な判断を示してください。"""

    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        # 新しいTemplateBuilderを使用した安全なテンプレート生成
        from .template_builder import TemplateBuilder
        
        # 他のSlotの出力をメッセージ形式に変換
        messages = []
        
        # システムプロンプトを追加
        messages.append({
            "role": "system",
            "content": self.build_system_prompt()
        })
        
        # 各Slotの出力をクリーンアップしてメッセージ化
        slot_entries = bb.get_slot_entries()
        other_entries = [e for e in slot_entries if e.slot_name != self.name]
        
        if other_entries:
            opinions_parts = [f"質問: {user_input}", "", "専門家の意見:"]
            
            for e in other_entries:
                role = e.slot_name.replace('Slot', '')
                # 重要：他Slotの出力を使用前に必ずクリーン
                content = self._clean_response(e.text).strip()
                if content:
                    opinions_parts.append(f"{role}: {content}")
            
            opinions_parts.extend([
                "",
                "上記の専門家の議論を統合し、責任ある判断を下してください。",
                "必ず「○○は正しいが、△△の懸念もあり、□□すべき」の形で明確な判断を示してください。"
            ])
            
            messages.append({
                "role": "user",
                "content": "\n".join(opinions_parts)
            })
        else:
            # 他の意見がない場合のフォールバック
            messages.append({
                "role": "user",
                "content": f"質問: {user_input}\n\n利用可能な専門家の意見がありません。一般的な知識に基づいて回答してください。"
            })
        
        # TemplateBuilderで安全なテンプレートを生成
        return TemplateBuilder.build(messages)

    def build_consensus_prompt(self, bb: SlotBlackboard, user_input: str, all_opinions: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Optional[str]:
        """統合・合意形成プロンプト"""
        if not all_opinions:
            return None
        
        # 簡潔な意見要約
        opinions_text = ""
        for op in all_opinions[:3]:  # 最大3つの意見
            role = op.get('role', 'Unknown')
            content = op.get('content', '').strip()[:40]  # 40文字に制限
            opinions_text += f"- {role}: {content}\n"
        
        return f"質問: {user_input}\n\n各専門家の意見:\n{opinions_text}\n最終的な統合結論（80文字以内）:"

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
                    logger.warning("従来Blackboardモードにフォールバック")
        else:
            self.blackboard_adapter = None
            if self.debug:
                logger.info("従来Blackboardモード")
        
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
    
    def _clean_response_for_synthesis(self, text: str) -> str:
        """統合用のテキストクリーニング（より厳格）"""
        if not isinstance(text, str):
            text = str(text)
        
        # 基本クリーニング
        text = text.strip()
        
        # 特殊トークン完全除去
        special_tokens = [
            r"<\|\s*[\w_]+?\s*\|>",  # 完全な特殊トークン
            r"<\|[^>]*$",            # 不完全な特殊トークン
            r"<\|[^>]*\|>",          # 従来パターン
            r"<\|end_of_turn\|>",    # 明示的除去
            r"<\|start_of_turn\|>",  # 明示的除去
            r"<\|assistant\|>",
            r"<\|user\|>",
            r"<\|system\|>",
            r"of_turn>",             # 不完全な断片
            r"start_of_",            # 不完全な断片
            r"end_of_",              # 不完全な断片
        ]
        
        for pattern in special_tokens:
            text = re.sub(pattern, "", text, flags=re.DOTALL)
        
        # 危険文字を全角化
        text = text.replace("<", "＜").replace(">", "＞")
        
        # 改行整理
        text = re.sub(r"\n{2,}", "\n", text)
        
        # 空の場合のフォールバック
        if not text.strip():
            return "(応答を生成できませんでした)"
        
        return text.strip()

    def _initialize_slots(self) -> Dict[str, BaseSlot]:
        """Slotの初期化"""
        
        # 基本Slotを作成
        slots = {}
        
        # IntelligentSynthesizerを使用するかどうかを決定
        use_intelligent_synthesizer = self.config.get('use_intelligent_synthesizer', True)
        
        if use_intelligent_synthesizer:
            # IntelligentSynthesizerを使用
            IntelligentSynthesizer = _import_synthesizer()
            if IntelligentSynthesizer:
                base_slots = {
                    'ReformulatorSlot': ReformulatorSlot,
                    'CriticSlot': CriticSlot,
                    'SupporterSlot': SupporterSlot,
                    'SynthesizerSlot': IntelligentSynthesizer  # 知的統合システム
                }
                if self.debug:
                    logger.info("🧠 IntelligentSynthesizer採用: 知的統合システムを使用")
            else:
                # フォールバック
                base_slots = {
                    'ReformulatorSlot': ReformulatorSlot,
                    'CriticSlot': CriticSlot,
                    'SupporterSlot': SupporterSlot,
                    'SynthesizerSlot': SynthesizerSlot
                }
                if self.debug:
                    logger.warning("⚠️ IntelligentSynthesizer利用不可: 従来のSynthesizerSlotを使用")
        else:
            # 従来のSynthesizerSlotを使用
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
            
            # IntelligentSynthesizerの場合、embedderを渡す
            if slot_name == 'SynthesizerSlot' and slot_class.__name__ == 'IntelligentSynthesizer':
                slots[slot_name] = slot_class(slot_name, merged_config, self.model_factory, self.embedder)
            else:
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
        全Slotを実行（デフォルトで協調モード）
        
        Args:
            bb: Slot Blackboard
            user_input: ユーザー入力
            embedder: 埋め込み生成器
        
        Returns:
            実行結果の辞書
        """
        # 協調モードの設定を確認（デフォルトはTrue）
        use_collaboration = self.config.get('use_collaboration', True)
        
        if use_collaboration:
            # 協調モードで実行（真の議論を実現）
            if self.debug:
                print("🔥 協調モード: 激しい議論を開始します")
            return self.run_collaborative_slots(bb, user_input, embedder)
        else:
            # 従来の並列実行
            if self.debug:
                print("📝 並列モード: 独立実行を開始します")
            return self._run_legacy_slots(bb, user_input, embedder)

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
    
    def _run_legacy_slots(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """
        並列モード：従来の順次実行（協調なし）
        """
        start_time = time.time()
        self.total_runs += 1
        
        try:
            results = {}
            execution_times = {}
            quality_scores = []
            
            if self.debug:
                print(f"SlotRunner（並列）実行開始: {len(self.execution_order)}個のSlot")
            
            # 構造化Blackboardの場合、ラウンドベース実行
            if self.use_structured_blackboard and self.blackboard_adapter:
                try:
                    results = self._run_all_slots_structured(bb, user_input, embedder)
                except Exception as e:
                    logger.error(f"構造化Blackboard実行エラー: {e}")
                    if self.debug:
                        import traceback
                        logger.debug(traceback.format_exc())
                    # フォールバックとして従来実行
                    logger.info("従来Blackboardにフォールバック")
                    results = self._run_all_slots_legacy_internal(bb, user_input, embedder)
            else:
                results = self._run_all_slots_legacy_internal(bb, user_input, embedder)
            
            # 共通の結果処理
            return self._process_slot_results(results, start_time, user_input)
            
        except Exception as e:
            return self._handle_slot_error(e, start_time)

    def _run_all_slots_legacy_internal(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """順次実行（後方互換性）"""
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

    def run_collaborative_slots(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """
        協調システム - 各Slotが議論・相互参照する実行モード
        
        各Slotが他の意見を読み、明示的に反応・議論する
        """
        start_time = time.time()
        self.total_runs += 1
        
        if self.debug:
            print(f"\n=== 協調システム開始 ===")
            print("各Slotが議論・相互参照を行います")
            print("=" * 60)
        
        try:
            # ===== フェーズ1: 初期分析（独立思考） =====
            if self.debug:
                print("📝 フェーズ1: 初期分析（独立思考）")
            
            # Reformulatorが問題を分析
            reformulator = self.slots.get('ReformulatorSlot')
            if not reformulator:
                return {'error': 'ReformulatorSlot not found'}
            
            reformulator_entry = reformulator.execute(bb, user_input, embedder)
            if not reformulator_entry:
                return {'error': 'Reformulator execution failed'}
            
            reformulator_entry.metadata['phase'] = 1
            reformulator_entry.metadata['phase_name'] = '初期分析'
            
            if self.debug:
                print(f"  💭 Reformulator: {reformulator_entry.text[:80]}...")
            
            # ===== フェーズ2: 他意見への直接的反応 =====
            if self.debug:
                print("\nフェーズ2: 他意見への直接的反応（議論開始）")
            
            # Criticが Reformulatorの意見を読んで批判
            critic = self.slots.get('CriticSlot')
            if critic:
                critic_prompt = f"""以下のReformulatorの分析について、厳しく批判的に検討してください:

【Reformulatorの分析】
{reformulator_entry.text}

【必須批判要求】
1. Reformulatorの分析の具体的な問題点を3つ以上指摘する
2. 「この分析は○○という点で間違っている」と明確に述べる
3. 実現不可能な理由を具体的に説明する
4. 「Reformulatorが見落としている重要な制約は△△である」と指摘する

必ず「Reformulatorの分析は間違っている。なぜなら」で始まり、具体的な反論を150文字以内で述べてください："""
                
                critic_entry = self._execute_targeted_response(critic, bb, critic_prompt, embedder)
                if critic_entry:
                    critic_entry.metadata.update({
                        'phase': 2, 'phase_name': '直接批判',
                        'targets': [reformulator_entry.entry_id],
                        'response_type': 'disagreement'
                    })
                    if self.debug:
                        print(f"  ⚔️ Critic → Reformulator: {critic_entry.text[:80]}...")
            
            # Supporterが Reformulatorの意見を読んで支持・拡張
            supporter = self.slots.get('SupporterSlot')
            if supporter:
                supporter_prompt = f"""以下のReformulatorの分析について、積極的に支持し発展させてください:

【Reformulatorの分析】
{reformulator_entry.text}

【必須支持要求】
1. 「Reformulatorの分析は正しい」と明確に述べる
2. なぜそれが実現可能なのか具体的な方法を3つ以上提示する
3. 成功事例や技術的解決策を示す
4. 「Reformulatorが指摘した○○は実際に△△によって実現できる」と説明する

必ず「Reformulatorの分析は正しい。実際に」で始め、具体的な支持理由を150文字以内で述べてください："""
                
                supporter_entry = self._execute_targeted_response(supporter, bb, supporter_prompt, embedder)
                if supporter_entry:
                    supporter_entry.metadata.update({
                        'phase': 2, 'phase_name': '積極的支持',
                        'targets': [reformulator_entry.entry_id],
                        'response_type': 'agreement'
                    })
                    if self.debug:
                        print(f"  🌟 Supporter → Reformulator: {supporter_entry.text[:80]}...")
            
            # ===== フェーズ3: 相互反応（議論の深化） =====
            if self.debug:
                print("\n⚡ フェーズ3: 相互反応（議論の深化）")
            
            # Reformulatorが Criticの批判に応答
            if critic_entry:
                reformulator_counter_prompt = f"""Criticから以下の厳しい批判を受けました:

【Criticの批判】
{critic_entry.text}

【必須反論要求】
1. 「Criticの批判は的外れである」と明確に述べる
2. Criticの批判のどの部分が間違っているか具体的に指摘する
3. あなたの分析がなぜ正しいのか、新しい根拠を3つ以上提示する
4. 「Criticが見落としている重要な点は○○である」と反論する

必ず「Criticの批判は間違っている。なぜなら」で始め、具体的な反論を150文字以内で述べてください："""
                
                reformulator_counter = self._execute_targeted_response(reformulator, bb, reformulator_counter_prompt, embedder)
                if reformulator_counter:
                    reformulator_counter.metadata.update({
                        'phase': 3, 'phase_name': '強烈な反駁',
                        'targets': [critic_entry.entry_id],
                        'response_type': 'strong_disagreement'
                    })
                    if self.debug:
                        print(f"  ⚡ Reformulator → Critic: {reformulator_counter.text[:80]}...")
            
            # Criticが Supporterの楽観論を批判
            if supporter_entry:
                critic_counter_prompt = f"""Supporterは以下のように楽観的に述べていますが、これを厳しく批判してください:

【Supporterの楽観論】
{supporter_entry.text}

【必須批判要求】
1. 「Supporterの楽観論は現実を無視している」と明確に述べる
2. この提案が失敗する具体的な理由を3つ以上示す
3. 過去の類似事例で失敗したものを引用する
4. 「Supporterが無視している現実的な制約は○○である」と指摘する

必ず「Supporterの楽観論は危険である。なぜなら」で始め、具体的な批判を150文字以内で述べてください："""
                
                critic_counter = self._execute_targeted_response(critic, bb, critic_counter_prompt, embedder)
                if critic_counter:
                    critic_counter.metadata.update({
                        'phase': 3, 'phase_name': '楽観論粉砕',
                        'targets': [supporter_entry.entry_id],
                        'response_type': 'strong_disagreement'
                    })
                    if self.debug:
                        print(f"  ⚔️ Critic → Supporter: {critic_counter.text[:80]}...")
            
            # ===== フェーズ4: 対立解決統合 =====
            if self.debug:
                print("\nフェーズ4: 対立解決統合（責任ある判断）")
            
            synthesizer = self.slots.get('SynthesizerSlot')
            if not synthesizer:
                return {'error': 'SynthesizerSlot not found'}
            
            # 全ての意見を収集
            all_entries = bb.get_slot_entries()
            recent_entries = [e for e in all_entries if e.metadata.get('phase', 0) >= 1]
            
            # 対立点の特定
            conflicts = self._identify_conflicts(recent_entries)
            
            synthesis_prompt = f"""以下の激しい議論を統合し、責任を持って最終判断を下してください:

【激しい議論の流れ】"""
            
            for entry in recent_entries:
                phase_name = entry.metadata.get('phase_name', '不明')
                response_type = entry.metadata.get('response_type', '')
                targets = entry.metadata.get('targets', [])
                
                # 議論の強度を表現
                intensity = ""
                if response_type == 'strong_disagreement':
                    intensity = "🔥"
                elif response_type == 'disagreement':
                    intensity = "⚔️"
                elif response_type == 'agreement':
                    intensity = "🌟"
                
                # エントリのテキストを安全にクリーン
                clean_text = self._clean_response_for_synthesis(entry.text)
                synthesis_prompt += f"\n{intensity} {entry.slot_name} ({phase_name}): {clean_text}"
            
            synthesis_prompt += f"""

【対立の状況】
{conflicts}

【あなたの統合責任】
この激しい議論を受けて、以下の責任を果たしてください：

1. 誰の意見が最も説得力があるか明確に判断する
2. 対立する意見について、どちらが正しいか決断する
3. 現実的で実行可能な解決策を提示する
4. なぜその判断を下すのか、責任を持って根拠を明示する

【必須形式】
必ず以下の形式で述べてください：
「この議論において、○○の指摘が最も妥当である。△△の懸念もあるが、□□の方法で解決可能である。最終的に◇◇すべきである。」

責任を持って200文字以内で最終判断を述べてください："""
            
            synthesis_entry = self._execute_targeted_response(synthesizer, bb, synthesis_prompt, embedder)
            if synthesis_entry:
                synthesis_entry.metadata.update({
                    'phase': 4, 'phase_name': '対立解決統合',
                    'targets': [e.entry_id for e in recent_entries],
                    'response_type': 'synthesis',
                    'conflicts_resolved': len(conflicts)
                })
                if self.debug:
                    print(f"  Synthesizer (対立解決): {synthesis_entry.text[:100]}...")
            
            # ===== 結果分析 =====
            final_time = time.time() - start_time
            
            # 真の協調メトリクス計算
            collaboration_quality = self._analyze_true_collaboration_quality(bb)
            
            result = {
                'success': True,
                'collaboration_mode': 'collaborative_discussion',
                'final_response': synthesis_entry.text if synthesis_entry else "統合に失敗",
                'collaboration_quality': collaboration_quality,
                'phases_executed': 4,
                'total_interactions': len([e for e in bb.get_slot_entries() if e.metadata.get('targets')]),
                'conflicts_detected': len(conflicts),
                'execution_time': final_time,
                'user_input': user_input
            }
            
            if self.debug:
                print(f"\n🎉 === 協調完了 ===")
                print(f"実行時間: {final_time:.2f}秒")
                print(f"相互作用: {result['total_interactions']}回")
                print(f"対立解決: {result['conflicts_detected']}件")
                print(f"協調品質: {collaboration_quality.get('overall_score', 0):.2f}")
                print(f"最終統合: {result['final_response'][:100]}...")
                print("=" * 60)
            
            return result
            
        except Exception as e:
            return self._handle_slot_error(e, start_time)
    
    def _execute_targeted_response(self, slot, bb: SlotBlackboard, prompt: str, embedder) -> Optional[SlotEntry]:
        """特定のプロンプトでSlotを実行"""
        try:
            sys_prompt = slot.build_system_prompt()
            # プロンプトの危険文字を事前に全角化（保険）
            safe_prompt = prompt.replace("<", "＜").replace(">", "＞")
            response = slot._generate_response(sys_prompt, safe_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # 応答を再度クリーン（二重保険）
            clean_response = slot._clean_response(response)
            
            metadata = {
                "role": slot.get_role_description(),
                "execution_time": 0,
                "targeted_response": True
            }
            
            return bb.add_slot_entry(slot.name, clean_response, None, metadata)
            
        except Exception as e:
            if self.debug:
                print(f"標的応答実行エラー ({slot.name}): {e}")
            return None
    
    def _identify_conflicts(self, entries: List[SlotEntry]) -> str:
        """議論から対立を特定"""
        conflicts = []
        
        disagreement_entries = [e for e in entries if e.metadata.get('response_type') == 'disagreement']
        
        for entry in disagreement_entries:
            targets = entry.metadata.get('targets', [])
            if targets:
                target_slots = [e.slot_name for e in entries if e.entry_id in targets]
                conflict_desc = f"{entry.slot_name} vs {', '.join(target_slots)}: {entry.text[:50]}..."
                conflicts.append(conflict_desc)
        
        return '\n'.join(conflicts) if conflicts else "明確な対立は検出されませんでした"
    
    def _analyze_true_collaboration_quality(self, bb: SlotBlackboard) -> Dict[str, float]:
        """協調品質を分析"""
        entries = bb.get_slot_entries()
        
        # 相互作用スコア
        interaction_count = len([e for e in entries if e.metadata.get('targets')])
        interaction_score = min(interaction_count / 6.0, 1.0)  # 6回の相互作用が理想
        
        # 引用スコア
        citation_keywords = ['について', 'が指摘した', 'の意見', 'の分析', 'の批判', 'の提案']
        citation_count = sum(1 for e in entries for keyword in citation_keywords if keyword in e.text)
        citation_score = min(citation_count / 12.0, 1.0)  # 各エントリ2回の引用が理想
        
        # 対立解決スコア
        disagreements = len([e for e in entries if e.metadata.get('response_type') == 'disagreement'])
        synthesis_entries = [e for e in entries if e.metadata.get('response_type') == 'synthesis']
        resolution_score = 1.0 if synthesis_entries and disagreements > 0 else 0.5
        
        # 総合スコア
        overall_score = (interaction_score * 0.4 + citation_score * 0.3 + resolution_score * 0.3)
        
        return {
            'interaction_score': interaction_score,
            'citation_score': citation_score,
            'resolution_score': resolution_score,
            'overall_score': overall_score,
            'total_interactions': interaction_count,
            'total_citations': citation_count,
            'disagreements_count': disagreements
        }
        
    def run_collaborative_slots_detailed(self, bb: SlotBlackboard, user_input: str, embedder=None) -> Dict[str, Any]:
        """
        協調システム - 多段階議論による協調実行
        
        フェーズ1: 初期意見提示
        フェーズ2: 相互参照・反論  
        フェーズ3: 引用付き最終統合
        """
        start_time = time.time()
        self.total_runs += 1
        
        if self.debug:
            print(f"\n=== 協調的議論システム開始 ===")
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
            print("フェーズ2: 相互参照・議論")
        
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
                    print(f"  {slot_name}: {entry.text[:60]}... (参照{len(other_opinions)}件)")
            else:
                results[slot_name] = {'entry': None, 'text': None, 'error': True, 'phase': 2}
                if self.debug:
                    print(f"  ❌ {slot_name}: 相互参照エラー")
        
        results['execution_times'] = execution_times
        return results
    
    def _run_phase3_synthesis(self, bb: SlotBlackboard, user_input: str, embedder, phase2_results: Dict[str, Any]) -> Dict[str, Any]:
        """フェーズ3: 引用付き最終統合"""
        if self.debug:
            print("フェーズ3: 引用付き最終統合")
        
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
                print(f"  最終統合完了: {entry.text[:80]}...")
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
        """相互参照モードでSlotを実行"""
        try:
            # Slotsの相互参照機能を使用
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
                "execution_time": 0,
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
            # SynthesizerSlotの引用機能を使用
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
