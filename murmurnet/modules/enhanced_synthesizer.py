#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Synthesizer
~~~~~~~~~~~~~~~~~~~
Boids則を活用した高度なSlot出力統合システム
意味ベクトル空間でのSwarm Intelligence統合

機能:
- BoidsBasedSynthesizer: Boids則による統合Slot
- AdaptiveSynthesisStrategy: 動的統合戦略選択
- QualityEvaluator: 統合結果の品質評価
- ConflictResolver: 矛盾解決機構
- SemanticUnderstandingEngine: 意味理解エンジン
- ArgumentEvaluator: 論証評価器

作者: Yuhi Sonoki
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
import re

from .boids import VectorSpace, BoidsController, SlotBoid
from .slots import BaseSlot
from .slot_blackboard import SlotBlackboard, SlotEntry

logger = logging.getLogger(__name__)


class TemplateBuilder:
    """
    テンプレート破損防止用ビルダー
    外部テンプレートに依存せず、自前で完全なテンプレートを構築
    """
    START = "<start_of_turn>"
    END = "<end_of_turn>"
    VALID_ROLES = {"user", "model", "system"}
    
    @staticmethod
    def build(messages: List[Dict[str, str]]) -> str:
        """メッセージリストから安全なテンプレートを構築"""
        if not messages:
            return f"<bos>\n{TemplateBuilder.START}model\n"
        
        out = ["<bos>"]
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            
            # roleの正規化
            if role == "assistant":
                role = "model"
            elif role not in TemplateBuilder.VALID_ROLES:
                role = "user"
            
            # contentのクリーニング
            content = TemplateBuilder._clean_content(content)
            
            out.append(f"{TemplateBuilder.START}{role}\n{content}{TemplateBuilder.END}")
        
        # 生成プロンプト
        out.append(f"{TemplateBuilder.START}model\n")
        return "\n".join(out)
    
    @staticmethod
    def _clean_content(content: str) -> str:
        """コンテンツの安全化"""
        # 制御文字やHTML的な要素を除去
        content = re.sub(r"[|=]{2,}", "", content)
        content = re.sub(r"<[^>]*>", "", content)
        content = re.sub(r"\s+", " ", content)
        return content.strip()


class TemplateValidator:
    """
    テンプレート破損検出・修復システム
    """
    PATTERN = re.compile(r"<start_of_turn>(\w+)\n([\s\S]*?)<end_of_turn>")
    VALID_ROLES = {"user", "model", "system"}
    
    @staticmethod
    def parse_chat(text: str) -> Optional[List[Tuple[str, str]]]:
        """チャットテンプレートをパース"""
        try:
            out = []
            for role, content in TemplateValidator.PATTERN.findall(text):
                if role not in TemplateValidator.VALID_ROLES:
                    logger.warning(f"無効なロール検出: {role}")
                    return None
                out.append((role, content.strip()))
            return out if out else None
        except Exception as e:
            logger.error(f"テンプレートパース失敗: {e}")
            return None
    
    @staticmethod
    def validate_response(response: str) -> bool:
        """応答の妥当性を検証"""
        if not response or len(response.strip()) == 0:
            return False
        
        # 異常な文字パターンの検出
        if re.search(r"[|]{3,}", response):
            logger.warning("縦棒洪水を検出")
            return False
        
        if re.search(r"[=]{5,}", response):
            logger.warning("等号洪水を検出")
            return False
        
        # 意味のあるテキストの検証
        japanese_chars = len(re.findall(r"[ひらがなカタカナ一-龯]", response))
        if japanese_chars < 5:
            logger.warning("日本語コンテンツ不足")
            return False
        
        return True


class TemplateFixer:
    """
    破損したテンプレートの修復
    """
    
    @staticmethod
    def fix_corrupted_response(raw: str) -> str:
        """破損した応答の修復"""
        if not raw:
            return "【統合結果】\n申し訳ございません。応答の生成に失敗しました。"
        
        # 1) 連続制御文字を削除
        cleaned = re.sub(r"[|=]{2,}", "", raw)
        cleaned = re.sub(r"[\n\r\t\s]{5,}", " ", cleaned)
        
        # 2) HTML的な要素をエスケープ
        cleaned = cleaned.replace("<", "＜").replace(">", "＞")
        
        # 3) 日本語部分を抽出
        japanese_parts = re.findall(r"[ひらがなカタカナ一-龯\s。、！？]+", cleaned)
        if japanese_parts:
            meaningful_text = "".join(japanese_parts)[:200]
        else:
            meaningful_text = cleaned[:200]
        
        # 4) 構造化して返却
        return f"【統合結果】\n{meaningful_text.strip()}"
    
    @staticmethod
    def create_fallback_response(user_input: str) -> str:
        """フォールバック応答の生成"""
        return f"""【統合結果】
ご質問「{user_input}」について、複数の視点から検討いたしました。

技術的な問題により詳細な統合結果を表示できませんでしたが、
この課題は多面的な検討が必要な重要なテーマです。

より詳細な回答については、お手数ですが再度お試しください。"""


class SynthesisStrategy:
    """統合戦略の列挙"""
    CONSENSUS = "consensus"           # コンセンサス重視
    DIVERSITY = "diversity"           # 多様性重視
    QUALITY = "quality"               # 品質重視
    BOIDS_COHERENCE = "boids_coherence"  # Boids結束重視
    ADAPTIVE = "adaptive"             # 動的選択


class IntelligentSynthesizer(BaseSlot):
    """
    知的統合システム - 真の議論統合を実現
    
    テンプレート破損対策を含む堅牢な統合処理
    """
    
    def __init__(self, name: str, config: Dict[str, Any], model_factory, embedder=None):
        super().__init__(name, config, model_factory)
        
        self.embedder = embedder
        
        # 知的統合設定
        self.evidence_weight = config.get('evidence_weight', 0.4)
        self.logic_weight = config.get('logic_weight', 0.3)
        self.relevance_weight = config.get('relevance_weight', 0.3)
        self.min_confidence_threshold = config.get('synthesis_confidence_threshold', 0.6)
        
        # テンプレート破損対策設定
        self.max_retry_attempts = config.get('max_retry_attempts', 1)
        self.use_template_builder = config.get('use_template_builder', True)
        self.enable_auto_recovery = config.get('enable_auto_recovery', True)
        
        # 知的分析エンジンの初期化
        self.understanding_engine = SemanticUnderstandingEngine(embedder, config)
        self.argument_evaluator = ArgumentEvaluator(config)
        
        logger.info("IntelligentSynthesizer初期化: 知的統合システム稼働（テンプレート破損対策有効）")
    
    def get_role_description(self) -> str:
        return "知的統合・論理的判断・責任ある結論"
    
    def build_system_prompt(self) -> str:
        """
        知的統合システム用のシステムプロンプト
        """
        return """あなたは知的統合システムです。複数の視点から提示された議論を論理的に評価し、責任ある判断を下してください。

【あなたの役割】
1. 各視点の論理的妥当性を評価する
2. 客観的証拠と主観的意見を区別する
3. 矛盾する主張を整理し、建設的な統合を行う
4. 確信度を含む明確な結論を提示する

【出力形式】
- 【判断】最も説得力のある論証とその理由
- 【根拠】客観的証拠と論理的妥当性
- 【結論】具体的で実行可能な最終回答
- 【確信度】この判断への確信レベル（高/中/低）

【注意事項】
- 単純な要約ではなく、知的な統合判断を行う
- 相反する意見がある場合は、論理的根拠に基づいて調停する
- 不確実な情報については確信度を適切に調整する
- 日本語で簡潔かつ明確に回答する"""

    def generate_with_recovery(self, messages: List[Dict[str, str]], user_input: str) -> str:
        """
        テンプレート破損対策付きの安全な生成
        """
        try:
            # 1. テンプレートを安全に構築
            prompt = TemplateBuilder.build(messages)
            
            # 2. 生成実行
            response = self.model.generate(
                prompt,
                max_tokens=self.config.get('max_tokens', 256),
                temperature=self.config.get('temperature', 0.7),
                stop=["<end_of_turn>", "</s>", "<|end_of_turn|>"]
            )
            
            # 3. 応答の検証
            if TemplateValidator.validate_response(response):
                return self._clean_response(response)
            
            # 4. 再試行（温度0で安全に）
            logger.warning("応答が異常です。再試行します。")
            response = self.model.generate(
                prompt,
                max_tokens=128,
                temperature=0.0,
                stop=["<end_of_turn>", "</s>"]
            )
            
            if TemplateValidator.validate_response(response):
                return self._clean_response(response)
            
            # 5. 最終フォールバック
            logger.error("応答生成に失敗しました。フォールバック応答を使用します。")
            return TemplateFixer.create_fallback_response(user_input)
            
        except Exception as e:
            logger.error(f"generate_with_recovery エラー: {e}")
            return TemplateFixer.create_fallback_response(user_input)

    def _clean_response(self, text: str) -> str:
        """
        応答のクリーニング（BaseSlotから継承）
        """
        if not text:
            return ""
        
        # 1. テンプレートトークンの除去
        text = re.sub(r"<start_of_turn>.*?<end_of_turn>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|.*?\|>", "", text)  # Gemmaトークン
        text = re.sub(r"</?s>", "", text)      # 特殊トークン
        text = re.sub(r"<bos>", "", text)      # Beginning of sequence
        
        # 2. 連続する制御文字の除去
        text = re.sub(r"[|=]{2,}", "", text)
        text = re.sub(r"^[| ]+$", "", text, flags=re.MULTILINE)
        
        # 3. 空行の正規化
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        # 4. 前後の空白除去
        text = text.strip()
        
        return text
    
    def generate_with_recovery(self, messages: List[Dict[str, str]], user_input: str) -> str:
        """
        テンプレート破損対策付きの生成処理
        """
        if not self.use_template_builder:
            # 従来方式（フォールバック）
            return self._generate_traditional(messages, user_input)
        
        # 新方式：TemplateBuilder使用
        try:
            # フェーズ1: 安全なテンプレート構築
            template = TemplateBuilder.build(messages)
            logger.debug(f"構築されたテンプレート: {template[:200]}...")
            
            # フェーズ2: 初回生成
            response = self._generate_with_template(template)
            
            # フェーズ3: 検証
            if TemplateValidator.validate_response(response):
                logger.info("テンプレート生成成功（初回）")
                return self._clean_response(response)
            
            # フェーズ4: 再試行
            logger.warning("応答品質不良、再試行実行")
            response = self._retry_generation(template)
            
            if TemplateValidator.validate_response(response):
                logger.info("テンプレート生成成功（再試行）")
                return self._clean_response(response)
            
            # フェーズ5: 修復
            if self.enable_auto_recovery:
                logger.warning("自動修復モード開始")
                return TemplateFixer.fix_corrupted_response(response)
            else:
                return TemplateFixer.create_fallback_response(user_input)
                
        except Exception as e:
            logger.error(f"テンプレート生成エラー: {e}")
            return TemplateFixer.create_fallback_response(user_input)
    
    def _generate_with_template(self, template: str) -> str:
        """テンプレートを使用した生成"""
        try:
            # 生成設定
            generation_params = {
                'temperature': 0.7,
                'max_new_tokens': 300,
                'stop': ["<end_of_turn>", "</s>", "<|end_of_turn|>"],
                'do_sample': True,
                'top_p': 0.9
            }
            
            # モデル生成
            response = self.model_factory.generate(
                template,
                **generation_params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"テンプレート生成エラー: {e}")
            raise
    
    def _retry_generation(self, template: str) -> str:
        """再試行生成（より保守的な設定）"""
        try:
            # より保守的な設定
            generation_params = {
                'temperature': 0.0,  # 決定論的
                'max_new_tokens': 128,  # 短縮
                'stop': ["<end_of_turn>", "</s>"],
                'do_sample': False
            }
            
            response = self.model_factory.generate(
                template,
                **generation_params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"再試行生成エラー: {e}")
            return ""
    
    def _generate_traditional(self, messages: List[Dict[str, str]], user_input: str) -> str:
        """従来の生成方式（フォールバック）"""
        try:
            # 従来のプロンプト構築
            prompt = self._build_traditional_prompt(messages)
            
            response = self.model_factory.generate(
                prompt,
                temperature=0.7,
                max_new_tokens=200
            )
            
            return self._clean_response(response)
            
        except Exception as e:
            logger.error(f"従来方式生成エラー: {e}")
            return TemplateFixer.create_fallback_response(user_input)
    
    def _build_traditional_prompt(self, messages: List[Dict[str, str]]) -> str:
        """従来のプロンプト構築"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        
        return "\n".join(parts) + "\nassistant:"
    
    def build_user_prompt(self, blackboard: SlotBlackboard, user_input: str) -> str:
        """
        新しいテンプレート方式でのプロンプト構築
        """
        # 他のSlotの出力を取得してクリーン化
        slot_entries = blackboard.get_slot_entries()
        relevant_entries = [entry for entry in slot_entries if entry.slot_name != self.name]
        
        if not relevant_entries:
            # 論証が不足している場合の簡易プロンプト
            return f"質問: {user_input}\n\n利用可能な他の視点がありません。一般的な知識に基づいて回答してください。"
        
        # メッセージ形式で構築
        messages = []
        
        # システムメッセージ
        messages.append({
            "role": "system",
            "content": self.build_system_prompt()
        })
        
        # 各Slotの出力を統合
        context_parts = [
            f"ユーザー質問: {user_input}",
            "",
            "=== 各視点の分析 ==="
        ]
        
        for i, entry in enumerate(relevant_entries):
            # 重要：他Slotの出力を必ずクリーン化
            clean_text = self._clean_response(entry.text) if hasattr(self, '_clean_response') else entry.text
            role_name = entry.slot_name.replace('Slot', '')
            
            context_parts.append(f"{i+1}. {role_name}の視点:")
            context_parts.append(f"   {clean_text[:200]}...")
            context_parts.append("")
        
        context_parts.extend([
            "=== 統合指針 ===",
            "上記の各視点を論理的に評価し、以下の形式で統合判断を行ってください：",
            "",
            "【判断】最も説得力のある論証とその理由",
            "【根拠】客観的証拠と論理的妥当性", 
            "【結論】具体的で実行可能な最終回答",
            "【確信度】この判断への確信レベル（高/中/低）",
            "",
            "責任を持って明確な知的判断を下してください。"
        ])
        
        messages.append({
            "role": "user",
            "content": "\n".join(context_parts)
        })
        
        # 新しいテンプレート方式で生成
        if self.use_template_builder:
            return self.generate_with_recovery(messages, user_input)
        else:
            # 従来方式（フォールバック）
            return "\n".join(context_parts)


class QualityEvaluator:
    """統合品質の評価器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coherence_weight = config.get('quality_coherence_weight', 0.4)
        self.diversity_weight = config.get('quality_diversity_weight', 0.3)
        self.relevance_weight = config.get('quality_relevance_weight', 0.3)
    
    def evaluate_synthesis_quality(self, boids: List[SlotBoid], user_input: str) -> float:
        """統合品質の総合評価"""
        if not boids:
            return 0.0
        
        # 1. 結束性評価
        coherence_score = self._evaluate_coherence(boids)
        
        # 2. 多様性評価
        diversity_score = self._evaluate_diversity(boids)
        
        # 3. 関連性評価
        relevance_score = self._evaluate_relevance(boids, user_input)
        
        # 重み付き総合スコア
        total_score = (
            coherence_score * self.coherence_weight +
            diversity_score * self.diversity_weight +
            relevance_score * self.relevance_weight
        )
        
        return float(np.clip(total_score, 0.0, 1.0))
    
    def _evaluate_coherence(self, boids: List[SlotBoid]) -> float:
        """結束性の評価"""
        if len(boids) < 2:
            return 1.0
        
        similarities = []
        for i, boid1 in enumerate(boids):
            for boid2 in boids[i+1:]:
                similarity = boid1.similarity_to(boid2)
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _evaluate_diversity(self, boids: List[SlotBoid]) -> float:
        """多様性の評価"""
        if len(boids) < 2:
            return 0.0
        
        # 平均距離が大きいほど多様性が高い
        distances = []
        for i, boid1 in enumerate(boids):
            for boid2 in boids[i+1:]:
                distance = boid1.distance_to(boid2)
                distances.append(distance)
        
        avg_distance = np.mean(distances) if distances else 0.0
        return float(min(avg_distance, 1.0))
    
    def _evaluate_relevance(self, boids: List[SlotBoid], user_input: str) -> float:
        """関連性の評価（簡易実装）"""
        # ここでは簡易的にテキスト長と内容の豊富さで評価
        relevance_scores = []
        
        for boid in boids:
            text_length_score = min(len(boid.text) / 200.0, 1.0)
            
            # ユーザー入力とのキーワード一致度
            user_words = set(user_input.lower().split())
            boid_words = set(boid.text.lower().split())
            keyword_match = len(user_words & boid_words) / max(len(user_words), 1)
            
            relevance = (text_length_score * 0.6 + keyword_match * 0.4)
            relevance_scores.append(relevance)
        
        return float(np.mean(relevance_scores)) if relevance_scores else 0.0


class ConflictResolver:
    """矛盾解決機構"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conflict_threshold = config.get('conflict_threshold', 0.3)
    
    def detect_conflicts(self, boids: List[SlotBoid]) -> Optional[str]:
        """矛盾の検出"""
        if len(boids) < 2:
            return None
        
        conflicts = []
        
        for i, boid1 in enumerate(boids):
            for boid2 in boids[i+1:]:
                similarity = boid1.similarity_to(boid2)
                
                # 低い類似度 = 潜在的な矛盾
                if similarity < self.conflict_threshold:
                    conflicts.append(f"{boid1.slot_name} ⟷ {boid2.slot_name}")
        
        if conflicts:
            return f"検出された視点の相違: {', '.join(conflicts)}"
        
        return None


class SemanticUnderstandingEngine:
    """
    意味理解エンジン - 真の相互理解を実現
    
    各Slotの出力を意味レベルで分析し、
    論点・根拠・結論を構造化して抽出
    """
    
    def __init__(self, embedder, config: Dict[str, Any]):
        self.embedder = embedder
        self.config = config
        self.debug = config.get('debug', False)
        
        # 意味構造テンプレート
        self.argument_structure = {
            'premise': None,      # 前提
            'evidence': [],       # 根拠
            'reasoning': None,    # 推論
            'conclusion': None,   # 結論
            'assumptions': [],    # 仮定
            'counterarguments': [] # 反駁可能性
        }
    
    def analyze_argument_structure(self, text: str, slot_role: str) -> Dict[str, Any]:
        """引数の論理構造を分析"""
        structure = self.argument_structure.copy()
        
        # 役割に応じた構造分析
        if 'Reformulator' in slot_role:
            structure = self._analyze_reformulator_structure(text)
        elif 'Critic' in slot_role:
            structure = self._analyze_critic_structure(text)
        elif 'Supporter' in slot_role:
            structure = self._analyze_supporter_structure(text)
        
        return structure
    
    def _analyze_reformulator_structure(self, text: str) -> Dict[str, Any]:
        """Reformulatorの構造分析"""
        # 問題定義の抽出
        premise = self._extract_problem_definition(text)
        assumptions = self._extract_hidden_assumptions(text)
        
        return {
            'premise': premise,
            'evidence': [],
            'reasoning': text,
            'conclusion': None,
            'assumptions': assumptions,
            'counterarguments': [],
            'type': 'problem_definition'
        }
    
    def _analyze_critic_structure(self, text: str) -> Dict[str, Any]:
        """Criticの構造分析"""
        # 批判の根拠を抽出
        evidence = self._extract_critical_evidence(text)
        counterarguments = self._extract_counterarguments(text)
        
        return {
            'premise': "既存の提案には問題がある",
            'evidence': evidence,
            'reasoning': text,
            'conclusion': self._extract_critical_conclusion(text),
            'assumptions': [],
            'counterarguments': counterarguments,
            'type': 'critical_analysis'
        }
    
    def _analyze_supporter_structure(self, text: str) -> Dict[str, Any]:
        """Supporterの構造分析"""
        # 支持の根拠を抽出
        evidence = self._extract_supportive_evidence(text)
        
        return {
            'premise': "提案は実現可能である",
            'evidence': evidence,
            'reasoning': text,
            'conclusion': self._extract_supportive_conclusion(text),
            'assumptions': [],
            'counterarguments': [],
            'type': 'supportive_analysis'
        }
    
    def _extract_problem_definition(self, text: str) -> str:
        """問題定義の抽出"""
        # キーワードベースの簡易抽出
        definition_keywords = ['とは', 'について', '問題', '課題', '定義']
        sentences = text.split('。')
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in definition_keywords):
                return sentence.strip()
        
        return text[:50] + "..."
    
    def _extract_hidden_assumptions(self, text: str) -> List[str]:
        """隠れた仮定の抽出"""
        assumption_patterns = [
            r'(\w+)が前提',
            r'(\w+)を仮定',
            r'(\w+)であれば',
            r'(\w+)の場合'
        ]
        
        assumptions = []
        for pattern in assumption_patterns:
            matches = re.findall(pattern, text)
            assumptions.extend(matches)
        
        return assumptions[:3]  # 最大3つ
    
    def _extract_critical_evidence(self, text: str) -> List[str]:
        """批判的根拠の抽出"""
        evidence_keywords = ['なぜなら', 'しかし', 'ただし', '問題は', 'リスクは']
        sentences = text.split('。')
        
        evidence = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in evidence_keywords):
                evidence.append(sentence.strip())
        
        return evidence[:2]  # 最大2つ
    
    def _extract_counterarguments(self, text: str) -> List[str]:
        """反駁の抽出"""
        counter_keywords = ['一方で', '逆に', 'むしろ', '反対に']
        sentences = text.split('。')
        
        counters = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in counter_keywords):
                counters.append(sentence.strip())
        
        return counters
    
    def _extract_critical_conclusion(self, text: str) -> str:
        """批判的結論の抽出"""
        conclusion_keywords = ['したがって', 'よって', '結論として', '最終的に']
        sentences = text.split('。')
        
        for sentence in reversed(sentences):
            if any(keyword in sentence for keyword in conclusion_keywords):
                return sentence.strip()
        
        return sentences[-1] if sentences else text
    
    def _extract_supportive_evidence(self, text: str) -> List[str]:
        """支持的根拠の抽出"""
        evidence_keywords = ['実際に', '例えば', '成功例', '証明', 'データ']
        sentences = text.split('。')
        
        evidence = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in evidence_keywords):
                evidence.append(sentence.strip())
        
        return evidence[:2]  # 最大2つ
    
    def _extract_supportive_conclusion(self, text: str) -> str:
        """支持的結論の抽出"""
        return self._extract_critical_conclusion(text)  # 同じロジック


class ArgumentEvaluator:
    """
    論証評価器 - 論理的妥当性を評価
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug = config.get('debug', False)
    
    def evaluate_argument_strength(self, argument: Dict[str, Any]) -> float:
        """論証の強度を評価"""
        score = 0.0
        
        # 前提の明確性
        if argument.get('premise'):
            score += 0.2
        
        # 根拠の豊富さ
        evidence_count = len(argument.get('evidence', []))
        score += min(evidence_count * 0.15, 0.3)
        
        # 推論の長さ（複雑性の代理指標）
        reasoning_length = len(argument.get('reasoning', ''))
        score += min(reasoning_length / 200.0, 0.2) * 0.2
        
        # 結論の存在
        if argument.get('conclusion'):
            score += 0.2
        
        # 反駁への対応
        counter_count = len(argument.get('counterarguments', []))
        score += min(counter_count * 0.1, 0.1)
        
        return min(score, 1.0)
    
    def evaluate_logical_consistency(self, arguments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """複数論証の論理的一貫性を評価"""
        if len(arguments) < 2:
            return {'consistency': 1.0, 'conflicts': []}
        
        conflicts = []
        consistency_scores = []
        
        for i, arg1 in enumerate(arguments):
            for j, arg2 in enumerate(arguments[i+1:], i+1):
                conflict = self._detect_logical_conflict(arg1, arg2)
                if conflict:
                    conflicts.append({
                        'arg1_index': i,
                        'arg2_index': j,
                        'conflict_type': conflict['type'],
                        'description': conflict['description']
                    })
                    consistency_scores.append(0.0)
                else:
                    consistency_scores.append(1.0)
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return {
            'consistency': overall_consistency,
            'conflicts': conflicts,
            'total_comparisons': len(consistency_scores)
        }
    
    def _detect_logical_conflict(self, arg1: Dict[str, Any], arg2: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """論理的対立の検出"""
        # 結論の対立
        conclusion1 = arg1.get('conclusion', '')
        conclusion2 = arg2.get('conclusion', '')
        
        # 簡易的な対立検出（キーワードベース）
        negative_keywords = ['ない', '困難', '不可能', '問題', 'リスク']
        positive_keywords = ['可能', '有効', '成功', '良い', '利点']
        
        arg1_negative = any(keyword in conclusion1 for keyword in negative_keywords)
        arg1_positive = any(keyword in conclusion1 for keyword in positive_keywords)
        
        arg2_negative = any(keyword in conclusion2 for keyword in negative_keywords)
        arg2_positive = any(keyword in conclusion2 for keyword in positive_keywords)
        
        if (arg1_negative and arg2_positive) or (arg1_positive and arg2_negative):
            return {
                'type': 'conclusion_conflict',
                'description': f"結論が対立: '{conclusion1}' vs '{conclusion2}'"
            }
        
        return None
