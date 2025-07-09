#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Synthesizer
~~~~~~~~~~~~~~~~~~~
Boids則を活用した高度なSlot出力統合システム
意味ベクトル空間でのSwarm Intelligence統合

機能:
- BoidsBasedSynthesizer: Boids則による統合Slot
- AdaptiveSynthesisStrategy: 動的統合戦略選択
- QualityEvaluator: 統合結果の品質評価
- ConflictResolver: 矛盾解決機構

作者: Yuhi Sonoki
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from .boids import VectorSpace, BoidsController, SlotBoid
from .slots import BaseSlot
from .slot_blackboard import SlotBlackboard, SlotEntry

logger = logging.getLogger(__name__)


class SynthesisStrategy:
    """統合戦略の列挙"""
    CONSENSUS = "consensus"           # コンセンサス重視
    DIVERSITY = "diversity"           # 多様性重視
    QUALITY = "quality"               # 品質重視
    BOIDS_COHERENCE = "boids_coherence"  # Boids結束重視
    ADAPTIVE = "adaptive"             # 動的選択


class BoidsBasedSynthesizer(BaseSlot):
    """
    Boids則を活用した統合Slot
    
    複数のSlot出力をベクトル空間でBoids則により調整し、
    最適な統合応答を生成する高度なSynthesizer
    """
    
    def __init__(self, name: str, config: Dict[str, Any], model_factory, embedder=None):
        super().__init__(name, config, model_factory)
        
        self.embedder = embedder
        self.vector_space = VectorSpace(embedder) if embedder else None
        self.boids_controller = BoidsController(config)
        
        # 統合戦略設定
        self.synthesis_strategy = config.get('synthesis_strategy', SynthesisStrategy.ADAPTIVE)
        self.quality_threshold = config.get('synthesis_quality_threshold', 0.6)
        self.max_iterations = config.get('boids_max_iterations', 3)
        
        # 品質評価設定
        self.evaluator = QualityEvaluator(config)
        self.conflict_resolver = ConflictResolver(config)
        
        logger.info(f"BoidsBasedSynthesizer初期化: {self.synthesis_strategy}戦略")
    
    def get_role_description(self) -> str:
        return "Boids則多視点統合・高品質応答生成"
    
    def build_system_prompt(self) -> str:
        return """あなたは最高度の統合専門家です。複数の異なる視点をBoids則（群知能）に基づいて統合し、以下を実現してください：

【統合原則】
1. **Separation（分離）**: 重複や矛盾を排除し、独自性を保持
2. **Alignment（整列）**: 共通の方向性と一貫性を確立  
3. **Cohesion（凝集）**: 散逸を防ぎ、中心的な価値を抽出

【統合品質基準】
- 全視点の価値を活かした包括性
- 論理的一貫性と実用性
- 独創的で建設的な洞察
- ユーザーにとっての明確な価値

謝罪や曖昧な表現は完全に排除し、確信に満ちた高品質な統合回答を生成してください。
各視点の強みを相乗効果的に組み合わせ、単純な並列ではない真の統合を実現してください。"""
    
    def build_user_prompt(self, blackboard: SlotBlackboard, user_input: str) -> str:
        # Slot出力をBoids空間で処理
        synthesis_data = self._perform_boids_synthesis(blackboard, user_input)
        
        # プロンプト構築
        context_parts = [
            f"ユーザー要求: {user_input}",
            "",
            "=== Boids則統合データ ===",
            f"統合戦略: {synthesis_data['strategy']}",
            f"品質スコア: {synthesis_data['quality_score']:.2f}",
            f"結束度: {synthesis_data['coherence']:.2f}",
            "",
            "=== 各視点の分析結果 ==="
        ]
        
        # 各Slot出力を統合結果に基づいて表示
        for slot_info in synthesis_data['processed_slots']:
            slot_name = slot_info['slot_name']
            text = slot_info['text']
            boids_score = slot_info.get('boids_score', 0.0)
            
            role_label = self._get_role_label(slot_name)
            context_parts.append(
                f"{role_label} (統合影響度: {boids_score:.2f})\n{text}\n"
            )
        
        # 統合指針
        if synthesis_data.get('conflicts'):
            context_parts.extend([
                "=== 検出された矛盾点 ===",
                synthesis_data['conflicts'],
                ""
            ])
        
        if synthesis_data.get('synthesis_direction'):
            context_parts.extend([
                "=== 統合方向性 ===",
                synthesis_data['synthesis_direction'],
                ""
            ])
        
        context = "\n".join(context_parts)
        
        return f"""以下のBoids則統合データに基づいて、最終的な統合回答を生成してください：

{context}

上記の多視点分析とBoids則統合結果を踏まえ、ユーザーに対する最高品質の統合回答を作成してください。
各視点の価値を相乗効果的に活かし、矛盾を解決した確信に満ちた回答を提供してください。"""
    
    def _perform_boids_synthesis(self, blackboard: SlotBlackboard, user_input: str) -> Dict[str, Any]:
        """Boids則を使用した統合処理"""
        if not self.vector_space:
            # フォールバック: 埋め込みなしの統合
            return self._fallback_synthesis(blackboard, user_input)
        
        # 1. Slot出力をベクトル空間に配置
        slot_entries = blackboard.get_slot_entries()
        if not slot_entries:
            return {'strategy': 'empty', 'quality_score': 0.0, 'coherence': 0.0, 'processed_slots': []}
        
        # 自分以外のSlotエントリを処理
        relevant_entries = [entry for entry in slot_entries if entry.slot_name != self.name]
        
        self.vector_space.clear()
        boids = []
        
        for entry in relevant_entries:
            boid = self.vector_space.add_slot_output(
                entry.slot_name, 
                entry.text, 
                entry.metadata
            )
            boids.append(boid)
        
        if len(boids) < 2:
            return self._simple_synthesis(relevant_entries)
        
        # 2. Boids則を反復適用
        for iteration in range(self.max_iterations):
            boids = self.boids_controller.apply_boids_rules(self.vector_space)
            
            # 収束判定
            coherence = self.boids_controller.evaluate_swarm_coherence(boids)
            if coherence['convergence'] > 0.8:
                logger.debug(f"Boids則収束: {iteration+1}回目で収束達成")
                break
        
        # 3. 統合戦略を決定
        strategy = self._select_synthesis_strategy(boids, coherence)
        
        # 4. 品質評価
        quality_score = self.evaluator.evaluate_synthesis_quality(boids, user_input)
        
        # 5. 矛盾検出と解決
        conflicts = self.conflict_resolver.detect_conflicts(boids)
        
        # 6. 統合方向性の決定
        synthesis_direction = self._determine_synthesis_direction(boids, strategy, coherence)
        
        return {
            'strategy': strategy,
            'quality_score': quality_score,
            'coherence': coherence['coherence'],
            'processed_slots': [
                {
                    'slot_name': boid.slot_name,
                    'text': boid.text,
                    'boids_score': self._calculate_boids_influence_score(boid, boids)
                }
                for boid in boids
            ],
            'conflicts': conflicts,
            'synthesis_direction': synthesis_direction,
            'boids_stats': coherence
        }
    
    def _fallback_synthesis(self, blackboard: SlotBlackboard, user_input: str) -> Dict[str, Any]:
        """埋め込みなしのフォールバック統合"""
        slot_entries = blackboard.get_slot_entries()
        relevant_entries = [entry for entry in slot_entries if entry.slot_name != self.name]
        
        return {
            'strategy': 'fallback',
            'quality_score': 0.5,
            'coherence': 0.5,
            'processed_slots': [
                {
                    'slot_name': entry.slot_name,
                    'text': entry.text,
                    'boids_score': 0.5
                }
                for entry in relevant_entries
            ],
            'conflicts': None,
            'synthesis_direction': "標準的な統合を実行します。"
        }
    
    def _simple_synthesis(self, entries: List[SlotEntry]) -> Dict[str, Any]:
        """単純統合（Boids適用不可時）"""
        return {
            'strategy': 'simple',
            'quality_score': 0.6,
            'coherence': 0.6,
            'processed_slots': [
                {
                    'slot_name': entry.slot_name,
                    'text': entry.text,
                    'boids_score': 0.6
                }
                for entry in entries
            ],
            'conflicts': None,
            'synthesis_direction': "シンプルな統合を実行します。"
        }
    
    def _select_synthesis_strategy(self, boids: List[SlotBoid], coherence: Dict[str, float]) -> str:
        """統合戦略の動的選択"""
        if self.synthesis_strategy != SynthesisStrategy.ADAPTIVE:
            return self.synthesis_strategy
        
        # 適応的戦略選択
        if coherence['coherence'] > 0.8:
            return SynthesisStrategy.CONSENSUS
        elif coherence['diversity'] > 0.7:
            return SynthesisStrategy.DIVERSITY
        elif len(boids) >= 3:
            return SynthesisStrategy.BOIDS_COHERENCE
        else:
            return SynthesisStrategy.QUALITY
    
    def _calculate_boids_influence_score(self, target_boid: SlotBoid, all_boids: List[SlotBoid]) -> float:
        """Boidの統合への影響度を計算"""
        if len(all_boids) <= 1:
            return 1.0
        
        # 他のBoidとの平均類似度
        similarities = []
        for other_boid in all_boids:
            if other_boid.slot_name != target_boid.slot_name:
                similarity = target_boid.similarity_to(other_boid)
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        avg_similarity = np.mean(similarities)
        
        # 速度の大きさ（変化の積極性）
        velocity_magnitude = np.linalg.norm(target_boid.velocity)
        
        # 影響度 = 平均類似度 * 0.7 + 速度正規化 * 0.3
        normalized_velocity = min(velocity_magnitude / 0.3, 1.0)
        influence_score = avg_similarity * 0.7 + normalized_velocity * 0.3
        
        return float(influence_score)
    
    def _determine_synthesis_direction(self, boids: List[SlotBoid], strategy: str, 
                                     coherence: Dict[str, float]) -> str:
        """統合の方向性を決定"""
        if strategy == SynthesisStrategy.CONSENSUS:
            return "全視点の共通点を重視し、コンセンサスベースの統合を行います。"
        elif strategy == SynthesisStrategy.DIVERSITY:
            return "多様な視点を並列保持し、多面的な統合回答を構成します。"
        elif strategy == SynthesisStrategy.QUALITY:
            return "最高品質の視点を核として、他の視点で補完する統合を行います。"
        elif strategy == SynthesisStrategy.BOIDS_COHERENCE:
            return "Boids群知能の結束性を活かし、相乗効果的な統合を実現します。"
        else:
            return "バランスの取れた統合を実行します。"
    
    def _get_role_label(self, slot_name: str) -> str:
        """Slot名から役割ラベルを取得"""
        role_map = {
            'ReformulatorSlot': '【再構成】',
            'CriticSlot': '【批評】',
            'SupporterSlot': '【支援】',
            'SynthesizerSlot': '【統合】'
        }
        return role_map.get(slot_name, f'【{slot_name}】')


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
