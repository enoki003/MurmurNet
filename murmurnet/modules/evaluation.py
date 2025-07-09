#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Engine
~~~~~~~~~~~~~~~~
MurmurNet Slot出力とBoids統合の評価・テスト機構

機能:
- SlotOutputEvaluator: 個別Slot出力の評価
- BoidsEvaluator: Boids統合効果の評価
- SystemPerformanceEvaluator: システム全体性能評価
- EvaluationReporter: 評価レポート生成

作者: Yuhi Sonoki
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """評価結果のデータ構造"""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    timestamp: float
    
    @property
    def normalized_score(self) -> float:
        """正規化されたスコア（0-1）"""
        return self.score / self.max_score if self.max_score > 0 else 0.0
    
    @property
    def grade(self) -> str:
        """スコアに基づく評価等級"""
        normalized = self.normalized_score
        if normalized >= 0.9:
            return "A+"
        elif normalized >= 0.8:
            return "A"
        elif normalized >= 0.7:
            return "B"
        elif normalized >= 0.6:
            return "C"
        elif normalized >= 0.5:
            return "D"
        else:
            return "F"


class SlotOutputEvaluator:
    """個別Slot出力の評価器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_length = config.get('eval_min_output_length', 20)
        self.max_length = config.get('eval_max_output_length', 500)
        self.keyword_weights = config.get('eval_keyword_weights', {})
    
    def evaluate_slot_output(self, slot_name: str, text: str, 
                           user_input: str, metadata: Dict[str, Any] = None) -> EvaluationResult:
        """個別Slot出力を評価"""
        metadata = metadata or {}
        
        # 基本品質評価
        length_score = self._evaluate_length(text)
        content_score = self._evaluate_content_quality(text)
        relevance_score = self._evaluate_relevance(text, user_input)
        role_adherence_score = self._evaluate_role_adherence(slot_name, text)
        
        # 総合スコア計算
        total_score = (
            length_score * 0.2 +
            content_score * 0.3 +
            relevance_score * 0.3 +
            role_adherence_score * 0.2
        )
        
        details = {
            'slot_name': slot_name,
            'text_length': len(text),
            'length_score': length_score,
            'content_score': content_score,
            'relevance_score': relevance_score,
            'role_adherence_score': role_adherence_score,
            'user_input_length': len(user_input),
            'metadata': metadata
        }
        
        return EvaluationResult(
            metric_name=f"{slot_name}_output_quality",
            score=total_score,
            max_score=1.0,
            details=details,
            timestamp=time.time()
        )
    
    def _evaluate_length(self, text: str) -> float:
        """出力長の評価"""
        length = len(text)
        
        if length < self.min_length:
            return 0.3  # 短すぎる
        elif length > self.max_length:
            return 0.7  # 長すぎる
        else:
            # 適切な長さの場合、理想長に近いほど高スコア
            ideal_length = (self.min_length + self.max_length) / 2
            distance = abs(length - ideal_length)
            max_distance = self.max_length - ideal_length
            return 1.0 - (distance / max_distance) * 0.3
    
    def _evaluate_content_quality(self, text: str) -> float:
        """内容品質の評価"""
        # 基本的な品質指標
        quality_score = 0.5  # ベーススコア
        
        # 文の多様性
        sentences = text.split('。')
        if len(sentences) > 1:
            quality_score += 0.2
        
        # 語彙の豊富さ
        words = text.split()
        unique_words = set(words)
        if len(words) > 0:
            vocab_diversity = len(unique_words) / len(words)
            quality_score += vocab_diversity * 0.2
        
        # 否定的表現の検出（減点）
        negative_patterns = ["申し訳", "わかりません", "できません", "すみません"]
        for pattern in negative_patterns:
            if pattern in text:
                quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _evaluate_relevance(self, text: str, user_input: str) -> float:
        """関連性の評価"""
        text_words = set(text.lower().split())
        input_words = set(user_input.lower().split())
        
        if not input_words:
            return 0.5
        
        # 共通語彙の割合
        common_words = text_words & input_words
        relevance = len(common_words) / len(input_words)
        
        # 関連語彙のボーナス
        if len(text_words) > 0:
            text_richness = len(text_words) / max(len(user_input.split()), 10)
            relevance += min(text_richness * 0.1, 0.3)
        
        return min(1.0, relevance)
    
    def _evaluate_role_adherence(self, slot_name: str, text: str) -> float:
        """役割遵守の評価"""
        role_keywords = {
            'ReformulatorSlot': ['言い換え', '別の表現', '具体例', '詳細', '再構成'],
            'CriticSlot': ['問題', '課題', '改善', '注意', '批判', '分析'],
            'SupporterSlot': ['良い点', '価値', '可能性', '励まし', '支援', '前向き'],
            'SynthesizerSlot': ['統合', '総合', 'まとめ', '結論', '全体', '包括']
        }
        
        keywords = role_keywords.get(slot_name, [])
        if not keywords:
            return 0.7  # 未知のSlotの場合は中間スコア
        
        text_lower = text.lower()
        matched_keywords = sum(1 for keyword in keywords if keyword in text_lower)
        
        # キーワード一致率
        keyword_score = matched_keywords / len(keywords)
        
        # 長さボーナス（役割に応じた適切な詳細度）
        length_bonus = min(len(text) / 100, 0.3)
        
        return min(1.0, keyword_score * 0.7 + length_bonus)


class BoidsEvaluator:
    """Boids統合効果の評価器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def evaluate_boids_integration(self, boids_result: Dict[str, Any], 
                                 user_input: str) -> EvaluationResult:
        """Boids統合結果の評価"""
        
        # Boids統計データから評価指標を抽出
        boids_stats = boids_result.get('boids_stats', {})
        coherence = boids_stats.get('coherence', 0.0)
        diversity = boids_stats.get('diversity', 0.0)
        convergence = boids_stats.get('convergence', 0.0)
        
        # 統合品質スコア
        synthesis_quality = boids_result.get('synthesis_quality', 0.0)
        
        # 処理されたSlot数
        processed_slots = boids_result.get('processed_slots', [])
        slot_count = len(processed_slots)
        
        # Boids効果スコアの計算
        boids_effectiveness = (
            coherence * 0.3 +
            diversity * 0.3 +
            convergence * 0.2 +
            synthesis_quality * 0.2
        )
        
        # Slot統合効果
        if slot_count > 1:
            slot_integration_score = min(slot_count / 4.0, 1.0)  # 最大4Slotで正規化
        else:
            slot_integration_score = 0.3
        
        # 総合Boidsスコア
        total_boids_score = (
            boids_effectiveness * 0.6 +
            slot_integration_score * 0.4
        )
        
        details = {
            'coherence': coherence,
            'diversity': diversity,
            'convergence': convergence,
            'synthesis_quality': synthesis_quality,
            'slot_count': slot_count,
            'boids_effectiveness': boids_effectiveness,
            'slot_integration_score': slot_integration_score,
            'strategy': boids_result.get('strategy', 'unknown'),
            'conflicts_detected': boids_result.get('conflicts') is not None
        }
        
        return EvaluationResult(
            metric_name="boids_integration_effectiveness",
            score=total_boids_score,
            max_score=1.0,
            details=details,
            timestamp=time.time()
        )


class SystemPerformanceEvaluator:
    """システム全体性能の評価器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = []
    
    def evaluate_system_performance(self, slot_results: Dict[str, Any], 
                                  total_time: float) -> EvaluationResult:
        """システム全体の性能評価"""
        
        # 成功率の評価
        successful_slots = sum(1 for result in slot_results.values() 
                             if result.get('text') and not result.get('error'))
        total_slots = len(slot_results)
        success_rate = successful_slots / total_slots if total_slots > 0 else 0.0
        
        # 実行時間の評価
        time_score = self._evaluate_execution_time(total_time)
        
        # 出力品質の平均
        quality_scores = []
        for result in slot_results.values():
            if result.get('metadata') and 'quality_score' in result['metadata']:
                quality_scores.append(result['metadata']['quality_score'])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # システム効率性スコア
        efficiency_score = (
            success_rate * 0.4 +
            time_score * 0.3 +
            avg_quality * 0.3
        )
        
        # 履歴に記録
        performance_data = {
            'timestamp': time.time(),
            'success_rate': success_rate,
            'execution_time': total_time,
            'avg_quality': avg_quality,
            'efficiency_score': efficiency_score
        }
        self.performance_history.append(performance_data)
        
        # 履歴サイズ制限
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        details = {
            'successful_slots': successful_slots,
            'total_slots': total_slots,
            'success_rate': success_rate,
            'execution_time': total_time,
            'time_score': time_score,
            'average_quality': avg_quality,
            'quality_scores': quality_scores,
            'history_size': len(self.performance_history)
        }
        
        return EvaluationResult(
            metric_name="system_performance",
            score=efficiency_score,
            max_score=1.0,
            details=details,
            timestamp=time.time()
        )
    
    def _evaluate_execution_time(self, execution_time: float) -> float:
        """実行時間の評価"""
        # 理想的な実行時間を設定
        ideal_time = self.config.get('ideal_execution_time', 10.0)
        max_acceptable_time = self.config.get('max_acceptable_time', 30.0)
        
        if execution_time <= ideal_time:
            return 1.0
        elif execution_time <= max_acceptable_time:
            # 線形減衰
            return 1.0 - (execution_time - ideal_time) / (max_acceptable_time - ideal_time) * 0.5
        else:
            return 0.3  # 遅すぎる場合の最低スコア
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """性能トレンドの分析"""
        if len(self.performance_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_data = self.performance_history[-10:]
        older_data = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else []
        
        recent_avg = sum(d['efficiency_score'] for d in recent_data) / len(recent_data)
        
        if older_data:
            older_avg = sum(d['efficiency_score'] for d in older_data) / len(older_data)
            trend = 'improving' if recent_avg > older_avg * 1.05 else 'declining' if recent_avg < older_avg * 0.95 else 'stable'
        else:
            trend = 'stable'
        
        return {
            'status': 'success',
            'trend': trend,
            'recent_average': recent_avg,
            'data_points': len(self.performance_history),
            'recent_scores': [d['efficiency_score'] for d in recent_data]
        }


class EvaluationReporter:
    """評価レポート生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reports = []
    
    def generate_comprehensive_report(self, slot_evaluations: List[EvaluationResult],
                                    boids_evaluation: Optional[EvaluationResult],
                                    system_evaluation: EvaluationResult) -> Dict[str, Any]:
        """包括的な評価レポートを生成"""
        
        report_timestamp = time.time()
        
        # Slot評価サマリー
        slot_summary = {
            'total_slots': len(slot_evaluations),
            'average_score': sum(e.normalized_score for e in slot_evaluations) / len(slot_evaluations) if slot_evaluations else 0.0,
            'best_slot': max(slot_evaluations, key=lambda e: e.normalized_score).details['slot_name'] if slot_evaluations else None,
            'slot_scores': {e.details['slot_name']: e.normalized_score for e in slot_evaluations}
        }
        
        # 総合評価
        overall_scores = [system_evaluation.normalized_score]
        if boids_evaluation:
            overall_scores.append(boids_evaluation.normalized_score)
        if slot_evaluations:
            overall_scores.extend(e.normalized_score for e in slot_evaluations)
        
        overall_score = sum(overall_scores) / len(overall_scores)
        overall_grade = self._score_to_grade(overall_score)
        
        report = {
            'timestamp': report_timestamp,
            'overall_score': overall_score,
            'overall_grade': overall_grade,
            'slot_summary': slot_summary,
            'system_performance': {
                'score': system_evaluation.normalized_score,
                'grade': system_evaluation.grade,
                'details': system_evaluation.details
            },
            'boids_integration': {
                'enabled': boids_evaluation is not None,
                'score': boids_evaluation.normalized_score if boids_evaluation else 0.0,
                'grade': boids_evaluation.grade if boids_evaluation else 'N/A',
                'details': boids_evaluation.details if boids_evaluation else {}
            },
            'individual_slots': [
                {
                    'slot_name': e.details['slot_name'],
                    'score': e.normalized_score,
                    'grade': e.grade,
                    'details': e.details
                }
                for e in slot_evaluations
            ],
            'recommendations': self._generate_recommendations(slot_evaluations, boids_evaluation, system_evaluation)
        }
        
        self.reports.append(report)
        return report
    
    def _score_to_grade(self, score: float) -> str:
        """スコアを評価等級に変換"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, slot_evaluations: List[EvaluationResult],
                                boids_evaluation: Optional[EvaluationResult],
                                system_evaluation: EvaluationResult) -> List[str]:
        """改善提案の生成"""
        recommendations = []
        
        # Slot別推奨事項
        for eval_result in slot_evaluations:
            if eval_result.normalized_score < 0.6:
                slot_name = eval_result.details['slot_name']
                recommendations.append(f"{slot_name}の出力品質向上が必要です（現在: {eval_result.grade}）")
        
        # Boids統合推奨事項
        if boids_evaluation and boids_evaluation.normalized_score < 0.7:
            coherence = boids_evaluation.details.get('coherence', 0.0)
            diversity = boids_evaluation.details.get('diversity', 0.0)
            
            if coherence < 0.6:
                recommendations.append("Slot間の結束性向上のため、alignment_weightの調整を検討してください")
            if diversity < 0.5:
                recommendations.append("出力の多様性確保のため、separation_radiusの調整を検討してください")
        
        # システム性能推奨事項
        if system_evaluation.normalized_score < 0.7:
            execution_time = system_evaluation.details.get('execution_time', 0.0)
            if execution_time > 15.0:
                recommendations.append("実行時間が長すぎます。並列処理やモデル最適化を検討してください")
        
        return recommendations
    
    def export_report(self, report: Dict[str, Any], file_path: str):
        """レポートをファイルに出力"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"評価レポートを出力: {file_path}")
        except Exception as e:
            logger.error(f"レポート出力エラー: {e}")
