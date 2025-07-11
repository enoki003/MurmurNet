#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slot → StructuredBlackboard アダプター
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
既存のSlotシステムを新しい構造化Blackboardで動作させるアダプター

機能:
- SlotからStructuredBlackboardへの透過的な移行
- 既存のSlotRunner APIとの互換性維持
- Boids統合の強化
- エージェント役割の自動マッピング

作者: Yuhi Sonoki
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from .structured_blackboard import StructuredBlackboard, AgentRole, BlackboardSnapshot
from .slot_blackboard import SlotBlackboard, SlotEntry  # 既存との互換性維持

logger = logging.getLogger(__name__)

class SlotBlackboardAdapter:
    """
    SlotシステムとStructuredBlackboardをブリッジするアダプター
    
    目的:
    1. 既存のSlotRunner APIとの後方互換性
    2. SlotRole → AgentRole の自動マッピング
    3. Boids統合での意味的情報の保持
    4. 段階的移行の支援
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug = config.get('debug', False)
        
        # 新しい構造化Blackboard
        self.structured_bb = StructuredBlackboard(config)
        
        # 既存システムとの互換性のために従来のSlotBlackboardも保持
        self.legacy_bb = SlotBlackboard(config) if 'SlotBlackboard' in str(type(SlotBlackboard)) else None
        
        # Slot名 → AgentRole マッピング
        self.slot_role_mapping = {
            'ReformulatorSlot': AgentRole.REFORMULATOR,
            'CriticSlot': AgentRole.CRITIC,
            'SupporterSlot': AgentRole.SUPPORTER,
            'SynthesizerSlot': AgentRole.SYNTHESIZER,
            'OutputSlot': AgentRole.OUTPUT,
            'OutputAgent': AgentRole.OUTPUT
        }
        
        if self.debug:
            logger.info("SlotBlackboardAdapter 初期化完了")
    
    # ================================================================
    # 既存 SlotBlackboard API の移行
    # ================================================================
    
    def add_slot_entry(self, slot_name: str, text: str, embedding=None, 
                      metadata: Optional[Dict[str, Any]] = None) -> 'SlotEntry':
        """
        既存のadd_slot_entry APIとの互換性を提供
        内部的にはStructuredBlackboardに適切に振り分け
        """
        # Slot名からエージェント役割を決定
        agent_role = self.slot_role_mapping.get(slot_name, AgentRole.REFORMULATOR)
        
        # 現在のバージョンを取得
        current_version = self.structured_bb.get_current_version()
        
        # メタデータの拡張
        enriched_metadata = metadata or {}
        if embedding is not None:
            enriched_metadata['embedding'] = embedding
        enriched_metadata['slot_name'] = slot_name
        
        # 役割に応じてStructuredBlackboardに追加
        success = False
        if agent_role in [AgentRole.CRITIC, AgentRole.SUPPORTER, AgentRole.REFORMULATOR]:
            success = self.structured_bb.add_opinion(
                agent_id=slot_name,
                agent_role=agent_role, 
                content=text,
                version_read=current_version,
                metadata=enriched_metadata
            )
        elif agent_role == AgentRole.SYNTHESIZER:
            success = self.structured_bb.update_summary(
                agent_id=slot_name,
                content=text,
                version_read=current_version,
                metadata=enriched_metadata
            )
        
        # 既存システムとの互換性のためにSlotEntryを作成して返す
        entry = SlotEntry(
            entry_id=f"{slot_name}_{int(time.time() * 1000)}",
            slot_name=slot_name,
            text=text,
            embedding=embedding,
            metadata=enriched_metadata,
            timestamp=time.time()
        )
        
        # 従来Blackboardにも追加（段階的移行中のみ）
        if self.legacy_bb:
            self.legacy_bb.add_slot_entry(slot_name, text, embedding, metadata)
        
        if self.debug:
            logger.info(f"Slot entry追加: {slot_name} → {agent_role.value} (成功: {success})")
        
        return entry
    
    def get_slot_entries(self, slot_name: Optional[str] = None) -> List['SlotEntry']:
        """既存のget_slot_entries APIとの互換性"""
        entries = []
        
        # StructuredBlackboardから取得
        snapshot = self.structured_bb.get_snapshot()
        
        # opinions から変換
        for opinion in snapshot.opinions:
            if slot_name is None or opinion.agent_id == slot_name:
                entry = SlotEntry(
                    entry_id=opinion.id,
                    slot_name=opinion.agent_id,
                    text=opinion.content,
                    embedding=opinion.metadata.get('embedding'),
                    metadata=opinion.metadata,
                    timestamp=opinion.timestamp
                )
                entries.append(entry)
        
        # summary から変換
        if snapshot.summary and (slot_name is None or snapshot.summary.agent_id == slot_name):
            entry = SlotEntry(
                entry_id=snapshot.summary.id,
                slot_name=snapshot.summary.agent_id,
                text=snapshot.summary.content,
                embedding=snapshot.summary.metadata.get('embedding'),
                metadata=snapshot.summary.metadata,
                timestamp=snapshot.summary.timestamp
            )
            entries.append(entry)
        
        return sorted(entries, key=lambda x: x.timestamp)
    
    # ================================================================
    # 新しい構造化API
    # ================================================================
    
    def start_collaborative_round(self) -> Tuple[BlackboardSnapshot, int]:
        """協調ラウンドを開始"""
        return self.structured_bb.start_round()
    
    def get_opinions_for_synthesis(self) -> List[str]:
        """統合用の意見リストを取得"""
        snapshot = self.structured_bb.get_snapshot()
        return [opinion.content for opinion in snapshot.opinions]
    
    def get_knowledge_for_context(self) -> List[str]:
        """コンテキスト用の知識リストを取得"""
        snapshot = self.structured_bb.get_snapshot()
        return [knowledge.content for knowledge in snapshot.external_knowledge]
    
    def update_synthesis_summary(self, agent_id: str, summary_text: str, 
                                version_read: int) -> bool:
        """統合要約を更新"""
        return self.structured_bb.update_summary(agent_id, summary_text, version_read)
    
    # ================================================================
    # Boids統合支援
    # ================================================================
    
    def get_boids_integration_data(self) -> Dict[str, Any]:
        """
        Boids統合のための構造化データを取得
        各Slotの出力をBoids則に適用するためのデータ形式
        """
        snapshot = self.structured_bb.get_snapshot()
        
        # Slot出力をBoids用に構造化
        boids_data = {
            'opinions': [],
            'summary': None,
            'knowledge_context': [],
            'version': snapshot.version,
            'strategy': 'boids_consensus'  # デフォルト戦略
        }
        
        # 意見をBoids用に変換
        for opinion in snapshot.opinions:
            boids_entry = {
                'agent_id': opinion.agent_id,
                'role': opinion.agent_role.value,
                'content': opinion.content,
                'embedding': opinion.metadata.get('embedding'),
                'timestamp': opinion.timestamp,
                'quality_score': self._calculate_quality_score(opinion)
            }
            boids_data['opinions'].append(boids_entry)
        
        # 要約情報
        if snapshot.summary:
            boids_data['summary'] = {
                'content': snapshot.summary.content,
                'agent_id': snapshot.summary.agent_id,
                'timestamp': snapshot.summary.timestamp
            }
        
        # 知識コンテキスト
        for knowledge in snapshot.external_knowledge:
            boids_data['knowledge_context'].append({
                'source': knowledge.metadata.get('source', 'unknown'),
                'content': knowledge.content[:200],  # 要約用に短縮
                'timestamp': knowledge.timestamp
            })
        
        return boids_data
    
    def _calculate_quality_score(self, entry) -> float:
        """エントリの品質スコアを計算（Boids用）"""
        # 簡易的な品質計算
        base_score = 0.5
        
        # 長さによる調整
        length_score = min(len(entry.content) / 200, 1.0)
        
        # 役割による調整
        role_weights = {
            AgentRole.CRITIC: 0.9,
            AgentRole.SUPPORTER: 0.8,
            AgentRole.REFORMULATOR: 0.7,
            AgentRole.SYNTHESIZER: 1.0
        }
        role_score = role_weights.get(entry.agent_role, 0.6)
        
        # メタデータによる調整
        metadata_score = 0.1 if entry.metadata.get('embedding') is not None else 0.0
        
        return min(base_score + length_score * 0.3 + role_score * 0.4 + metadata_score, 1.0)
    
    # ================================================================
    # 移行支援・統計
    # ================================================================
    
    def get_migration_status(self) -> Dict[str, Any]:
        """移行状況を報告"""
        structured_stats = self.structured_bb.get_statistics()
        
        return {
            'structured_blackboard': {
                'active': True,
                'version': structured_stats['current_version'],
                'operations': structured_stats['total_operations'],
                'opinions': structured_stats['opinions_count'],
                'knowledge': structured_stats['knowledge_count'],
                'summary_updates': structured_stats['summary_updates']
            },
            'compatibility': self.legacy_bb is not None,
            'slot_mappings': len(self.slot_role_mapping)
        }
    
    def clear_turn(self) -> None:
        """ターンクリア（既存APIとの互換性）"""
        # Blackboard version維持
        if self.debug:
            logger.info("Turn cleared - StructuredBlackboard version維持")
        
        # 従来システムのクリアは行うが、StructuredBlackboardは維持
        if self.legacy_bb:
            self.legacy_bb.clear_turn()
    
    # ================================================================
    # SlotRunner統合メソッド
    # ================================================================
    
    def connect_legacy_blackboard(self, legacy_bb: 'SlotBlackboard'):
        """従来のSlotBlackboardを接続（移行期間中の互換性）"""
        self.legacy_bb = legacy_bb
        if self.debug:
            logger.info("SlotBlackboard接続完了")
    
    def _get_agent_role(self, slot_name: str) -> AgentRole:
        """Slot名からAgentRoleを取得"""
        return self.slot_role_mapping.get(slot_name, AgentRole.REFORMULATOR)
    
    def get_structured_blackboard(self) -> StructuredBlackboard:
        """構造化Blackboardへの直接アクセス"""
        return self.structured_bb
    
    def get_blackboard_snapshot(self) -> BlackboardSnapshot:
        """現在のBlackboard状態スナップショット"""
        return self.structured_bb.get_snapshot()
    
    def start_collaborative_round(self) -> Tuple[BlackboardSnapshot, int]:
        """協調ラウンドの開始"""
        return self.structured_bb.start_round()
    
    def commit_collaborative_round(self, commits: List[Tuple[str, str, Any]]) -> bool:
        """協調ラウンドのコミット"""
        return self.structured_bb.commit_round(commits)

# ================================================================
# SlotEntry クラス（互換性のため）
# ================================================================

import time

class SlotEntry:
    """SlotBlackboard互換性のためのエントリクラス"""
    
    def __init__(self, entry_id: str, slot_name: str, text: str, 
                 embedding=None, metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[float] = None):
        self.entry_id = entry_id
        self.slot_name = slot_name
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
