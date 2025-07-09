#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structured Blackboard モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
エージェント間協調のための構造化された黒板パターン実装

目的: 情報の役割分離、バージョン管理、アクセス制御を備えた真のBlackboard

機能:
- 構造化データ: opinions, external_knowledge, summary の3セクション
- バージョン管理: 自動インクリメント版数、古い版への書き込み防止
- 専用API: add_opinion, add_knowledge, update_summary
- 履歴管理: 全操作の時系列履歴
- 擬似非同期: スナップショット配布 → 各エージェント生成 → まとめてコミット

作者: Yuhi Sonoki
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """エージェントの役割定義（読み書き規約）"""
    RESEARCHER = "researcher"      # external_knowledge を書き込み
    CRITIC = "critic"             # opinions を書き込み  
    SUPPORTER = "supporter"       # opinions を書き込み
    REFORMULATOR = "reformulator" # opinions を書き込み
    SYNTHESIZER = "synthesizer"   # summary を書き込み
    OUTPUT = "output"             # 全てを読み取り専用

@dataclass
class BlackboardEntry:
    """Blackboard エントリの標準構造"""
    id: str
    agent_id: str
    agent_role: AgentRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    version_read: int = 0  # エージェントが読んだBlackboard版数

@dataclass
class BlackboardSnapshot:
    """Blackboard の状態スナップショット"""
    version: int
    opinions: List[BlackboardEntry]
    external_knowledge: List[BlackboardEntry] 
    summary: Optional[BlackboardEntry]
    timestamp: float = field(default_factory=time.time)

class StructuredBlackboard:
    """
    構造化された協調Blackboard
    
    アーキテクチャ:
    1. 3つの専用セクション (opinions, external_knowledge, summary)
    2. バージョン管理による一貫性保証
    3. エージェント役割に基づくアクセス制御
    4. 完全な操作履歴
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug = config.get('debug', False)
        
        # === 構造化データストレージ ===
        self.opinions: List[BlackboardEntry] = []
        self.external_knowledge: List[BlackboardEntry] = []
        self.summary: Optional[BlackboardEntry] = None
        
        # === バージョン管理 ===
        self.version = 1
        self.version_lock = threading.RLock()
        
        # === 履歴管理 ===
        self.history: deque = deque(maxlen=config.get('max_history', 1000))
        
        # === スレッドセーフティ ===
        self.data_lock = threading.RLock()
        
        # === 統計 ===
        self.stats = {
            'total_operations': 0,
            'opinions_count': 0,
            'knowledge_count': 0,
            'summary_updates': 0,
            'version_conflicts': 0
        }
        
        if self.debug:
            logger.info(f"StructuredBlackboard 初期化完了 (version: {self.version})")
    
    # ================================================================
    # 専用 API (role-based access control)
    # ================================================================
    
    def add_opinion(self, agent_id: str, agent_role: AgentRole, content: str, 
                   version_read: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        意見を追加 (CRITIC, SUPPORTER, REFORMULATOR 専用)
        
        Args:
            agent_id: エージェントID
            agent_role: エージェント役割
            content: 意見内容
            version_read: エージェントが読んだBlackboard版数
            metadata: 追加メタデータ
            
        Returns:
            成功時True、バージョン競合時False
        """
        if agent_role not in [AgentRole.CRITIC, AgentRole.SUPPORTER, AgentRole.REFORMULATOR]:
            logger.warning(f"不正な役割によるopinion追加試行: {agent_role}")
            return False
        
        return self._add_entry('opinions', agent_id, agent_role, content, version_read, metadata)
    
    def add_knowledge(self, agent_id: str, content: str, source: str,
                     version_read: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        外部知識を追加 (RESEARCHER 専用)
        
        Args:
            agent_id: エージェントID  
            content: 知識内容
            source: 知識ソース
            version_read: エージェントが読んだBlackboard版数
            metadata: 追加メタデータ
            
        Returns:
            成功時True、バージョン競合時False
        """
        enriched_metadata = metadata or {}
        enriched_metadata['source'] = source
        
        return self._add_entry('external_knowledge', agent_id, AgentRole.RESEARCHER, 
                             content, version_read, enriched_metadata)
    
    def update_summary(self, agent_id: str, content: str, version_read: int,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        要約を更新 (SYNTHESIZER 専用)
        
        Args:
            agent_id: エージェントID
            content: 要約内容  
            version_read: エージェントが読んだBlackboard版数
            metadata: 追加メタデータ
            
        Returns:
            成功時True、バージョン競合時False
        """
        if not self._check_version(version_read):
            return False
        
        with self.data_lock:
            entry = BlackboardEntry(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                agent_role=AgentRole.SYNTHESIZER,
                content=content,
                metadata=metadata or {},
                version_read=version_read
            )
            
            self.summary = entry
            self._increment_version()
            self._add_to_history('update_summary', agent_id, entry.id)
            self.stats['summary_updates'] += 1
            self.stats['total_operations'] += 1
            
            if self.debug:
                logger.info(f"要約更新: {agent_id} (version: {self.version})")
            
            return True
    
    # ================================================================
    # 読み取り API (全エージェント共通)
    # ================================================================
    
    def get_snapshot(self) -> BlackboardSnapshot:
        """現在のBlackboard状態のスナップショットを取得"""
        with self.data_lock:
            return BlackboardSnapshot(
                version=self.version,
                opinions=self.opinions.copy(),
                external_knowledge=self.external_knowledge.copy(),
                summary=self.summary
            )
    
    def get_current_version(self) -> int:
        """現在のバージョン番号を取得"""
        return self.version
    
    def get_latest_opinions(self, limit: int = 10) -> List[BlackboardEntry]:
        """最新の意見を取得"""
        with self.data_lock:
            return sorted(self.opinions, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_knowledge_by_source(self, source: str) -> List[BlackboardEntry]:
        """ソース別の外部知識を取得"""
        with self.data_lock:
            return [k for k in self.external_knowledge 
                   if k.metadata.get('source') == source]
    
    def get_current_summary(self) -> Optional[BlackboardEntry]:
        """現在の要約を取得"""
        return self.summary
    
    # ================================================================
    # 擬似非同期ステップ制御
    # ================================================================
    
    def start_round(self) -> Tuple[BlackboardSnapshot, int]:
        """
        新しいラウンドを開始
        
        Returns:
            (現在のスナップショット, 現在のバージョン)
        """
        snapshot = self.get_snapshot()
        if self.debug:
            logger.info(f"ラウンド開始: version {snapshot.version}")
        return snapshot, snapshot.version
    
    def commit_round(self, agent_commits: List[Tuple[str, str, Any]]) -> bool:
        """
        ラウンドの一括コミット
        
        Args:
            agent_commits: [(operation, agent_id, data), ...] のリスト
            
        Returns:
            全て成功時True
        """
        success_count = 0
        
        with self.data_lock:
            for operation, agent_id, data in agent_commits:
                try:
                    if operation == 'add_opinion':
                        # SlotRunner からのデータ構造に合わせる
                        agent_role = data.get('agent_role', AgentRole.REFORMULATOR)
                        if self.add_opinion(agent_id, agent_role, data['content'], 
                                          data['version_read'], data.get('metadata')):
                            success_count += 1
                    elif operation == 'add_knowledge':
                        if self.add_knowledge(agent_id, data['content'], data['source'],
                                            data['version_read'], data.get('metadata')):
                            success_count += 1
                    elif operation == 'update_summary':
                        if self.update_summary(agent_id, data['content'], 
                                             data['version_read'], data.get('metadata')):
                            success_count += 1
                except Exception as e:
                    logger.error(f"コミットエラー {operation}/{agent_id}: {e}")
        
        if self.debug:
            logger.info(f"ラウンドコミット完了: {success_count}/{len(agent_commits)} 成功")
        
        return success_count == len(agent_commits)
    
    # ================================================================
    # 内部実装
    # ================================================================
    
    def _add_entry(self, section: str, agent_id: str, agent_role: AgentRole, 
                  content: str, version_read: int, metadata: Optional[Dict[str, Any]]) -> bool:
        """エントリ追加の共通実装"""
        if not self._check_version(version_read):
            return False
        
        with self.data_lock:
            entry = BlackboardEntry(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                agent_role=agent_role,
                content=content,
                metadata=metadata or {},
                version_read=version_read
            )
            
            if section == 'opinions':
                self.opinions.append(entry)
                self.stats['opinions_count'] += 1
            elif section == 'external_knowledge':
                self.external_knowledge.append(entry)
                self.stats['knowledge_count'] += 1
            
            self._increment_version()
            self._add_to_history(f'add_{section}', agent_id, entry.id)
            self.stats['total_operations'] += 1
            
            if self.debug:
                logger.info(f"{section} 追加: {agent_id} (version: {self.version})")
            
            return True
    
    def _check_version(self, version_read: int) -> bool:
        """バージョン競合チェック"""
        if version_read < self.version - 1:  # 1版の遅れは許容
            self.stats['version_conflicts'] += 1
            if self.debug:
                logger.warning(f"バージョン競合: read={version_read}, current={self.version}")
            return False
        return True
    
    def _increment_version(self) -> None:
        """バージョンをインクリメント"""
        with self.version_lock:
            self.version += 1
    
    def _add_to_history(self, operation: str, agent_id: str, entry_id: str) -> None:
        """履歴に操作を記録"""
        self.history.append({
            'timestamp': time.time(),
            'operation': operation,
            'agent_id': agent_id,
            'entry_id': entry_id,
            'version': self.version
        })
    
    # ================================================================
    # 統計・デバッグ
    # ================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self.data_lock:
            return {
                **self.stats,
                'current_version': self.version,
                'opinions_count': len(self.opinions),
                'knowledge_count': len(self.external_knowledge),
                'has_summary': self.summary is not None,
                'history_size': len(self.history)
            }
    
    def get_agent_activity(self) -> Dict[str, int]:
        """エージェント別活動統計"""
        activity = {}
        for record in self.history:
            agent = record['agent_id']
            activity[agent] = activity.get(agent, 0) + 1
        return activity
    
    def clear_all(self) -> None:
        """全データをクリア（テスト用）"""
        with self.data_lock:
            self.opinions.clear()
            self.external_knowledge.clear()
            self.summary = None
            self.history.clear()
            self.version = 1
            self.stats = {k: 0 for k in self.stats}
            
            if self.debug:
                logger.info("Blackboard 全クリア完了")
    
    def analyze_opinion_diversity(self) -> Dict[str, Any]:
        """意見の多様性と統合度を分析"""
        with self.data_lock:
            if not self.opinions:
                return {'diversity_score': 0.0, 'consensus_areas': [], 'conflict_areas': []}
            
            # 意見の内容分析（簡単なキーワードベース）
            opinion_keywords = []
            for opinion in self.opinions:
                # 重要なキーワードを抽出（日本語対応）
                import re
                keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{2,}', opinion.content)
                opinion_keywords.append(set(keywords))
            
            # 多様性スコア計算（ジャッカード類似度ベース）
            diversity_scores = []
            for i in range(len(opinion_keywords)):
                for j in range(i+1, len(opinion_keywords)):
                    intersection = len(opinion_keywords[i] & opinion_keywords[j])
                    union = len(opinion_keywords[i] | opinion_keywords[j])
                    similarity = intersection / union if union > 0 else 0
                    diversity_scores.append(1 - similarity)  # 多様性 = 1 - 類似度
            
            avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
            
            # コンセンサス領域と対立領域の特定
            common_keywords = set.intersection(*opinion_keywords) if opinion_keywords else set()
            unique_keywords = []
            for keywords in opinion_keywords:
                unique_keywords.extend(keywords - common_keywords)
            
            return {
                'diversity_score': avg_diversity,
                'consensus_areas': list(common_keywords)[:5],  # 上位5つ
                'conflict_areas': list(set(unique_keywords))[:5],  # 上位5つ
                'opinion_count': len(self.opinions),
                'knowledge_count': len(self.external_knowledge)
            }
    
    def get_synthesis_context(self) -> Dict[str, Any]:
        """Synthesizer用の統合コンテキストを提供"""
        analysis = self.analyze_opinion_diversity()
        
        # 最新の意見とナレッジを構造化
        recent_opinions = sorted(self.opinions, key=lambda x: x.timestamp, reverse=True)[:5]
        recent_knowledge = sorted(self.external_knowledge, key=lambda x: x.timestamp, reverse=True)[:3]
        
        # 役割別の意見分類
        role_opinions = {}
        for opinion in recent_opinions:
            role = opinion.agent_role.value
            if role not in role_opinions:
                role_opinions[role] = []
            role_opinions[role].append(opinion.content)
        
        return {
            'analysis': analysis,
            'recent_opinions': [{'role': op.agent_role.value, 'content': op.content} for op in recent_opinions],
            'recent_knowledge': [{'source': kn.metadata.get('source', 'unknown'), 'content': kn.content} for kn in recent_knowledge],
            'role_opinions': role_opinions,
            'current_summary': self.summary.content if self.summary else None
        }


# ================================================================
# 便利関数
# ================================================================

def create_structured_blackboard(config: Optional[Dict[str, Any]] = None) -> StructuredBlackboard:
    """構造化Blackboardの便利作成関数"""
    default_config = {
        'debug': True,
        'max_history': 1000
    }
    
    if config:
        default_config.update(config)
    
    return StructuredBlackboard(default_config)
