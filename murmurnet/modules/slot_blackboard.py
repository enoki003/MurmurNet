#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slot Blackboard モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~
Slot アーキテクチャ対応の拡張黒板実装
複数の擬似エージェント（Slot）間でのコンテキスト共有と協調を支援

機能:
- Slot 名とテキストの関連付け
- 埋め込みベクトルによる意味的類似度計算
- Slot間の協調パターンサポート
- 従来のBlackboard機能の継承

作者: Yuhi Sonoki
"""

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .blackboard import Blackboard


class SlotEntry:
    """Slot エントリのデータ構造"""
    
    def __init__(self, slot_name: str, text: str, embedding: Optional[np.ndarray] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.slot_name = slot_name
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.entry_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """一意のエントリIDを生成"""
        content = f"{self.slot_name}_{self.text}_{self.timestamp}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'entry_id': self.entry_id,
            'slot_name': self.slot_name,
            'text': self.text,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        return f"SlotEntry({self.slot_name}: {self.text[:50]}...)"


class SlotBlackboard(Blackboard):
    """
    Slot アーキテクチャ対応の拡張黒板
    
    機能:
    - 従来のBlackboard機能
    - Slot エントリの管理
    - 埋め込みベクトルによる類似度計算
    - Slot間協調サポート
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Slot 専用ストレージ
        self.slot_entries: List[SlotEntry] = []
        self.slot_lock = threading.RLock()
        
        # Slot 名の管理
        self.registered_slots = set()
        self.slot_order = []  # 実行順序の記録
        
        # 類似度計算設定
        self.similarity_threshold = config.get('slot_similarity_threshold', 0.7)
        self.max_slot_entries = config.get('max_slot_entries', 100)
        
        if self.debug:
            print(f"SlotBlackboard初期化完了: 最大エントリ{self.max_slot_entries}, 類似度閾値{self.similarity_threshold}")
    
    def register_slot(self, slot_name: str) -> None:
        """Slot を登録"""
        with self.slot_lock:
            if slot_name not in self.registered_slots:
                self.registered_slots.add(slot_name)
                self.slot_order.append(slot_name)
                if self.debug:
                    print(f"Slot '{slot_name}' が登録されました")
    
    def add_slot_entry(self, slot_name: str, text: str, embedding: Optional[np.ndarray] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> SlotEntry:
        """Slot エントリを追加"""
        # Slot を自動登録
        self.register_slot(slot_name)
        
        # エントリ作成
        entry = SlotEntry(slot_name, text, embedding, metadata)
        
        with self.slot_lock:
            self.slot_entries.append(entry)
            
            # 最大エントリ数制限
            if len(self.slot_entries) > self.max_slot_entries:
                # 古いエントリを削除（FIFO）
                removed = self.slot_entries.pop(0)
                if self.debug:
                    print(f"古いSlotエントリを削除: {removed}")
        
        # 従来の黒板にも保存（互換性のため）
        self.write(f"slot_{slot_name}_{entry.entry_id}", entry.to_dict())
        
        if self.debug:
            print(f"SlotEntry追加: {slot_name} - {text[:30]}...")
        
        return entry
    
    def get_slot_entries(self, slot_name: Optional[str] = None) -> List[SlotEntry]:
        """Slot エントリを取得"""
        with self.slot_lock:
            if slot_name is None:
                return self.slot_entries.copy()
            else:
                return [entry for entry in self.slot_entries if entry.slot_name == slot_name]
    
    def get_latest_slot_entry(self, slot_name: str) -> Optional[SlotEntry]:
        """指定Slotの最新エントリを取得"""
        entries = self.get_slot_entries(slot_name)
        return entries[-1] if entries else None
    
    def calculate_similarity(self, entry1: SlotEntry, entry2: SlotEntry) -> float:
        """2つのSlotエントリ間の類似度を計算"""
        if entry1.embedding is None or entry2.embedding is None:
            # テキストベースの簡易類似度（共通単語の割合）
            words1 = set(entry1.text.lower().split())
            words2 = set(entry2.text.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0
        
        # 埋め込みベクトルでのコサイン類似度
        try:
            embedding1 = entry1.embedding.reshape(1, -1)
            embedding2 = entry2.embedding.reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            if self.debug:
                print(f"類似度計算エラー: {e}")
            return 0.0
    
    def find_similar_entries(self, target_entry: SlotEntry, exclude_same_slot: bool = True) -> List[Tuple[SlotEntry, float]]:
        """類似するエントリを検索"""
        similar_entries = []
        
        with self.slot_lock:
            for entry in self.slot_entries:
                if exclude_same_slot and entry.slot_name == target_entry.slot_name:
                    continue
                
                if entry.entry_id == target_entry.entry_id:
                    continue
                
                similarity = self.calculate_similarity(target_entry, entry)
                if similarity >= self.similarity_threshold:
                    similar_entries.append((entry, similarity))
        
        # 類似度の降順でソート
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries
    
    def get_slot_context(self, slot_name: str, include_similar: bool = True) -> Dict[str, Any]:
        """指定Slotのコンテキストを構築"""
        latest_entry = self.get_latest_slot_entry(slot_name)
        if not latest_entry:
            return {}
        
        context = {
            'slot_name': slot_name,
            'latest_entry': latest_entry.to_dict(),
            'all_entries': [entry.to_dict() for entry in self.get_slot_entries(slot_name)],
            'similar_entries': []
        }
        
        if include_similar:
            similar = self.find_similar_entries(latest_entry)
            context['similar_entries'] = [
                {'entry': entry.to_dict(), 'similarity': sim}
                for entry, sim in similar[:5]  # 上位5件
            ]
        
        return context
    
    def get_all_slots_summary(self) -> Dict[str, Any]:
        """全Slotの要約を取得"""
        summary = {
            'registered_slots': list(self.registered_slots),
            'slot_order': self.slot_order.copy(),
            'total_entries': len(self.slot_entries),
            'slot_summaries': {}
        }
        
        for slot_name in self.registered_slots:
            entries = self.get_slot_entries(slot_name)
            latest = entries[-1] if entries else None
            
            summary['slot_summaries'][slot_name] = {
                'entry_count': len(entries),
                'latest_text': latest.text if latest else None,
                'latest_timestamp': latest.timestamp if latest else None
            }
        
        return summary
    
    def clear_slot_entries(self, slot_name: Optional[str] = None) -> None:
        """Slot エントリをクリア"""
        with self.slot_lock:
            if slot_name is None:
                # 全エントリクリア
                cleared_count = len(self.slot_entries)
                self.slot_entries.clear()
                self.registered_slots.clear()
                self.slot_order.clear()
            else:
                # 特定Slotのエントリのみクリア
                before_count = len(self.slot_entries)
                self.slot_entries = [entry for entry in self.slot_entries if entry.slot_name != slot_name]
                cleared_count = before_count - len(self.slot_entries)
                
                # Slot登録情報も削除
                if slot_name in self.registered_slots:
                    self.registered_slots.remove(slot_name)
                    if slot_name in self.slot_order:
                        self.slot_order.remove(slot_name)
        
        if self.debug:
            target = slot_name or "全Slot"
            print(f"{target}のエントリ{cleared_count}件をクリアしました")
    
    def clear_current_turn(self):
        """新しいターンのためにクリア（Slot情報は保持）"""
        # 親クラスのクリア実行
        super().clear_current_turn()
        
        # Slot エントリもクリア（登録Slot情報は保持）
        with self.slot_lock:
            cleared_count = len(self.slot_entries)
            self.slot_entries.clear()
        
        if self.debug:
            print(f"ターンクリア: Slotエントリ{cleared_count}件クリア、登録Slot{len(self.registered_slots)}個保持")
    
    def get_slot_debug_view(self) -> Dict[str, Any]:
        """Slot用のデバッグビューを取得"""
        debug_view = {
            'base_blackboard': self.get_debug_view(),
            'slot_info': self.get_all_slots_summary(),
            'recent_entries': []
        }
        
        # 最新5件のエントリを表示用に整形
        with self.slot_lock:
            recent = self.slot_entries[-5:] if len(self.slot_entries) >= 5 else self.slot_entries
            for entry in recent:
                debug_view['recent_entries'].append({
                    'slot_name': entry.slot_name,
                    'text_preview': entry.text[:100],
                    'timestamp': entry.timestamp,
                    'has_embedding': entry.embedding is not None
                })
        
        return debug_view
    
    def get_structured_context(self) -> Dict[str, Any]:
        """構造化されたコンテキストを取得"""
        # SlotBlackboardAdapterが利用可能な場合は構造化Blackboardから取得
        if hasattr(self, '_structured_adapter') and self._structured_adapter:
            return self._structured_adapter.structured_bb.get_synthesis_context()
        
        # フォールバック: 基本形式でコンテキストを構築
        with self.slot_lock:
            recent_entries = sorted(self.slot_entries, key=lambda x: x.timestamp, reverse=True)[:5]
        
        role_opinions = {}
        for entry in recent_entries:
            # Slot名から役割を推定
            role = 'unknown'
            if 'Reformulator' in entry.slot_name:
                role = 'reformulator'
            elif 'Critic' in entry.slot_name:
                role = 'critic'
            elif 'Supporter' in entry.slot_name:
                role = 'supporter'
            elif 'Synthesizer' in entry.slot_name:
                role = 'synthesizer'
            
            if role not in role_opinions:
                role_opinions[role] = []
            role_opinions[role].append(entry.text)
        
        return {
            'recent_opinions': [{'role': self._get_role_from_slot(e.slot_name), 'content': e.text} for e in recent_entries],
            'role_opinions': role_opinions,
            'analysis': {
                'diversity_score': 0.5,  # デフォルト値
                'consensus_areas': [],
                'conflict_areas': [],
                'opinion_count': len(recent_entries)
            }
        }
    
    def _get_role_from_slot(self, slot_name: str) -> str:
        """Slot名から役割名を取得"""
        if 'Reformulator' in slot_name:
            return 'reformulator'
        elif 'Critic' in slot_name:
            return 'critic'
        elif 'Supporter' in slot_name:
            return 'supporter'
        elif 'Synthesizer' in slot_name:
            return 'synthesizer'
        else:
            return 'unknown'
    
    def connect_structured_adapter(self, adapter):
        """構造化Blackboardアダプターを接続"""
        self._structured_adapter = adapter
    
    # ===== 協調議論サポート機能 =====
    
    def get_cross_reference_context(self, requesting_slot: str) -> Dict[str, Any]:
        """他Slotの意見を参照するためのコンテキストを取得"""
        with self.slot_lock:
            other_entries = [entry for entry in self.slot_entries if entry.slot_name != requesting_slot]
            
            # 最新のエントリを役割別に整理
            latest_by_role = {}
            for entry in reversed(other_entries):  # 新しい順
                role = self._get_role_from_slot(entry.slot_name)
                if role not in latest_by_role:
                    latest_by_role[role] = entry
            
            cross_reference = {
                'other_opinions': [
                    {
                        'slot_name': entry.slot_name,
                        'role': self._get_role_from_slot(entry.slot_name),
                        'content': entry.text,
                        'timestamp': entry.timestamp,
                        'similarity': self._calculate_text_similarity(entry.text, self.get_latest_slot_entry(requesting_slot))
                    }
                    for entry in latest_by_role.values()
                ],
                'conflict_indicators': self._detect_conflicts(other_entries),
                'consensus_indicators': self._detect_consensus(other_entries)
            }
            
            return cross_reference
    
    def _calculate_text_similarity(self, text1: str, entry2: Optional[SlotEntry]) -> float:
        """テキスト間の類似度を計算（簡易版）"""
        if not entry2 or not text1:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(entry2.text.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def _detect_conflicts(self, entries: List[SlotEntry]) -> List[Dict[str, Any]]:
        """意見の対立を検出"""
        conflicts = []
        conflict_keywords = ['しかし', 'だが', '一方で', '反対に', '問題は', '課題は', 'リスク']
        
        for entry in entries:
            for keyword in conflict_keywords:
                if keyword in entry.text:
                    conflicts.append({
                        'slot_name': entry.slot_name,
                        'conflict_indicator': keyword,
                        'text_snippet': entry.text[:100]
                    })
                    break
        
        return conflicts
    
    def _detect_consensus(self, entries: List[SlotEntry]) -> List[Dict[str, Any]]:
        """意見の合意を検出"""
        consensus = []
        consensus_keywords = ['同様に', '同じく', '賛成', '支持', '確かに', 'その通り']
        
        for entry in entries:
            for keyword in consensus_keywords:
                if keyword in entry.text:
                    consensus.append({
                        'slot_name': entry.slot_name,
                        'consensus_indicator': keyword,
                        'text_snippet': entry.text[:100]
                    })
                    break
        
        return consensus
    
    def add_discussion_round(self, round_number: int, phase: str) -> None:
        """議論ラウンドの記録を追加"""
        if not hasattr(self, 'discussion_rounds'):
            self.discussion_rounds = []
        
        self.discussion_rounds.append({
            'round': round_number,
            'phase': phase,
            'timestamp': time.time(),
            'entries_count': len(self.slot_entries)
        })
    
    def get_discussion_history(self) -> Dict[str, Any]:
        """議論履歴を取得"""
        with self.slot_lock:
            # 時系列順にエントリを整理
            timeline = []
            for entry in self.slot_entries:
                timeline.append({
                    'timestamp': entry.timestamp,
                    'slot_name': entry.slot_name,
                    'role': self._get_role_from_slot(entry.slot_name),
                    'content': entry.text,
                    'phase': entry.metadata.get('phase', 1)
                })
            
            # フェーズ別に整理
            phases = {}
            for item in timeline:
                phase = item['phase']
                if phase not in phases:
                    phases[phase] = []
                phases[phase].append(item)
            
            return {
                'timeline': sorted(timeline, key=lambda x: x['timestamp']),
                'phases': phases,
                'total_entries': len(timeline),
                'discussion_rounds': getattr(self, 'discussion_rounds', [])
            }
    
    def calculate_collaboration_metrics(self) -> Dict[str, float]:
        """協調度メトリクスを計算"""
        with self.slot_lock:
            if len(self.slot_entries) < 2:
                return {'collaboration_score': 0.0, 'diversity_score': 0.0, 'consensus_score': 0.0}
            
            # 多様性スコア（異なるSlotからの意見の数）
            unique_slots = len(set(entry.slot_name for entry in self.slot_entries))
            diversity_score = min(unique_slots / 4.0, 1.0)  # 4つのSlotが理想
            
            # 相互参照スコア（他Slotを明示的に言及する頻度）
            reference_count = 0
            total_entries = len(self.slot_entries)
            
            for entry in self.slot_entries:
                other_slots = [slot for slot in self.registered_slots if slot != entry.slot_name]
                for other_slot in other_slots:
                    slot_keywords = ['Reformulator', 'Critic', 'Supporter', 'Synthesizer']
                    if any(keyword in entry.text for keyword in slot_keywords):
                        reference_count += 1
                        break
            
            reference_score = reference_count / total_entries if total_entries > 0 else 0.0
            
            # コンセンサススコア（合意指標の検出）
            consensus_indicators = self._detect_consensus(self.slot_entries)
            conflict_indicators = self._detect_conflicts(self.slot_entries)
            
            if len(consensus_indicators) + len(conflict_indicators) > 0:
                consensus_score = len(consensus_indicators) / (len(consensus_indicators) + len(conflict_indicators))
            else:
                consensus_score = 0.5  # 中立
            
            # 総合協調スコア
            collaboration_score = (diversity_score * 0.3 + reference_score * 0.4 + consensus_score * 0.3)
            
            return {
                'collaboration_score': collaboration_score,
                'diversity_score': diversity_score,
                'reference_score': reference_score,
                'consensus_score': consensus_score,
                'consensus_indicators': len(consensus_indicators),
                'conflict_indicators': len(conflict_indicators)
            }
