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
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .blackboard import Blackboard
from .redis_blackboard import RedisBlackboard, create_blackboard


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


class SlotBlackboard:
    """
    Slot アーキテクチャ対応の分散黒板
    
    RedisBlackboardベースで複数プロセス・ホスト間での
    Slotデータ共有を実現
    
    機能:
    - Redis分散ブラックボード機能
    - Slot エントリの管理
    - 埋め込みベクトルによる類似度計算
    - Slot間協調サポート
    """
    
    def __init__(self, config):
        # ベースブラックボードを作成（Redis優先）
        self.base_blackboard = create_blackboard(config)
        self.config = config
        self.debug = config.get('debug', False)
        
        # Slot 専用設定
        self.slot_key_prefix = "slot_entry:"
        self.slot_meta_prefix = "slot_meta:"
        
        # Slot 名の管理（Redis SET使用）
        self.slots_set_key = "registered_slots"
        self.slot_order_key = "slot_execution_order"
        
        # 類似度計算設定
        self.similarity_threshold = config.get('slot_similarity_threshold', 0.7)
        self.max_slot_entries = config.get('max_slot_entries', 100)
        
        if self.debug:
            print(f"SlotBlackboard初期化完了: 最大エントリ{self.max_slot_entries}, 類似度閾値{self.similarity_threshold}")
    
    def register_slot(self, slot_name: str) -> None:
        """Slot を登録（Redis SET使用）"""
        try:
            # Redis SETに追加
            if hasattr(self.base_blackboard, 'redis_client'):
                # Redis版の場合
                redis_client = self.base_blackboard.redis_client
                redis_client.sadd(self.base_blackboard._make_key(self.slots_set_key), slot_name)
                redis_client.lpush(self.base_blackboard._make_key(self.slot_order_key), slot_name)
            else:
                # ローカル版の場合
                existing_slots = self.base_blackboard.read(self.slots_set_key) or set()
                existing_slots.add(slot_name)
                self.base_blackboard.write(self.slots_set_key, list(existing_slots))
                
                existing_order = self.base_blackboard.read(self.slot_order_key) or []
                if slot_name not in existing_order:
                    existing_order.append(slot_name)
                    self.base_blackboard.write(self.slot_order_key, existing_order)
            
            if self.debug:
                print(f"Slot '{slot_name}' が登録されました")
                
        except Exception as e:
            print(f"Slot登録エラー: {e}")
    
    def add_slot_entry(self, slot_name: str, text: str, embedding: Optional[np.ndarray] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> SlotEntry:
        """Slot エントリを追加（Redis分散対応）"""
        # Slot を自動登録
        self.register_slot(slot_name)
        
        # エントリ作成
        entry = SlotEntry(slot_name, text, embedding, metadata)
        
        # Redis保存用にエントリをシリアライズ
        entry_data = entry.to_dict()
        
        # 埋め込みベクトルは別途保存（バイナリ形式）
        if embedding is not None:
            embedding_key = f"{self.slot_key_prefix}embedding_{entry.entry_id}"
            # NumPy配列をRedisに保存するためにbase64エンコード
            import base64
            embedding_b64 = base64.b64encode(embedding.tobytes()).decode('utf-8')
            embedding_info = {
                'shape': embedding.shape,
                'dtype': str(embedding.dtype),
                'data': embedding_b64
            }
            self.base_blackboard.write(embedding_key, embedding_info)
        
        # エントリ本体を保存
        entry_key = f"{self.slot_key_prefix}{entry.entry_id}"
        self.base_blackboard.write(entry_key, entry_data)
        
        # インデックス管理（エントリ数制限）
        self._manage_slot_entries_limit(slot_name)
        
        if self.debug:
            print(f"Slotエントリ追加: {slot_name} -> {len(text)}文字")
        
        return entry
    
    def _manage_slot_entries_limit(self, slot_name: str) -> None:
        """Slotエントリ数制限管理"""
        try:
            # 該当Slotのエントリ一覧取得
            entries = self.get_slot_entries_from_redis(slot_name)
            
            if len(entries) > self.max_slot_entries:
                # 古いエントリを削除
                excess_count = len(entries) - self.max_slot_entries
                oldest_entries = sorted(entries, key=lambda e: e.timestamp)[:excess_count]
                
                for old_entry in oldest_entries:
                    # エントリ削除
                    entry_key = f"{self.slot_key_prefix}{old_entry.entry_id}"
                    embedding_key = f"{self.slot_key_prefix}embedding_{old_entry.entry_id}"
                    
                    if hasattr(self.base_blackboard, 'redis_client'):
                        redis_client = self.base_blackboard.redis_client
                        redis_client.delete(
                            self.base_blackboard._make_key(entry_key),
                            self.base_blackboard._make_key(embedding_key)
                        )
                    
                if self.debug:
                    print(f"古いSlotエントリ削除: {excess_count}件")
                    
        except Exception as e:
            print(f"エントリ制限管理エラー: {e}")
    
    def get_slot_entries_from_redis(self, slot_name: Optional[str] = None) -> List[SlotEntry]:
        """RedisからSlotエントリを取得"""
        entries = []
        
        try:
            if hasattr(self.base_blackboard, 'redis_client'):
                # Redis版での取得
                redis_client = self.base_blackboard.redis_client
                pattern = f"{self.base_blackboard.key_prefix}{self.slot_key_prefix}*"
                keys = redis_client.keys(pattern)
                
                for key in keys:
                    if key.endswith('_embedding'):  # 埋め込みキーは除外
                        continue
                        
                    entry_data = self.base_blackboard.read(key[len(self.base_blackboard.key_prefix):])
                    if entry_data and isinstance(entry_data, dict):
                        # slot_nameフィルタリング
                        if slot_name is None or entry_data.get('slot_name') == slot_name:
                            entry = self._restore_slot_entry(entry_data)
                            if entry:
                                entries.append(entry)
            else:
                # ローカル版での取得
                all_data = self.base_blackboard.read_all()
                for key, value in all_data.items():
                    if key.startswith(self.slot_key_prefix) and not key.endswith('_embedding'):
                        if isinstance(value, dict):
                            if slot_name is None or value.get('slot_name') == slot_name:
                                entry = self._restore_slot_entry(value)
                                if entry:
                                    entries.append(entry)
                                    
        except Exception as e:
            print(f"Slotエントリ取得エラー: {e}")
        
        # タイムスタンプでソート
        return sorted(entries, key=lambda e: e.timestamp)
    
    def _restore_slot_entry(self, entry_data: Dict[str, Any]) -> Optional[SlotEntry]:
        """辞書データからSlotEntryを復元"""
        try:
            entry = SlotEntry(
                slot_name=entry_data['slot_name'],
                text=entry_data['text'],
                metadata=entry_data.get('metadata', {})
            )
            entry.timestamp = entry_data['timestamp']
            entry.entry_id = entry_data['entry_id']
            
            # 埋め込みベクトルの復元
            embedding_key = f"{self.slot_key_prefix}embedding_{entry.entry_id}"
            embedding_info = self.base_blackboard.read(embedding_key)
            
            if embedding_info and isinstance(embedding_info, dict):
                import base64
                import numpy as np
                
                # base64デコードしてNumPy配列に復元
                data_bytes = base64.b64decode(embedding_info['data'].encode('utf-8'))
                shape = tuple(embedding_info['shape'])
                dtype = np.dtype(embedding_info['dtype'])
                
                embedding = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                entry.embedding = embedding
            
            return entry
            
        except Exception as e:
            print(f"Slotエントリ復元エラー: {e}")
            return None
    
    def get_registered_slots(self) -> Set[str]:
        """登録済みSlot一覧を取得（Redis対応版）"""
        try:
            if hasattr(self.base_blackboard, 'redis_client'):
                slot_set = self.base_blackboard.redis_client.smembers(
                    self.base_blackboard._make_key(self.slots_set_key)
                )
                return {slot.decode('utf-8') if isinstance(slot, bytes) else slot for slot in slot_set}
            else:
                existing_slots = self.base_blackboard.read(self.slots_set_key) or []
                return set(existing_slots)
        except Exception as e:
            if self.debug:
                print(f"登録済みSlot取得エラー: {e}")
            return set()
    
    def get_slot_order(self) -> List[str]:
        """Slot順序を取得（Redis対応版）"""
        try:
            if hasattr(self.base_blackboard, 'redis_client'):
                slot_list = self.base_blackboard.redis_client.lrange(
                    self.base_blackboard._make_key(self.slot_order_key), 0, -1
                )
                return [slot.decode('utf-8') if isinstance(slot, bytes) else slot for slot in slot_list]
            else:
                existing_order = self.base_blackboard.read(self.slot_order_key) or []
                return existing_order
        except Exception as e:
            if self.debug:
                print(f"Slot順序取得エラー: {e}")
            return []
    
    def get_slot_entries(self, slot_name: Optional[str] = None) -> List[SlotEntry]:
        """Slot エントリを取得（Redis対応版）"""
        return self.get_slot_entries_from_redis(slot_name)
    
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
        """類似するエントリを検索（Redis対応版）"""
        similar_entries = []
        
        # Redisから全エントリを取得
        all_entries = self.get_slot_entries_from_redis()
        
        for entry in all_entries:
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
        """全Slotの要約を取得（Redis対応版）"""
        registered_slots = self.get_registered_slots()
        
        summary = {
            'registered_slots': list(registered_slots),
            'slot_order': self.get_slot_order(),
            'total_entries': len(self.get_slot_entries_from_redis()),
            'slot_summaries': {}
        }
        
        for slot_name in registered_slots:
            entries = self.get_slot_entries(slot_name)
            latest = entries[-1] if entries else None
            
            summary['slot_summaries'][slot_name] = {
                'entry_count': len(entries),
                'latest_text': latest.text if latest else None,
                'latest_timestamp': latest.timestamp if latest else None
            }
        
        return summary
    
    def clear_slot_entries(self, slot_name: Optional[str] = None) -> None:
        """Slot エントリをクリア（Redis対応版）"""
        try:
            if hasattr(self.base_blackboard, 'redis_client'):
                if slot_name is None:
                    # 全エントリクリア
                    pattern = f"{self.base_blackboard.key_prefix}{self.slot_key_prefix}*"
                    keys = self.base_blackboard.redis_client.keys(pattern)
                    if keys:
                        self.base_blackboard.redis_client.delete(*keys)
                    
                    self.base_blackboard.redis_client.delete(
                        self.base_blackboard._make_key(self.slots_set_key),
                        self.base_blackboard._make_key(self.slot_order_key)
                    )
                    if self.debug:
                        print("全Slotエントリをクリアしました")
                else:
                    # 特定Slotのエントリのみクリア
                    entries_to_keep = []
                    all_entries = self.get_slot_entries_from_redis()
                    for entry in all_entries:
                        if entry.slot_name != slot_name:
                            entries_to_keep.append(entry)
                    
                    # Redisを更新
                    pattern = f"{self.base_blackboard.key_prefix}{self.slot_key_prefix}*"
                    keys = self.base_blackboard.redis_client.keys(pattern)
                    if keys:
                        self.base_blackboard.redis_client.delete(*keys)
                    
                    for entry in entries_to_keep:
                        entry_key = f"{self.slot_key_prefix}{entry.entry_id}"
                        self.base_blackboard.write(entry_key, entry.to_dict())
                    
                    # Slot登録情報も削除
                    self.base_blackboard.redis_client.srem(
                        self.base_blackboard._make_key(self.slots_set_key), slot_name
                    )
                    self.base_blackboard.redis_client.lrem(
                        self.base_blackboard._make_key(self.slot_order_key), 0, slot_name
                    )
                    
                    cleared_count = len(all_entries) - len(entries_to_keep)
                    if self.debug:
                        print(f"Slot {slot_name} のエントリ{cleared_count}件をクリアしました")
            else:
                # ローカル版での処理
                if slot_name is None:
                    # 全データから slot_key_prefix で始まるものを削除
                    all_data = self.base_blackboard.read_all()
                    for key in list(all_data.keys()):
                        if key.startswith(self.slot_key_prefix):
                            self.base_blackboard.delete(key)
                    self.base_blackboard.delete(self.slots_set_key)
                    self.base_blackboard.delete(self.slot_order_key)
                    if self.debug:
                        print("全Slotエントリをクリアしました")
                        
        except Exception as e:
            if self.debug:
                print(f"Slotエントリクリアエラー: {e}")
    
    def clear_current_turn(self):
        """新しいターンのためにクリア（Slot情報は保持）（Redis対応版）"""
        # 親クラスのクリア実行
        self.base_blackboard.clear_current_turn()
        
        # Slot エントリもクリア（登録Slot情報は保持）
        try:
            all_entries = self.get_slot_entries_from_redis()
            cleared_count = len(all_entries)
            
            if hasattr(self.base_blackboard, 'redis_client'):
                # Redis版：slot_entry:で始まるキーを削除
                pattern = f"{self.base_blackboard.key_prefix}{self.slot_key_prefix}*"
                keys = self.base_blackboard.redis_client.keys(pattern)
                if keys:
                    self.base_blackboard.redis_client.delete(*keys)
            else:
                # ローカル版：slot_entry:で始まるキーを削除
                all_data = self.base_blackboard.read_all()
                for key in list(all_data.keys()):
                    if key.startswith(self.slot_key_prefix):
                        self.base_blackboard.delete(key)
            
            if self.debug:
                registered_count = len(self.get_registered_slots())
                print(f"ターンクリア: Slotエントリ{cleared_count}件クリア、登録Slot{registered_count}個保持")
        except Exception as e:
            if self.debug:
                print(f"Redisターンクリアエラー: {e}")
    
    def get_slot_debug_view(self) -> Dict[str, Any]:
        """Slot用のデバッグビューを取得（Redis対応版）"""
        debug_view = {
            'base_blackboard': self.get_debug_view(),
            'slot_info': self.get_all_slots_summary(),
            'recent_entries': []
        }
        
        # 最新5件のエントリを表示用に整形
        try:
            all_entries = self.get_slot_entries_from_redis()
            recent = all_entries[-5:] if len(all_entries) >= 5 else all_entries
            for entry in recent:
                debug_view['recent_entries'].append({
                    'slot_name': entry.slot_name,
                    'text_preview': entry.text[:100],
                    'timestamp': entry.timestamp,
                    'has_embedding': entry.embedding is not None
                })
        except Exception as e:
            if self.debug:
                print(f"デバッグビュー取得エラー: {e}")
            debug_view['recent_entries'] = []
        
        return debug_view
    
    def get_structured_context(self) -> Dict[str, Any]:
        """構造化されたコンテキストを取得（Redis対応版）"""
        # SlotBlackboardAdapterが利用可能な場合は構造化Blackboardから取得
        if hasattr(self, '_structured_adapter') and self._structured_adapter:
            return self._structured_adapter.structured_bb.get_synthesis_context()
        
        # フォールバック: 基本形式でコンテキストを構築
        try:
            all_entries = self.get_slot_entries_from_redis()
            recent_entries = sorted(all_entries, key=lambda x: x.timestamp, reverse=True)[:5]
        except Exception as e:
            if self.debug:
                print(f"構造化コンテキスト取得エラー: {e}")
            recent_entries = []
        
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
        """他Slotの意見を参照するためのコンテキストを取得（Redis対応版）"""
        try:
            all_entries = self.get_slot_entries_from_redis()
            other_entries = [entry for entry in all_entries if entry.slot_name != requesting_slot]
            
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
        except Exception as e:
            if self.debug:
                print(f"相互参照コンテキスト取得エラー: {e}")
            return {'other_opinions': [], 'conflict_indicators': [], 'consensus_indicators': []}
    
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
        """議論ラウンドの記録を追加（Redis対応版）"""
        try:
            existing_rounds = self.base_blackboard.read('discussion_rounds') or []
            new_round = {
                'round': round_number,
                'phase': phase,
                'timestamp': time.time(),
                'entries_count': len(self.get_slot_entries_from_redis())
            }
            existing_rounds.append(new_round)
            self.base_blackboard.write('discussion_rounds', existing_rounds)
            
            if self.debug:
                print(f"議論ラウンド追加: Round {round_number}, Phase {phase}")
        except Exception as e:
            if self.debug:
                print(f"議論ラウンド追加エラー: {e}")
    
    def get_discussion_history(self) -> Dict[str, Any]:
        """議論履歴を取得（Redis対応版）"""
        try:
            all_entries = self.get_slot_entries_from_redis()
            
            # 時系列順にエントリを整理
            timeline = []
            for entry in all_entries:
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
                'discussion_rounds': self.base_blackboard.read('discussion_rounds') or []
            }
        except Exception as e:
            if self.debug:
                print(f"議論履歴取得エラー: {e}")
            return {'timeline': [], 'phases': {}, 'total_entries': 0, 'discussion_rounds': []}
    
    def calculate_collaboration_metrics(self) -> Dict[str, float]:
        """協調度メトリクスを計算（Redis対応版）"""
        try:
            all_entries = self.get_slot_entries_from_redis()
            
            if len(all_entries) < 2:
                return {'collaboration_score': 0.0, 'diversity_score': 0.0, 'consensus_score': 0.0}
            
            # 多様性スコア（異なるSlotからの意見の数）
            unique_slots = len(set(entry.slot_name for entry in all_entries))
            diversity_score = min(unique_slots / 4.0, 1.0)  # 4つのSlotが理想
            
            # 相互参照スコア（他Slotを明示的に言及する頻度）
            reference_count = 0
            total_entries = len(all_entries)
            registered_slots = self.get_registered_slots()
            
            for entry in all_entries:
                other_slots = [slot for slot in registered_slots if slot != entry.slot_name]
                for other_slot in other_slots:
                    slot_keywords = ['Reformulator', 'Critic', 'Supporter', 'Synthesizer']
                    if any(keyword in entry.text for keyword in slot_keywords):
                        reference_count += 1
                        break
            
            reference_score = reference_count / total_entries if total_entries > 0 else 0.0
            
            # コンセンサススコア（合意指標の検出）
            consensus_indicators = self._detect_consensus(all_entries)
            conflict_indicators = self._detect_conflicts(all_entries)
            
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
        except Exception as e:
            if self.debug:
                print(f"協調度メトリクス計算エラー: {e}")
            return {'collaboration_score': 0.0, 'diversity_score': 0.0, 'consensus_score': 0.0}
    
    # ===== 基本Blackboardメソッドの委譲 =====
    
    def write(self, key: str, data: Any) -> None:
        """基本blackboardへの書き込み委譲"""
        return self.base_blackboard.write(key, data)
    
    def read(self, key: str) -> Any:
        """基本blackboardからの読み取り委譲"""
        return self.base_blackboard.read(key)
    
    def read_all(self) -> Dict[str, Any]:
        """基本blackboardの全データ読み取り委譲"""
        return self.base_blackboard.read_all()
    
    def delete(self, key: str) -> None:
        """基本blackboardからの削除委譲"""
        return self.base_blackboard.delete(key)
    
    def exists(self, key: str) -> bool:
        """基本blackboardでの存在確認委譲"""
        return self.base_blackboard.exists(key)
    
    def clear(self) -> None:
        """基本blackboardのクリア委譲"""
        return self.base_blackboard.clear()
    
    def get_debug_view(self) -> Dict[str, Any]:
        """基本blackboardのデバッグビュー委譲"""
        return self.base_blackboard.get_debug_view()
    
    def health_check(self) -> bool:
        """基本blackboardのヘルスチェック委譲"""
        if hasattr(self.base_blackboard, 'health_check'):
            return self.base_blackboard.health_check()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """基本blackboardの統計情報委譲"""
        if hasattr(self.base_blackboard, 'get_stats'):
            return self.base_blackboard.get_stats()
        return {}
