#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Boids アルゴリズム実装
~~~~~~~~~~~~~~~~~~~
MurmurNet Slot出力に対するBoids則（分離・整列・凝集）の適用
意味的ベクトル空間上でのSwarm Intelligence実現

機能:
- VectorSpace: 出力の意味的ベクトル空間管理
- BoidsController: 3つのBoids則の実装
- SlotBoid: Slot出力のBoid表現
- BoidsEvaluator: Boids適用効果の評価

作者: Yuhi Sonoki
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlotBoid:
    """
    Slot出力のBoid表現
    
    Boidsアルゴリズムにおける「個体」に相当
    位置=出力の意味ベクトル、速度=次の出力意図
    """
    slot_name: str
    text: str
    position: np.ndarray  # 現在の意味ベクトル
    velocity: np.ndarray  # 出力意図ベクトル
    timestamp: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.velocity is None:
            # 初期速度は位置ベクトルと同じ方向に設定
            self.velocity = self.position * 0.1
    
    @property
    def dimension(self) -> int:
        """ベクトル次元数"""
        return len(self.position) if self.position is not None else 0
    
    def distance_to(self, other: 'SlotBoid') -> float:
        """他のBoidとの距離（コサイン距離）"""
        if self.position is None or other.position is None:
            return float('inf')
        
        similarity = cosine_similarity(
            self.position.reshape(1, -1),
            other.position.reshape(1, -1)
        )[0][0]
        
        # コサイン距離 = 1 - コサイン類似度
        return 1.0 - similarity
    
    def similarity_to(self, other: 'SlotBoid') -> float:
        """他のBoidとの類似度"""
        return 1.0 - self.distance_to(other)


class VectorSpace:
    """
    出力の意味的ベクトル空間管理
    
    Slot出力をベクトル化し、空間内での位置・関係を管理
    """
    
    def __init__(self, embedder, dimension: int = 384):
        self.embedder = embedder
        self.dimension = dimension
        self.boids: List[SlotBoid] = []
        self.lock = threading.RLock()
        
        # 空間統計
        self.stats = {
            'total_boids': 0,
            'average_distance': 0.0,
            'max_similarity': 0.0,
            'min_similarity': 1.0,
            'cluster_count': 0
        }
    
    def add_slot_output(self, slot_name: str, text: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> SlotBoid:
        """Slot出力をBoidとして空間に追加"""
        try:
            # テキストをベクトル化
            if hasattr(self.embedder, 'embed_text'):
                embedding = self.embedder.embed_text(text)
            else:
                # フォールバック：簡易ベクトル化
                embedding = self._fallback_embedding(text)
            
            # Boid作成
            boid = SlotBoid(
                slot_name=slot_name,
                text=text,
                position=embedding,
                velocity=None,  # __post_init__で自動設定
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            with self.lock:
                self.boids.append(boid)
                self._update_stats()
            
            logger.debug(f"VectorSpace: {slot_name} Boidを追加 (次元: {boid.dimension})")
            return boid
            
        except Exception as e:
            logger.error(f"VectorSpace: Boid追加エラー: {e}")
            # エラー時はゼロベクトルでBoidを作成
            zero_embedding = np.zeros(self.dimension)
            return SlotBoid(
                slot_name=slot_name,
                text=text,
                position=zero_embedding,
                velocity=zero_embedding * 0.1,
                timestamp=time.time(),
                metadata=metadata or {'error': str(e)}
            )
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """埋め込み失敗時のフォールバック"""
        # 簡易的なハッシュベースベクトル化
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # ハッシュを数値ベクトルに変換
        vector = np.array([
            ord(char) / 255.0 for char in text_hash[:self.dimension]
        ] + [0.0] * max(0, self.dimension - len(text_hash)))
        
        # 正規化
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    def get_neighbors(self, target_boid: SlotBoid, radius: float = 0.5, 
                     max_neighbors: int = 5) -> List[SlotBoid]:
        """指定Boidの近隣Boidを取得"""
        neighbors = []
        
        with self.lock:
            for boid in self.boids:
                if boid.slot_name != target_boid.slot_name:  # 自分以外
                    distance = target_boid.distance_to(boid)
                    if distance <= radius:
                        neighbors.append((boid, distance))
        
        # 距離でソートして上位N件を返す
        neighbors.sort(key=lambda x: x[1])
        return [boid for boid, _ in neighbors[:max_neighbors]]
    
    def get_all_boids(self) -> List[SlotBoid]:
        """全Boidのコピーを取得"""
        with self.lock:
            return self.boids.copy()
    
    def _update_stats(self):
        """空間統計を更新"""
        if len(self.boids) < 2:
            return
        
        distances = []
        similarities = []
        
        for i, boid1 in enumerate(self.boids):
            for boid2 in self.boids[i+1:]:
                distance = boid1.distance_to(boid2)
                similarity = boid1.similarity_to(boid2)
                
                distances.append(distance)
                similarities.append(similarity)
        
        if distances:
            self.stats.update({
                'total_boids': len(self.boids),
                'average_distance': np.mean(distances),
                'max_similarity': max(similarities),
                'min_similarity': min(similarities)
            })
    
    def clear(self):
        """空間をクリア"""
        with self.lock:
            self.boids.clear()
            self.stats = {
                'total_boids': 0,
                'average_distance': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 1.0,
                'cluster_count': 0
            }


class BoidsController:
    """
    Boids則の実装コントローラー
    
    Separation（分離）、Alignment（整列）、Cohesion（凝集）の
    3つのルールを意味ベクトル空間に適用
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Boidsパラメータ
        self.separation_radius = config.get('separation_radius', 0.8)
        self.alignment_radius = config.get('alignment_radius', 0.6)
        self.cohesion_radius = config.get('cohesion_radius', 0.7)
        
        # 重み係数
        self.separation_weight = config.get('separation_weight', 1.5)
        self.alignment_weight = config.get('alignment_weight', 1.0)
        self.cohesion_weight = config.get('cohesion_weight', 1.2)
        
        # 制御パラメータ
        self.max_velocity = config.get('max_velocity', 0.3)
        self.min_distance = config.get('min_distance', 0.1)
        
        logger.info(f"BoidsController初期化: sep={self.separation_radius}, "
                   f"align={self.alignment_radius}, coh={self.cohesion_radius}")
    
    def apply_boids_rules(self, vector_space: VectorSpace) -> List[SlotBoid]:
        """
        全Boidに対してBoids則を適用し、更新されたBoidリストを返す
        """
        boids = vector_space.get_all_boids()
        if len(boids) < 2:
            return boids
        
        updated_boids = []
        
        for boid in boids:
            # 各ルールを適用
            separation_force = self._separation(boid, boids)
            alignment_force = self._alignment(boid, boids)
            cohesion_force = self._cohesion(boid, boids)
            
            # 合成力を計算
            total_force = (
                separation_force * self.separation_weight +
                alignment_force * self.alignment_weight +
                cohesion_force * self.cohesion_weight
            )
            
            # 速度更新
            new_velocity = boid.velocity + total_force
            
            # 最大速度制限
            velocity_magnitude = np.linalg.norm(new_velocity)
            if velocity_magnitude > self.max_velocity:
                new_velocity = new_velocity / velocity_magnitude * self.max_velocity
            
            # 位置更新
            new_position = boid.position + new_velocity
            
            # 新しいBoidを作成
            updated_boid = SlotBoid(
                slot_name=boid.slot_name,
                text=boid.text,
                position=new_position,
                velocity=new_velocity,
                timestamp=boid.timestamp,
                metadata={
                    **boid.metadata,
                    'boids_applied': True,
                    'forces': {
                        'separation': np.linalg.norm(separation_force),
                        'alignment': np.linalg.norm(alignment_force),
                        'cohesion': np.linalg.norm(cohesion_force)
                    }
                }
            )
            
            updated_boids.append(updated_boid)
        
        logger.debug(f"Boids則適用完了: {len(updated_boids)}個のBoidを更新")
        return updated_boids
    
    def _separation(self, boid: SlotBoid, all_boids: List[SlotBoid]) -> np.ndarray:
        """分離力の計算 - 近すぎるBoidとの距離を取る"""
        separation_force = np.zeros_like(boid.position)
        count = 0
        
        for other in all_boids:
            if other.slot_name == boid.slot_name:
                continue
            
            distance = boid.distance_to(other)
            
            if distance < self.separation_radius and distance > self.min_distance:
                # 距離が近いほど強い反発力
                direction = boid.position - other.position
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    # 正規化された方向ベクトル × 距離の逆数
                    force = direction / direction_norm / (distance + 0.01)
                    separation_force += force
                    count += 1
        
        if count > 0:
            separation_force /= count
        
        return separation_force
    
    def _alignment(self, boid: SlotBoid, all_boids: List[SlotBoid]) -> np.ndarray:
        """整列力の計算 - 近隣Boidの速度方向に合わせる"""
        avg_velocity = np.zeros_like(boid.velocity)
        count = 0
        
        for other in all_boids:
            if other.slot_name == boid.slot_name:
                continue
            
            distance = boid.distance_to(other)
            
            if distance < self.alignment_radius:
                avg_velocity += other.velocity
                count += 1
        
        if count > 0:
            avg_velocity /= count
            # 平均速度と現在速度の差を整列力とする
            alignment_force = avg_velocity - boid.velocity
        else:
            alignment_force = np.zeros_like(boid.velocity)
        
        return alignment_force
    
    def _cohesion(self, boid: SlotBoid, all_boids: List[SlotBoid]) -> np.ndarray:
        """凝集力の計算 - 近隣Boidの重心に向かう"""
        center_of_mass = np.zeros_like(boid.position)
        count = 0
        
        for other in all_boids:
            if other.slot_name == boid.slot_name:
                continue
            
            distance = boid.distance_to(other)
            
            if distance < self.cohesion_radius:
                center_of_mass += other.position
                count += 1
        
        if count > 0:
            center_of_mass /= count
            # 重心に向かう方向を凝集力とする
            cohesion_force = center_of_mass - boid.position
        else:
            cohesion_force = np.zeros_like(boid.position)
        
        return cohesion_force
    
    def evaluate_swarm_coherence(self, boids: List[SlotBoid]) -> Dict[str, float]:
        """群れの結束度を評価"""
        if len(boids) < 2:
            return {'coherence': 0.0, 'diversity': 0.0, 'convergence': 0.0}
        
        # 平均距離（多様性の指標）
        total_distance = 0
        distance_count = 0
        
        # 速度の一致度（収束の指標）
        velocities = [boid.velocity for boid in boids]
        avg_velocity = np.mean(velocities, axis=0)
        velocity_variance = np.mean([
            np.linalg.norm(v - avg_velocity) for v in velocities
        ])
        
        for i, boid1 in enumerate(boids):
            for boid2 in boids[i+1:]:
                total_distance += boid1.distance_to(boid2)
                distance_count += 1
        
        avg_distance = total_distance / distance_count if distance_count > 0 else 0
        
        return {
            'coherence': 1.0 / (1.0 + avg_distance),  # 距離が近いほど高い結束度
            'diversity': min(avg_distance, 1.0),      # 適度な距離は多様性
            'convergence': 1.0 / (1.0 + velocity_variance)  # 速度が揃うほど高い収束度
        }
