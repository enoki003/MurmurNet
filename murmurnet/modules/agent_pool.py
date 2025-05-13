#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Pool モジュール
~~~~~~~~~~~~~~~~~
Boids型自己増殖エージェントプールの管理
- エージェントの作成と削除
- Boids的なルールに基づくエージェントの行動制御
- エージェント間の協調と反発の管理

作者: Yuhi Sonoki
"""

import uuid
import time
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
import json
import logging
from collections import deque

class Agent:
    """個々のエージェントを表現するクラス"""
    
    def __init__(self, agent_id: str, role: str, config: Dict[str, Any],
                personality: Dict[str, Any] = None):
        """
        エージェントの初期化
        
        引数:
            agent_id: エージェントのID
            role: エージェントの役割
            config: 設定辞書
            personality: パーソナリティ設定
        """
        self.id = agent_id
        self.role = role
        self.config = config
        self.created_at = time.time()
        self.last_active = time.time()
        
        # パーソナリティ特性（なければデフォルト値を使用）
        self.personality = personality or {
            'openness': random.uniform(0.3, 0.8),       # 開放性
            'conscientiousness': random.uniform(0.4, 0.9), # 誠実性
            'extraversion': random.uniform(0.2, 0.8),   # 外向性
            'agreeableness': random.uniform(0.3, 0.7),  # 協調性
            'neuroticism': random.uniform(0.1, 0.6)     # 神経症的傾向
        }
        
        # Boidsパラメータ
        self.position = np.random.rand(config.get('vector_dim', 384))  # 意見空間内の位置
        self.velocity = np.zeros(config.get('vector_dim', 384))  # 意見空間内の速度
        
        # エージェント状態
        self.messages = []  # 発言履歴
        self.contribution_score = 0.0  # 貢献度スコア
        self.active = True  # アクティブかどうか
        self.lifespan = config.get('default_agent_lifespan', 10)  # 寿命（ターン数）
        self.generation = 0  # 世代（親から派生した場合に増加）
        self.parent_id = None  # 親エージェントのID（存在する場合）
        
        # 活性化履歴
        self.activation_history = deque(maxlen=10)  # 最近のアクティベーション履歴
    
    def update_activity(self) -> None:
        """エージェントの活動時間を更新"""
        self.last_active = time.time()
    
    def add_message(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """
        エージェントの発言を記録
        
        引数:
            message: 発言テキスト
            metadata: 発言に関するメタデータ
        """
        if not metadata:
            metadata = {}
            
        self.messages.append({
            'text': message,
            'timestamp': time.time(),
            'metadata': metadata
        })
        self.update_activity()
    
    def add_contribution(self, score: float) -> None:
        """
        エージェントの貢献度を加算
        
        引数:
            score: 追加する貢献度スコア
        """
        self.contribution_score += score
    
    def update_position(self, new_position: np.ndarray) -> None:
        """
        エージェントの意見空間上の位置を更新
        
        引数:
            new_position: 新しい位置ベクトル
        """
        self.position = new_position
        self.update_activity()
    
    def decrement_lifespan(self) -> int:
        """
        エージェントの寿命を1減らし、残りの寿命を返す
        
        戻り値:
            int: 残りの寿命
        """
        self.lifespan -= 1
        if self.lifespan <= 0:
            self.active = False
        return self.lifespan
    
    def reset_lifespan(self) -> None:
        """エージェントの寿命をリセット"""
        self.lifespan = self.config.get('default_agent_lifespan', 10)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        エージェントの情報を辞書として返す
        
        戻り値:
            Dict[str, Any]: エージェント情報
        """
        return {
            'id': self.id,
            'role': self.role,
            'created_at': self.created_at,
            'last_active': self.last_active,
            'personality': self.personality,
            'contribution_score': self.contribution_score,
            'active': self.active,
            'lifespan': self.lifespan,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'message_count': len(self.messages)
        }


class AgentPool:
    """
    Boids型自己増殖エージェントプールを管理するクラス
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        AgentPoolの初期化
        
        引数:
            config: 設定辞書
            logger: ロガーインスタンス
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.debug = config.get('debug', False)
        
        # エージェント管理
        self.agents: Dict[str, Agent] = {}  # ID → エージェントのマッピング
        self.active_agents: List[str] = []  # アクティブなエージェントのIDリスト
        self.inactive_agents: List[str] = []  # 非アクティブなエージェントのIDリスト
        
        # パフォーマンス設定
        self.min_agents = config.get('min_agents', 3)  # 最小エージェント数
        self.max_agents = config.get('max_agents', 12)  # 最大エージェント数
        self.threshold_remove = config.get('threshold_remove', -5.0)  # 削除閾値
        self.threshold_reproduce = config.get('threshold_reproduce', 8.0)  # 複製閾値
        
        # ターン管理
        self.current_turn = 0
        self.turn_history = []  # ターン情報の履歴
        
        # Boidsルール重み
        self.boids_weights = {
            'cohesion': config.get('weight_cohesion', 0.3),  # 結合ルール
            'separation': config.get('weight_separation', 0.2),  # 分離ルール
            'alignment': config.get('weight_alignment', 0.4),  # 整列ルール
            'innovation': config.get('weight_innovation', 0.1)  # 革新性ルール
        }
        
        if self.debug:
            self.logger.info("AgentPool: 初期化完了")
        
        # 初期エージェント作成
        self._create_initial_agents()
    
    def _create_initial_agents(self) -> None:
        """初期エージェントを作成"""
        initial_count = self.config.get('initial_agents', 3)
        initial_roles = ["systemアナリスト", "質問者", "議論調整者"]
        
        for i in range(min(initial_count, len(initial_roles))):
            agent_id = self._generate_agent_id()
            role = initial_roles[i]
            
            # パーソナリティ設定
            personality = self._generate_personality(role)
            
            # エージェント作成
            agent = Agent(agent_id, role, self.config, personality)
            
            # プールに追加
            self.agents[agent_id] = agent
            self.active_agents.append(agent_id)
            
            if self.debug:
                self.logger.info(f"初期エージェント作成: ID={agent_id}, 役割={role}")
    
    def _generate_agent_id(self) -> str:
        """
        一意のエージェントIDを生成
        
        戻り値:
            str: エージェントID
        """
        return f"agent_{uuid.uuid4().hex[:8]}"
    
    def _generate_personality(self, role: str = None) -> Dict[str, float]:
        """
        エージェントのパーソナリティを生成
        
        引数:
            role: エージェントの役割
            
        戻り値:
            Dict[str, float]: パーソナリティ特性
        """
        # 基本的な特性（ランダム生成）
        personality = {
            'openness': random.uniform(0.3, 0.8),       # 開放性
            'conscientiousness': random.uniform(0.4, 0.9), # 誠実性
            'extraversion': random.uniform(0.2, 0.8),   # 外向性
            'agreeableness': random.uniform(0.3, 0.7),  # 協調性
            'neuroticism': random.uniform(0.1, 0.6)     # 神経症的傾向
        }
        
        # 役割に応じた特性の調整
        if role:
            if "アナリスト" in role or "分析" in role:
                personality['openness'] = min(personality['openness'] * 1.3, 1.0)
                personality['conscientiousness'] = min(personality['conscientiousness'] * 1.2, 1.0)
                
            elif "質問" in role or "クエリ" in role:
                personality['extraversion'] = min(personality['extraversion'] * 1.2, 1.0)
                personality['openness'] = min(personality['openness'] * 1.1, 1.0)
                
            elif "調整" in role or "ファシリテータ" in role:
                personality['agreeableness'] = min(personality['agreeableness'] * 1.3, 1.0)
                personality['extraversion'] = min(personality['extraversion'] * 1.1, 1.0)
                
            elif "批評" in role or "批判" in role:
                personality['neuroticism'] = min(personality['neuroticism'] * 1.2, 1.0)
                personality['agreeableness'] *= 0.8
        
        return personality
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        IDからエージェントを取得
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            Optional[Agent]: エージェントオブジェクト（存在しない場合はNone）
        """
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        全エージェントの情報を取得
        
        戻り値:
            List[Dict[str, Any]]: エージェント情報のリスト
        """
        return [agent.to_dict() for agent in self.agents.values()]
    
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """
        アクティブなエージェントの情報を取得
        
        戻り値:
            List[Dict[str, Any]]: アクティブなエージェント情報のリスト
        """
        return [self.agents[agent_id].to_dict() for agent_id in self.active_agents]
    
    def create_agent(self, role: str, personality: Dict[str, Any] = None,
                   parent_id: str = None) -> str:
        """
        新しいエージェントを作成
        
        引数:
            role: エージェントの役割
            personality: パーソナリティ設定
            parent_id: 親エージェントのID
            
        戻り値:
            str: 作成されたエージェントのID
        """
        # すでに最大エージェント数に達している場合
        if len(self.active_agents) >= self.max_agents:
            if self.debug:
                self.logger.warning(f"エージェント作成失敗: 最大数({self.max_agents})に達しています")
            return None
            
        # IDと役割の生成
        agent_id = self._generate_agent_id()
        
        # パーソナリティが指定されていない場合は生成
        if not personality:
            personality = self._generate_personality(role)
        
        # エージェント作成
        agent = Agent(agent_id, role, self.config, personality)
        
        # 親エージェントが指定されている場合
        if parent_id and parent_id in self.agents:
            parent = self.agents[parent_id]
            agent.generation = parent.generation + 1
            agent.parent_id = parent_id
            
            # パーソナリティに若干のバリエーションを加える
            for key in agent.personality:
                variation = random.uniform(-0.1, 0.1)
                agent.personality[key] = max(0.0, min(1.0, agent.personality[key] + variation))
        
        # プールに追加
        self.agents[agent_id] = agent
        self.active_agents.append(agent_id)
        
        if self.debug:
            parent_info = f", 親={parent_id}" if parent_id else ""
            self.logger.info(f"エージェント作成: ID={agent_id}, 役割={role}{parent_info}")
        
        return agent_id
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        エージェントを非アクティブに設定
        
        引数:
            agent_id: 削除するエージェントのID
            
        戻り値:
            bool: 削除に成功したかどうか
        """
        if agent_id not in self.agents:
            return False
        
        # エージェントを非アクティブに
        agent = self.agents[agent_id]
        agent.active = False
        
        # アクティブリストから削除
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
        
        # 非アクティブリストに追加
        if agent_id not in self.inactive_agents:
            self.inactive_agents.append(agent_id)
        
        if self.debug:
            self.logger.info(f"エージェント削除: ID={agent_id}, 役割={agent.role}")
        
        return True
    
    def next_turn(self) -> Dict[str, Any]:
        """
        次のターンに進む
        - エージェントの寿命を減らす
        - 貢献度に基づいてエージェントの追加・削除を検討
        - Boidsルールを適用して意見空間上の位置を更新
        
        戻り値:
            Dict[str, Any]: ターンの要約情報
        """
        self.current_turn += 1
        turn_start = time.time()
        
        # ターン情報
        turn_info = {
            'turn': self.current_turn,
            'timestamp': turn_start,
            'active_agents': len(self.active_agents),
            'inactive_agents': len(self.inactive_agents),
            'added_agents': [],
            'removed_agents': []
        }
        
        # 各エージェントの寿命を減らす
        expired_agents = []
        for agent_id in self.active_agents.copy():
            agent = self.agents[agent_id]
            remaining = agent.decrement_lifespan()
            
            if remaining <= 0:
                expired_agents.append(agent_id)
        
        # 貢献度に基づく評価
        self._evaluate_agents(turn_info)
        
        # 期限切れエージェントの処理
        for agent_id in expired_agents:
            if agent_id in self.active_agents:  # すでに削除済みでないことを確認
                self.remove_agent(agent_id)
                turn_info['removed_agents'].append({
                    'id': agent_id,
                    'reason': 'lifespan_expired',
                    'role': self.agents[agent_id].role
                })
        
        # Boidsルールの適用
        self._apply_boids_rules()
        
        # 最小エージェント数を確保
        while len(self.active_agents) < self.min_agents:
            role = random.choice(["情報分析者", "批評者", "質問者", "調整役"])
            new_id = self.create_agent(role)
            if new_id:
                turn_info['added_agents'].append({
                    'id': new_id,
                    'reason': 'min_agents',
                    'role': role
                })
            else:
                break
        
        # ターン情報を履歴に追加
        self.turn_history.append(turn_info)
        
        if self.debug:
            self.logger.info(f"ターン {self.current_turn} 完了: "
                          f"活性={len(self.active_agents)}エージェント, "
                          f"追加={len(turn_info['added_agents'])}, "
                          f"削除={len(turn_info['removed_agents'])}")
        
        return turn_info
    
    def _evaluate_agents(self, turn_info: Dict[str, Any]) -> None:
        """
        エージェントを評価し、追加・削除を行う
        
        引数:
            turn_info: 現在のターン情報
        """
        # スコアでソート
        scored_agents = [(agent_id, self.agents[agent_id].contribution_score)
                         for agent_id in self.active_agents]
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # 最良のエージェントを増殖
        best_agents = [agent_id for agent_id, score in scored_agents 
                      if score >= self.threshold_reproduce]
        
        for agent_id in best_agents:
            parent = self.agents[agent_id]
            
            # エージェント数上限チェック
            if len(self.active_agents) >= self.max_agents:
                break
                
            # 親の貢献度を少し下げる
            parent.contribution_score *= 0.8
            
            # 新しいエージェントの役割を選択
            role = parent.role
            if random.random() < 0.3:  # 30%の確率で新しい役割
                possible_roles = ["質問者", "総合分析", "要約担当", "批評者", "統合者"]
                role = random.choice(possible_roles)
            
            # 新しいエージェントを作成
            new_id = self.create_agent(role, parent.personality.copy(), agent_id)
            if new_id:
                turn_info['added_agents'].append({
                    'id': new_id,
                    'reason': 'high_contribution',
                    'parent_id': agent_id,
                    'role': role
                })
        
        # 最悪のエージェントを削除（ただし、最小エージェント数は確保）
        worst_agents = [agent_id for agent_id, score in reversed(scored_agents) 
                       if score <= self.threshold_remove]
                       
        for agent_id in worst_agents:
            # 最小エージェント数チェック
            if len(self.active_agents) <= self.min_agents:
                break
                
            if self.remove_agent(agent_id):
                turn_info['removed_agents'].append({
                    'id': agent_id,
                    'reason': 'low_contribution',
                    'role': self.agents[agent_id].role,
                    'score': self.agents[agent_id].contribution_score
                })
    
    def _apply_boids_rules(self) -> None:
        """Boidsルールを適用してエージェントの意見空間上の位置を更新"""
        if len(self.active_agents) <= 1:
            return
            
        # 各エージェントの現在位置をベクトルとして収集
        positions = []
        velocities = []
        agent_refs = []
        
        for agent_id in self.active_agents:
            agent = self.agents[agent_id]
            positions.append(agent.position)
            velocities.append(agent.velocity)
            agent_refs.append(agent)
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # 各エージェントに対してBoidsルールを適用
        for i, agent in enumerate(agent_refs):
            # 現在の位置と速度
            current_pos = positions[i]
            current_vel = velocities[i]
            
            # ルール1: 結合 - 群れの中心に向かう
            center_of_mass = np.mean(positions, axis=0)
            cohesion = (center_of_mass - current_pos) * self.boids_weights['cohesion']
            
            # ルール2: 分離 - 近すぎるエージェントから離れる
            separation = np.zeros_like(current_pos)
            for j, other_pos in enumerate(positions):
                if i != j:
                    diff = current_pos - other_pos
                    distance = np.linalg.norm(diff)
                    if distance < 0.1:  # 近接閾値
                        # 距離に反比例する分離ベクトル
                        sep_factor = min(1.0 / max(0.001, distance), 10)
                        separation += diff * sep_factor
            separation *= self.boids_weights['separation']
            
            # ルール3: 整列 - 他のエージェントと同じ方向に進む
            avg_velocity = np.mean(velocities, axis=0)
            alignment = avg_velocity * self.boids_weights['alignment']
            
            # ルール4: 革新性 - 少しランダムな方向に進む
            innovation = np.random.rand(len(current_pos)) * 2 - 1  # -1〜1のランダムベクトル
            innovation *= self.boids_weights['innovation']
            
            # 速度更新
            new_velocity = current_vel * 0.5 + cohesion + separation + alignment + innovation
            
            # 速度の大きさに制限
            speed = np.linalg.norm(new_velocity)
            max_speed = 0.2
            if speed > max_speed:
                new_velocity = new_velocity * (max_speed / speed)
            
            # 位置更新
            new_position = current_pos + new_velocity
            
            # 範囲を0〜1に制限
            new_position = np.clip(new_position, 0, 1)
            
            # エージェントのデータを更新
            agent.velocity = new_velocity
            agent.position = new_position
    
    def assign_contributions(self, contributions: Dict[str, float]) -> None:
        """
        各エージェントに貢献度スコアを割り当てる
        
        引数:
            contributions: エージェントIDとスコアのマッピング
        """
        for agent_id, score in contributions.items():
            if agent_id in self.agents:
                self.agents[agent_id].add_contribution(score)
                
                if self.debug and abs(score) > 0.5:
                    direction = "ポジティブ" if score > 0 else "ネガティブ"
                    self.logger.debug(f"貢献度割当: ID={agent_id}, {direction}={score:.2f}")
    
    def add_agent_message(self, agent_id: str, message: str, metadata: Dict[str, Any] = None) -> bool:
        """
        エージェントにメッセージを追加
        
        引数:
            agent_id: エージェントID
            message: メッセージテキスト
            metadata: メタデータ
            
        戻り値:
            bool: 成功したかどうか
        """
        if agent_id not in self.agents or agent_id not in self.active_agents:
            return False
            
        agent = self.agents[agent_id]
        agent.add_message(message, metadata)
        
        # 貢献度を少し増やす（メッセージを出したことに対する報酬）
        agent.add_contribution(0.1)
        
        return True
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """
        エージェントプールのメトリクスを計算
        
        戻り値:
            Dict[str, Any]: メトリクス情報
        """
        if not self.active_agents:
            return {
                'agent_count': 0,
                'avg_contribution': 0.0,
                'avg_lifespan': 0.0,
                'diversity': 0.0,
                'generation_stats': {}
            }
            
        # 基本メトリクスを収集
        contributions = []
        lifespans = []
        generations = []
        
        for agent_id in self.active_agents:
            agent = self.agents[agent_id]
            contributions.append(agent.contribution_score)
            lifespans.append(agent.lifespan)
            generations.append(agent.generation)
        
        # 世代別統計
        gen_stats = {}
        for gen in set(generations):
            count = generations.count(gen)
            gen_stats[str(gen)] = {
                'count': count,
                'percentage': count / len(self.active_agents)
            }
        
        # 多様性指標（パーソナリティの標準偏差の平均）
        personality_vectors = []
        for agent_id in self.active_agents:
            agent = self.agents[agent_id]
            p_vector = [
                agent.personality.get('openness', 0.5),
                agent.personality.get('conscientiousness', 0.5),
                agent.personality.get('extraversion', 0.5),
                agent.personality.get('agreeableness', 0.5),
                agent.personality.get('neuroticism', 0.5)
            ]
            personality_vectors.append(p_vector)
            
        personality_array = np.array(personality_vectors)
        personality_diversity = float(np.mean(np.std(personality_array, axis=0)))
        
        return {
            'agent_count': len(self.active_agents),
            'avg_contribution': np.mean(contributions) if contributions else 0.0,
            'max_contribution': np.max(contributions) if contributions else 0.0,
            'min_contribution': np.min(contributions) if contributions else 0.0,
            'avg_lifespan': np.mean(lifespans) if lifespans else 0.0,
            'avg_generation': np.mean(generations) if generations else 0.0,
            'max_generation': np.max(generations) if generations else 0.0,
            'personality_diversity': personality_diversity,
            'generation_stats': gen_stats
        }
    
    def export_to_json(self, filepath: str) -> bool:
        """
        エージェントプールの状態をJSONファイルにエクスポート
        
        引数:
            filepath: ファイルパス
            
        戻り値:
            bool: エクスポートに成功したかどうか
        """
        try:
            export_data = {
                'turn': self.current_turn,
                'timestamp': time.time(),
                'active_agents': len(self.active_agents),
                'inactive_agents': len(self.inactive_agents),
                'agents': [],
                'metrics': self.get_agent_metrics(),
                'config': {k: v for k, v in self.config.items() if isinstance(v, (int, float, str, bool, list, dict))}
            }
            
            # エージェント情報を追加
            for agent_id, agent in self.agents.items():
                agent_data = agent.to_dict()
                
                # 追加情報
                agent_data['messages'] = [
                    {'text': m['text'], 'timestamp': m['timestamp']} 
                    for m in agent.messages[-5:]  # 最新の5メッセージのみ
                ]
                
                # numpy配列をリストに変換
                agent_data['position'] = agent.position.tolist()
                
                export_data['agents'].append(agent_data)
            
            # JSONファイルに保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            if self.debug:
                self.logger.info(f"エージェントプール状態を {filepath} にエクスポート")
                
            return True
            
        except Exception as e:
            if self.debug:
                self.logger.error(f"エクスポートエラー: {str(e)}")
            return False

    def _process_opinion_vector(self, agent_id: int, opinion_text: str) -> None:
        """
        エージェントの意見をベクトル化し、意見空間に追加（内部メソッド）
        
        引数:
            agent_id: エージェントID
            opinion_text: エージェントの意見テキスト
        """
        if not self.use_self_replication or not opinion_text:
            return
            
        try:
            # エージェント情報の取得
            agent_info = None
            if 0 <= agent_id < len(self.agents):
                agent_info = self.agents[agent_id]
            else:
                return
                
            # 意見のベクトル化
            opinion_vector = self.vectorizer.vectorize_text(opinion_text)
            
            if opinion_vector is not None:
                # エージェント情報をメタデータとして付与
                metadata = {
                    'agent_id': agent_id,
                    'role': agent_info['role'],
                    'principle': agent_info.get('principle', 'unknown'),
                    'strategy': agent_info.get('strategy', 'default'),
                    'text': opinion_text[:100]  # テキスト冒頭を保存
                }
                
                # 意見空間に追加
                self.opinion_space.add_opinion_vector(opinion_vector, metadata)
                
                if self.debug:
                    print(f"意見ベクトル追加: エージェント{agent_id}({agent_info['role']})")
                    
        except Exception as e:
            if self.debug:
                print(f"意見ベクトル処理エラー: {str(e)}")
                import traceback
                traceback.print_exc()

class AgentPoolManager:
    """
    エージェントプールを管理するクラス
    """
    
    def __init__(self, config: Dict[str, Any], llm, blackboard=None, opinion_space_manager=None):
        """
        初期化
        
        引数:
            config: 設定辞書
            llm: 言語モデルインスタンス
            blackboard: 黒板インスタンス
            opinion_space_manager: 意見空間マネージャ（省略可能）
        """
        self.config = config or {}
        self.llm = llm
        self.blackboard = blackboard
        self.debug = self.config.get('debug', False)
        
        # Boids型自己増殖機能のための設定
        self.use_self_replication = self.config.get('use_self_replication', False)
        self.opinion_space = opinion_space_manager
        self.vectorizer = None
        
        # エージェント設定
        self.num_agents = self.config.get('num_agents', 3)
        self.current_turn = 0
        self.agent_roles = []
        self.agents = []  # Boids型自己増殖機能用エージェントリスト
        
        # エージェント調整機能設定
        self.enable_agent_growth = self.config.get('enable_agent_growth', False)
        self.max_agents = self.config.get('max_agents', 8)
        self.min_agents = self.config.get('min_agents', 2)
        
        # Boids型自己増殖機能が有効の場合
        if self.use_self_replication:
            # エージェントファクトリ初期化
            from .boids_agent_factory import BoidsAgentFactory
            self.agent_factory = BoidsAgentFactory(self.config, opinion_space_manager)
            
            # 意見ベクトル化モジュール初期化
            from .opinion_vectorizer import OpinionVectorizer
            self.vectorizer = OpinionVectorizer(self.config)
            
            if opinion_space_manager:
                self.agent_factory.set_opinion_space_manager(opinion_space_manager)
            
            if self.debug:
                print("Boids型自己増殖機能を有効化")
        
        # グローバルロック初期化
        self._initialize_lock()
    
    def set_opinion_space_manager(self, opinion_space_manager) -> None:
        """
        意見空間マネージャを設定
        
        引数:
            opinion_space_manager: OpinionSpaceManagerインスタンス
        """
        self.opinion_space = opinion_space_manager
        
        # Boids型自己増殖機能が有効の場合、ファクトリにも設定
        if self.use_self_replication and hasattr(self, 'agent_factory'):
            self.agent_factory.set_opinion_space_manager(opinion_space_manager)
            
        if self.debug:
            print("意見空間マネージャを設定しました")

    def _initialize_lock(self):
        """非同期処理用のロック初期化"""
        try:
            import threading
            self.lock = threading.Lock()
        except ImportError:
            # シンプルなロック実装（スレッドセーフでない環境用）
            class DummyLock:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def acquire(self, *args, **kwargs): return True
                def release(self): pass
            self.lock = DummyLock()

    async def run_discussion_rounds(self, user_query: str, max_turns: int = 3) -> Dict[str, Any]:
        """
        議論ラウンドを実行し、複数エージェントによる対話を進行させる
        
        引数:
            user_query: ユーザーからの質問/クエリ
            max_turns: 最大ターン数
            
        戻り値:
            Dict[str, Any]: 最終的な応答情報を含む辞書
        """
        if self.debug:
            print(f"議論開始: ターン数上限={max_turns}")
            
        # 初期設定
        conversation_history = [user_query]
        self.current_turn = 0
        
        # 初期エージェント配置（ユーザークエリに応じて）
        if not self.agents:
            self._initialize_agents(user_query)
            
        # メインループ
        for turn in range(max_turns):
            self.current_turn += 1
            turn_start = time.time()
            
            if self.debug:
                print(f"ターン {self.current_turn}/{max_turns} 開始")
                
            # 1. エージェント発言収集（並列または逐次）
            outputs = await self._collect_agent_responses(conversation_history)
            
            # 会話履歴に追加（失敗したエージェントの応答は除外）
            valid_outputs = []
            for agent_id, output_text in outputs:
                # 応答が有効かチェック（Noneや空文字は失敗とみなす）
                if output_text is not None and output_text.strip() != "":
                    conversation_history.append(f"{self.agents[agent_id]['role']}: {output_text}")
                    valid_outputs.append((agent_id, output_text))
                    
                    # 黒板があれば書き込み
                    if self.blackboard:
                        self.blackboard.write('agent_message', {
                            'type': 'agent',
                            'agent_id': agent_id,
                            'role': self.agents[agent_id]['role'],
                            'text': output_text,
                            'turn': self.current_turn
                        })
            
            # 2. 発言のベクトル化（有効な応答のみ）
            if self.use_self_replication and self.vectorizer and self.opinion_space:
                for agent_id, output_text in valid_outputs:
                    try:
                        # 出力がテキストかどうかを確認し、辞書なら適切なフィールドを抽出
                        text_to_vectorize = output_text
                        if isinstance(output_text, dict) and 'text' in output_text:
                            text_to_vectorize = output_text['text']
                        elif not isinstance(output_text, str):
                            if self.debug:
                                print(f"警告: 出力テキストが文字列でもなく適切な辞書でもありません: {type(output_text)}")
                            continue
                            
                        # まずvectorizeメソッドを試す（標準的な命名規則）
                        if hasattr(self.vectorizer, 'vectorize'):
                            vec = self.vectorizer.vectorize(text_to_vectorize)
                        # 次にvectorize_textメソッドを試す
                        elif hasattr(self.vectorizer, 'vectorize_text'):
                            vec = self.vectorizer.vectorize_text(text_to_vectorize)
                        else:
                            # どちらのメソッドもなければモックベクトルを使用
                            if self.debug:
                                print("警告: 適切なベクトル化メソッドが見つかりません。モックベクトルを使用します。")
                            import numpy as np
                            vec = np.random.rand(self.config.get('vector_dim', 64))
                        
                        self.opinion_space.add_vector(agent_id, vec, 
                                                    {'turn': self.current_turn, 'text': str(text_to_vectorize)[:100]})
                    except Exception as e:
                        if self.debug:
                            print(f"ベクトル化エラー: {str(e)}")
                            import traceback
                            traceback.print_exc()
            
            # 3. 自己診断
            diagnosis = self._run_self_diagnosis()
            
            # 4. エージェント追加・削除
            if self.use_self_replication:
                self._process_agent_adjustments(diagnosis)
                
            # 5. 終了判定
            if diagnosis.get('is_fully_converged', False):
                if self.debug:
                    print("意見が収束したため対話を終了します")
                break
                
            if self.debug:
                print(f"ターン {self.current_turn} 完了: 処理時間={time.time() - turn_start:.2f}秒")
        
        # 6. 最終応答生成
        final_answer = await self._produce_final_answer(conversation_history)
        
        result = {
            'answer': final_answer,
            'conversation': conversation_history,
            'turns': self.current_turn,
            'agents': len(self.agents)
        }
        
        return result
    
    async def _collect_agent_responses(self, conversation_history: List[str]) -> List[Tuple[str, str]]:
        """
        全エージェントの発言を収集する
        
        引数:
            conversation_history: 現在までの会話履歴
            
        戻り値:
            List[Tuple[str, str]]: (agent_id, response_text) のリスト
        """
        outputs = []
        use_parallel = self.config.get('use_parallel', True)
        safe_parallel = self.config.get('safe_parallel', False)
        
        if use_parallel and not safe_parallel:
            # 完全な並列処理
            try:
                import asyncio
                tasks = []
                for agent_id, agent_info in self.agents.items():
                    task = self._generate_agent_response(agent_id, conversation_history)
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks)
                outputs = [(agent_id, response) for agent_id, response in zip(self.agents.keys(), responses)]
                
            except ImportError:
                # asyncioが使えない場合はフォールバック
                print("警告: asyncioが利用できないため、逐次実行にフォールバックします")
                use_parallel = False
        
        if not use_parallel or safe_parallel:
            # 逐次処理または安全な並列処理
            for agent_id, agent_info in self.agents.items():
                if safe_parallel:
                    # 少し間隔をあけて起動して競合を避ける
                    time.sleep(0.1)
                response = await self._generate_agent_response(agent_id, conversation_history)
                outputs.append((agent_id, response))
                
                # 安全な並列処理の場合は会話履歴を更新
                if safe_parallel:
                    conversation_history.append(f"{agent_info['role']}: {response}")
        
        return outputs
        
    async def _generate_agent_response(self, agent_id: str, conversation_history: List[str], max_retries: int = 2) -> str:
        """
        特定のエージェントからの応答を生成する
        
        引数:
            agent_id: エージェントID
            conversation_history: 会話履歴
            max_retries: 最大リトライ回数
            
        戻り値:
            str: 生成された応答テキスト
        """
        if agent_id not in self.agents:
            if self.debug:
                print(f"警告: 存在しないエージェントID ({agent_id}) の応答を要求されました")
            return None
            
        agent_info = self.agents[agent_id]
        role = agent_info.get('role', 'アドバイザー')
        
        try:
            # エージェントプロンプト作成 (メタ発言禁止と次の話題誘導の強化)
            prompt_elements = [
                f"あなたは {role} です。他のエージェントと共同で議論に参加しています。",
                "現在進行中の議論に参加し、専門的かつ有益な視点を提供してください。",
                f"会話履歴: {conversation_history[-min(5, len(conversation_history)):]}", # 直近5発言まで
                
                # メタ発言禁止の強化指示
                "【厳守すべき禁止事項】",
                "・他のエージェントについて言及することは絶対に禁止です（「他のエージェント」「〇〇さん」などへの言及は避ける）",
                "・「～さんが応答できなかった」「エラーが発生した」などのメタ発言は厳禁です",
                "・システム内部や会話の進行状況についての発言は避けてください",
                "・エラーメッセージや「考え中」などの処理中の状態を示す表現は使わないでください",
                "・「私はAIアシスタントです」「私はLLMです」などの自己言及も避けてください",
                "・「申し訳ありません」「すみません」などの謝罪表現も避けてください",
                "・「続行しますか？」「次へ進みますか？」などの質問も避けてください",
                "・議論の内容そのものだけに焦点を当ててください",
                
                # 次の話題誘導を必須タスクに設定（強化指示）
                "【必ず実行すべき必須タスク】",
                "1. ユーザーの質問/議論に対する具体的かつ有用な回答を提供する",
                "2. 必ず議論を発展させる新しい切り口や質問を1つ以上提案してください（※これは必ず含めてください）",
                "3. 発言は簡潔に300字以内でまとめ、新たな視点を入れつつ回答を深める",
                "4. 結論だけでなく、理由や根拠も簡潔に示してください",
                "5. 回答の最後には必ず次に議論すべき視点や疑問点を示して発言を終えてください",
                
                f"現在の担当役割: {role}",
                f"以上の制約に従い、{role}として回答してください:"
            ]

            user_prompt = "\n".join(prompt_elements)
            
            # エージェント応答生成
            for attempt in range(max_retries):
                if self.debug and attempt > 0:
                    print(f"{role} リトライ #{attempt+1}")
                    
                try:
                    # モデルから応答テキスト生成
                    if hasattr(self, 'model_clients') and self.model_clients and agent_id in self.model_clients:
                        model_client = self.model_clients[agent_id]
                        response = await model_client.generate(user_prompt, 
                                                            max_tokens=self.config.get('max_tokens', 200),
                                                            temperature=agent_info.get('temperature', 0.7),
                                                            stop_sequences=["\n\n", "##", "エージェント:"])
                    elif hasattr(self, 'llm') and self.llm:
                        # LLMインスタンスを使って応答生成
                        response = await self.llm.generate(user_prompt,
                                                        max_tokens=self.config.get('max_tokens', 200),
                                                        temperature=agent_info.get('temperature', 0.7),
                                                        stop_sequences=["\n\n", "##", "エージェント:"])
                    else:
                        # デフォルトあるいはモック応答
                        response = f"{role}としての考え: この議論について、重要な視点は...。さらに考えるべき点として..."
                        time.sleep(0.5)  # モック応答の場合は少し待つ
                        
                    # 応答後処理
                    if response and isinstance(response, str):
                        response = response.strip()
                        # プレフィックスや余分な修飾を削除
                        prefixes = [f"{role}: ", f"{role}として、", f"{role}からの応答: ", "回答: ", f"{role}の視点: "]
                        for prefix in prefixes:
                            if response.startswith(prefix):
                                response = response[len(prefix):]
                        
                        # メタ発言のチェックと除去（強化版）
                        meta_phrases = [
                            "他のエージェント", "応答できません", "回答できません", 
                            "AIとして", "アシスタントとして", "私はAIです", 
                            "申し訳ありません", "エラー", "考え中",
                            "すみません", "対応できかねます", "モデルとして",
                            "エラーが発生", "回答が生成できません", "別のエージェント",
                            "エージェントの", "さんは", "さんが", "続行しますか",
                            "次へ進みますか", "処理を続けますか", "反復処理", "反復"
                        ]
                        
                        contains_meta = any(phrase in response.lower() for phrase in meta_phrases)
                        
                        # 次の話題への誘導が含まれているか確認（強化版）
                        next_topic_phrases = [
                            "さらに", "次の視点", "考えるべき点", "別の角度", "新たな質問", 
                            "議論を発展", "検討すべき", "考察すべき", "次に議論すべき",
                            "さらなる考察", "加えて考えるべき", "今後の展開として", 
                            "別の観点", "次のステップ", "重要な論点として", "疑問点として"
                        ]
                        has_next_topic = any(phrase in response for phrase in next_topic_phrases)
                        
                        # メタ発言を含む場合はリトライ
                        if contains_meta and attempt < max_retries - 1:
                            if self.debug:
                                print(f"メタ発言検出のためリトライ: {response[:30]}...")
                            continue
                            
                        # 次の話題誘導がない場合は追加（必ず追加するように強化）
                        if not has_next_topic and len(response) > 20:
                            # 役割に応じた次の話題誘導を追加
                            next_topic_suggestions = [
                                f"\n\nさらに検討すべき点として、この議論を深めるためには具体例や異なる視点からの分析も重要ではないでしょうか？",
                                f"\n\n次に考えるべきこととして、この問題の長期的な影響について議論する価値があると思います。",
                                f"\n\n議論を進めるために、この考えを実践する際の課題について検討してみてはいかがでしょう？"
                            ]
                            import random
                            response = response + random.choice(next_topic_suggestions)
                        
                        # 「反復処理を続行しますか?」などの処理継続確認を除去
                        continuation_patterns = [
                            r"反復処理を続行します(か|か？|か\?)?", 
                            r"処理を続行します(か|か？|か\?)?",
                            r"続行します(か|か？|か\?)?",
                            r"続けます(か|か？|か\?)?"
                        ]
                        for pattern in continuation_patterns:
                            response = re.sub(pattern, "", response)
                        
                        # 成功したらそのまま返却
                        if response:
                            return response
                            
                except Exception as e:
                    if self.debug:
                        print(f"エージェント応答生成エラー (リトライ {attempt+1}/{max_retries}): {str(e)}")
                    if attempt == max_retries - 1:
                        raise  # 最後のリトライでエラーが出たら例外送出
            
            # すべてのリトライに失敗した場合のフォールバック応答
            import random
            fallback_responses = [
                f"この議論では、{role}として重要なのは幅広い視点から考察することです。次に、具体的な事例を検討してみてはどうでしょうか？",
                f"この問題については様々な要素を考慮する必要があります。次のステップとして、実際の応用例について議論を深めてみましょう。",
                f"この課題について考えると、いくつかの重要な側面があります。さらに、長期的な影響についても検討する価値があるでしょう。"
            ]
            return random.choice(fallback_responses)
            
        except Exception as e:
            if self.debug:
                print(f"エージェント応答生成の致命的エラー: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 致命的エラー時のフォールバック（エラーメッセージを含まない）
            import random
            emergency_responses = [
                f"多角的な視点からこの問題を検討すると、いくつかの重要な側面が浮かび上がります。次に、具体例を交えて議論を深めましょう。",
                f"この議論の核心部分を考えると、いくつかの重要な要素があります。次のステップとして、実践的な応用について考察を深めてはいかがでしょう？"
            ]
            return random.choice(emergency_responses)
    
    def _get_default_system_prompt(self, role: str) -> str:
        """
        役割に応じたデフォルトのシステムプロンプトを取得
        
        引数:
            role: エージェントの役割
            
        戻り値:
            str: システムプロンプト
        """
        if "contrarian" in role.lower() or "批判" in role or "反対" in role:
            return "あなたは会議でのデビルズアドボケート（あえて反対意見を述べる役）です。他の発言者の意見に疑問を呈し、代替案や潜在的な問題点を指摘してください。"
        
        elif "mediator" in role.lower() or "調停" in role or "調整" in role:
            return "あなたは議論の調停者です。各発言者の主張を要約し、共通点を見つけて統合案を提案してください。対立がある場合は妥協点を示してください。"
        
        elif "align" in role.lower() or "同調" in role or "深掘り" in role:
            return "あなたは他の意見を深掘りする役割です。他の発言者の提案を支持し、それをさらに発展させる詳細な情報や理由を述べてください。"
            
        elif "分析" in role or "analyst" in role.lower():
            return "あなたは分析者です。データに基づいて冷静に事実を分析し、明確で論理的な視点を提供してください。"
            
        elif "質問" in role or "query" in role.lower():
            return "あなたは質問者です。議論の論点を明確にし、他の参加者の意見をより深く理解するための質問を投げかけてください。"
            
        else:
            return "あなたは議論の参加者です。他の発言者の意見を尊重しつつ、自分の考えを述べてください。"
    
    def _initialize_agents(self, user_query: str) -> None:
        """
        ユーザークエリに基づいて初期エージェントを設定
        
        引数:
            user_query: ユーザーからの質問/クエリ
        """
        # 基本の役割を設定
        initial_roles = ["分析者", "質問者", "調整者"]
        num_agents = min(self.num_agents, len(initial_roles))
        
        # エージェント作成
        self.agents = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            role = initial_roles[i]
            
            # エージェント情報
            self.agents[agent_id] = {
                'id': agent_id,
                'role': role,
                'created_at': time.time(),
                'generation': 0,
                'lifespan': self.config.get('default_agent_lifespan', 10)
            }
        
        if self.debug:
            print(f"初期エージェント {num_agents}体を作成しました")
    
    def _run_self_diagnosis(self) -> Dict[str, Any]:
        """
        現在の意見空間について自己診断を実行
        
        戻り値:
            Dict[str, Any]: 診断結果
        """
        # デフォルト値
        diagnosis = {
            'stagnation_detected': False,
            'bias_detected': False, 
            'divergence_detected': False,
            'suggested_action': None,
            'is_fully_converged': False
        }
        
        # 自己増殖機能が無効、または意見空間がない場合は早期リターン
        if not (self.use_self_replication and self.opinion_space):
            return diagnosis
            
        try:
            # 最新ベクトルを取得
            latest_vectors = self.opinion_space.get_latest_vectors()
            if not latest_vectors or len(latest_vectors) < 2:
                return diagnosis
                
            # 平均距離と最大距離を計算
            avg_distance, max_distance = self.opinion_space.calculate_distance_metrics()
            prev_centroid_distance = self.opinion_space.calculate_centroid_movement()
            
            # 閾値設定
            threshold_low = self.config.get('threshold_similarity_low', 0.2)  # 高類似度閾値
            threshold_high = self.config.get('threshold_similarity_high', 0.7)  # 低類似度閾値
            threshold_stagnation = self.config.get('threshold_stagnation', 0.1)  # 停滞閾値
            
            # 偏り検出
            if avg_distance < threshold_low:  # 小さな平均距離は意見の収束を示す
                diagnosis['bias_detected'] = True
            
            # 発散検出
            if max_distance > threshold_high:  # 大きな最大距離は意見の乖離を示す
                diagnosis['divergence_detected'] = True
                
            # 停滞検出
            if prev_centroid_distance < threshold_stagnation:  # 小さな移動距離は停滞を示す
                diagnosis['stagnation_detected'] = True
                
            # 推奨アクションの決定
            if diagnosis['bias_detected'] and diagnosis['stagnation_detected']:
                diagnosis['suggested_action'] = "add_contrarian_agent"
            elif diagnosis['divergence_detected']:
                diagnosis['suggested_action'] = "add_mediator_agent"
            elif self.opinion_space.has_redundant_vectors(0.9):  # 冗長なベクトルがある
                diagnosis['suggested_action'] = "remove_redundant_agent"
                
            # 収束判定（全員が近い意見で、かつ前回からほとんど変化がない）
            if avg_distance < threshold_low * 0.7 and prev_centroid_distance < threshold_stagnation * 0.5:
                diagnosis['is_fully_converged'] = True
                
            if self.debug:
                print(f"自己診断結果: bias={diagnosis['bias_detected']}, "
                      f"divergence={diagnosis['divergence_detected']}, "
                      f"stagnation={diagnosis['stagnation_detected']}, "
                      f"action={diagnosis['suggested_action']}")
                
            return diagnosis
                
        except Exception as e:
            if self.debug:
                print(f"自己診断エラー: {str(e)}")
            return diagnosis
    
    def _process_agent_adjustments(self, diagnosis: Dict[str, Any]) -> None:
        """
        診断結果に基づきエージェントの追加・削除を行う
        
        引数:
            diagnosis: 診断結果
        """
        if not self.use_self_replication:
            return
            
        suggested_action = diagnosis.get('suggested_action')
        if not suggested_action:
            return
            
        try:
            if suggested_action == "add_contrarian_agent" and len(self.agents) < self.max_agents:
                # 反対意見エージェントを追加
                new_id = f"agent_{len(self.agents)}"
                self.agents[new_id] = {
                    'id': new_id,
                    'role': "反対意見者",
                    'system_prompt': "あなたは会議でのデビルズアドボケートです。他の発言者の意見に疑問を呈し、"
                                    "代替案や潜在的な問題点を指摘してください。",
                    'created_at': time.time(),
                    'generation': self.current_turn,
                    'lifespan': self.config.get('default_agent_lifespan', 10)
                }
                if self.debug:
                    print(f"反対意見エージェント追加: ID={new_id}")
                    
            elif suggested_action == "add_mediator_agent" and len(self.agents) < self.max_agents:
                # 調停エージェントを追加
                new_id = f"agent_{len(self.agents)}"
                self.agents[new_id] = {
                    'id': new_id,
                    'role': "調停者",
                    'system_prompt': "あなたは議論の調停者です。各発言者の主張を要約し、共通点を見つけて統合案を提案してください。"
                                    "対立がある場合は妥協点を示してください。",
                    'created_at': time.time(),
                    'generation': self.current_turn,
                    'lifespan': self.config.get('default_agent_lifespan', 10)
                }
                if self.debug:
                    print(f"調停エージェント追加: ID={new_id}")
                    
            elif suggested_action == "remove_redundant_agent" and len(self.agents) > self.min_agents:
                # 冗長なエージェントを削除
                agent_to_remove = self._select_removal_candidate()
                if agent_to_remove:
                    if self.debug:
                        print(f"エージェント削除: ID={agent_to_remove}, 役割={self.agents[agent_to_remove]['role']}")
                    self.agents.pop(agent_to_remove, None)
        
        except Exception as e:
            if self.debug:
                print(f"エージェント調整エラー: {str(e)}")
    
    def _select_removal_candidate(self) -> Optional[str]:
        """
        削除すべきエージェントを選定
        
        戻り値:
            Optional[str]: 削除すべきエージェントのID、適切な候補がなければNone
        """
        if not self.opinion_space or len(self.agents) <= self.min_agents:
            return None
            
        try:
            # 類似度の高いペアを探す
            max_sim = 0.0
            redundant_pair = None
            
            latest_vectors = self.opinion_space.get_latest_vectors()
            agent_ids = list(latest_vectors.keys())
            
            for i, agent_id1 in enumerate(agent_ids):
                for j in range(i+1, len(agent_ids)):
                    agent_id2 = agent_ids[j]
                    vec1 = latest_vectors[agent_id1]
                    vec2 = latest_vectors[agent_id2]
                    
                    # コサイン類似度を計算
                    sim = self.opinion_space.calculate_similarity(vec1, vec2)
                    
                    if sim > max_sim:
                        max_sim = sim
                        redundant_pair = (agent_id1, agent_id2)
            
            # 類似性が閾値を超える場合、寿命の長い方を削除
            if max_sim > 0.9 and redundant_pair:
                agent1, agent2 = redundant_pair
                
                if self.agents[agent1].get('lifespan', 0) > self.agents[agent2].get('lifespan', 0):
                    return agent1
                else:
                    return agent2
            
            # それ以外の場合、一番古いエージェントを返す
            oldest_agent = None
            oldest_time = float('inf')
            
            for agent_id, agent_info in self.agents.items():
                if agent_info.get('created_at', float('inf')) < oldest_time:
                    oldest_time = agent_info.get('created_at', float('inf'))
                    oldest_agent = agent_id
                    
            return oldest_agent
            
        except Exception as e:
            if self.debug:
                print(f"削除候補選定エラー: {str(e)}")
            return None
    
    async def _produce_final_answer(self, conversation_history: List[str]) -> str:
        """
        最終的な応答を生成
        
        引数:
            conversation_history: 会話履歴
            
        戻り値:
            str: 生成された最終応答
        """
        if self.debug:
            print("最終応答の生成を開始します")
            
        # 最終応答生成用プロンプト
        prompt = "あなたは最終回答を生成する出力エージェントです。以下の会話履歴を要約し、"
        prompt += "ユーザーの質問に対する包括的な回答を生成してください。\n\n"
        prompt += "会話履歴:\n" + "\n".join(conversation_history)
        prompt += "\n\n上記の議論を踏まえてユーザーの質問に対する回答をまとめてください。"
        
        try:
            final_answer = await self.llm.generate(prompt)
            return final_answer
        except Exception as e:
            if self.debug:
                print(f"最終応答生成エラー: {str(e)}")
            return "申し訳ありませんが、回答の生成中にエラーが発生しました。"

    def update_roles_based_on_question(self, question: str) -> None:
        """
        質問内容に基づいてエージェントの役割を更新する
        
        引数:
            question: ユーザーからの質問/クエリ
        """
        if self.debug:
            print(f"質問に基づいて役割を更新: {question[:30]}...")
        
        # 質問タイプを分類
        question_type = self._classify_question_type(question)
        
        # 黒板があれば質問タイプを記録
        if self.blackboard:
            self.blackboard.write('question_type', question_type)
            
        # Boids型自己増殖機能を使用している場合は早期リターン
        if self.use_self_replication:
            return
            
        # 質問タイプに応じた役割を設定
        if question_type == 'discussion':
            # 議論型質問 - 多様な視点や意見が必要
            roles = [
                {"role": "議論調整者", "description": "議論を整理し、異なる視点を統合する"},
                {"role": "批判的思考者", "description": "異論を唱え、議論の盲点を指摘する"},
                {"role": "創造的思考者", "description": "既存の枠組みを超えた新しい視点を提供する"}
            ]
        elif question_type == 'planning':
            # 計画・構想型質問 - 実現性と創造性のバランスが必要
            roles = [
                {"role": "計画立案者", "description": "現実的で実行可能な計画を提案する"},
                {"role": "リスク分析者", "description": "潜在的な問題点や障害を特定する"},
                {"role": "創造的発想者", "description": "革新的なアイデアや可能性を提案する"}
            ]
        elif question_type == 'informational':
            # 情報提供型質問 - 事実と多様な視点のバランスが必要
            roles = [
                {"role": "知識提供者", "description": "関連する事実や情報を提供する"},
                {"role": "文脈解説者", "description": "背景や関連する文脈を説明する"},
                {"role": "異なる視点提供者", "description": "複数の観点から情報を解釈する"}
            ]
        else:  # conversational または default
            # 一般会話型質問 - シンプルな対応
            roles = [
                {"role": "対話者", "description": "自然な対話を行う"},
                {"role": "情報提供者", "description": "必要に応じて情報を提供する"}
            ]
        
        # エージェント役割を更新
        self.agent_roles = roles
        
        if self.debug:
            print(f"質問タイプ: {question_type}, 役割数: {len(roles)}")
            
    def _classify_question_type(self, question: str) -> str:
        """
        質問内容を分析し、タイプを分類する（内部メソッド）
        
        引数:
            question: ユーザーからの質問/クエリ
            
        戻り値:
            str: 質問タイプ ('discussion', 'planning', 'informational', 'conversational')
        """
        # 質問の特徴を分析
        question_lower = question.lower()
        
        # 議論型質問の特徴語
        discussion_keywords = ["議論", "討論", "考察", "意見", "どう思う", "異なる視点", "debate", "discuss", 
                              "意見を聞かせて", "考え方", "思想", "哲学", "倫理", "議論してください"]
        
        # 計画・構想型質問の特徴語
        planning_keywords = ["計画", "構想", "設計", "方法", "どうすれば", "ステップ", "プラン", "戦略", "方針", 
                            "アイデア", "改善", "最適化", "開発", "実装", "設計", "計画を立てて", "方法を考えて"]
        
        # 情報提供型質問の特徴語
        info_keywords = ["何ですか", "説明", "教えて", "情報", "事実", "歴史", "いつ", "どこで", "なぜ", "どのように", 
                        "who", "what", "when", "where", "why", "how", "とは", "について教えて"]
        
        # 一致する特徴語の数をカウント
        discussion_count = sum(1 for kw in discussion_keywords if kw in question_lower)
        planning_count = sum(1 for kw in planning_keywords if kw in question_lower)
        info_count = sum(1 for kw in info_keywords if kw in question_lower)
        
        # 疑問文の特徴も確認
        has_question_mark = "?" in question or "？" in question
        ends_with_question = any(question.endswith(q) for q in ["か", "の", "でしょうか", "ですか"])
        
        # 質問の長さも特徴として使用
        is_long = len(question) > 30
        
        # 分類ロジック
        if discussion_count > max(planning_count, info_count) or "議論" in question_lower or "討論" in question_lower:
            return "discussion"
        elif planning_count > max(discussion_count, info_count) or any(kw in question_lower for kw in planning_keywords):
            return "planning"
        elif info_count > max(discussion_count, planning_count) or (has_question_mark and not is_long) or ends_with_question:
            return "informational"
        else:
            return "conversational"  # デフォルト

    async def run_agents(self, blackboard) -> List[Dict[str, Any]]:
        """
        すべてのエージェントを実行して応答を生成する
        
        引数:
            blackboard: 黒板インスタンス
            
        戻り値:
            List[Dict[str, Any]]: エージェント応答のリスト
        """
        if self.debug:
            print(f"エージェント実行中...")
            
        # 再帰呼び出し防止用のグローバルフラグ
        # スレッドIDとコールIDでユニークなキーを作成
        import threading
        import uuid
        
        # グローバル実行状態管理用の辞書がなければ作成
        if not hasattr(AgentPoolManager, '_run_agents_status'):
            AgentPoolManager._run_agents_status = {}
            
        # 現在のスレッドID
        thread_id = threading.get_ident()
        
        # 現在のコールスタックに既存の実行がないか確認
        if thread_id in AgentPoolManager._run_agents_status:
            if self.debug:
                print(f"警告: run_agentsはすでにスレッド {thread_id} で実行中です。再帰呼び出しを防止します。")
            return []

        # 実行状態を記録
        call_id = str(uuid.uuid4())
        AgentPoolManager._run_agents_status[thread_id] = {
            'call_id': call_id,
            'start_time': time.time()
        }
        
        try:
            use_parallel = self.config.get('use_parallel', False)
            responses = []
            
            # 入力を取得
            input_data = blackboard.read('input')
            if not input_data:
                if self.debug:
                    print("警告: 入力データがありません")
                return []
                
            # 入力テキストを取得
            input_text = ""
            if isinstance(input_data, dict) and 'normalized' in input_data:
                input_text = input_data['normalized']
            elif isinstance(input_data, str):
                input_text = input_data
                
            # システムプロンプト（あれば）
            system_prompt = blackboard.read('system_prompt') or ""
            
            # 会話コンテキスト（あれば）- 過去の会話履歴を制限
            conversation_context = blackboard.read('conversation_context') or ""
            if len(conversation_context) > 1000:  # 1000文字を超える場合は最新部分のみを使用
                conversation_context = "...(過去の会話は省略)...\n" + conversation_context[-1000:]
                # 黒板の内容も更新
                blackboard.write('conversation_context', conversation_context)
                if self.debug:
                    print(f"会話コンテキストを短縮: {len(conversation_context)} 文字")
            
            # 前のエージェント応答（あれば）- 最新の3つのみを使用
            prev_responses = blackboard.read('agent_responses') or []
            if len(prev_responses) > 3:
                prev_responses = prev_responses[-3:]
                
            # エージェント数を確認
            num_agents = 0
            if isinstance(self.agents, dict):
                num_agents = len(self.agents)
            elif isinstance(self.agents, list):
                num_agents = len(self.agents)
            else:
                num_agents = self.num_agents
                
            if self.debug:
                print(f"実行するエージェント数: {num_agents}")
                
            # 並列実行
            if use_parallel and num_agents > 1:
                try:
                    import asyncio
                    
                    # 各エージェントのプロンプトを生成
                    agent_tasks = []
                    
                    if isinstance(self.agents, dict):
                        for agent_id in self.agents:
                            agent_task = self._generate_agent_response(
                                agent_id, input_text, system_prompt, 
                                conversation_context, prev_responses
                            )
                            agent_tasks.append(agent_task)
                    elif isinstance(self.agents, list):
                        for i, agent_info in enumerate(self.agents):
                            agent_id = f"agent_{i}"
                            agent_task = self._generate_agent_response(
                                agent_id, input_text, system_prompt, 
                                conversation_context, prev_responses
                            )
                            agent_tasks.append(agent_task)
                    else:
                        # デフォルトのエージェント数
                        for i in range(self.num_agents):
                            agent_id = f"agent_{i}"
                            agent_task = self._generate_agent_response(
                                agent_id, input_text, system_prompt, 
                                conversation_context, prev_responses
                            )
                            agent_tasks.append(agent_task)
                        
                    # 並列実行
                    if agent_tasks:
                        # セマフォを使って同時実行数を制限
                        semaphore = asyncio.Semaphore(min(3, len(agent_tasks)))
                        
                        async def run_with_semaphore(task):
                            async with semaphore:
                                return await task
                                
                        limited_tasks = [run_with_semaphore(task) for task in agent_tasks]
                        agent_responses = await asyncio.gather(*limited_tasks)
                        
                        for response in agent_responses:
                            if response:
                                responses.append(response)
                                
                except (ImportError, Exception) as e:
                    if self.debug:
                        print(f"並列実行エラー: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    # 並列処理に失敗したら逐次処理にフォールバック
                    use_parallel = False
                    
            # 逐次実行
            if not use_parallel:
                if isinstance(self.agents, dict):
                    for agent_id in self.agents:
                        response = await self._generate_agent_response(
                            agent_id, input_text, system_prompt, 
                            conversation_context, prev_responses
                        )
                        if response:
                            responses.append(response)
                elif isinstance(self.agents, list):
                    for i, agent_info in enumerate(self.agents):
                        agent_id = f"agent_{i}"
                        response = await self._generate_agent_response(
                            agent_id, input_text, system_prompt, 
                            conversation_context, prev_responses
                        )
                        if response:
                            responses.append(response)
                else:
                    # デフォルトのエージェント数
                    for i in range(self.num_agents):
                        agent_id = f"agent_{i}"
                        response = await self._generate_agent_response(
                            agent_id, input_text, system_prompt, 
                            conversation_context, prev_responses
                        )
                        if response:
                            responses.append(response)
                        
            # 応答結果を黒板に書き込む
            blackboard.write('agent_responses', responses)
            
            if self.debug:
                print(f"{len(responses)}個のエージェント応答を生成")
                
            return responses
        finally:
            # 必ず実行されるクリーンアップ処理
            # 実行状態をクリア
            if thread_id in AgentPoolManager._run_agents_status:
                del AgentPoolManager._run_agents_status[thread_id]
                
    async def _generate_agent_response(self, agent_id, input_text, 
                                     system_prompt="", conversation_context="", 
                                     prev_responses=None) -> Dict[str, Any]:
        """
        エージェント応答を生成する（内部メソッド）
        
        引数:
            agent_id: エージェントID
            input_text: 入力テキスト
            system_prompt: システムプロンプト
            conversation_context: 会話コンテキスト
            prev_responses: 前回の応答リスト
            
        戻り値:
            Dict[str, Any]: 応答情報
        """
        # 再帰呼び出し防止用セーフガード
        # スレッドローカル変数を使用してコールスタックを追跡
        import threading
        
        # スレッドローカルストレージが初期化されていなければ初期化
        if not hasattr(threading.current_thread(), '_agent_call_stack'):
            threading.current_thread()._agent_call_stack = set()
            
        # エージェントIDを文字列化
        agent_id_str = str(agent_id)
        
        # すでにこのエージェントIDを処理中なら再帰呼び出しとみなして終了
        if agent_id_str in threading.current_thread()._agent_call_stack:
            if self.debug:
                print(f"再帰呼び出し検出: agent_id={agent_id_str}")
            return {'agent_id': agent_id_str, 'role': '不明', 'text': '再帰エラー防止のため応答生成をスキップしました', 'timestamp': time.time()}
        
        # コールスタックに追加
        threading.current_thread()._agent_call_stack.add(agent_id_str)
        
        try:
            # エージェント情報を取得
            agent_info = None
            if isinstance(self.agents, dict):
                # 辞書型の場合
                # agent_idが辞書の場合、'id'フィールドを取得
                if isinstance(agent_id, dict) and 'id' in agent_id:
                    agent_id = agent_id['id']
                    
                if isinstance(agent_id, str) and agent_id in self.agents:
                    agent_info = self.agents[agent_id]
            else:
                # リスト型の場合
                try:
                    idx = -1
                    # agent_idが辞書の場合、'id'フィールドを取得し処理
                    if isinstance(agent_id, dict) and 'id' in agent_id:
                        agent_id_str = agent_id['id']
                        if agent_id_str.startswith("agent_"):
                            try:
                                idx = int(agent_id_str.replace("agent_", ""))
                            except ValueError:
                                idx = -1
                    elif isinstance(agent_id, str) and agent_id.startswith("agent_"):
                        try:
                            idx = int(agent_id.replace("agent_", ""))
                        except ValueError:
                            idx = -1
                    elif isinstance(agent_id, int):
                        idx = agent_id
                        
                    if 0 <= idx < len(self.agents):
                        agent_info = self.agents[idx]
                except (ValueError, IndexError, TypeError):
                    agent_info = None
            
            if not agent_info:
                # エージェント情報が取得できなければデフォルト情報を使用
                agent_info = {
                    'role': 'アシスタント',
                    'system_prompt': ''
                }
                
            # エージェントの役割を取得
            role = agent_info.get('role', 'アシスタント')
            
            # エージェント固有のシステムプロンプト
            agent_system = agent_info.get('system_prompt', '')
            if not agent_system:
                agent_system = self._get_default_system_prompt(role)
                
            # プロンプトの組み立て
            prompt = agent_system
            if system_prompt and not prompt:
                prompt = system_prompt
                
            # 会話履歴を追加
            if conversation_context:
                if prompt:
                    prompt += "\n\n"
                prompt += conversation_context
                
            # 前の応答を追加
            if prev_responses:
                if prompt:
                    prompt += "\n\n"
                prompt += "これまでの議論:\n"
                for resp in prev_responses[-3:]:  # 最新の3つのみ
                    agent_role = resp.get('role', 'アシスタント')
                    agent_text = resp.get('text', '')
                    if agent_text:
                        prompt += f"{agent_role}: {agent_text}\n"
                        
            # 入力を追加
            if prompt:
                prompt += "\n\n"
            prompt += f"クエリ: {input_text}\n"
            prompt += f"\nあなたは{role}として発言してください。"
            
            try:
                # LLM呼び出し
                response_text = await self.llm.generate(prompt)
                
                # 応答情報の構築
                response = {
                    'agent_id': agent_id_str,
                    'role': role,
                    'text': response_text,
                    'timestamp': time.time()
                }
                
                return response
                
            except Exception as e:
                if self.debug:
                    print(f"エージェント応答生成エラー: {str(e)}")
                # エラー時の自然な応答を返す
                fallback_responses = [
                    f"（{role}は考え中...）",
                    f"（{role}は議論を聞いています）",
                    "（他のエージェントの意見を参考にしています...）",
                    f"（{role}からの応答を待っています...）"
                ]
                import random
                return {
                    'agent_id': agent_id_str,
                    'role': role,
                    'text': random.choice(fallback_responses),
                    'timestamp': time.time(),
                    'error': True  # エラーフラグを追加
                }
        finally:
            # 必ず実行されるクリーンアップ処理
            # コールスタックから削除
            threading.current_thread()._agent_call_stack.discard(agent_id_str)