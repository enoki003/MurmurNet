#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 分散SLMシステム
~~~~~~~~~~~~~~~~~~~
複数の小型言語モデルを組み合わせた分散創発型アーキテクチャ
黒板設計・要約・RAGを統合した協調パターン

作者: Yuhi Sonoki
"""

# distributed_slm.py
import os
import yaml
import logging
import asyncio
import re
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from MurmurNet.modules.input_reception import InputReception
from MurmurNet.modules.blackboard import Blackboard
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.modules.output_agent import OutputAgent
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.conversation_memory import ConversationMemory

# Boids型自己増殖エージェント関連のモジュールをインポート
from MurmurNet.modules.opinion_vectorizer import OpinionVectorizer
from MurmurNet.modules.opinion_space_manager import OpinionSpaceManager
from MurmurNet.modules.self_diagnosis import SelfDiagnosis
from MurmurNet.modules.boids_agent_factory import BoidsAgentFactory

class DistributedSLM:
    """
    分散創発型言語モデルメインクラス
    単一の関数呼び出しで高度な対話生成機能を提供するブラックボックス型モジュール
    
    特徴:
    - 複数の小規模モデルが協調的に動作
    - 黒板を通じた情報共有
    - 反復的な知識交換で知性を創発
    - Boids型自己増殖による動的エージェント管理（オプション）
    """
    def __init__(self, config: dict = None, blackboard=None):
        """
        各モジュール初期化
        
        引数:
            config: 設定辞書
            blackboard: Blackboardインスタンス（省略時は内部で作成）
        """
        self.config = config or {}
        
        # 動作パラメータ
        self.num_agents = self.config.get('num_agents', 2)
        self.iterations = self.config.get('iterations', 1)  # 反復回数
        self.use_summary = self.config.get('use_summary', True)  # 要約を使うか
        self.use_parallel = self.config.get('use_parallel', False)  # 並列処理を使うか
        self.use_memory = self.config.get('use_memory', True)  # 会話履歴を使うか
        
        # Boids型自己増殖エージェント機能の設定
        self.use_boids = self.config.get('use_boids', False)  # Boids型自己増殖機能を使うか
        if self.use_boids:
            # AgentPoolManagerに自己増殖機能を有効化する設定を追加
            self.config['use_self_replication'] = True
            
            # 最大エージェント数の設定（デフォルト5）
            if 'max_agents' not in self.config:
                self.config['max_agents'] = 5
                
            # 反復回数を最低でも2以上にする（自己増殖効果を発揮するため）
            if self.iterations < 2:
                self.iterations = 2

            # エージェント設定の調整
            self.config['min_agents'] = self.config.get('min_agents', 3)
            self.config['enable_agent_growth'] = True
            self.config['vector_dim'] = self.config.get('vector_dim', 384)  # 意見空間のベクトルサイズ
                
        # 各モジュールの初期化
        self.blackboard = blackboard if blackboard is not None else Blackboard(self.config)
        self.input_reception = InputReception(self.config)
        
        # Boids型自己増殖エージェント機能が有効な場合、OpinionSpaceManagerを初期化
        self.opinion_space = None
        self.vectorizer = None
        if self.use_boids:
            # まず意見ベクトル化モジュールを初期化
            self.vectorizer = OpinionVectorizer(self.config)
            if self.config.get('debug', False):
                print("OpinionVectorizerを初期化しました")
                
            # 次に意見空間マネージャを初期化（ベクトル化モジュールを渡す）
            self.opinion_space = OpinionSpaceManager(self.config, self.vectorizer)
            if self.config.get('debug', False):
                print("OpinionSpaceManagerを初期化しました")

        # AgentPoolManagerの初期化（必要に応じて意見空間を渡す）
        if self.use_boids and self.opinion_space:
            self.agent_pool = AgentPoolManager(self.config, self, self.blackboard, self.opinion_space)
        else:
            self.agent_pool = AgentPoolManager(self.config, self, self.blackboard)
            
        self.rag_retriever = RAGRetriever(self.config)
        self.summary_engine = SummaryEngine(self.config)
        self.output_agent = OutputAgent(self.config)
        self.conversation_memory = ConversationMemory(self.config)
        
        # Boids型自己増殖エージェント機能が有効な場合、初期エージェントをAgentPoolクラスで作成
        if self.use_boids and hasattr(self.agent_pool, 'agent_factory'):
            self._initialize_boids_agents()
        
        # ロガー設定
        self._setup_logger()
        self.logger.info(f"Initialized with {self.num_agents} agents, {self.iterations} iterations")
        if self.use_boids:
            self.logger.info("Boids型自己増殖エージェント機能を有効化しました")
            
    def _initialize_boids_agents(self):
        """Boids型自己増殖エージェントの初期化（内部メソッド）"""
        if not self.use_boids or not hasattr(self.agent_pool, 'agent_factory'):
            return
            
        try:
            # 初期エージェントを作成
            initial_count = self.config.get('min_agents', 3)
            initial_roles = ["アナリスト", "質問者", "調整者"]
            
            # AgentPoolを直接操作する
            if hasattr(self.agent_pool, 'agents'):
                if isinstance(self.agent_pool.agents, list):
                    self.agent_pool.agents = []  # リストをクリア
                elif isinstance(self.agent_pool.agents, dict):
                    self.agent_pool.agents = {}  # 辞書をクリア
                    
                # 各エージェントを作成
                for i in range(min(initial_count, len(initial_roles))):
                    role = initial_roles[i]
                    agent_info = {
                        'id': f"agent_{i}",
                        'role': role,
                        'created_at': time.time(),
                        'generation': 0,
                        'lifespan': self.config.get('default_agent_lifespan', 10),
                        'position': np.random.rand(self.config.get('vector_dim', 384)),
                        'velocity': np.zeros(self.config.get('vector_dim', 384)),
                        'contribution_score': 0.0,
                        'active': True
                    }
                    
                    # エージェントを追加
                    if isinstance(self.agent_pool.agents, list):
                        self.agent_pool.agents.append(agent_info)
                    elif isinstance(self.agent_pool.agents, dict):
                        self.agent_pool.agents[f"agent_{i}"] = agent_info
                
                if self.config.get('debug', False):
                    print(f"Boids型自己増殖エージェント {len(initial_roles)} 体を初期化しました")
        except Exception as e:
            # エラーハンドリング
            if self.config.get('debug', False):
                print(f"Boidsエージェント初期化エラー: {str(e)}")
                import traceback
                traceback.print_exc()
        
    def _setup_logger(self):
        """ロガー初期化（内部メソッド）"""
        self.logger = logging.getLogger('DistributedSLM')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if not self.config.get('debug') else logging.DEBUG)

    async def _run_iteration(self, iteration: int) -> None:
        """単一の反復サイクルを実行（内部メソッド）"""
        self.logger.info(f"Starting iteration {iteration}")
        
        # 質問に基づいて役割を更新（初回のみ）
        if iteration == 0:
            input_data = self.blackboard.read('input')
            if input_data:
                question = input_data.get('normalized') if isinstance(input_data, dict) else str(input_data)
                # 役割の更新を明示的に実行
                self.agent_pool.update_roles_based_on_question(question)
                self.logger.debug(f"Updated agent roles based on question type")
                
        # Boids型自己増殖エージェント機能を使用している場合、BOIDSアルゴリズムの処理を実行
        if self.use_boids and iteration > 0:
            self._apply_boids_cycle(iteration)
        
        # 1. エージェント実行（並列または逐次）
        if self.use_parallel:
            await self._run_agents_parallel()
        else:
            # 非並列実行の場合はawaitをつける必要がある
            await self.agent_pool.run_agents(self.blackboard)
            
        # 2. エージェント出力収集
        agent_entries = []
        # エージェント数を取得
        num_agents = self.num_agents
        if self.use_boids and hasattr(self.agent_pool, 'agents'):
            # AgentPoolクラスのagents属性から直接取得
            if isinstance(self.agent_pool.agents, dict):
                # agents辞書型の場合はその長さを使用
                num_agents = len(self.agent_pool.agents)
            elif isinstance(self.agent_pool.agents, list):
                # agentsリスト型の場合はその長さを使用
                num_agents = len(self.agent_pool.agents)
        
        for i in range(num_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            if agent_output:
                agent_entries.append({"agent": i, "text": agent_output})
        
        # 3. 出力の要約（使用する場合）
        if self.use_summary and agent_entries:
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            self.blackboard.write(f'summary_{iteration}', summary)
            self.logger.debug(f"Created summary for iteration {iteration}")
        
        # 4. エージェント出力をベクトル化し意見空間に追加（Boids型自己増殖機能時）
        if self.use_boids and self.opinion_space:
            self._vectorize_agent_outputs(agent_entries, iteration)
        
        # 5. エージェントの貢献度評価
        if self.use_boids and iteration > 0:
            self._evaluate_agent_contributions(agent_entries)
            
        # 6. Boids型自己増殖機能の診断結果記録
        if self.use_boids:
            # 自己診断実行
            diagnosis = self._run_self_diagnosis(iteration)
            diagnosis_key = f'diagnosis_{iteration}'
            self.blackboard.write(diagnosis_key, diagnosis)
            
            if diagnosis:
                action = diagnosis.get('action')
                reason = diagnosis.get('reason', '')
                if action:
                    self.logger.debug(f"Iteration {iteration} diagnosis: {action} - {reason}")
    
    def _apply_boids_cycle(self, iteration: int) -> None:
        """
        Boids型自己増殖エージェント機能のサイクルを実行（内部メソッド）
        
        引数:
            iteration: 現在の反復番号
        """
        if not self.use_boids or not hasattr(self.agent_pool, 'agents'):
            return
            
        try:
            # 1. 必要に応じてエージェントの寿命を減らす
            if hasattr(self.agent_pool, '_apply_boids_rules'):
                # AgentPoolクラスのメソッドが直接利用可能な場合
                self.agent_pool._apply_boids_rules()
            else:
                # 独自に実装する場合
                self._apply_boids_rules()
                
            # 2. エージェントのターンを進める
            if hasattr(self.agent_pool, 'next_turn'):
                # AgentPoolの標準メソッドがある場合
                turn_info = self.agent_pool.next_turn()
                if self.config.get('debug', False):
                    self.logger.debug(f"ボイズターン {iteration}: "
                                    f"追加={len(turn_info.get('added_agents', []))}エージェント, "
                                    f"削除={len(turn_info.get('removed_agents', []))}エージェント")
            
        except Exception as e:
            if self.config.get('debug', False):
                self.logger.error(f"BOIDSサイクルエラー: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
    
    def _apply_boids_rules(self) -> None:
        """
        Boidsアルゴリズムルールを適用してエージェントの位置を更新（内部メソッド）
        AgentPoolのメソッドが利用できない場合にフォールバックとして使用
        """
        if not self.use_boids or not hasattr(self.agent_pool, 'agents') or not self.opinion_space:
            return
            
        try:
            import numpy as np
            
            # エージェントリストの取得
            agents = []
            if isinstance(self.agent_pool.agents, dict):
                agents = list(self.agent_pool.agents.values())
            elif isinstance(self.agent_pool.agents, list):
                agents = self.agent_pool.agents
                
            if len(agents) <= 1:
                return  # エージェントが1以下の場合は処理不要
                
            # 各エージェントの現在位置をベクトルとして収集
            positions = []
            velocities = []
            
            for agent in agents:
                # エージェントの位置と速度を取得
                if isinstance(agent, dict):
                    pos = agent.get('position', np.random.rand(self.config.get('vector_dim', 384)))
                    vel = agent.get('velocity', np.zeros(self.config.get('vector_dim', 384)))
                else:
                    pos = getattr(agent, 'position', np.random.rand(self.config.get('vector_dim', 384)))
                    vel = getattr(agent, 'velocity', np.zeros(self.config.get('vector_dim', 384)))
                    
                positions.append(pos)
                velocities.append(vel)
            
            # numpy配列に変換
            positions = np.array(positions)
            velocities = np.array(velocities)
            
            # 重み係数の設定
            boids_weights = {
                'cohesion': self.config.get('weight_cohesion', 0.3),    # 結合
                'separation': self.config.get('weight_separation', 0.2), # 分離
                'alignment': self.config.get('weight_alignment', 0.4),   # 整列
                'innovation': self.config.get('weight_innovation', 0.1)  # 革新性
            }
            
            # 各エージェントにBOIDSルールを適用
            for i, agent in enumerate(agents):
                # 現在の位置と速度
                current_pos = positions[i]
                current_vel = velocities[i]
                
                # ルール1: 群れの中心に向かう（結合）
                center_of_mass = np.mean(positions, axis=0)
                cohesion = (center_of_mass - current_pos) * boids_weights['cohesion']
                
                # ルール2: 近すぎるエージェントから離れる（分離）
                separation = np.zeros_like(current_pos)
                for j, other_pos in enumerate(positions):
                    if i != j:
                        diff = current_pos - other_pos
                        distance = np.linalg.norm(diff)
                        if distance < 0.1:  # 近接閾値
                            # 距離に反比例する分離ベクトル
                            sep_factor = min(1.0 / max(0.001, distance), 10)
                            separation += diff * sep_factor
                separation *= boids_weights['separation']
                
                # ルール3: 他のエージェントと同じ方向に進む（整列）
                avg_velocity = np.mean(velocities, axis=0)
                alignment = avg_velocity * boids_weights['alignment']
                
                # ルール4: 少しランダムな変化を加える（革新性）
                innovation = np.random.rand(len(current_pos)) * 2 - 1  # -1〜1のランダムベクトル
                innovation *= boids_weights['innovation']
                
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
                
                # エージェントの位置と速度を更新
                if isinstance(agent, dict):
                    agent['position'] = new_position
                    agent['velocity'] = new_velocity
                else:
                    agent.position = new_position
                    agent.velocity = new_velocity
                    
            if self.config.get('debug', False):
                self.logger.debug(f"BOIDSルール適用: {len(agents)}エージェントを更新")
                
        except Exception as e:
            if self.config.get('debug', False):
                self.logger.error(f"BOIDSルール適用エラー: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
    
    def _vectorize_agent_outputs(self, agent_entries: list, iteration: int) -> None:
        """
        エージェント出力をベクトル化し意見空間に追加（内部メソッド）
        
        引数:
            agent_entries: エージェント出力のリスト
            iteration: 現在の反復番号
        """
        if not self.use_boids or not self.opinion_space or not hasattr(self.agent_pool, 'agents'):
            return
            
        # 再帰呼び出し防止のためのスレッドローカル変数
        import threading
        
        # スレッドローカルストレージが初期化されていなければ初期化
        if not hasattr(threading.current_thread(), '_vectorizing_agents'):
            threading.current_thread()._vectorizing_agents = set()
            
        try:
            from MurmurNet.modules.opinion_vectorizer import OpinionVectorizer
            
            # ベクトル化モジュールの確認
            vectorizer = None
            if hasattr(self.agent_pool, 'vectorizer') and self.agent_pool.vectorizer:
                # AgentPoolのベクトル化モジュールを使用
                vectorizer = self.agent_pool.vectorizer
            elif not hasattr(self, 'vectorizer') or not self.vectorizer:
                # 必要に応じて新しく作成
                self.vectorizer = OpinionVectorizer(self.config)
                vectorizer = self.vectorizer
            else:
                vectorizer = self.vectorizer
                
            for entry in agent_entries:
                agent_id = entry.get('agent')
                text = entry.get('text', '')
                
                if not text:
                    continue
                    
                # 再帰呼び出し検出 - 同じエージェントIDを処理中ならスキップ
                agent_id_str = f"agent_{agent_id}"
                if agent_id_str in threading.current_thread()._vectorizing_agents:
                    if self.config.get('debug', False):
                        print(f"再帰呼び出し検出: ベクトル化をスキップ agent_id={agent_id_str}")
                    continue
                
                # 処理中マークを設定
                threading.current_thread()._vectorizing_agents.add(agent_id_str)
                
                try:
                    # テキストをベクトル化
                    if hasattr(vectorizer, 'vectorize_text'):
                        vector = vectorizer.vectorize_text(text)
                    elif hasattr(vectorizer, 'vectorize'):
                        vector = vectorizer.vectorize(text)
                    else:
                        if self.config.get('debug', False):
                            self.logger.warning("適切なベクトル化メソッドがありません")
                        continue
                    
                    # 意見空間に追加
                    metadata = {
                        'agent_id': agent_id_str,
                        'text': text[:100] if len(text) > 100 else text,
                        'iteration': iteration,
                        'timestamp': time.time()
                    }
                    
                    # エージェントメタデータの追加
                    if isinstance(self.agent_pool.agents, dict) and agent_id_str in self.agent_pool.agents:
                        agent = self.agent_pool.agents[agent_id_str]
                        if isinstance(agent, dict):
                            metadata['role'] = agent.get('role', 'Unknown')
                    elif isinstance(self.agent_pool.agents, list) and agent_id < len(self.agent_pool.agents):
                        agent = self.agent_pool.agents[agent_id]
                        if isinstance(agent, dict):
                            metadata['role'] = agent.get('role', 'Unknown')
                    
                    # 意見空間に追加（直接add_vectorメソッドを使う）
                    if hasattr(self.opinion_space, 'add_vector'):
                        self.opinion_space.add_vector(agent_id_str, vector, metadata)
                    # 互換性のためadd_opinionメソッドも試す
                    elif hasattr(self.opinion_space, 'add_opinion'):
                        self.opinion_space.add_opinion(text, metadata)
                
                finally:
                    # 処理終了時に必ずマークを削除
                    threading.current_thread()._vectorizing_agents.discard(agent_id_str)
                
            if self.config.get('debug', False) and agent_entries:
                self.logger.debug(f"{len(agent_entries)}個の意見ベクトルを追加")
                
        except Exception as e:
            if self.config.get('debug', False):
                self.logger.error(f"ベクトル化エラー: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
        finally:
            # 念のため、スレッドローカル変数をクリア
            if hasattr(threading.current_thread(), '_vectorizing_agents'):
                threading.current_thread()._vectorizing_agents.clear()
    
    def _evaluate_agent_contributions(self, agent_entries: list) -> None:
        """
        エージェントの貢献度を評価（内部メソッド）
        
        引数:
            agent_entries: エージェント出力のリスト
        """
        if not self.use_boids or not hasattr(self.agent_pool, 'agents'):
            return
            
        try:
            # 各エージェントの貢献度スコアを計算
            contributions = {}
            
            # 簡易的な貢献度計算（文章の長さとトピックの適合性）
            for entry in agent_entries:
                agent_id = entry.get('agent')
                text = entry.get('text', '')
                
                if not text:
                    continue
                    
                # 基本点（文章の長さに基づく）
                score = min(len(text) / 100, 3.0)  # 最大3点
                
                # 追加点（内容に基づく）
                has_facts = any(marker in text.lower() for marker in ['研究によると', '調査結果', '統計', 'データ'])
                has_structure = any(marker in text for marker in ['第一に', '一方で', 'しかし', 'また', 'ただし'])
                has_examples = any(marker in text for marker in ['例えば', '具体的には', '一例として'])
                
                if has_facts:
                    score += 0.5
                if has_structure:
                    score += 0.5
                if has_examples:
                    score += 0.5
                
                # 貢献度スコアを記録
                contributions[f"agent_{agent_id}"] = score
            
            # AgentPoolに貢献度を割り当て
            if hasattr(self.agent_pool, 'assign_contributions'):
                self.agent_pool.assign_contributions(contributions)
                
                if self.config.get('debug', False) and contributions:
                    self.logger.debug(f"{len(contributions)}エージェントに貢献度を割り当て")
                    
        except Exception as e:
            if self.config.get('debug', False):
                self.logger.error(f"貢献度評価エラー: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                
    def _run_self_diagnosis(self, iteration: int) -> dict:
        """
        自己診断を実行（内部メソッド）
        
        引数:
            iteration: 現在の反復番号
            
        戻り値:
            診断結果の辞書
        """
        if not self.use_boids or not self.opinion_space:
            return {}
        
        try:
            # 意見空間のメトリクスを取得
            avg_distance = self.opinion_space.calculate_average_distance() if hasattr(self.opinion_space, 'calculate_average_distance') else None
            centroid_movement = self.opinion_space.calculate_centroid_movement() if hasattr(self.opinion_space, 'calculate_centroid_movement') else None
            clusters = self.opinion_space.detect_clusters(threshold=0.2) if hasattr(self.opinion_space, 'detect_clusters') else []
            
            # 診断の初期化
            diagnosis = {
                'iteration': iteration,
                'timestamp': time.time(),
                'metrics': {
                    'avg_distance': avg_distance,
                    'centroid_movement': centroid_movement,
                    'cluster_count': len(clusters) if clusters else 0
                }
            }
            
            # エージェント数の取得
            num_agents = 0
            if hasattr(self.agent_pool, 'agents'):
                if isinstance(self.agent_pool.agents, dict):
                    num_agents = len(self.agent_pool.agents)
                elif isinstance(self.agent_pool.agents, list):
                    num_agents = len(self.agent_pool.agents)
            
            # 閾値を設定
            threshold_stagnation = self.config.get('threshold_stagnation', 0.1)  # 停滞閾値
            threshold_diversity_low = self.config.get('threshold_diversity_low', 0.2)  # 多様性低下閾値
            threshold_diversity_high = self.config.get('threshold_diversity_high', 0.7)  # 多様性過剰閾値
            
            # 状態を判断
            is_stagnant = centroid_movement is not None and centroid_movement < threshold_stagnation
            is_converged = avg_distance is not None and avg_distance < threshold_diversity_low
            is_diverged = avg_distance is not None and avg_distance > threshold_diversity_high
            
            # 診断結果を決定
            if num_agents < self.config.get('min_agents', 3):
                action = 'add_agent'
                reason = 'エージェント数が最小値を下回っています'
            elif is_stagnant and is_converged:
                action = 'add_contrarian'
                reason = '議論が停滞し意見が収束しています'
            elif is_diverged:
                action = 'add_mediator'
                reason = '意見が発散しています'
            elif is_stagnant:
                action = 'add_innovation'
                reason = '議論が停滞しています'
            elif num_agents > self.config.get('max_agents', 5):
                action = 'remove_agent'
                reason = 'エージェント数が過剰です'
            else:
                action = None
                reason = '特に問題ありません'
            
            diagnosis['action'] = action
            diagnosis['reason'] = reason
            
            return diagnosis
            
        except Exception as e:
            if self.config.get('debug', False):
                self.logger.error(f"自己診断エラー: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
            return {}
    
    async def _run_agents_parallel(self) -> None:
        """エージェントを並列実行（内部メソッド）"""
        self.logger.info("Running agents in parallel")
        
        # run_agents関数に適切に渡す
        try:
            # まず、エージェントを並列実行
            results = await self.agent_pool.run_agents(self.blackboard)
            
            # 応答を黒板に書き込み
            if results:
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        # 結果が辞書の場合
                        agent_id = result.get('agent_id', i)
                        role = result.get('role', f'エージェント{i}')
                        text = result.get('text', '')
                        self.blackboard.write(f'agent_{i}_output', text)
                        self.blackboard.write(f'agent_{i}_role', role)
                    elif result:
                        # 結果が文字列などの場合
                        self.blackboard.write(f'agent_{i}_output', str(result))
            else:
                # 結果がなかった場合のフォールバック処理
                # Boids型自己増殖エージェント機能を使用している場合の処理を修正
                num_agents = self.num_agents
                if self.use_boids and hasattr(self.agent_pool, 'agents'):
                    # AgentPoolクラスのagents属性から直接取得
                    if isinstance(self.agent_pool.agents, dict):
                        num_agents = len(self.agent_pool.agents)
                    elif isinstance(self.agent_pool.agents, list):
                        num_agents = len(self.agent_pool.agents)
                
                for i in range(num_agents):
                    self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
                    
        except Exception as e:
            self.logger.error(f"Parallel execution error: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
            # エラーが発生した場合は個別に実行を試みる
            # Boids型自己増殖エージェント機能を使用している場合の処理を修正
            num_agents = self.num_agents
            if self.use_boids and hasattr(self.agent_pool, 'agents'):
                # AgentPoolクラスのagents属性から直接取得
                if isinstance(self.agent_pool.agents, dict):
                    num_agents = len(self.agent_pool.agents)
                elif isinstance(self.agent_pool.agents, list):
                    num_agents = len(self.agent_pool.agents)
                    
            for i in range(num_agents):
                try:
                    # 個別に応答生成を試みる
                    input_data = self.blackboard.read('input')
                    input_text = input_data.get('normalized') if isinstance(input_data, dict) else str(input_data)
                    system_prompt = self.blackboard.read('system_prompt') or ""
                    conversation_context = self.blackboard.read('conversation_context') or ""
                    
                    # エージェントIDを文字列にして、AgentPoolManagerのAPIを直接使用
                    agent_id = f"agent_{i}"
                    response = await self.agent_pool._generate_agent_response(
                        agent_id, input_text, system_prompt, conversation_context
                    )
                    
                    if response:
                        if isinstance(response, dict):
                            self.blackboard.write(f'agent_{i}_output', response.get('text', ''))
                            self.blackboard.write(f'agent_{i}_role', response.get('role', f'エージェント{i}'))
                        else:
                            self.blackboard.write(f'agent_{i}_output', str(response))
                    else:
                        self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
                        
                except Exception as inner_e:
                    self.logger.error(f"Agent {i} fallback execution failed: {str(inner_e)}")
                    self.blackboard.write(f'agent_{i}_output', f"エージェント{i}の実行に失敗しました: {str(inner_e)[:50]}")
    
    def _is_memory_related_question(self, text: str) -> bool:
        """
        入力テキストが会話記憶に関連する質問かどうかを判定
        
        引数:
            text: 入力テキスト
        
        戻り値:
            会話記憶に関連する質問ならTrue
        """
        # 記憶関連の質問パターン
        memory_patterns = [
            r"(私|僕|俺|わたし|ぼく|おれ)の名前(は|を)(なに|何|なん)",
            r"(私|僕|俺|わたし|ぼく|おれ)(の|は|を)(なに|何|なん)と(言い|いい|呼び|よび|よん)",
            r"(私|僕|俺|わたし|ぼく|おれ)の名前(は|を)(覚え|おぼえ)",
            r"(私|僕|俺|わたし|ぼく|おれ)の(趣味|しゅみ)(は|を)(なに|何|なん)",
            r"(私|僕|俺|わたし|ぼく|おれ)(は|が)(なに|何|なん)(が好き|を好きと言いました)",
            r"(覚え|おぼえ)てる",
            r"(覚え|おぼえ)てます",
            r"(私|僕|俺|わたし|ぼく|おれ)について",
        ]
        
        # いずれかのパターンにマッチしたら記憶関連の質問と判定
        for pattern in memory_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
        
    def _collect_boids_information(self) -> dict:
        """
        Boids型自己増殖エージェント機構の情報を収集
        
        戻り値:
            収集した情報の辞書
        """
        if not self.use_boids:
            return {}
            
        info = {
            'active_agents': [],
            'iterations': self.iterations,
            'diagnoses': []
        }
        
        # アクティブなエージェント情報
        if hasattr(self.agent_pool, 'agents'):
            if isinstance(self.agent_pool.agents, dict):
                # 辞書形式のエージェント
                for agent_id, agent_data in self.agent_pool.agents.items():
                    if isinstance(agent_data, dict):
                        # 辞書形式
                        info['active_agents'].append({
                            'id': agent_id,
                            'role': agent_data.get('role', 'Unknown'),
                            'strategy': agent_data.get('strategy', 'Unknown'),
                            'principle': agent_data.get('principle', 'Unknown')
                        })
                    else:
                        # Agentクラスのインスタンス
                        info['active_agents'].append({
                            'id': agent_id,
                            'role': getattr(agent_data, 'role', 'Unknown'),
                            'strategy': getattr(agent_data, 'strategy', 'Unknown'),
                            'principle': getattr(agent_data, 'principle', 'Unknown')
                        })
            elif isinstance(self.agent_pool.agents, list):
                # リスト形式のエージェント
                for i, agent_data in enumerate(self.agent_pool.agents):
                    if isinstance(agent_data, dict):
                        info['active_agents'].append({
                            'id': f"agent_{i}",
                            'role': agent_data.get('role', 'Unknown'),
                            'strategy': agent_data.get('strategy', 'Unknown'),
                            'principle': agent_data.get('principle', 'Unknown')
                        })
                    else:
                        info['active_agents'].append({
                            'id': f"agent_{i}",
                            'role': getattr(agent_data, 'role', 'Unknown'),
                            'strategy': getattr(agent_data, 'strategy', 'Unknown'),
                            'principle': getattr(agent_data, 'principle', 'Unknown')
                        })
        
        # 各イテレーションの診断結果
        for i in range(self.iterations):
            diagnosis_key = f'diagnosis_{i}'
            diagnosis = self.blackboard.read(diagnosis_key)
            if diagnosis:
                info['diagnoses'].append({
                    'iteration': i,
                    'action': diagnosis.get('action'),
                    'reason': diagnosis.get('reason', '')
                })
        
        return info
        
    async def generate(self, input_text: str) -> str:
        """
        外部公開API: 入力文字列から最終応答を生成
        引数：
            input_text: ユーザー入力文字列
        戻り値：
            生成された応答文字列
        """
        self.logger.info("Starting generation process")
        
        # 新しいターンのために黒板をクリア（会話履歴は保持）
        self.blackboard.clear_current_turn()
        
        # 会話記憶に関連する質問かチェック
        if self.use_memory and self._is_memory_related_question(input_text):
            # 会話記憶から回答を取得
            memory_answer = self.conversation_memory.get_answer_for_question(input_text)
            if memory_answer:
                self.logger.info("Found answer from conversation memory")
                # 会話記憶を更新
                self.conversation_memory.add_conversation_entry(
                    user_input=input_text,
                    system_response=memory_answer
                )
                return memory_answer
        
        # 1. 入力受付・前処理
        processed = self.input_reception.process(input_text)
        self.blackboard.write('input', processed)
        
        # 2. 会話履歴の追加（使用する場合）
        if self.use_memory:
            conversation_context = self.conversation_memory.get_conversation_context()
            if conversation_context and conversation_context != "過去の会話はありません。":
                # 会話コンテキストを黒板に書き込む
                self.blackboard.write('conversation_context', conversation_context)
                self.logger.debug(f"Added conversation context: {len(conversation_context)} chars")
                
                # 会話コンテキストを入力処理に統合
                if isinstance(processed, dict) and 'normalized' in processed:
                    processed['context'] = conversation_context
                    self.blackboard.write('input', processed)
        
        # 3. RAG検索
        rag_result = self.rag_retriever.retrieve(input_text)
        self.blackboard.write('rag', rag_result)
        
        # 4. 初期要約の実行（オプション）
        if self.use_summary:
            initial_summary = f"ユーザー入力: {processed['normalized'] if isinstance(processed, dict) else processed}\n\n検索情報: {rag_result}"
            if self.use_memory and 'conversation_context' in self.blackboard.memory:
                initial_summary += f"\n\n会話コンテキスト: {self.blackboard.read('conversation_context')}"
            self.blackboard.write('initial_summary', initial_summary)
            
        # 5. 反復サイクルの実行
        for i in range(self.iterations):
            await self._run_iteration(i)
            
        # 6. 最終要約とエージェント出力の収集
        entries = []
        # 各イテレーションの要約を収集
        if self.use_summary:
            for i in range(self.iterations):
                summary = self.blackboard.read(f'summary_{i}')
                if summary:
                    entries.append({"type": "summary", "iteration": i, "text": summary})
        
        # 現在アクティブなエージェント数を取得
        num_active_agents = self.num_agents
        if self.use_boids and hasattr(self.agent_pool, 'agents'):
            # AgentPoolクラスのagents属性から直接取得
            if isinstance(self.agent_pool.agents, dict):
                num_active_agents = len(self.agent_pool.agents)
            elif isinstance(self.agent_pool.agents, list):
                num_active_agents = len(self.agent_pool.agents)
        
        # 最終エージェント出力も収集
        for i in range(num_active_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            if agent_output:
                agent_role = self.blackboard.read(f"agent_{i}_role") or f"エージェント{i}"
                entries.append({"type": "agent", "agent": i, "role": agent_role, "text": agent_output})
                
        # Boids型自己増殖機構の情報を収集（デバッグ用）
        if self.use_boids and self.config.get('debug'):
            boids_info = self._collect_boids_information()
            self.blackboard.write('boids_info', boids_info)
            self.logger.debug(f"Collected Boids information: {len(boids_info.get('active_agents', []))} active agents")
                
        # 7. 最終応答生成
        final_response = self.output_agent.generate(self.blackboard, entries)
        self.blackboard.write('final_response', final_response)
        
        # 8. 会話履歴に追加（使用する場合）
        if self.use_memory:
            self.conversation_memory.add_conversation_entry(
                user_input=input_text, 
                system_response=final_response
            )
            self.logger.debug("Updated conversation memory")
        
        return final_response
        
    def reset_memory(self):
        """会話履歴をリセット"""
        if hasattr(self, 'conversation_memory'):
            self.conversation_memory.clear_memory()
            self.logger.info("Conversation memory cleared")

    def save_log(self, filepath: str = None) -> str:
        """
        現在のセッションのログを保存
        
        引数:
            filepath: 保存先ファイルパス（省略時は自動生成）
            
        戻り値:
            保存したファイルパス
        """
        import json
        import datetime
        
        # ファイルパス未指定の場合は自動生成
        if not filepath:
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join('./log', f'turn_{now}.json')
            
        # ディレクトリがなければ作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存するデータの整理
        log_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'blackboard': self.blackboard.memory,
        }
        
        # Boids型自己増殖機構の情報を追加
        if self.use_boids:
            log_data['boids_info'] = self._collect_boids_information()
        
        # 情報をJSON形式で書き出し
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Log saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save log: {str(e)}")
            return ""