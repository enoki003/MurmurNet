#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self Diagnosis モジュール
~~~~~~~~~~~~~~~~~~~
議論の状態を診断し、エージェントの増減や役割の修正を判断する

作者: Yuhi Sonoki
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import time
import random
import threading
import traceback

class SelfDiagnosis:
    """
    議論の状態を診断するモジュール
    - 意見の多様性や収束状況を分析
    - エージェントの貢献度を評価
    - 最適なエージェント操作（増加・減少・修正）を判断
    """
    
    def __init__(self, config: Dict[str, Any], opinion_space_manager=None):
        """
        初期化
        
        引数:
            config: 設定辞書
            opinion_space_manager: OpinionSpaceManagerインスタンス
        """
        self.config = config
        self.debug = config.get('debug', False)
        self.opinion_space = opinion_space_manager
        
        # 診断閾値
        self.diversity_threshold = config.get('diversity_threshold', 0.4)  # 多様性が足りない判定閾値
        self.convergence_threshold = config.get('convergence_threshold', 0.1)  # 収束したと判定する閾値
        self.similarity_threshold = config.get('similarity_threshold', 0.7)  # エージェントが類似していると判定する閾値
        
        # 診断間隔（秒）
        self.diagnosis_interval = config.get('diagnosis_interval', 10)
        
        # 最大/最小エージェント数
        self.max_agents = config.get('max_agents', 5)
        self.min_agents = config.get('min_agents', 2)
        
        # 前回の診断時刻
        self.last_diagnosis_time = 0
        
        # 前回の診断結果
        self.last_diagnosis = None
        
        # 連続同一判断のカウンタ
        self.repeat_counter = {}
        
        if self.debug:
            print("SelfDiagnosis初期化完了")
    
    def set_opinion_space_manager(self, opinion_space_manager):
        """
        OpinionSpaceManagerを設定
        
        引数:
            opinion_space_manager: OpinionSpaceManagerインスタンス
        """
        self.opinion_space = opinion_space_manager
    
    def diagnose(self, agents: List[Dict], blackboard=None) -> Dict[str, Any]:
        """
        議論の状態を診断し、必要な対応を判断
        
        引数:
            agents: 現在アクティブなエージェントリスト
            blackboard: 黒板オブジェクト（省略可能）
            
        戻り値:
            診断結果と推奨アクション
        """
        # 再帰処理の防止
        import threading
        current_thread = threading.current_thread()
        
        # スレッドローカルストレージを初期化
        if not hasattr(current_thread, '_diagnosing'):
            current_thread._diagnosing = False
        
        # 既に診断中なら処理をスキップ
        if getattr(current_thread, '_diagnosing', False):
            if self.debug:
                print("警告: diagnoseの再帰呼び出しを検出。処理をスキップします。")
            return {'action': 'none', 'reason': '診断処理の再帰呼び出し回避', 'timestamp': time.time()}
        
        # 診断中フラグを設定
        current_thread._diagnosing = True
        
        try:
            # 前回の診断から十分な時間が経過していない場合はスキップ
            current_time = time.time()
            if current_time - self.last_diagnosis_time < self.diagnosis_interval and self.last_diagnosis:
                return self.last_diagnosis
                
            # 議論状態の包括的な分析を実行
            discussion_analysis = self._analyze_discussion_state(agents)
            
            # 分析結果に基づいて推奨アクションを決定
            recommendation = self._determine_recommended_action(discussion_analysis, agents)
              # 診断結果を作成
            result = {
                **recommendation,
                'timestamp': current_time,
                'analysis': discussion_analysis
            }
            
            # 結果をキャッシュし、返却
            self.last_diagnosis = result
            self.last_diagnosis_time = current_time
                
            # 意見空間マネージャがない場合はデフォルト診断
            if not self.opinion_space:
                result = self._default_diagnosis(agents)
                self.last_diagnosis = result
                self.last_diagnosis_time = current_time
                return result
                
            # エージェント数が上下限を超えるケースを先にチェック
            num_agents = len(agents)
            if num_agents > self.max_agents:
                result = {
                    'action': 'reduce',
                    'reason': f'エージェント数が上限({self.max_agents})を超えています',
                    'target': self._select_agent_for_reduction_with_details(agents),
                    'timestamp': current_time
                }
                self.last_diagnosis = result
                self.last_diagnosis_time = current_time
                return result
                
            if num_agents < self.min_agents:
                result = {
                    'action': 'add',
                    'reason': f'エージェント数が下限({self.min_agents})を下回っています',
                    'properties': self._generate_diverse_agent_properties(agents),
                    'timestamp': current_time
                }
                self.last_diagnosis = result
                self.last_diagnosis_time = current_time
                return result
            
            # 意見空間メトリクスの取得（安全に）
            try:
                # 1. 多様性メトリクスの取得
                diversity_metrics = {}
                if hasattr(self.opinion_space, 'compute_diversity_metrics'):
                    diversity_metrics = self.opinion_space.compute_diversity_metrics()
                else:
                    diversity_metrics = {'mean_distance': 0.5, 'variance': 0.1}
                
                # 2. 変化メトリクスの取得
                change_metrics = {}
                if hasattr(self.opinion_space, 'compute_change_metrics'):
                    change_metrics = self.opinion_space.compute_change_metrics()
                else:
                    change_metrics = {
                        'centroid_movement': 0.05,
                        'average_agent_movement': 0.05,
                        'opinion_convergence': 0.0
                    }                # 3. クラスタリングの実行（安全に）
                clusters = None
                if hasattr(self.opinion_space, 'cluster_opinions'):
                    try:
                        # n_clustersパラメータを使用
                        clusters = self.opinion_space.cluster_opinions(n_clusters=min(3, num_agents))
                    except Exception as e:
                        if self.debug and self.config.get('verbose_logging', False):
                            print(f"クラスタリングエラー: {str(e)}")
                
                # 4. クラスタ間距離（安全に）
                inter_cluster_distances = []
                if clusters and hasattr(self.opinion_space, 'get_inter_cluster_distances'):
                    try:
                        inter_cluster_distances = self.opinion_space.get_inter_cluster_distances()
                    except Exception as e:
                        if self.debug:
                            print(f"クラスタ間距離計算エラー: {str(e)}")
                
                avg_cluster_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0.0
                
                # 5. エージェント貢献度の計算（安全に）
                agent_contributions = {}
                if hasattr(self.opinion_space, 'get_agent_contribution_score'):
                    for i, agent in enumerate(agents):
                        agent_id = agent.get('id', i)
                        try:
                            contribution = self.opinion_space.get_agent_contribution_score(agent_id)
                            agent_contributions[agent_id] = contribution
                        except Exception as e:
                            if self.debug:
                                print(f"エージェント{agent_id}の貢献度計算エラー: {str(e)}")
                            agent_contributions[agent_id] = 0.5  # デフォルト値
            
            except Exception as e:
                if self.debug:
                    print(f"意見空間メトリクス計算エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # エラーが発生した場合はデフォルト診断に切り替え
                result = self._default_diagnosis(agents)
                self.last_diagnosis = result
                self.last_diagnosis_time = current_time
                return result
            
            # 診断ロジック
            
            # 1. 意見の多様性が足りない場合、新しいエージェントを追加
            diversity_score = diversity_metrics.get('mean_distance', 0.5)
            if diversity_score < self.diversity_threshold and num_agents < self.max_agents:
                # 多様性不足が連続して検出された場合のみ
                if self._check_repeated_diagnosis('low_diversity', 2):
                    result = {
                        'action': 'add',
                        'reason': '意見の多様性が不足しているため新しい視点を導入',
                        'properties': self._generate_diverse_agent_properties(agents),
                        'metrics': {
                            'diversity': diversity_metrics,
                            'change': change_metrics
                        },
                        'timestamp': current_time
                    }
                    self.last_diagnosis = result
                    self.last_diagnosis_time = current_time
                    return result
            
            # 2. 類似したエージェントが存在する場合、冗長なエージェントを削除
            try:
                distance_matrix = np.ones((len(agents), len(agents)))  # デフォルトは最大距離
                if hasattr(self.opinion_space, 'compute_distance_matrix'):
                    try:
                        distance_matrix = self.opinion_space.compute_distance_matrix()
                    except Exception as e:
                        if self.debug:
                            print(f"距離行列計算エラー: {str(e)}")
                
                if distance_matrix.size > 0 and num_agents > self.min_agents:
                    for i in range(len(agents)):
                        for j in range(i+1, len(agents)):
                            # エージェント間の距離が閾値より小さい（類似している）
                            if distance_matrix[i, j] < (1.0 - self.similarity_threshold):
                                # どちらを残すかを決定（貢献度が低い方を削除）
                                agent1_id = agents[i].get('id', i)
                                agent2_id = agents[j].get('id', j)
                                
                                agent1_score = agent_contributions.get(agent1_id, 0.5)
                                agent2_score = agent_contributions.get(agent2_id, 0.5)
                                
                                target_idx = i if agent1_score < agent2_score else j
                                target_id = agents[target_idx].get('id', target_idx)
                                
                                result = {
                                    'action': 'reduce',
                                    'reason': '類似のエージェントが存在し冗長',
                                    'target': target_id,
                                    'similarity_score': 1.0 - distance_matrix[i, j],
                                    'timestamp': current_time
                                }
                                self.last_diagnosis = result
                                self.last_diagnosis_time = current_time
                                return result
            except Exception as e:
                if self.debug:
                    print(f"類似エージェント検出エラー: {str(e)}")
            
            # 3. エージェントの意見が均等に分散しすぎている場合、収束を促すエージェントを追加
            diversity_score = diversity_metrics.get('mean_distance', 0.5)
            variance = diversity_metrics.get('variance', 0.1)
            if (diversity_score > 0.7 and variance > 0.15 and num_agents < self.max_agents):
                
                # 分散しすぎが連続して検出された場合のみ
                if self._check_repeated_diagnosis('high_dispersion', 2):
                    result = {
                        'action': 'add',
                        'reason': '意見が過度に分散しており収束が必要',
                        'properties': {
                            'role': '調停者',
                            'strategy': '論点整理',
                            'principle': '相違点を認めつつ共通基盤を見つける'
                        },
                        'metrics': {
                            'diversity': diversity_metrics,
                            'change': change_metrics
                        },
                        'timestamp': current_time
                    }
                    self.last_diagnosis = result
                    self.last_diagnosis_time = current_time
                    return result
            
            # 4. 議論が停滞している場合、エージェントの役割を変更
            centroid_movement = change_metrics.get('centroid_movement', 0.05)
            agent_movement = change_metrics.get('average_agent_movement', 0.05)
            
            # 停滞判定の閾値を調整（より保守的に）
            stagnation_threshold = 0.05  # 以前は0.1
            
            # もうひとつの判定基準として繰り返し検出回数を使用
            if (centroid_movement < stagnation_threshold and agent_movement < stagnation_threshold and len(agents) >= 2):
                
                # 停滞が連続検出回数を増加
                if 'stagnation' not in self.repeat_counter:
                    self.repeat_counter['stagnation'] = 0
                else:
                    self.repeat_counter['stagnation'] += 1
                    
                # 3回以上連続で検出された場合のみアクション（以前は2回）
                if self.repeat_counter['stagnation'] >= 3:
                    # カウンターをリセット
                    self.repeat_counter['stagnation'] = 0
                    
                    # 貢献度が最も低いエージェントを見つける
                    if agent_contributions:
                        target_id = min(agent_contributions.items(), key=lambda x: x[1])[0]
                    else:
                        # 貢献度情報がなければランダム選択
                        random_idx = random.randint(0, len(agents) - 1)
                        target_id = agents[random_idx].get('id', random_idx)
                    
                    # エージェントの役割などを特定
                    target_idx = -1
                    for i, agent in enumerate(agents):
                        if agent.get('id', i) == target_id:
                            target_idx = i
                            break
                    
                    if target_idx >= 0:
                        result = {
                            'action': 'modify',
                            'reason': '議論が停滞しているため役割変更',
                            'target': target_id,
                            'properties': self._generate_challenging_agent_properties(agents[target_idx]),
                            'metrics': {
                                'diversity': diversity_metrics,
                                'change': change_metrics
                            },
                            'timestamp': current_time
                        }
                        if self.debug:
                            print(f"議論が停滞しています - エージェント{target_id}の役割変更")
                        self.last_diagnosis = result
                        self.last_diagnosis_time = current_time
                        return result
                else:
                    # まだアクションを起こす段階ではないが、停滞の可能性を記録
                    if self.debug:
                        print(f"議論停滞の可能性を検出: {self.repeat_counter['stagnation']}/3回目")
            else:
                # 停滞条件が満たされない場合はカウンターをリセット
                self.repeat_counter['stagnation'] = 0
            
            # 5. 議論が十分収束し、かつエージェント数が最小値より多い場合はエージェント削減
            convergence = change_metrics.get('opinion_convergence', 0.0)
            diversity_score = diversity_metrics.get('mean_distance', 0.5)
            if (convergence > self.convergence_threshold and
                diversity_score < 0.3 and
                num_agents > self.min_agents):
                
                # 収束が連続して検出された場合のみ
                if self._check_repeated_diagnosis('convergence', 2):
                    target_details = self._select_agent_for_reduction_with_details(agents)
                    
                    result = {
                        'action': 'reduce',
                        'reason': '議論が十分に収束したため',
                        'target': target_details,
                        'metrics': {
                            'diversity': diversity_metrics,
                            'change': change_metrics
                        },
                        'timestamp': current_time
                    }
                    self.last_diagnosis = result
                    self.last_diagnosis_time = current_time
                    return result
            
            # 6. デフォルト：必要なアクションなし
            result = {
                'action': 'none',
                'reason': '議論は良好に進行中',
                'metrics': {
                    'diversity': diversity_metrics,
                    'change': change_metrics
                },
                'timestamp': current_time
            }
            
            # チャンスベースの介入（ランダム要素）
            if random.random() < 0.15 and num_agents < self.max_agents:  # 15%の確率で
                if self._check_repeated_diagnosis('none', 3):  # 何もしないが3回続いた場合
                    result = {
                        'action': 'add',
                        'reason': '議論を活性化するため定期的な刷新',
                        'properties': self._generate_diverse_agent_properties(agents, force_diverse=True),
                        'metrics': {
                            'diversity': diversity_metrics,
                            'change': change_metrics
                        },
                        'timestamp': current_time
                    }
            
            self.last_diagnosis = result
            self.last_diagnosis_time = current_time
            return result
        
        except Exception as e:
            # 予期せぬエラーが発生した場合
            if self.debug:
                print(f"診断処理中の予期せぬエラー: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # 安全なデフォルト診断を返す
            result = {
                'action': 'none',
                'reason': 'エラー発生のため診断を中断',
                'error': str(e),
                'timestamp': time.time()
            }
            self.last_diagnosis = result
            self.last_diagnosis_time = time.time()
            return result
            
        finally:
            # 必ず実行されるクリーンアップ
            current_thread._diagnosing = False
    
    def _default_diagnosis(self, agents: List[Dict]) -> Dict[str, Any]:
        """
        意見空間情報なしの基本診断ロジック
        
        引数:
            agents: アクティブなエージェントリスト
            
        戻り値:
            診断結果
        """
        num_agents = len(agents)
        
        # エージェント数の簡易チェック
        if num_agents > self.max_agents:
            return {
                'action': 'reduce',
                'reason': f'エージェント数が上限({self.max_agents})を超えています',
                'target': random.randint(0, num_agents - 1),
                'timestamp': time.time()
            }
            
        if num_agents < self.min_agents:
            return {
                'action': 'add',
                'reason': f'エージェント数が下限({self.min_agents})を下回っています',
                'properties': self._generate_diverse_agent_properties(agents),
                'timestamp': time.time()
            }
            
        # ランダムな介入（確率で決定）
        r = random.random()
        
        # 10%の確率でエージェントを追加
        if r < 0.1 and num_agents < self.max_agents:
            return {
                'action': 'add',
                'reason': '議論の活性化のため',
                'properties': self._generate_diverse_agent_properties(agents),
                'timestamp': time.time()
            }
        # 5%の確率でエージェント削除（最小数以上ある場合）
        elif r < 0.15 and num_agents > self.min_agents:
            return {
                'action': 'reduce',
                'reason': '議論の効率化のため',
                'target': random.randint(0, num_agents - 1),
                'timestamp': time.time()
            }
        # 10%の確率でエージェント修正
        elif r < 0.25 and num_agents > 0:
            target_idx = random.randint(0, num_agents - 1)
            return {
                'action': 'modify',
                'reason': '新しい視点の導入のため',
                'target': agents[target_idx].get('id', target_idx),
                'properties': self._generate_challenging_agent_properties(agents[target_idx]),
                'timestamp': time.time()
            }
        # 75%の確率で何もしない
        else:
            return {
                'action': 'none',
                'reason': '現状のエージェント構成で議論を継続',
                'timestamp': time.time()
            }
    
    def _generate_diverse_agent_properties(self, agents: List[Dict], force_diverse: bool = False) -> Dict[str, str]:
        """
        既存のエージェントと異なる特性を持つエージェント設定を生成
        
        引数:
            agents: 現在のエージェントリスト
            force_diverse: 強制的に大きく異なるエージェントを作成するフラグ
            
        戻り値:
            新エージェントの特性辞書
        """
        # 現在のエージェントの役割・戦略・原則をリストアップ
        current_roles = [agent.get('role', '') for agent in agents if 'role' in agent]
        current_strategies = [agent.get('strategy', '') for agent in agents if 'strategy' in agent]
        current_principles = [agent.get('principle', '') for agent in agents if 'principle' in agent]
        
        # 意見ベクトル空間があれば中心から最も遠い位置を計算
        if self.opinion_space:            try:
                # クラスタリング実行 - 修正済みメソッドを使用
                clusters = self.opinion_space.cluster_opinions()
                
                # 最大の意見クラスタを特定
                max_cluster = max(clusters, key=len) if clusters else None
                
                if max_cluster and force_diverse:
                    # 主流意見クラスタの中心
                    centroid = self.opinion_space.get_centroid(max_cluster)
                    
                    # 中心から最も遠い方向に新しいエージェントを設定
                    opposite_direction = -centroid
                    # 正規化
                    norm = np.linalg.norm(opposite_direction)
                    if norm > 0:
                        opposite_direction = opposite_direction / norm
                    
                    # 反対方向の特性を選択
                    # ここでは単純なマッピングだが、実際は複雑な処理が必要
                    if any('批判' in role for role in current_roles):
                        new_role = '協調者'
                    elif any('協調' in role for role in current_roles):
                        new_role = '批判者'
                    else:
                        new_role = '統合者'
                        
                    if any('分析' in strategy for strategy in current_strategies):
                        new_strategy = '直感的な思考'
                    elif any('直感' in strategy for strategy in current_strategies):
                        new_strategy = '論理的な分析'
                    else:
                        new_strategy = '多角的な視点'
                        
                    return {
                        'role': new_role,
                        'strategy': new_strategy,
                        'principle': '既存の枠組みにとらわれない自由な発想'
                    }
            except Exception as e:
                if self.debug:
                    print(f"意見空間分析でのエージェント生成に失敗: {str(e)}")
                # 失敗した場合は以下の基本処理で続行
        
        # 基本的な役割候補
        role_options = [
            '質問者', '批判者', '協調者', '統合者', '提案者', 
            '分析者', '創造者', '調停者', '弁護者', '実用主義者'
        ]
        
        # 基本的な戦略候補
        strategy_options = [
            '論理的な分析', '直感的な思考', '多角的な視点', '事実の検証', 
            '感情的な共感', '専門的な知識の提供', '問題の再定義', 
            '経験に基づく判断', '比喩による説明', '未来予測'
        ]
        
        # 基本的な原則候補
        principle_options = [
            '客観性を重視', '多様性を尊重', '明確な表現を心がける', 
            '過去の事例から学ぶ', '普遍的な原則を探求', '矛盾点を指摘', 
            '共通基盤を見つける', '具体例で説明する', 
            '想定外の可能性を考慮', '最善の解決策を模索'
        ]
        
        # 既存のエージェントと異なる特性を選択
        new_role = self._select_different_option(role_options, current_roles)
        new_strategy = self._select_different_option(strategy_options, current_strategies)
        new_principle = self._select_different_option(principle_options, current_principles)
        
        return {
            'role': new_role,
            'strategy': new_strategy,
            'principle': new_principle
        }
    
    def _generate_challenging_agent_properties(self, current_agent: Dict) -> Dict[str, str]:
        """
        現在のエージェントに対する挑戦的な特性を持つ設定を生成
        
        引数:
            current_agent: 現在のエージェント情報
            
        戻り値:
            変更後のエージェント特性
        """
        current_role = current_agent.get('role', '')
        current_strategy = current_agent.get('strategy', '')
        
        # 現在の役割に対する挑戦的な役割を設定
        if '批判' in current_role:
            new_role = '協調者'
        elif '協調' in current_role:
            new_role = '批判者'
        elif '質問' in current_role:
            new_role = '回答者'
        elif '分析' in current_role:
            new_role = '直感思考者'
        elif '統合' in current_role:
            new_role = '対立点発見者'
        else:
            # その他の場合はランダムに変更
            role_options = ['批判者', '協調者', '統合者', '提案者', '創造者']
            new_role = random.choice(role_options)
        
        # 現在の戦略に対する挑戦的な戦略を設定
        if '論理' in current_strategy:
            new_strategy = '直感的な思考'
        elif '直感' in current_strategy:
            new_strategy = '論理的な分析'
        elif '多角' in current_strategy:
            new_strategy = '一点突破的な深掘り'
        elif '検証' in current_strategy:
            new_strategy = '想像力を活かした展開'
        else:
            # その他の場合はランダムに変更
            strategy_options = [
                '論理的な分析', '直感的な思考', '批判的思考', 
                '創造的アイディア', '事例ベースの説明'
            ]
            new_strategy = random.choice(strategy_options)
        
        # 原則は適宜調整
        principle_options = [
            '既存の枠組みにとらわれない自由な発想',
            '異なる視点からの再考を促す',
            '隠れた前提を明らかにする',
            '対話の流れを変える',
            '議論の進展を促す新しい観点を提供する'
        ]
        new_principle = random.choice(principle_options)
        
        return {
            'role': new_role,
            'strategy': new_strategy,
            'principle': new_principle
        }
    def _select_agent_for_reduction_with_details(self, agents: List[Dict]) -> Dict:
        """
        削減すべきエージェントを選択し、詳細な理由を返す（内部メソッド）
        
        引数:
            agents: エージェントリスト
            
        戻り値:
            削減すべきエージェント情報（詳細な理由を含む）
        """
        if not agents:
            return None
            
        # 意見ベクトル空間がある場合、類似度の高いエージェントペアを特定
        if self.opinion_space:
            try:
                # 最新の意見空間分析を取得
                space_analysis = self.opinion_space.analyze_opinion_space()
                
                if space_analysis.get('valid', False) and 'distances' in space_analysis:
                    # 距離行列を取得
                    distances = np.array(space_analysis['distances'])
                    
                    if distances.size > 0:
                        n = distances.shape[0]
                        
                        # 最も類似度が高い（距離が小さい）ペアを見つける
                        min_dist = float('inf')
                        similar_pair = None
                        
                        for i in range(n):
                            for j in range(i+1, n):
                                if distances[i, j] < min_dist and distances[i, j] < self.similarity_threshold:
                                    min_dist = distances[i, j]
                                    similar_pair = (i, j)
                        
                        if similar_pair:
                            i, j = similar_pair
                            # ペアのうちいずれかを選択（より貢献度が低いほう）
                            agent_i = agents[i] if i < len(agents) else None
                            agent_j = agents[j] if j < len(agents) else None
                            
                            if agent_i and agent_j:
                                # 貢献度や生存期間で比較
                                target = agent_i if (agent_i.get('contribution_score', 0) < 
                                                   agent_j.get('contribution_score', 0)) else agent_j
                                return {
                                    'agent_id': target.get('id'),
                                    'reason': '他のエージェントと意見が類似しており冗長です'
                                }
            except Exception as e:
                if self.debug:
                    print(f"エージェント選択エラー: {e}")
        
        # 類似性による選択ができない場合は、最も古いエージェントを選択
        oldest_agent = None
        oldest_time = float('inf')
        
        for agent in agents:
            created_at = agent.get('created_at', float('inf'))
            if created_at < oldest_time:
                oldest_time = created_at
                oldest_agent = agent
                
        if oldest_agent:
            return {
                'agent_id': oldest_agent.get('id'),
                'reason': '最も長く議論に参加している古参エージェントです'
            }
            
        # それもダメならランダム選択
        target = random.choice(agents)
        return {
            'agent_id': target.get('id'),
            'reason': 'リソース削減のためランダムに選択されました'
        }
    
    def _select_role_for_balanced_team(self) -> str:
        """
        バランスの取れたチーム構成のための役割を選択（内部メソッド）
        
        戻り値:
            推奨される役割
        """
        # 基本的な役割セットからランダム選択
        roles = ['アナリスト', '質問者', '批判者', '総合者', '詳細化']
        return random.choice(roles)
    
    def _select_agent_for_reduction(self, agents: List[Dict]) -> Union[int, str]:
        """
        削除対象のエージェントを選択
        
        引数:
            agents: アクティブなエージェントリスト
            
        戻り値:
            削除対象のエージェントID
        """
        # 意見空間があれば貢献度に基づいて選択
        if self.opinion_space:
            agent_contributions = {}
            for i, agent in enumerate(agents):
                agent_id = agent.get('id', i)
                agent_contributions[agent_id] = self.opinion_space.get_agent_contribution_score(agent_id)
                
            # 貢献度が最も低いエージェントを選択
            if agent_contributions:
                target_id = min(agent_contributions.items(), key=lambda x: x[1])[0]
                return target_id
        
        # 意見空間がなければランダム選択
        num_agents = len(agents)
        if num_agents == 0:
            return 0
        
        target_idx = random.randint(0, num_agents - 1)
        return agents[target_idx].get('id', target_idx)
    
    def _select_different_option(self, options: List[str], current_values: List[str]) -> str:
        """
        既存の値と異なるオプションを選択
        
        引数:
            options: 選択肢リスト
            current_values: 既存の値リスト
            
        戻り値:
            選択したオプション
        """
        # 既存の値と重複しないオプションをフィルタ
        available_options = [opt for opt in options if not any(opt in val for val in current_values)]
        
        # 選択肢がなければすべてから選択
        if not available_options:
            available_options = options
            
        # ランダム選択
        return random.choice(available_options)
    
    def _check_repeated_diagnosis(self, diagnosis_type: str, required_repeats: int) -> bool:
        """
        特定の診断が連続して発生しているかチェック
        
        引数:
            diagnosis_type: 診断タイプ
            required_repeats: 必要な連続回数
            
        戻り値:
            条件を満たしていればTrue
        """
        # カウンタ初期化
        if diagnosis_type not in self.repeat_counter:
            self.repeat_counter[diagnosis_type] = 1
            return False
            
        # 現在の診断と前回の診断タイプが一致
        if self.last_diagnosis and self.last_diagnosis.get('action') == diagnosis_type:
            self.repeat_counter[diagnosis_type] += 1
        else:
            # 一致しなければリセット
            self.repeat_counter[diagnosis_type] = 1
            
        # 他の診断カウンタをリセット
        for k in self.repeat_counter.keys():
            if k != diagnosis_type:
                self.repeat_counter[k] = 0
                
        # 条件チェック
        return self.repeat_counter[diagnosis_type] >= required_repeats

    def _analyze_discussion_state(self, agents: List[Dict]) -> Dict[str, Any]:
        """
        議論状態の包括的な分析を行う（内部メソッド）
        
        引数:
            agents: エージェントリスト
            
        戻り値:
            分析結果
        """
        # 意見空間の分析
        if self.opinion_space:
            try:
                # 最新の意見空間分析を取得
                space_analysis = self.opinion_space.analyze_opinion_space()
                
                if space_analysis.get('valid', False):
                    # 多様性・収束性を取得
                    diversity = space_analysis.get('diversity', 0)
                    convergence = space_analysis.get('convergence', 0) 
                    clusters = space_analysis.get('clusters', 1)
                    max_distance = space_analysis.get('max_distance', 0)
                    
                    # 多様性フラグ
                    low_diversity = diversity < self.diversity_threshold
                    high_convergence = convergence > self.convergence_threshold
                    
                    # 多クラスタフラグ（意見が割れている状態）
                    opinion_divergence = clusters >= 2 and max_distance > 0.5
                    
                    # 停滞検知
                    stagnation = False
                    
                    return {
                        'space_valid': True,
                        'diversity': diversity,
                        'convergence': convergence,
                        'clusters': clusters,
                        'max_distance': max_distance,
                        'low_diversity': low_diversity,
                        'high_convergence': high_convergence,
                        'opinion_divergence': opinion_divergence,
                        'stagnation': stagnation
                    }
            except Exception as e:
                if self.debug:
                    print(f"意見空間分析エラー: {e}")
        
        # 意見空間分析ができない場合はデフォルト値
        return {
            'space_valid': False,
            'diversity': 0.5,
            'convergence': 0.5,
            'clusters': 1,            'max_distance': 0.0,
            'low_diversity': False,
            'high_convergence': False,
            'opinion_divergence': False,
            'stagnation': False
        }
        
    def _determine_recommended_action(self, analysis: Dict[str, Any], agents: List[Dict]) -> Dict[str, Any]:
        """
        分析結果に基づいて推奨アクションを決定する（内部メソッド）
        
        引数:
            analysis: 議論状態分析結果
            agents: エージェントリスト
            
        戻り値:
            推奨アクションを含む辞書
        """
        # 分析が有効でない場合はデフォルト推奨（何もしない）
        if not analysis.get('space_valid', False):
            return {
                'action': 'none',
                'reason': '十分な情報がなく診断できません'
            }
        
        num_agents = len(agents)
        
        # 多様性が低い場合は新しいエージェントを追加
        if analysis.get('low_diversity', False) and num_agents < self.max_agents:
            return {
                'action': 'add',
                'reason': '意見の多様性が不足しています'
            }
        
        # 収束性が高い場合はエージェントを削減
        if analysis.get('high_convergence', False) and num_agents > self.min_agents:
            return {
                'action': 'reduce',
                'reason': '意見が十分に収束しています'
            }
        
        # 意見が分散しすぎている場合は調停エージェントを追加
        if analysis.get('opinion_divergence', False) and num_agents < self.max_agents:
            return {
                'action': 'add_mediator',
                'reason': '意見が過度に分散しています'
            }
        
        # 停滞している場合は反対意見エージェントを追加
        if analysis.get('stagnation', False):
            if num_agents < self.max_agents:
                return {
                    'action': 'add_contrarian',
                    'reason': '議論が停滞しています'
                }
            else:
                # エージェント数上限に達している場合は既存エージェントを修正
                return {
                    'action': 'modify',
                    'reason': '議論が停滞しています（エージェント数上限）'
                }
        
        # 特に問題がなければ何もしない
        return {
            'action': 'none',
            'reason': '議論は良好に進行中です'
        }