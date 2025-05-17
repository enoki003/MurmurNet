#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpinionSpaceManager モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~
エージェントの意見ベクトル空間を管理し、多様性・収束性を分析する

作者: Yuhi Sonoki
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import time
from collections import deque
import threading
import traceback

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class OpinionSpaceManager:
    """
    エージェントの意見ベクトル空間を管理するクラス
    - 意見のベクトル表現を蓄積・管理
    - クラスタリングによる意見グループの検出
    - 意見空間の多様性・収束性の分析
    """
    
    def __init__(self, config: Dict[str, Any], vectorizer):
        """
        OpinionSpaceManagerの初期化
        
        引数:
            config: 設定辞書
            vectorizer: 意見をベクトル化するためのモジュール
        """
        self.config = config
        self.debug = config.get('debug', False)
        self.vectorizer = vectorizer
        self.dimension = config.get('vector_dim', 384)  # ベクトル次元数
        
        # 意見ベクトル空間の初期化
        self.vectors = []  # ベクトルデータ
        self.metadata = []  # 各ベクトルのメタデータ
        self.opinions = []  # 元の意見テキスト
        
        # 履歴管理
        self.max_history = config.get('history_size', 50)  # 保持する最大履歴数
        self.centroid_history = deque(maxlen=self.max_history)  # 中心点の履歴
        self.diversity_history = deque(maxlen=self.max_history)  # 多様性の履歴
        
        # ターン管理
        self.current_turn = 0
        self.turn_boundaries = [0]  # 各ターンの境界インデックス
        
        # クラスタリング結果
        self.clusters = None
        self.cluster_centers = None
        self.silhouette = -1.0
        self.optimal_k = 1
        
        # 再帰呼び出し防止用フラグ
        self.is_processing = False
        self.is_adding_vector = False
        
        # スレッドローカル用のIDセット
        self._agent_id_processing_set = set()
        
        if self.debug:
            print("OpinionSpaceManager: 初期化完了")
    
    def add_opinion(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        意見テキストをベクトル化して空間に追加
        
        引数:
            text: 意見テキスト
            metadata: 意見に関するメタデータ（エージェントID、ターンなど）
            
        戻り値:
            bool: 追加に成功したかどうか
        """
        if not text:
            return False
            
        # 再帰呼び出し防止
        if self.is_processing:
            if self.debug:
                print("警告: add_opinion の再帰呼び出しを検出。処理をスキップします。")
            return False
        
        self.is_processing = True
            
        try:
            # テキストをベクトル化
            vector = self.vectorizer.vectorize(text)
            if vector is None or len(vector) != self.dimension:
                if self.debug:
                    print(f"ベクトル化失敗: 無効なベクトル次元 {len(vector) if vector is not None else 'None'}")
                return False
                
            # メタデータにタイムスタンプを追加
            metadata = metadata or {}
            if 'timestamp' not in metadata:
                metadata['timestamp'] = int(time.time())
            
            # ターン情報を設定
            if 'turn' not in metadata:
                metadata['turn'] = self.current_turn
                
            # データを追加
            self.vectors.append(vector)
            self.metadata.append(metadata)
            self.opinions.append(text)
            
            if self.debug:
                print(f"意見をベクトル空間に追加: {text[:30]}... (ターン: {metadata.get('turn', 'N/A')})")
                
            return True
            
        except Exception as e:
            if self.debug:
                print(f"意見追加エラー: {str(e)}")
            return False
        finally:
            # 処理完了フラグを戻す
            self.is_processing = False
    
    def add_vector(self, agent_id, vector: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        意見ベクトルを直接空間に追加
        
        引数:
            agent_id: エージェントID
            vector: 意見ベクトル
            metadata: 意見に関するメタデータ
            
        戻り値:
            bool: 追加に成功したかどうか
        """
        # 再帰呼び出し防止（スレッドセーフ処理）
        
        # スレッドローカルストレージを初期化
        if not hasattr(threading.current_thread(), '_opinion_processing_set'):
            threading.current_thread()._opinion_processing_set = set()
        
        # エージェントIDを文字列化
        agent_id_str = str(agent_id)
        
        # 既に処理中のエージェントIDかチェック
        if agent_id_str in threading.current_thread()._opinion_processing_set:
            if self.debug:
                print(f"警告: add_vector の再帰呼び出しを検出 (agent_id={agent_id_str})。処理をスキップします。")
            return False
        
        # 処理中セットに追加
        threading.current_thread()._opinion_processing_set.add(agent_id_str)
        
        try:
            # ベクトルの検証
            if vector is None:
                if self.debug:
                    print("ベクトル追加失敗: ベクトルがNoneです")
                return False
            
            # 次元のチェックとリサイズ
            vector_dim = len(vector)
            if vector_dim != self.dimension:
                if self.debug:
                    print(f"ベクトル次元不一致: {vector_dim} (期待値: {self.dimension})")
                # リサイズを試みる
                try:
                    if vector_dim > self.dimension:
                        # 切り捨て
                        vector = vector[:self.dimension]
                    else:
                        # ゼロパディング
                        padded = np.zeros(self.dimension)
                        padded[:vector_dim] = vector
                        vector = padded
                except Exception as resize_err:
                    if self.debug:
                        print(f"ベクトルリサイズ失敗: {str(resize_err)}")
                    return False
                
            # メタデータにタイムスタンプを追加
            metadata = metadata or {}
            if 'timestamp' not in metadata:
                metadata['timestamp'] = int(time.time())
            
            # エージェントIDを追加
            if 'agent_id' not in metadata:
                metadata['agent_id'] = agent_id
            
            # ターン情報を設定
            if 'turn' not in metadata:
                metadata['turn'] = self.current_turn
                
            # データを追加
            self.vectors.append(vector)
            self.metadata.append(metadata)
            # 意見テキストフィールドには簡易情報を設定
            self.opinions.append(f"Agent {agent_id} vector ({metadata.get('text', 'no text')})")
            
            if self.debug:
                print(f"ベクトルを空間に追加: エージェントID={agent_id} (ターン: {metadata.get('turn', 'N/A')})")
                
            return True
            
        except Exception as e:
            if self.debug:
                print(f"ベクトル追加エラー: {str(e)}")
                traceback.print_exc()
            return False
        finally:
            # 必ず実行されるクリーンアップ処理
            # 処理中セットから削除
            threading.current_thread()._opinion_processing_set.discard(agent_id_str)
    
    def next_turn(self) -> None:
        """新しいターンに移行"""
        self.current_turn += 1
        # 現在の意見数を境界として記録
        self.turn_boundaries.append(len(self.vectors))
        
        if self.debug:
            print(f"ターン {self.current_turn} に移行 (累積意見数: {len(self.vectors)})")
    
    def get_turn_opinions(self, turn: int = None) -> List[Tuple[str, Dict[str, Any], np.ndarray]]:
        """
        指定したターンの意見を取得
        
        引数:
            turn: ターン番号（Noneなら現在のターン）
            
        戻り値:
            List[Tuple[str, Dict, np.ndarray]]: (意見テキスト, メタデータ, ベクトル)のリスト
        """
        if turn is None:
            turn = self.current_turn
            
        if turn < 0 or turn >= len(self.turn_boundaries):
            return []
            
        start_idx = self.turn_boundaries[turn]
        end_idx = self.turn_boundaries[turn + 1] if turn + 1 < len(self.turn_boundaries) else len(self.vectors)
        
        result = []
        for i in range(start_idx, end_idx):
            result.append((self.opinions[i], self.metadata[i], self.vectors[i]))
            
        return result
        
    def cluster_opinions(self, n_clusters: int = None, max_iter: int = 300, n_init: int = 10) -> Dict[str, Any]:
        """
        意見ベクトル空間をクラスタリング
        
        引数:
            n_clusters: クラスタ数（Noneなら最適なクラスタ数を自動決定）
            max_iter: KMeansの最大反復回数
            n_init: KMeansの初期化回数
            
        戻り値:
            Dict[str, Any]: クラスタリング結果
        """
        # 再帰呼び出し防止
        if getattr(self, '_is_clustering', False):
            if self.debug:
                print("警告: cluster_opinionsの再帰呼び出しを検出。処理をスキップします。")
            return {
                'clusters': [],
                'centers': [],
                'silhouette': -1.0,
                'k': 1
            }
            
        self._is_clustering = True
        
        try:
            if len(self.vectors) < 2:
                # 意見が不足している場合
                self.clusters = np.zeros(len(self.vectors), dtype=int)
                self.cluster_centers = np.zeros((1, self.dimension)) if len(self.vectors) > 0 else np.array([])
                self.silhouette = -1.0
                self.optimal_k = 1
                
                return {
                    'clusters': self.clusters.tolist() if len(self.vectors) > 0 else [],
                    'centers': self.cluster_centers.tolist() if len(self.vectors) > 0 else [],
                    'silhouette': -1.0,
                    'k': 1
                }
            
            vectors = np.array(self.vectors)
            
            # 最適なクラスタ数を決定（n_clustersが指定されていない場合）
            k = n_clusters
            if k is None:
                k = self._find_optimal_k(vectors)
            
            # クラスタ数は少なくとも1、最大でもデータ点の数
            k = max(1, min(k, len(vectors) - 1))
            
            # クラスタリング実行
            start_time = time.time()
            
            # scikit-learn 1.3+との互換性のため、init_と_n_initの両方をサポート
            kmeans_params = {'n_clusters': k, 'random_state': 42, 'max_iter': max_iter}
            if hasattr(KMeans, 'n_init'):
                kmeans_params['n_init'] = n_init
            elif hasattr(KMeans, '_n_init'):
                kmeans_params['_n_init'] = n_init
            else:
                # n_initを指定しない（デフォルト値を使用）
                pass
                
            kmeans = KMeans(**kmeans_params)
            self.clusters = kmeans.fit_predict(vectors)
            self.cluster_centers = kmeans.cluster_centers_
            
            # シルエットスコアの計算（クラスタが2つ以上ある場合のみ）
            if k > 1 and len(vectors) > k:
                self.silhouette = silhouette_score(vectors, self.clusters)
            else:
                self.silhouette = -1.0  # 意味のあるシルエットスコアを計算できない
                
            # 結果を保存
            self.optimal_k = k
            
            # 詳細デバッグが有効な場合のみログ出力
            if self.debug and self.config.get('verbose_logging', False):
                print(f"意見クラスタリング完了: {k}クラスタ, シルエットスコア={self.silhouette:.3f}, 処理時間={time.time()-start_time:.3f}秒")
            
            # 結果を返す
            return {
                'clusters': self.clusters.tolist(),
                'centers': self.cluster_centers.tolist(),
                'silhouette': self.silhouette,
                'k': k
            }
            
        except Exception as e:
            if self.debug:
                print(f"クラスタリングエラー: {str(e)}")
                traceback.print_exc()
                
            # エラー発生時は空結果を返す
            self.clusters = np.zeros(len(self.vectors), dtype=int)
            self.cluster_centers = np.zeros((1, self.dimension))
            self.silhouette = -1.0
            self.optimal_k = 1
            
            return {
                'clusters': self.clusters.tolist(),
                'centers': self.cluster_centers.tolist(),
                'silhouette': -1.0,
                'k': 1
            }
        finally:
            # 必ず実行処理
            self._is_clustering = False

    def _find_optimal_k(self, vectors: np.ndarray, n_init: int = 5, max_iter: int = 200) -> int:
        """
        最適なクラスタ数を決定（内部メソッド）
        
        引数:
            vectors: 意見ベクトルの配列
            n_init: KMeansの初期化回数
            max_iter: KMeansの最大反復回数
            
        戻り値:
            int: 最適なクラスタ数
        """
        # データセットが小さい場合は少数のクラスタを使用
        n_samples = len(vectors)
        if n_samples <= 3:
            return 1
        elif n_samples <= 10:
            return min(2, n_samples - 1)
            
        # 検討するクラスタ数の範囲を決定
        max_k = min(8, n_samples - 1)  # クラスタ数の上限（最大8）
        range_k = range(2, max_k + 1)  # 検討するクラスタ数の範囲
        
        # 各クラスタ数でのシルエットスコアを計算
        silhouette_scores = []
        for k in range_k:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init, max_iter=max_iter)
                clusters = kmeans.fit_predict(vectors)
                score = silhouette_score(vectors, clusters)
                silhouette_scores.append((k, score))
            except Exception:
                # エラーが発生した場合は低スコアを割り当て
                silhouette_scores.append((k, -1.0))
        
        # スコアでソート（降順）
        silhouette_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 最良のクラスタ数を選択
        best_k, best_score = silhouette_scores[0]
        if self.debug:
            print(f"最適なクラスタ数を判定: {best_k} (シルエットスコア={best_score:.3f})")
            
        return best_k
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        意見空間の特性を計算
        - 多様性
        - 収束性
        - クラスタ構造
        
        戻り値:
            Dict[str, Any]: 各種メトリクス
        """
        # 意見が不足している場合
        if len(self.vectors) < 2:
            return {
                'diversity': 0.0,
                'convergence': 1.0,
                'centroid': [0.0] * self.dimension,
                'clusters': 1,
                'silhouette': -1.0,
                'outliers': 0
            }
            
        try:
            vectors = np.array(self.vectors)
            
            # 中心点の計算
            centroid = np.mean(vectors, axis=0)
            
            # 多様性の計算（標準偏差の平均）
            diversity = np.mean(np.std(vectors, axis=0))
            
            # クラスタリングの実行（まだ実行されていなければ）
            if self.clusters is None or self.cluster_centers is None:
                self.cluster_opinions()
                
            # 収束度の計算（中心からの平均距離の逆数）
            distances = np.linalg.norm(vectors - centroid, axis=1)
            mean_distance = np.mean(distances)
            convergence = 1.0 / (1.0 + mean_distance)  # 0〜1に正規化
            
            # 外れ値の検出
            threshold = np.mean(distances) + 1.5 * np.std(distances)
            outliers = np.sum(distances > threshold)
            
            # 履歴の更新
            self.centroid_history.append(centroid)
            self.diversity_history.append(diversity)
            
            result = {
                'diversity': diversity,
                'convergence': convergence,
                'centroid': centroid.tolist(),
                'clusters': self.optimal_k,
                'silhouette': self.silhouette,
                'outliers': int(outliers)
            }
            
            if self.debug:
                print(f"意見空間メトリクス: 多様性={diversity:.3f}, 収束度={convergence:.3f}, クラスタ数={self.optimal_k}")
                
            return result
            
        except Exception as e:
            if self.debug:
                print(f"メトリクス計算エラー: {str(e)}")
                traceback.print_exc()
                
            # エラー発生時はデフォルト値を返す
            return {
                'diversity': 0.0,
                'convergence': 0.0,
                'centroid': [0.0] * self.dimension,
                'clusters': 1,
                'silhouette': -1.0,
                'outliers': 0
            }
    
    def calculate_convergence_trend(self) -> Dict[str, Any]:
        """
        収束トレンドの計算
        - 中心点の移動
        - 多様性の変化
        
        戻り値:
            Dict[str, Any]: トレンド情報
        """
        # 履歴が不足している場合
        if len(self.centroid_history) < 2:
            return {
                'centroid_movement': 0.0,
                'diversity_change': 0.0,
                'is_converging': True
            }
            
        try:
            # 中心点の移動距離
            centroids = np.array(list(self.centroid_history))
            centroid_diffs = centroids[1:] - centroids[:-1]
            centroid_movement = np.mean([np.linalg.norm(diff) for diff in centroid_diffs])
            
            # 多様性の変化率
            diversity_values = np.array(list(self.diversity_history))
            if len(diversity_values) >= 2:
                recent_diversity_change = (diversity_values[-1] - diversity_values[-2]) / max(0.0001, diversity_values[-2])
                overall_diversity_change = (diversity_values[-1] - diversity_values[0]) / max(0.0001, diversity_values[0])
            else:
                recent_diversity_change = 0.0
                overall_diversity_change = 0.0
                
            # 収束傾向の判定
            is_converging = centroid_movement < 0.1 and recent_diversity_change < 0.0
            
            result = {
                'centroid_movement': float(centroid_movement),
                'recent_diversity_change': float(recent_diversity_change),
                'overall_diversity_change': float(overall_diversity_change),
                'is_converging': bool(is_converging)
            }
            
            if self.debug:
                print(f"収束トレンド: 中心移動={centroid_movement:.3f}, 多様性変化={recent_diversity_change:.3f}, "
                      f"収束中={is_converging}")
                
            return result
            
        except Exception as e:
            if self.debug:
                print(f"収束トレンド計算エラー: {str(e)}")
                traceback.print_exc()
                
            # エラー発生時はデフォルト値を返す
            return {
                'centroid_movement': 0.0,
                'diversity_change': 0.0,
                'is_converging': False
            }
    
    def find_outlier_opinions(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        """
        外れ値となっている意見を検出
        
        戻り値:
            List[Tuple[int, str, Dict[str, Any]]]: (インデックス, 意見テキスト, メタデータ)のリスト
        """
        if len(self.vectors) < 3:
            return []
            
        try:
            vectors = np.array(self.vectors)
            
            # 中心点の計算
            centroid = np.mean(vectors, axis=0)
            
            # 各ベクトルと中心点の距離
            distances = np.linalg.norm(vectors - centroid, axis=1)
            
            # 外れ値の閾値を設定
            threshold = np.mean(distances) + 1.5 * np.std(distances)
            
            # 外れ値のインデックスを取得
            outlier_indices = np.where(distances > threshold)[0]
            
            # 外れ値の意見を収集
            outliers = []
            for idx in outlier_indices:
                outliers.append((int(idx), self.opinions[idx], self.metadata[idx]))
                
            if self.debug and outliers:
                print(f"{len(outliers)}件の外れ値意見を検出")
                
            return outliers
            
        except Exception as e:
            if self.debug:
                print(f"外れ値検出エラー: {str(e)}")
            return []
    
    def get_cluster_representatives(self) -> List[Tuple[int, str]]:
        """
        各クラスタの代表的な意見を取得
        
        戻り値:
            List[Tuple[int, str]]: (クラスタID, 代表意見)のリスト
        """
        if len(self.vectors) < 2 or self.clusters is None or self.cluster_centers is None:
            if len(self.opinions) > 0:
                return [(0, self.opinions[0])]
            return []
            
        try:
            vectors = np.array(self.vectors)
            
            # クラスタごとの代表意見を見つける
            representatives = []
            for cluster_id in range(self.optimal_k):
                # クラスタに属する意見のインデックスを取得
                indices = np.where(self.clusters == cluster_id)[0]
                
                if len(indices) == 0:
                    continue
                    
                # クラスタ中心
                center = self.cluster_centers[cluster_id]
                
                # 中心に最も近い意見を探す
                distances = np.linalg.norm(vectors[indices] - center, axis=1)
                closest_idx = indices[np.argmin(distances)]
                
                representatives.append((cluster_id, self.opinions[closest_idx]))
                
            return representatives
            
        except Exception as e:
            if self.debug:
                print(f"クラスタ代表取得エラー: {str(e)}")
            return []
    
    def get_opinion_distances(self, query_vector: np.ndarray) -> List[Tuple[int, float]]:
        """
        指定したベクトルと全意見の距離を計算
        
        引数:
            query_vector: 検索クエリベクトル
            
        戻り値:
            List[Tuple[int, float]]: (インデックス, 距離)のリスト（距離の昇順）
        """
        if len(self.vectors) == 0:
            return []
            
        try:
            vectors = np.array(self.vectors)
            
            # 各ベクトルとクエリベクトルの距離を計算
            distances = np.linalg.norm(vectors - query_vector, axis=1)
            
            # インデックスと距離をペアにしてソート
            pairs = [(i, float(d)) for i, d in enumerate(distances)]
            pairs.sort(key=lambda x: x[1])
            
            return pairs
            
        except Exception as e:
            if self.debug:
                print(f"意見距離計算エラー: {str(e)}")
            return []
    
    def export_data(self) -> Dict[str, Any]:
        """
        意見空間データをエクスポート
        
        戻り値:
            Dict[str, Any]: エクスポートされたデータ
        """
        metrics = self.calculate_metrics()
        trend = self.calculate_convergence_trend()
        
        result = {
            'metrics': metrics,
            'trend': trend,
            'opinion_count': len(self.vectors),
            'turn': self.current_turn,
            'turn_boundaries': self.turn_boundaries,
            'latest_opinions': []
        }
        
        # 最新のターンの意見を追加
        latest_opinions = self.get_turn_opinions(self.current_turn)
        for i, (text, meta, _) in enumerate(latest_opinions):
            result['latest_opinions'].append({
                'index': i,
                'text': text,
                'agent_id': meta.get('agent_id'),
                'role': meta.get('role', 'unknown')
            })
            
        # クラスタ代表も追加
        result['cluster_representatives'] = [
            {'cluster': c, 'text': t} 
            for c, t in self.get_cluster_representatives()
        ]
        
        return result
    
    def get_latest_vectors(self) -> Dict[str, np.ndarray]:
        """
        各エージェントの最新ベクトルを取得
        
        戻り値:
            Dict[str, np.ndarray]: エージェントIDとベクトルのマッピング
        """
        agent_latest_vectors = {}
        
        try:
            # メタデータを逆順に走査して各エージェントの最新ベクトルを見つける
            for i in range(len(self.metadata) - 1, -1, -1):
                meta = self.metadata[i]
                agent_id = meta.get('agent_id')
                
                if agent_id and agent_id not in agent_latest_vectors:
                    agent_latest_vectors[str(agent_id)] = self.vectors[i]
                    
            return agent_latest_vectors
        
        except Exception as e:
            if self.debug:
                print(f"最新ベクトル取得エラー: {str(e)}")
            return {}
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        2つのベクトル間の類似度を計算
        
        引数:
            vec1: 1つ目のベクトル
            vec2: 2つ目のベクトル
            
        戻り値:
            float: コサイン類似度（0～1）
        """
        try:
            # コサイン類似度を計算
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # ゼロベクトルチェック
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
        
        except Exception as e:
            if self.debug:
                print(f"類似度計算エラー: {str(e)}")
            return 0.0
    
    def has_redundant_vectors(self, threshold: float = 0.9) -> bool:
        """
        冗長なベクトルが存在するかチェック
        
        引数:
            threshold: 類似度閾値（この値以上で冗長と判断）
            
        戻り値:
            bool: 冗長ベクトルが存在するか
        """
        try:
            latest_vectors = self.get_latest_vectors()
            agent_ids = list(latest_vectors.keys())
            
            # エージェントが少なすぎる場合は冗長性なしと判断
            if len(agent_ids) < 2:
                return False
                
            # 類似度の高いペアを探す
            for i in range(len(agent_ids)):
                for j in range(i+1, len(agent_ids)):
                    agent_id1 = agent_ids[i]
                    agent_id2 = agent_ids[j]
                    
                    vec1 = latest_vectors[agent_id1]
                    vec2 = latest_vectors[agent_id2]
                    
                    sim = self.calculate_similarity(vec1, vec2)
                    if sim >= threshold:
                        if self.debug:
                            print(f"冗長ベクトル検出: エージェント{agent_id1}とエージェント{agent_id2} (類似度={sim:.3f})")
                        return True
            
            return False
            
        except Exception as e:
            if self.debug:
                print(f"冗長ベクトル検出エラー: {str(e)}")
            return False
    
    def calculate_distance_metrics(self) -> Tuple[float, float]:
        """
        最新のエージェントベクトル間の距離メトリクスを計算
        
        戻り値:
            Tuple[float, float]: (平均距離, 最大距離)
        """
        try:
            latest_vectors = self.get_latest_vectors()
            vectors = np.array(list(latest_vectors.values()))
            
            if len(vectors) < 2:
                return 0.0, 0.0
                
            # ペアごとの距離を計算
            distances = []
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    dist = 1.0 - self.calculate_similarity(vectors[i], vectors[j])
                    distances.append(dist)
                    
            avg_distance = np.mean(distances) if distances else 0.0
            max_distance = np.max(distances) if distances else 0.0
            
            return float(avg_distance), float(max_distance)
            
        except Exception as e:
            if self.debug:
                print(f"距離メトリクス計算エラー: {str(e)}")
            return 0.0, 0.0
    
    def calculate_centroid_movement(self) -> float:
        """
        前回ターンからの中心点の移動距離を計算
        
        戻り値:
            float: 中心点の移動距離
        """
        try:
            if len(self.centroid_history) < 2:
                return 0.0
                
            # 最新の2つの中心点を取得
            prev_centroid = self.centroid_history[-2]
            curr_centroid = self.centroid_history[-1]
            
            # ユークリッド距離を計算
            movement = np.linalg.norm(curr_centroid - prev_centroid)
            return float(movement)
            
        except Exception as e:
            if self.debug:
                print(f"中心点移動計算エラー: {str(e)}")
            return 0.0
    
    def analyze_opinion_space(self, turn=None) -> Dict[str, Any]:
        """
        意見空間の分析を行う
        
        引数:
            turn: 分析対象のターン（Noneの場合は最新ターン）
            
        戻り値:
            分析結果（多様性、クラスタ数、収束度合いなど）
        """
        # 分析対象のベクトル群を取得
        vectors, meta = self._get_vectors_for_turn(turn)
        
        if len(vectors) <= 1:
            # ベクトルが1つ以下の場合は分析不能
            return {
                'diversity': 0.0,
                'convergence': 1.0,
                'clusters': 1,
                'valid': False,
                'centroid': np.zeros(self.dimension) if len(vectors) == 0 else vectors[0],
                'agent_count': len(vectors),
                'distances': []
            }
            
        try:
            # 全ベクトルの平均（重心）を計算
            centroid = np.mean(vectors, axis=0)
            
            # 各ベクトル間の距離行列を計算
            distances = self._calculate_distance_matrix(vectors)
            
            # 多様性指標（平均距離）を計算
            diversity = np.mean(distances)
            
            # 最大距離（最も異なる意見間）
            max_distance = np.max(distances)
            
            # クラスタリング分析
            cluster_results = self._cluster_opinions(vectors)
            n_clusters = cluster_results['n_clusters']
            cluster_centers = cluster_results['cluster_centers']
            labels = cluster_results['labels']
            silhouette = cluster_results['silhouette']
            
            # 収束度合い（1-多様性）
            convergence = 1.0 - min(1.0, diversity)
            
            # 分析結果をまとめる
            result = {
                'diversity': float(diversity),
                'convergence': float(convergence),
                'max_distance': float(max_distance),
                'clusters': n_clusters,
                'cluster_centers': cluster_centers,
                'cluster_labels': labels,
                'silhouette': float(silhouette),
                'centroid': centroid,
                'agent_count': len(vectors),
                'distances': distances.tolist(),
                'valid': True
            }
            
            # 履歴に追加
            self.centroid_history.append(centroid)
            self.diversity_history.append(diversity)
            
            return result
        except Exception as e:
            if self.debug:
                print(f"意見空間分析エラー: {e}")
                traceback.print_exc()
            return {
                'diversity': 0.0,
                'convergence': 0.0,
                'clusters': 1,
                'valid': False,
                'error': str(e)
            }
    
    def _get_vectors_for_turn(self, turn=None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        特定ターンのベクトルとメタデータを取得（内部メソッド）
        
        引数:
            turn: 対象ターン（Noneの場合は最新ターン）
            
        戻り値:
            (ベクトルのリスト, メタデータのリスト)
        """
        if not self.vectors:
            return [], []
            
        # ターンが指定されていない場合は最新ターン
        if turn is None:
            turn = self.current_turn
        
        # 指定ターンのベクトルとメタデータを抽出
        turn_vectors = []
        turn_metadata = []
        
        for i, metadata in enumerate(self.metadata):
            if i < len(self.vectors) and metadata.get('turn') == turn:
                turn_vectors.append(self.vectors[i])
                turn_metadata.append(metadata)
                
        # 各エージェントの最新ベクトルのみを取得
        if not turn_vectors:
            # エージェントIDでグループ化して最新のものだけ取得
            agent_latest = {}
            for i, metadata in enumerate(self.metadata):
                if i >= len(self.vectors):
                    continue
                agent_id = metadata.get('agent_id')
                if not agent_id:
                    continue
                timestamp = metadata.get('timestamp', 0)
                
                if agent_id not in agent_latest or timestamp > agent_latest[agent_id]['timestamp']:
                    agent_latest[agent_id] = {
                        'index': i,
                        'timestamp': timestamp
                    }
            
            # 最新ベクトルのみを追加
            for agent_data in agent_latest.values():
                idx = agent_data['index']
                if 0 <= idx < len(self.vectors):
                    turn_vectors.append(self.vectors[idx])
                    turn_metadata.append(self.metadata[idx])
        
        return turn_vectors, turn_metadata
    
    def _calculate_distance_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        ベクトル間の距離行列を計算（内部メソッド）
        
        引数:
            vectors: ベクトルのリスト
            
        戻り値:
            距離行列（NumPy配列）
        """
        n = len(vectors)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # コサイン距離を計算
                distance = self._cosine_distance(vectors[i], vectors[j])
                distances[i, j] = distance
                distances[j, i] = distance  # 対称行列
                
        return distances
    
    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        2つのベクトル間のコサイン距離を計算（内部メソッド）
        
        引数:
            vec1, vec2: 比較するベクトル
            
        戻り値:
            距離値（0.0-2.0の範囲、0.0が同一）
        """
        # ゼロベクトルチェック
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 1.0  # デフォルト距離
            
        # コサイン類似度の計算
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
            
        similarity = dot_product / (norm1 * norm2)
        
        # 類似度を[-1, 1]の範囲に制限
        similarity = max(-1.0, min(1.0, similarity))
        
        # コサイン距離に変換（1 - 類似度）
        return 1.0 - similarity
        
    def _cluster_opinions(self, vectors: List[np.ndarray], n_clusters: int = None, n_init: int = 10, max_iter: int = 300) -> Dict:
        """
        意見ベクトルのクラスタリングを行う（内部メソッド）
        
        引数:
            vectors: ベクトルのリスト
            n_clusters: 最大クラスタ数（Noneの場合はベクトル数/2）
            n_init: KMeansの初期化回数
            max_iter: KMeansの最大反復回数
            
        戻り値:
            クラスタリング結果
        """
        n_vectors = len(vectors)
        if n_vectors <= 1:
            return {
                'n_clusters': 1,
                'cluster_centers': [vectors[0]] if n_vectors == 1 else [],
                'labels': [0] if n_vectors == 1 else [],
                'silhouette': 0.0
            }
            
        # 最大クラスタ数の決定（デフォルトはベクトル数の半分）
        if n_clusters is None:
            n_clusters = max(2, n_vectors // 2)
        n_clusters = min(n_clusters, n_vectors - 1)
        
        # クラスタ数1から順にシルエットスコアを計算し最適なkを見つける
        best_silhouette = -1.0
        best_k = 1
        best_labels = None
        best_centers = None
        
        try:
            # ベクトルが少なければ全探索、多ければ効率化
            if n_vectors <= 10:
                # 探索範囲: 1〜n_clusters
                search_range = range(1, n_clusters + 1)
            else:
                # 効率化のため少数のkのみ評価
                search_range = [1, 2, 3, min(5, n_clusters)]
                
            for k in search_range:
                if k == 1:
                    # k=1の場合は全て同じクラスタ
                    labels = np.zeros(n_vectors, dtype=int)
                    centers = [np.mean(vectors, axis=0)]
                    silhouette = 0.0  # 単一クラスタの場合シルエットスコアは計算不能
                else:
                    # k-meansクラスタリング - n_initとmax_iterを明示的に指定
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init, max_iter=max_iter)
                    labels = kmeans.fit_predict(vectors)
                    centers = kmeans.cluster_centers_
                    
                    # シルエットスコア計算
                    if len(set(labels)) > 1:  # 複数のクラスタに分かれていれば
                        silhouette = silhouette_score(vectors, labels)
                    else:
                        silhouette = 0.0
                
                # より良いスコアを持つkを記録
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
                    best_labels = labels
                    best_centers = centers
        except Exception as e:
            if self.debug:
                print(f"クラスタリングエラー: {e}")
                traceback.print_exc()
            # エラー時は単一クラスタとして扱う
            best_k = 1
            best_labels = np.zeros(n_vectors, dtype=int)
            best_centers = [np.mean(vectors, axis=0)]
            best_silhouette = 0.0
            
        # 結果を返す
        return {
            'n_clusters': best_k,
            'cluster_centers': best_centers,
            'labels': best_labels.tolist(),
            'silhouette': best_silhouette
        }
    
    def compute_diversity_metrics(self) -> Dict[str, float]:
        """
        意見の多様性に関連するメトリクスを計算
        
        戻り値:
            Dict[str, float]: 多様性関連のメトリクス
        """
        try:
            if len(self.vectors) < 2:
                return {'mean_distance': 0.0, 'variance': 0.0, 'max_distance': 0.0}
                
            vectors = np.array(self.vectors)
            
            # 中心点の計算
            centroid = np.mean(vectors, axis=0)
            
            # 中心からの距離
            distances = np.linalg.norm(vectors - centroid, axis=1)
            
            # 各距離の統計量
            mean_distance = float(np.mean(distances))
            variance = float(np.var(distances))
            max_distance = float(np.max(distances))
            
            return {
                'mean_distance': mean_distance,
                'variance': variance,
                'max_distance': max_distance
            }
            
        except Exception as e:
            if self.debug:
                print(f"多様性メトリクス計算エラー: {str(e)}")
            return {'mean_distance': 0.5, 'variance': 0.1, 'max_distance': 0.0}
            
    def compute_change_metrics(self) -> Dict[str, float]:
        """
        意見空間の変化に関連するメトリクスを計算
        
        戻り値:
            Dict[str, float]: 変化関連のメトリクス
        """
        try:
            # 中心点の移動
            centroid_movement = self.calculate_centroid_movement()
            
            # エージェントの移動量
            agent_movement = 0.05  # デフォルト値
            
            # 最新のエージェント分布状態を取得
            latest_vectors = self.get_latest_vectors()
            if len(latest_vectors) >= 2:
                vectors = np.array(list(latest_vectors.values()))
                
                # 意見の収束度合い（距離の逆数）
                distances = []
                centroid = np.mean(vectors, axis=0)
                for v in vectors:
                    distances.append(np.linalg.norm(v - centroid))
                
                mean_distance = np.mean(distances)
                opinion_convergence = 1.0 / (1.0 + mean_distance)
                
                # エージェント間の平均移動量（前回と今回の距離）
                # TODO: 実際の移動量の計算を実装
                if len(self.centroid_history) >= 2:
                    agent_movement = centroid_movement
                
            else:
                opinion_convergence = 0.0
            
            return {
                'centroid_movement': float(centroid_movement),
                'average_agent_movement': float(agent_movement),
                'opinion_convergence': float(opinion_convergence)
            }
            
        except Exception as e:
            if self.debug:
                print(f"変化メトリクス計算エラー: {str(e)}")
            return {
                'centroid_movement': 0.05,
                'average_agent_movement': 0.05,
                'opinion_convergence': 0.0
            }
    
    def compute_distance_matrix(self) -> np.ndarray:
        """
        最新の意見ベクトル間の距離行列を計算
        
        戻り値:
            np.ndarray: 距離行列
        """
        try:
            # 最新のエージェントベクトルを取得
            latest_vectors = self.get_latest_vectors()
            vectors = list(latest_vectors.values())
            
            if len(vectors) < 2:
                return np.zeros((1, 1))
                
            # 距離行列の計算
            return self._calculate_distance_matrix(vectors)
            
        except Exception as e:
            if self.debug:
                print(f"距離行列計算エラー: {str(e)}")
            return np.ones((1, 1))  # エラー時は1x1の単位行列
    
    def get_agent_contribution_score(self, agent_id) -> float:
        """
        エージェントの貢献度スコアを計算
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            float: 貢献度スコア（0-1）
        """
        try:
            agent_id_str = str(agent_id)
            
            # 最新のエージェントベクトルを取得
            latest_vectors = self.get_latest_vectors()
            
            if agent_id_str not in latest_vectors:
                return 0.5  # デフォルト値
                
            agent_vector = latest_vectors[agent_id_str]
            all_vectors = list(latest_vectors.values())
            
            if len(all_vectors) < 2:
                return 0.5  # 比較対象がない場合
                
            # 全体の中心を計算
            centroid = np.mean(all_vectors, axis=0)
            
            # このエージェントの中心からの距離
            distance = np.linalg.norm(agent_vector - centroid)
            
            # 全エージェントの中心からの距離
            all_distances = []
            for v in all_vectors:
                all_distances.append(np.linalg.norm(v - centroid))
                
            # 平均距離と標準偏差
            mean_distance = np.mean(all_distances)
            std_distance = np.std(all_distances) if len(all_distances) > 1 else 0.1
            
            # 多様性への貢献度（z-score）
            if std_distance > 0:
                diversity_score = (distance - mean_distance) / std_distance
            else:
                diversity_score = 0.0
                
            # 値を0-1に正規化
            normalized_score = 1.0 / (1.0 + np.exp(-diversity_score))
            
            return float(normalized_score)
            
        except Exception as e:
            if self.debug:
                print(f"エージェント貢献度計算エラー: {str(e)}")
            return 0.5  # エラー時はデフォルト値
            
    def get_centroid(self, vectors) -> np.ndarray:
        """
        ベクトル群の中心点を計算
        
        引数:
            vectors: ベクトルのリスト
            
        戻り値:
            np.ndarray: 中心点ベクトル
        """
        if not vectors:
            return np.zeros(self.dimension)
            
        return np.mean(vectors, axis=0)
        
    def get_inter_cluster_distances(self) -> List[float]:
        """
        クラスタ間の距離を計算
        
        戻り値:
            List[float]: クラスタ間距離のリスト
        """
        try:
            if self.cluster_centers is None or len(self.cluster_centers) < 2:
                return []
                
            # クラスタ中心間の距離を計算
            distances = []
            for i in range(len(self.cluster_centers)):
                for j in range(i+1, len(self.cluster_centers)):
                    dist = np.linalg.norm(self.cluster_centers[i] - self.cluster_centers[j])
                    distances.append(float(dist))
                    
            return distances
            
        except Exception as e:
            if self.debug:
                print(f"クラスタ間距離計算エラー: {str(e)}")
            return []
