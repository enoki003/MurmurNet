#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
オートスケーリングシステム
~~~~~~~~~~~~~~~~~~~~~~~~
動的ワーカー追加/削除、リソース使用率ベースのスケーリング、Kubernetesとの統合

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

# Kubernetes クライアント（オプション）
try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# Ray クライアント（オプション）
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

class ScalingDirection(Enum):
    """スケーリング方向"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingStrategy(Enum):
    """スケーリング戦略"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    TASK_QUEUE_BASED = "task_queue_based"
    HYBRID = "hybrid"
    PREDICTIVE = "predictive"

@dataclass
class ScalingMetrics:
    """スケーリングメトリクス"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    task_queue_length: int = 0
    active_workers: int = 0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingDecision:
    """スケーリング判定結果"""
    direction: ScalingDirection
    target_workers: int
    current_workers: int
    reason: str
    confidence: float
    metrics: ScalingMetrics

@dataclass
class WorkerTemplate:
    """ワーカーテンプレート"""
    name: str
    image: str
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    environment: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

class AutoScaler:
    """
    オートスケーリングシステム
    
    リソース使用率とタスクキューに基づいて動的にワーカーをスケール
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # スケーリング設定
        self.strategy = ScalingStrategy(config.get('scaling_strategy', 'hybrid'))
        self.min_workers = config.get('min_workers', 1)
        self.max_workers = config.get('max_workers', 10)
        self.target_cpu_utilization = config.get('target_cpu_utilization', 0.7)
        self.target_memory_utilization = config.get('target_memory_utilization', 0.8)
        self.max_queue_length = config.get('max_queue_length', 50)
        
        # スケーリング制約
        self.scale_up_threshold = config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = config.get('scale_down_threshold', 0.3)
        self.scale_up_cooldown = config.get('scale_up_cooldown', 300)  # 5分
        self.scale_down_cooldown = config.get('scale_down_cooldown', 600)  # 10分
        self.evaluation_period = config.get('evaluation_period', 60)  # 1分
        
        # 状態管理
        self.current_workers = self.min_workers
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        self.metrics_history: List[ScalingMetrics] = []
        self.max_history_size = config.get('max_history_size', 100)
        
        # Kubernetes設定
        self.use_kubernetes = config.get('use_kubernetes', False) and KUBERNETES_AVAILABLE
        self.namespace = config.get('kubernetes_namespace', 'default')
        self.deployment_name = config.get('deployment_name', 'murmurnet-workers')
        
        # ワーカーテンプレート
        self.worker_template = WorkerTemplate(
            name=config.get('worker_template_name', 'murmurnet-worker'),
            image=config.get('worker_image', 'murmurnet:latest'),
            cpu_request=config.get('worker_cpu_request', '100m'),
            cpu_limit=config.get('worker_cpu_limit', '500m'),
            memory_request=config.get('worker_memory_request', '128Mi'),
            memory_limit=config.get('worker_memory_limit', '512Mi'),
            environment=config.get('worker_environment', {}),
            labels=config.get('worker_labels', {})
        )
        
        # 予測モデル設定
        self.enable_predictive_scaling = config.get('enable_predictive_scaling', False)
        self.prediction_window = config.get('prediction_window', 300)  # 5分先を予測
        
        # コールバック
        self.scale_callbacks: List[Callable] = []
        
        self._running = False
        
        # Kubernetesクライアント初期化
        if self.use_kubernetes:
            self._init_kubernetes_client()
        
        self.logger.info(f"オートスケーラー初期化完了 - 戦略: {self.strategy.value}")

    def _init_kubernetes_client(self):
        """Kubernetesクライアントを初期化"""
        try:
            # クラスター内実行時は自動設定、ローカル実行時はkubeconfigを使用
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount'):
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
            
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            
            self.logger.info("Kubernetesクライアント初期化成功")
            
        except Exception as e:
            self.logger.error(f"Kubernetesクライアント初期化エラー: {e}")
            self.use_kubernetes = False

    async def start(self):
        """オートスケーラー開始"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("オートスケーラー開始")
        
        # 初期ワーカー数を設定
        await self._ensure_initial_workers()
        
        # スケーリングループ開始
        asyncio.create_task(self._scaling_loop())

    async def stop(self):
        """オートスケーラー停止"""
        self._running = False
        self.logger.info("オートスケーラー停止")

    def add_scale_callback(self, callback: Callable):
        """スケーリングコールバックを追加"""
        self.scale_callbacks.append(callback)

    async def update_metrics(self, metrics: ScalingMetrics):
        """メトリクスを更新"""
        self.metrics_history.append(metrics)
        
        # 履歴サイズ制限
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        self.logger.debug(f"メトリクス更新: CPU={metrics.cpu_usage:.2f}, Memory={metrics.memory_usage:.2f}, Queue={metrics.task_queue_length}")

    async def _scaling_loop(self):
        """スケーリング判定ループ"""
        while self._running:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(self.evaluation_period)
            except Exception as e:
                self.logger.error(f"スケーリング評価エラー: {e}")

    async def _evaluate_scaling(self):
        """スケーリング判定を実行"""
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # スケーリング戦略に基づいて判定
        if self.strategy == ScalingStrategy.CPU_BASED:
            decision = self._evaluate_cpu_based(current_metrics)
        elif self.strategy == ScalingStrategy.MEMORY_BASED:
            decision = self._evaluate_memory_based(current_metrics)
        elif self.strategy == ScalingStrategy.TASK_QUEUE_BASED:
            decision = self._evaluate_queue_based(current_metrics)
        elif self.strategy == ScalingStrategy.HYBRID:
            decision = self._evaluate_hybrid(current_metrics)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            decision = self._evaluate_predictive(current_metrics)
        else:
            return
        
        # クールダウン期間をチェック
        if not self._is_cooldown_expired(decision.direction):
            self.logger.debug(f"クールダウン期間中、スケーリングを見送り: {decision.direction.value}")
            return
        
        # スケーリング実行
        if decision.direction != ScalingDirection.STABLE:
            await self._execute_scaling(decision)

    def _evaluate_cpu_based(self, metrics: ScalingMetrics) -> ScalingDecision:
        """CPU使用率ベースの判定"""
        cpu_usage = metrics.cpu_usage
        
        if cpu_usage > self.scale_up_threshold:
            target = min(self.current_workers + 1, self.max_workers)
            return ScalingDecision(
                direction=ScalingDirection.UP,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"CPU使用率が高い: {cpu_usage:.2f}",
                confidence=min(cpu_usage / self.scale_up_threshold, 1.0),
                metrics=metrics
            )
        elif cpu_usage < self.scale_down_threshold:
            target = max(self.current_workers - 1, self.min_workers)
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"CPU使用率が低い: {cpu_usage:.2f}",
                confidence=1.0 - (cpu_usage / self.scale_down_threshold),
                metrics=metrics
            )
        
        return ScalingDecision(
            direction=ScalingDirection.STABLE,
            target_workers=self.current_workers,
            current_workers=self.current_workers,
            reason="CPU使用率が適正範囲",
            confidence=1.0,
            metrics=metrics
        )

    def _evaluate_memory_based(self, metrics: ScalingMetrics) -> ScalingDecision:
        """メモリ使用率ベースの判定"""
        memory_usage = metrics.memory_usage
        
        if memory_usage > self.scale_up_threshold:
            target = min(self.current_workers + 1, self.max_workers)
            return ScalingDecision(
                direction=ScalingDirection.UP,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"メモリ使用率が高い: {memory_usage:.2f}",
                confidence=min(memory_usage / self.scale_up_threshold, 1.0),
                metrics=metrics
            )
        elif memory_usage < self.scale_down_threshold:
            target = max(self.current_workers - 1, self.min_workers)
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"メモリ使用率が低い: {memory_usage:.2f}",
                confidence=1.0 - (memory_usage / self.scale_down_threshold),
                metrics=metrics
            )
        
        return ScalingDecision(
            direction=ScalingDirection.STABLE,
            target_workers=self.current_workers,
            current_workers=self.current_workers,
            reason="メモリ使用率が適正範囲",
            confidence=1.0,
            metrics=metrics
        )

    def _evaluate_queue_based(self, metrics: ScalingMetrics) -> ScalingDecision:
        """タスクキュー長ベースの判定"""
        queue_length = metrics.task_queue_length
        optimal_queue_per_worker = self.max_queue_length / self.current_workers
        
        if queue_length > self.max_queue_length:
            # キューが溢れそうな場合はスケールアップ
            recommended_workers = math.ceil(queue_length / optimal_queue_per_worker)
            target = min(recommended_workers, self.max_workers)
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"タスクキューが満杯: {queue_length}",
                confidence=min(queue_length / self.max_queue_length, 1.0),
                metrics=metrics
            )
        elif queue_length < optimal_queue_per_worker * 0.3:
            # キューが少ない場合はスケールダウン
            target = max(self.current_workers - 1, self.min_workers)
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"タスクキューが少ない: {queue_length}",
                confidence=1.0 - (queue_length / optimal_queue_per_worker),
                metrics=metrics
            )
        
        return ScalingDecision(
            direction=ScalingDirection.STABLE,
            target_workers=self.current_workers,
            current_workers=self.current_workers,
            reason="タスクキューが適正範囲",
            confidence=1.0,
            metrics=metrics
        )

    def _evaluate_hybrid(self, metrics: ScalingMetrics) -> ScalingDecision:
        """ハイブリッド判定（複数指標の組み合わせ）"""
        cpu_decision = self._evaluate_cpu_based(metrics)
        memory_decision = self._evaluate_memory_based(metrics)
        queue_decision = self._evaluate_queue_based(metrics)
        
        # 重み付けスコア計算
        cpu_weight = 0.4
        memory_weight = 0.3
        queue_weight = 0.3
        
        scale_up_score = 0
        scale_down_score = 0
        
        if cpu_decision.direction == ScalingDirection.UP:
            scale_up_score += cpu_weight * cpu_decision.confidence
        elif cpu_decision.direction == ScalingDirection.DOWN:
            scale_down_score += cpu_weight * cpu_decision.confidence
        
        if memory_decision.direction == ScalingDirection.UP:
            scale_up_score += memory_weight * memory_decision.confidence
        elif memory_decision.direction == ScalingDirection.DOWN:
            scale_down_score += memory_weight * memory_decision.confidence
        
        if queue_decision.direction == ScalingDirection.UP:
            scale_up_score += queue_weight * queue_decision.confidence
        elif queue_decision.direction == ScalingDirection.DOWN:
            scale_down_score += queue_weight * queue_decision.confidence
        
        # 判定
        threshold = 0.6
        if scale_up_score > threshold:
            target = min(self.current_workers + 1, self.max_workers)
            return ScalingDecision(
                direction=ScalingDirection.UP,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"ハイブリッド判定でスケールアップ (スコア: {scale_up_score:.2f})",
                confidence=scale_up_score,
                metrics=metrics
            )
        elif scale_down_score > threshold:
            target = max(self.current_workers - 1, self.min_workers)
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_workers=target,
                current_workers=self.current_workers,
                reason=f"ハイブリッド判定でスケールダウン (スコア: {scale_down_score:.2f})",
                confidence=scale_down_score,
                metrics=metrics
            )
        
        return ScalingDecision(
            direction=ScalingDirection.STABLE,
            target_workers=self.current_workers,
            current_workers=self.current_workers,
            reason="ハイブリッド判定で安定",
            confidence=1.0 - max(scale_up_score, scale_down_score),
            metrics=metrics
        )

    def _evaluate_predictive(self, metrics: ScalingMetrics) -> ScalingDecision:
        """予測ベースの判定"""
        if len(self.metrics_history) < 10:
            # 履歴不足の場合はハイブリッド判定にフォールバック
            return self._evaluate_hybrid(metrics)
        
        # 簡単な線形予測（実際の実装ではより高度な予測モデルを使用）
        predicted_cpu = self._predict_metric([m.cpu_usage for m in self.metrics_history[-10:]])
        predicted_memory = self._predict_metric([m.memory_usage for m in self.metrics_history[-10:]])
        predicted_queue = self._predict_metric([m.task_queue_length for m in self.metrics_history[-10:]])
        
        # 予測値に基づいて判定
        predicted_metrics = ScalingMetrics(
            cpu_usage=predicted_cpu,
            memory_usage=predicted_memory,
            task_queue_length=int(predicted_queue),
            active_workers=metrics.active_workers,
            avg_response_time=metrics.avg_response_time,
            throughput=metrics.throughput
        )
        
        decision = self._evaluate_hybrid(predicted_metrics)
        decision.reason = f"予測判定: {decision.reason}"
        
        return decision

    def _predict_metric(self, values: List[float]) -> float:
        """簡単な線形予測"""
        if len(values) < 2:
            return values[-1] if values else 0.0
        
        # 線形回帰による予測
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return values[-1]
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 予測ポイント数（現在から予測期間分先）
        prediction_steps = self.prediction_window // self.evaluation_period
        future_x = n + prediction_steps
        
        predicted_value = slope * future_x + intercept
        return max(0, predicted_value)  # 負の値は0にクリップ

    def _is_cooldown_expired(self, direction: ScalingDirection) -> bool:
        """クールダウン期間が終了しているかチェック"""
        current_time = time.time()
        
        if direction == ScalingDirection.UP:
            return current_time - self.last_scale_up_time >= self.scale_up_cooldown
        elif direction == ScalingDirection.DOWN:
            return current_time - self.last_scale_down_time >= self.scale_down_cooldown
        
        return True

    async def _execute_scaling(self, decision: ScalingDecision):
        """スケーリングを実行"""
        if decision.target_workers == self.current_workers:
            return
        
        self.logger.info(f"スケーリング実行: {self.current_workers} -> {decision.target_workers} ({decision.reason})")
        
        try:
            if self.use_kubernetes:
                success = await self._scale_kubernetes_deployment(decision.target_workers)
            else:
                success = await self._scale_local_workers(decision.target_workers)
            
            if success:
                self.current_workers = decision.target_workers
                
                # クールダウン時間を更新
                current_time = time.time()
                if decision.direction == ScalingDirection.UP:
                    self.last_scale_up_time = current_time
                elif decision.direction == ScalingDirection.DOWN:
                    self.last_scale_down_time = current_time
                
                # コールバック実行
                await self._execute_scale_callbacks(decision)
                
                self.logger.info(f"スケーリング成功: 現在のワーカー数 {self.current_workers}")
            else:
                self.logger.error("スケーリング失敗")
                
        except Exception as e:
            self.logger.error(f"スケーリング実行エラー: {e}")

    async def _scale_kubernetes_deployment(self, target_workers: int) -> bool:
        """Kubernetesデプロイメントをスケール"""
        try:
            # デプロイメントの現在の状態を取得
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            # レプリカ数を更新
            deployment.spec.replicas = target_workers
            
            # デプロイメントを更新
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            self.logger.info(f"Kubernetesデプロイメント更新: {self.deployment_name} -> {target_workers} replicas")
            return True
            
        except ApiException as e:
            self.logger.error(f"Kubernetes API エラー: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Kubernetesスケーリングエラー: {e}")
            return False

    async def _scale_local_workers(self, target_workers: int) -> bool:
        """ローカルワーカーをスケール（Ray使用）"""
        if not RAY_AVAILABLE:
            self.logger.warning("Ray未インストール、ローカルスケーリングをスキップ")
            return False
        
        try:
            # Ray クラスターのリソース確認
            cluster_resources = ray.cluster_resources()
            available_cpus = cluster_resources.get('CPU', 0)
            
            if target_workers > available_cpus:
                self.logger.warning(f"利用可能CPU数 ({available_cpus}) より多いワーカー ({target_workers}) が要求されました")
                target_workers = int(available_cpus)
            
            # 実際のワーカー数調整はRayの自動スケーリングに委ねる
            # ここでは設定を更新するのみ
            self.logger.info(f"ローカルワーカー目標数を {target_workers} に設定")
            return True
            
        except Exception as e:
            self.logger.error(f"ローカルスケーリングエラー: {e}")
            return False

    async def _ensure_initial_workers(self):
        """初期ワーカー数を確保"""
        if self.use_kubernetes:
            await self._scale_kubernetes_deployment(self.min_workers)
        else:
            await self._scale_local_workers(self.min_workers)

    async def _execute_scale_callbacks(self, decision: ScalingDecision):
        """スケーリングコールバックを実行"""
        for callback in self.scale_callbacks:
            try:
                await callback(decision)
            except Exception as e:
                self.logger.error(f"スケーリングコールバックエラー: {e}")

    def get_scaling_status(self) -> Dict[str, Any]:
        """スケーリング状態を取得"""
        recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'strategy': self.strategy.value,
            'last_scale_up_time': self.last_scale_up_time,
            'last_scale_down_time': self.last_scale_down_time,
            'recent_metrics': {
                'cpu_usage': recent_metrics.cpu_usage if recent_metrics else 0,
                'memory_usage': recent_metrics.memory_usage if recent_metrics else 0,
                'task_queue_length': recent_metrics.task_queue_length if recent_metrics else 0,
                'throughput': recent_metrics.throughput if recent_metrics else 0
            } if recent_metrics else None,
            'use_kubernetes': self.use_kubernetes,
            'enable_predictive_scaling': self.enable_predictive_scaling
        }

class KubernetesManager:
    """
    Kubernetes統合管理
    
    Kubernetesクラスターでの分散ワーカー管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not KUBERNETES_AVAILABLE:
            raise ImportError("Kubernetesクライアントライブラリが必要です: pip install kubernetes")
        
        # 設定
        self.namespace = config.get('namespace', 'default')
        self.service_account = config.get('service_account', 'murmurnet-worker')
        self.image_pull_policy = config.get('image_pull_policy', 'IfNotPresent')
        
        # Kubernetesクライアント初期化
        self._init_kubernetes_client()
        
        self.logger.info("Kubernetes管理システム初期化完了")

    def _init_kubernetes_client(self):
        """Kubernetesクライアントを初期化"""
        try:
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount'):
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
            
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.batch_v1 = client.BatchV1Api()
            
        except Exception as e:
            self.logger.error(f"Kubernetesクライアント初期化エラー: {e}")
            raise

    async def create_worker_deployment(self, template: WorkerTemplate, replicas: int = 1) -> bool:
        """ワーカーデプロイメントを作成"""
        try:
            deployment = self._create_deployment_spec(template, replicas)
            
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            self.logger.info(f"ワーカーデプロイメント作成成功: {template.name}")
            return True
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info(f"デプロイメント既存: {template.name}")
                return True
            else:
                self.logger.error(f"デプロイメント作成エラー: {e}")
                return False
        except Exception as e:
            self.logger.error(f"デプロイメント作成例外: {e}")
            return False

    def _create_deployment_spec(self, template: WorkerTemplate, replicas: int):
        """デプロイメント仕様を作成"""
        labels = {'app': template.name, **template.labels}
        
        container = client.V1Container(
            name=template.name,
            image=template.image,
            image_pull_policy=self.image_pull_policy,
            env=[
                client.V1EnvVar(name=k, value=v)
                for k, v in template.environment.items()
            ],
            resources=client.V1ResourceRequirements(
                requests={
                    'cpu': template.cpu_request,
                    'memory': template.memory_request
                },
                limits={
                    'cpu': template.cpu_limit,
                    'memory': template.memory_limit
                }
            )
        )
        
        pod_spec = client.V1PodSpec(
            containers=[container],
            service_account_name=self.service_account,
            restart_policy='Always'
        )
        
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )
        
        deployment_spec = client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(match_labels={'app': template.name}),
            template=pod_template
        )
        
        return client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(name=template.name, labels=labels),
            spec=deployment_spec
        )

    async def get_worker_pods(self, deployment_name: str) -> List[Dict[str, Any]]:
        """ワーカーポッド一覧を取得"""
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f'app={deployment_name}'
            )
            
            result = []
            for pod in pods.items:
                result.append({
                    'name': pod.metadata.name,
                    'phase': pod.status.phase,
                    'node': pod.spec.node_name,
                    'created': pod.metadata.creation_timestamp,
                    'ready': self._is_pod_ready(pod)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"ポッド一覧取得エラー: {e}")
            return []

    def _is_pod_ready(self, pod) -> bool:
        """ポッドが準備完了状態かチェック"""
        if not pod.status.conditions:
            return False
        
        for condition in pod.status.conditions:
            if condition.type == 'Ready':
                return condition.status == 'True'
        
        return False

    async def delete_worker_deployment(self, deployment_name: str) -> bool:
        """ワーカーデプロイメントを削除"""
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            self.logger.info(f"ワーカーデプロイメント削除成功: {deployment_name}")
            return True
            
        except ApiException as e:
            if e.status == 404:  # Not found
                self.logger.info(f"デプロイメント未存在: {deployment_name}")
                return True
            else:
                self.logger.error(f"デプロイメント削除エラー: {e}")
                return False
        except Exception as e:
            self.logger.error(f"デプロイメント削除例外: {e}")
            return False

def create_autoscaler(config: Dict[str, Any]) -> AutoScaler:
    """オートスケーラーを作成"""
    return AutoScaler(config)

def create_kubernetes_manager(config: Dict[str, Any]) -> Optional[KubernetesManager]:
    """Kubernetes管理システムを作成"""
    if not KUBERNETES_AVAILABLE:
        return None
    return KubernetesManager(config)
