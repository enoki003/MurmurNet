#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分散システム監視・メトリクス
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prometheusメトリクス統合、分散システム可視化、パフォーマンス監視とアラート

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os

# Prometheusクライアントライブラリ（オプション）
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, REGISTRY
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Grafana API クライアント（オプション）
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

@dataclass
class MetricPoint:
    """メトリクスデータポイント"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """アラート情報"""
    name: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    threshold: float
    comparison: str  # '>', '<', '==', '!='
    metric_name: str
    triggered: bool = False
    trigger_time: Optional[float] = None
    resolved_time: Optional[float] = None

class MetricsCollector:
    """
    メトリクス収集器
    
    システム全体のメトリクスを収集、保存、配信
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # メトリクス保存
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_values: Dict[str, float] = {}
        
        # Prometheusメトリクス（利用可能な場合）
        self.prometheus_metrics: Dict[str, Any] = {}
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        # 設定
        self.collection_interval = config.get('metrics_collection_interval', 10.0)
        self.retention_hours = config.get('metrics_retention_hours', 24)
        self.enable_prometheus = config.get('enable_prometheus', True) and PROMETHEUS_AVAILABLE
        self.prometheus_port = config.get('prometheus_port', 8000)
        
        # 同期プリミティブ
        self._lock = threading.Lock()
        self._running = False
        
        # Prometheusメトリクス初期化
        if self.enable_prometheus:
            self._init_prometheus_metrics()
        
        self.logger.info("メトリクス収集器初期化完了")

    def _init_prometheus_metrics(self):
        """Prometheusメトリクスを初期化"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # 基本的なシステムメトリクス
        self.prometheus_metrics = {
            # カウンター
            'requests_total': Counter(
                'murmurnet_requests_total',
                'Total number of requests',
                ['method', 'status'],
                registry=self.registry
            ),
            'tasks_processed_total': Counter(
                'murmurnet_tasks_processed_total',
                'Total number of tasks processed',
                ['node_id', 'task_type'],
                registry=self.registry
            ),
            'errors_total': Counter(
                'murmurnet_errors_total',
                'Total number of errors',
                ['component', 'error_type'],
                registry=self.registry
            ),
            
            # ゲージ
            'active_nodes': Gauge(
                'murmurnet_active_nodes',
                'Number of active nodes',
                registry=self.registry
            ),
            'current_load': Gauge(
                'murmurnet_current_load',
                'Current system load',
                ['node_id'],
                registry=self.registry
            ),
            'memory_usage_bytes': Gauge(
                'murmurnet_memory_usage_bytes',
                'Memory usage in bytes',
                ['component'],
                registry=self.registry
            ),
            'pending_tasks': Gauge(
                'murmurnet_pending_tasks',
                'Number of pending tasks',
                registry=self.registry
            ),
            
            # ヒストグラム
            'request_duration_seconds': Histogram(
                'murmurnet_request_duration_seconds',
                'Request duration in seconds',
                ['method'],
                registry=self.registry
            ),
            'task_execution_duration_seconds': Histogram(
                'murmurnet_task_execution_duration_seconds',
                'Task execution duration in seconds',
                ['task_type'],
                registry=self.registry
            ),
            
            # サマリー
            'response_size_bytes': Summary(
                'murmurnet_response_size_bytes',
                'Response size in bytes',
                registry=self.registry
            )
        }
        
        self.logger.info("Prometheusメトリクス初期化完了")

    async def start(self):
        """メトリクス収集開始"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("メトリクス収集開始")
        
        # Prometheusサーバー開始
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port, registry=self.registry)
                self.logger.info(f"Prometheusメトリクスサーバー開始: ポート {self.prometheus_port}")
            except Exception as e:
                self.logger.error(f"Prometheusサーバー開始エラー: {e}")
        
        # バックグラウンド収集タスク
        asyncio.create_task(self._collection_loop())
        asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """メトリクス収集停止"""
        self._running = False
        self.logger.info("メトリクス収集停止")

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """メトリクスを記録"""
        if labels is None:
            labels = {}
        
        timestamp = time.time()
        point = MetricPoint(timestamp, value, labels)
        
        with self._lock:
            self.metrics[name].append(point)
            self.current_values[name] = value
        
        # Prometheusメトリクスも更新
        if self.enable_prometheus:
            self._update_prometheus_metric(name, value, labels)

    def _update_prometheus_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Prometheusメトリクスを更新"""
        try:
            # メトリクス名をPrometheus形式にマッピング
            prometheus_name = self._map_to_prometheus_name(name)
            
            if prometheus_name in self.prometheus_metrics:
                metric = self.prometheus_metrics[prometheus_name]
                
                if hasattr(metric, 'labels'):
                    # ラベル付きメトリクス
                    labeled_metric = metric.labels(**labels)
                    if hasattr(labeled_metric, 'set'):
                        labeled_metric.set(value)
                    elif hasattr(labeled_metric, 'inc'):
                        labeled_metric.inc(value)
                else:
                    # ラベルなしメトリクス
                    if hasattr(metric, 'set'):
                        metric.set(value)
                    elif hasattr(metric, 'inc'):
                        metric.inc(value)
                        
        except Exception as e:
            self.logger.debug(f"Prometheusメトリクス更新エラー: {e}")

    def _map_to_prometheus_name(self, name: str) -> str:
        """内部メトリクス名をPrometheus名にマッピング"""
        mapping = {
            'active_nodes_count': 'active_nodes',
            'node_load': 'current_load',
            'memory_usage': 'memory_usage_bytes',
            'pending_tasks_count': 'pending_tasks',
            'requests_count': 'requests_total',
            'tasks_processed_count': 'tasks_processed_total',
            'errors_count': 'errors_total',
            'request_duration': 'request_duration_seconds',
            'task_duration': 'task_execution_duration_seconds',
            'response_size': 'response_size_bytes'
        }
        return mapping.get(name, name)

    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """指定時間内のメトリクス履歴を取得"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            return [
                point for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]

    def get_current_value(self, name: str) -> Optional[float]:
        """現在のメトリクス値を取得"""
        return self.current_values.get(name)

    def get_aggregated_metrics(self, hours: int = 1) -> Dict[str, Dict[str, float]]:
        """集約メトリクスを取得"""
        result = {}
        
        for name in self.metrics.keys():
            history = self.get_metric_history(name, hours)
            if not history:
                continue
            
            values = [point.value for point in history]
            result[name] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'current': values[-1] if values else 0,
                'count': len(values)
            }
        
        return result

    async def _collection_loop(self):
        """メトリクス収集ループ"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"メトリクス収集エラー: {e}")

    async def _collect_system_metrics(self):
        """システムメトリクスを収集"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent()
            self.record_metric('cpu_usage_percent', cpu_percent)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage_percent', memory.percent)
            self.record_metric('memory_usage', memory.used, {'component': 'system'})
            
            # ディスク使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric('disk_usage_percent', disk_percent)
            
        except ImportError:
            # psutilが利用できない場合は基本的なメトリクスのみ
            pass
        except Exception as e:
            self.logger.error(f"システムメトリクス収集エラー: {e}")

    async def _cleanup_loop(self):
        """古いメトリクスデータのクリーンアップ"""
        while self._running:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                with self._lock:
                    for name, points in self.metrics.items():
                        # 古いデータポイントを削除
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()
                
                # 1時間ごとにクリーンアップ
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"メトリクスクリーンアップエラー: {e}")

class AlertManager:
    """
    アラート管理システム
    
    メトリクスベースのアラート発報・管理
    """
    
    def __init__(self, config: Dict[str, Any], metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # アラート設定
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        self.check_interval = config.get('alert_check_interval', 30.0)
        
        # 通知設定
        self.notification_channels = config.get('notification_channels', [])
        
        self._running = False
        
        # デフォルトアラートを設定
        self._setup_default_alerts()
        
        self.logger.info("アラート管理システム初期化完了")

    def _setup_default_alerts(self):
        """デフォルトアラートを設定"""
        default_alerts = [
            Alert(
                name='high_cpu_usage',
                description='CPU使用率が高い',
                severity='warning',
                threshold=80.0,
                comparison='>',
                metric_name='cpu_usage_percent'
            ),
            Alert(
                name='high_memory_usage',
                description='メモリ使用率が高い',
                severity='warning',
                threshold=85.0,
                comparison='>',
                metric_name='memory_usage_percent'
            ),
            Alert(
                name='no_active_nodes',
                description='アクティブなノードがない',
                severity='critical',
                threshold=1.0,
                comparison='<',
                metric_name='active_nodes_count'
            ),
            Alert(
                name='high_error_rate',
                description='エラー率が高い',
                severity='critical',
                threshold=10.0,
                comparison='>',
                metric_name='error_rate_percent'
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.name] = alert

    async def start(self):
        """アラート管理開始"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("アラート管理開始")
        
        # アラートチェックループ
        asyncio.create_task(self._alert_check_loop())

    async def stop(self):
        """アラート管理停止"""
        self._running = False
        self.logger.info("アラート管理停止")

    def add_alert(self, alert: Alert):
        """アラートを追加"""
        self.alerts[alert.name] = alert
        self.logger.info(f"アラート追加: {alert.name}")

    def remove_alert(self, name: str):
        """アラートを削除"""
        if name in self.alerts:
            del self.alerts[name]
            self.logger.info(f"アラート削除: {name}")

    def add_alert_handler(self, handler: Callable):
        """アラートハンドラーを追加"""
        self.alert_handlers.append(handler)

    async def _alert_check_loop(self):
        """アラートチェックループ"""
        while self._running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"アラートチェックエラー: {e}")

    async def _check_alerts(self):
        """すべてのアラートをチェック"""
        for alert in self.alerts.values():
            try:
                await self._check_single_alert(alert)
            except Exception as e:
                self.logger.error(f"アラート {alert.name} チェックエラー: {e}")

    async def _check_single_alert(self, alert: Alert):
        """単一アラートをチェック"""
        current_value = self.metrics_collector.get_current_value(alert.metric_name)
        
        if current_value is None:
            return
        
        # しきい値と比較
        triggered = self._evaluate_condition(current_value, alert.threshold, alert.comparison)
        
        # アラート状態の変化をチェック
        if triggered and not alert.triggered:
            # アラート発火
            alert.triggered = True
            alert.trigger_time = time.time()
            await self._fire_alert(alert, current_value)
            
        elif not triggered and alert.triggered:
            # アラート解決
            alert.triggered = False
            alert.resolved_time = time.time()
            await self._resolve_alert(alert, current_value)

    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """条件を評価"""
        if comparison == '>':
            return value > threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '==':
            return abs(value - threshold) < 0.001
        elif comparison == '!=':
            return abs(value - threshold) >= 0.001
        return False

    async def _fire_alert(self, alert: Alert, current_value: float):
        """アラートを発火"""
        self.logger.warning(f"アラート発火: {alert.name} - {alert.description}")
        
        alert_data = {
            'name': alert.name,
            'description': alert.description,
            'severity': alert.severity,
            'current_value': current_value,
            'threshold': alert.threshold,
            'trigger_time': alert.trigger_time,
            'status': 'fired'
        }
        
        # アラートハンドラーを実行
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                self.logger.error(f"アラートハンドラーエラー: {e}")
        
        # 通知チャンネルに送信
        await self._send_notifications(alert_data)

    async def _resolve_alert(self, alert: Alert, current_value: float):
        """アラートを解決"""
        self.logger.info(f"アラート解決: {alert.name}")
        
        alert_data = {
            'name': alert.name,
            'description': alert.description,
            'severity': alert.severity,
            'current_value': current_value,
            'threshold': alert.threshold,
            'resolved_time': alert.resolved_time,
            'status': 'resolved'
        }
        
        # 通知チャンネルに送信
        await self._send_notifications(alert_data)

    async def _send_notifications(self, alert_data: Dict[str, Any]):
        """通知を送信"""
        for channel in self.notification_channels:
            try:
                if channel['type'] == 'webhook':
                    await self._send_webhook_notification(channel, alert_data)
                elif channel['type'] == 'slack':
                    await self._send_slack_notification(channel, alert_data)
                elif channel['type'] == 'email':
                    await self._send_email_notification(channel, alert_data)
            except Exception as e:
                self.logger.error(f"通知送信エラー ({channel['type']}): {e}")

    async def _send_webhook_notification(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Webhook通知を送信"""
        if not REQUESTS_AVAILABLE:
            return
        
        url = channel.get('url')
        if not url:
            return
        
        payload = {
            'alert': alert_data,
            'timestamp': time.time()
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Webhook通知エラー: {e}")

    async def _send_slack_notification(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Slack通知を送信"""
        # Slack Webhook実装
        webhook_url = channel.get('webhook_url')
        if not webhook_url or not REQUESTS_AVAILABLE:
            return
        
        color = 'danger' if alert_data['severity'] == 'critical' else 'warning'
        status_emoji = '🔥' if alert_data['status'] == 'fired' else '✅'
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"{status_emoji} {alert_data['name']}",
                'text': alert_data['description'],
                'fields': [
                    {
                        'title': 'Current Value',
                        'value': str(alert_data['current_value']),
                        'short': True
                    },
                    {
                        'title': 'Threshold',
                        'value': str(alert_data['threshold']),
                        'short': True
                    }
                ],
                'timestamp': int(time.time())
            }]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Slack通知エラー: {e}")

    async def _send_email_notification(self, channel: Dict[str, Any], alert_data: Dict[str, Any]):
        """Email通知を送信"""
        # 簡単なEmail実装（実際のプロダクションではより堅牢な実装が必要）
        self.logger.info(f"Email通知（未実装）: {alert_data['name']}")

    def get_alert_status(self) -> Dict[str, Any]:
        """アラート状態を取得"""
        active_alerts = [
            {
                'name': alert.name,
                'description': alert.description,
                'severity': alert.severity,
                'triggered': alert.triggered,
                'trigger_time': alert.trigger_time
            }
            for alert in self.alerts.values()
            if alert.triggered
        ]
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'alerts': active_alerts
        }

class DashboardManager:
    """
    ダッシュボード管理システム
    
    Grafana等の可視化ダッシュボードとの統合
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Grafana設定
        self.grafana_url = config.get('grafana_url')
        self.grafana_api_key = config.get('grafana_api_key')
        
        # ダッシュボード設定
        self.dashboard_config = {
            'title': 'MurmurNet 分散システム監視',
            'tags': ['murmurnet', 'distributed'],
            'refresh': '30s',
            'time': {
                'from': 'now-1h',
                'to': 'now'
            }
        }
        
        self.logger.info("ダッシュボード管理システム初期化完了")

    async def create_dashboard(self) -> bool:
        """Grafanaダッシュボードを作成"""
        if not self.grafana_url or not self.grafana_api_key or not REQUESTS_AVAILABLE:
            self.logger.warning("Grafana設定が不完全、ダッシュボード作成をスキップ")
            return False
        
        try:
            dashboard_json = self._generate_dashboard_json()
            
            headers = {
                'Authorization': f'Bearer {self.grafana_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f'{self.grafana_url}/api/dashboards/db',
                json=dashboard_json,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info("Grafanaダッシュボード作成成功")
                return True
            else:
                self.logger.error(f"ダッシュボード作成エラー: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"ダッシュボード作成例外: {e}")
            return False

    def _generate_dashboard_json(self) -> Dict[str, Any]:
        """ダッシュボードJSON定義を生成"""
        return {
            'dashboard': {
                'title': self.dashboard_config['title'],
                'tags': self.dashboard_config['tags'],
                'refresh': self.dashboard_config['refresh'],
                'time': self.dashboard_config['time'],
                'panels': [
                    self._create_system_overview_panel(),
                    self._create_node_status_panel(),
                    self._create_task_metrics_panel(),
                    self._create_performance_panel(),
                    self._create_alert_panel()
                ]
            },
            'overwrite': True
        }

    def _create_system_overview_panel(self) -> Dict[str, Any]:
        """システム概要パネル"""
        return {
            'title': 'システム概要',
            'type': 'stat',
            'targets': [
                {
                    'expr': 'murmurnet_active_nodes',
                    'legendFormat': 'アクティブノード'
                },
                {
                    'expr': 'murmurnet_pending_tasks',
                    'legendFormat': '待機タスク'
                }
            ],
            'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0}
        }

    def _create_node_status_panel(self) -> Dict[str, Any]:
        """ノード状態パネル"""
        return {
            'title': 'ノード負荷',
            'type': 'graph',
            'targets': [
                {
                    'expr': 'murmurnet_current_load',
                    'legendFormat': 'ノード {{node_id}}'
                }
            ],
            'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0}
        }

    def _create_task_metrics_panel(self) -> Dict[str, Any]:
        """タスクメトリクスパネル"""
        return {
            'title': 'タスク処理状況',
            'type': 'graph',
            'targets': [
                {
                    'expr': 'rate(murmurnet_tasks_processed_total[5m])',
                    'legendFormat': '処理レート'
                },
                {
                    'expr': 'rate(murmurnet_errors_total[5m])',
                    'legendFormat': 'エラーレート'
                }
            ],
            'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
        }

    def _create_performance_panel(self) -> Dict[str, Any]:
        """パフォーマンスパネル"""
        return {
            'title': 'レスポンス時間',
            'type': 'graph',
            'targets': [
                {
                    'expr': 'histogram_quantile(0.95, murmurnet_request_duration_seconds_bucket)',
                    'legendFormat': '95%ile'
                },
                {
                    'expr': 'histogram_quantile(0.50, murmurnet_request_duration_seconds_bucket)',
                    'legendFormat': '50%ile'
                }
            ],
            'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
        }

    def _create_alert_panel(self) -> Dict[str, Any]:
        """アラートパネル"""
        return {
            'title': 'アクティブアラート',
            'type': 'table',
            'targets': [
                {
                    'expr': 'ALERTS{alertstate="firing"}',
                    'format': 'table'
                }
            ],
            'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 16}
        }

def create_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """メトリクス収集器を作成"""
    return MetricsCollector(config)

def create_alert_manager(config: Dict[str, Any], metrics_collector: MetricsCollector) -> AlertManager:
    """アラート管理システムを作成"""
    return AlertManager(config, metrics_collector)

def create_dashboard_manager(config: Dict[str, Any]) -> DashboardManager:
    """ダッシュボード管理システムを作成"""
    return DashboardManager(config)
