#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分散協調メカニズム
~~~~~~~~~~~~~~~~
分散ワーカー間の協調プロトコル、合意アルゴリズム、負荷分散、フェイルオーバー機構

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json

# Rayがインストールされている場合のみインポート
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

class ConsensusAlgorithm(Enum):
    """合意アルゴリズムタイプ"""
    RAFT = "raft"
    PBFT = "pbft"
    SIMPLE_MAJORITY = "simple_majority"

class NodeState(Enum):
    """ノード状態"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPECT = "suspect"
    DEAD = "dead"

@dataclass
class WorkerNode:
    """ワーカーノード情報"""
    node_id: str
    address: str
    load: float = 0.0
    last_heartbeat: float = 0.0
    state: NodeState = NodeState.ACTIVE
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

@dataclass
class Task:
    """分散タスク"""
    task_id: str
    payload: Dict[str, Any]
    priority: int = 0
    timestamp: float = 0.0
    assigned_node: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

class DistributedCoordinator:
    """
    分散協調システム
    
    分散ワーカー間の協調、負荷分散、障害処理を管理
    """
    
    def __init__(self, config: Dict[str, Any], node_id: Optional[str] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ノード情報
        self.node_id = node_id or str(uuid.uuid4())
        self.nodes: Dict[str, WorkerNode] = {}
        self.is_leader = False
        self.leader_id: Optional[str] = None
        
        # 協調設定
        self.heartbeat_interval = config.get('heartbeat_interval', 5.0)
        self.failure_timeout = config.get('failure_timeout', 15.0)
        self.consensus_algorithm = ConsensusAlgorithm(
            config.get('consensus_algorithm', 'simple_majority')
        )
        
        # 負荷分散設定
        self.load_balance_strategy = config.get('load_balance_strategy', 'round_robin')
        self.max_load_threshold = config.get('max_load_threshold', 0.8)
        
        # タスクキュー
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        # 同期プリミティブ
        self._coordination_lock = asyncio.Lock()
        self._running = False
        
        # メトリクス
        self.metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'consensus_operations': 0,
            'leader_elections': 0
        }
        
        self.logger.info(f"分散協調システム初期化完了 - ノードID: {self.node_id}")

    async def start(self):
        """協調システム開始"""
        if self._running:
            return
            
        self._running = True
        self.logger.info("分散協調システム開始")
        
        # 現在のノードを登録
        self.nodes[self.node_id] = WorkerNode(
            node_id=self.node_id,
            address=f"node://{self.node_id}",
            last_heartbeat=time.time(),
            capabilities=['general']
        )
        
        # バックグラウンドタスク開始
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._failure_detection_loop())
        asyncio.create_task(self._task_scheduler_loop())
        
        # リーダー選出
        await self._elect_leader()

    async def stop(self):
        """協調システム停止"""
        self._running = False
        self.logger.info("分散協調システム停止")

    async def register_node(self, node: WorkerNode):
        """新しいノードを登録"""
        async with self._coordination_lock:
            self.nodes[node.node_id] = node
            node.last_heartbeat = time.time()
            self.logger.info(f"ノード登録: {node.node_id}")
            
            # 負荷再分散をトリガー
            await self._rebalance_tasks()

    async def unregister_node(self, node_id: str):
        """ノードの登録解除"""
        async with self._coordination_lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.logger.info(f"ノード登録解除: {node_id}")
                
                # リーダーノードが削除された場合、再選出
                if node_id == self.leader_id:
                    await self._elect_leader()
                
                # タスクの再配布
                await self._redistribute_tasks(node_id)

    async def submit_task(self, payload: Dict[str, Any], priority: int = 0) -> str:
        """タスクを分散システムに投入"""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            payload=payload,
            priority=priority,
            timestamp=time.time()
        )
        
        async with self._coordination_lock:
            self.pending_tasks[task_id] = task
            self.logger.debug(f"タスク投入: {task_id}")
        
        return task_id

    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """タスク結果を取得"""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return task.payload.get('result')
        return None

    async def achieve_consensus(self, proposal: Dict[str, Any]) -> bool:
        """分散合意を実行"""
        if not self.is_leader:
            self.logger.warning("リーダーノードのみが合意を開始できます")
            return False
            
        self.metrics['consensus_operations'] += 1
        
        if self.consensus_algorithm == ConsensusAlgorithm.SIMPLE_MAJORITY:
            return await self._simple_majority_consensus(proposal)
        elif self.consensus_algorithm == ConsensusAlgorithm.RAFT:
            return await self._raft_consensus(proposal)
        elif self.consensus_algorithm == ConsensusAlgorithm.PBFT:
            return await self._pbft_consensus(proposal)
        
        return False

    async def _simple_majority_consensus(self, proposal: Dict[str, Any]) -> bool:
        """単純過半数による合意"""
        active_nodes = [n for n in self.nodes.values() if n.state == NodeState.ACTIVE]
        required_votes = len(active_nodes) // 2 + 1
        
        # 提案をハッシュ化
        proposal_hash = hashlib.sha256(
            json.dumps(proposal, sort_keys=True).encode()
        ).hexdigest()
        
        votes = 1  # リーダーの票
        
        # 他のノードから投票を収集（実際の実装では分散投票が必要）
        for node in active_nodes:
            if node.node_id != self.node_id:
                # シミュレートされた投票（実際の実装ではネットワーク通信）
                vote = await self._simulate_node_vote(node.node_id, proposal_hash)
                if vote:
                    votes += 1
        
        consensus_achieved = votes >= required_votes
        self.logger.info(f"単純過半数合意: {votes}/{len(active_nodes)} 票, 結果: {consensus_achieved}")
        
        return consensus_achieved

    async def _simulate_node_vote(self, node_id: str, proposal_hash: str) -> bool:
        """ノード投票をシミュレート（実装用プレースホルダー）"""
        # 実際の実装では、ネットワークを通じて他のノードに投票を要求
        # ここではランダムに投票結果を生成
        await asyncio.sleep(0.1)  # ネットワーク遅延をシミュレート
        return random.random() > 0.2  # 80%の確率で賛成票

    async def _raft_consensus(self, proposal: Dict[str, Any]) -> bool:
        """Raftアルゴリズムによる合意（簡略版）"""
        # Raftの詳細実装は複雑なため、ここでは基本的な構造のみ
        self.logger.info("Raft合意アルゴリズム実行中...")
        
        # ログエントリとして提案を追加
        log_entry = {
            'term': getattr(self, '_current_term', 1),
            'proposal': proposal,
            'timestamp': time.time()
        }
        
        # フォロワーノードに複製（簡略版）
        success_count = 1  # リーダー自身
        active_followers = [n for n in self.nodes.values() 
                          if n.state == NodeState.ACTIVE and n.node_id != self.node_id]
        
        for follower in active_followers:
            success = await self._replicate_log_entry(follower.node_id, log_entry)
            if success:
                success_count += 1
        
        # 過半数の合意が得られた場合、コミット
        majority = len(self.nodes) // 2 + 1
        consensus_achieved = success_count >= majority
        
        self.logger.info(f"Raft合意: {success_count}/{len(self.nodes)} ノード, 結果: {consensus_achieved}")
        return consensus_achieved

    async def _replicate_log_entry(self, node_id: str, log_entry: Dict[str, Any]) -> bool:
        """ログエントリを他のノードに複製"""
        # 実際の実装ではネットワーク通信
        await asyncio.sleep(0.05)  # ネットワーク遅延をシミュレート
        return random.random() > 0.1  # 90%の成功率

    async def _pbft_consensus(self, proposal: Dict[str, Any]) -> bool:
        """PBFT（Byzantine Fault Tolerance）アルゴリズム（簡略版）"""
        self.logger.info("PBFT合意アルゴリズム実行中...")
        
        # 3段階のプロトコル: pre-prepare, prepare, commit
        phases = ['pre-prepare', 'prepare', 'commit']
        
        for phase in phases:
            success = await self._pbft_phase(phase, proposal)
            if not success:
                self.logger.warning(f"PBFT {phase} フェーズで合意失敗")
                return False
        
        self.logger.info("PBFT合意成功")
        return True

    async def _pbft_phase(self, phase: str, proposal: Dict[str, Any]) -> bool:
        """PBFTの各フェーズを実行"""
        active_nodes = [n for n in self.nodes.values() if n.state == NodeState.ACTIVE]
        f = (len(active_nodes) - 1) // 3  # Byzantine障害を許容できるノード数
        required_votes = len(active_nodes) - f
        
        votes = 1  # リーダーの票
        
        for node in active_nodes:
            if node.node_id != self.node_id:
                vote = await self._simulate_pbft_vote(node.node_id, phase, proposal)
                if vote:
                    votes += 1
        
        success = votes >= required_votes
        self.logger.debug(f"PBFT {phase}: {votes}/{len(active_nodes)} 票")
        return success

    async def _simulate_pbft_vote(self, node_id: str, phase: str, proposal: Dict[str, Any]) -> bool:
        """PBFT投票をシミュレート"""
        await asyncio.sleep(0.02)
        return random.random() > 0.05  # 95%の成功率

    async def _elect_leader(self):
        """リーダー選出"""
        if not self.nodes:
            return
            
        # アクティブなノードから選出
        active_nodes = [n for n in self.nodes.values() if n.state == NodeState.ACTIVE]
        if not active_nodes:
            return
        
        # 最小のnode_idを持つノードをリーダーに選出（簡単なアルゴリズム）
        leader_node = min(active_nodes, key=lambda n: n.node_id)
        old_leader = self.leader_id
        self.leader_id = leader_node.node_id
        self.is_leader = (self.leader_id == self.node_id)
        
        if old_leader != self.leader_id:
            self.metrics['leader_elections'] += 1
            self.logger.info(f"新しいリーダー選出: {self.leader_id}")

    async def _heartbeat_loop(self):
        """ハートビートループ"""
        while self._running:
            try:
                # 自ノードのハートビートを更新
                if self.node_id in self.nodes:
                    self.nodes[self.node_id].last_heartbeat = time.time()
                
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"ハートビートエラー: {e}")

    async def _failure_detection_loop(self):
        """障害検出ループ"""
        while self._running:
            try:
                current_time = time.time()
                failed_nodes = []
                
                for node_id, node in self.nodes.items():
                    if node_id == self.node_id:
                        continue  # 自ノードはスキップ
                    
                    time_since_heartbeat = current_time - node.last_heartbeat
                    
                    if time_since_heartbeat > self.failure_timeout:
                        if node.state == NodeState.ACTIVE:
                            node.state = NodeState.SUSPECT
                            self.logger.warning(f"ノード {node_id} が応答不能の疑い")
                        elif node.state == NodeState.SUSPECT and time_since_heartbeat > self.failure_timeout * 2:
                            node.state = NodeState.DEAD
                            failed_nodes.append(node_id)
                            self.logger.error(f"ノード {node_id} が停止と判定")
                
                # 失敗したノードを処理
                for node_id in failed_nodes:
                    await self.unregister_node(node_id)
                
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"障害検出エラー: {e}")

    async def _task_scheduler_loop(self):
        """タスクスケジューラーループ"""
        while self._running:
            try:
                if self.is_leader and self.pending_tasks:
                    await self._schedule_tasks()
                
                await asyncio.sleep(1.0)
            except Exception as e:
                self.logger.error(f"タスクスケジューラーエラー: {e}")

    async def _schedule_tasks(self):
        """タスクをワーカーノードに割り当て"""
        async with self._coordination_lock:
            available_nodes = [
                n for n in self.nodes.values() 
                if n.state == NodeState.ACTIVE and n.load < self.max_load_threshold
            ]
            
            if not available_nodes:
                return
            
            # 優先度順にタスクをソート
            sorted_tasks = sorted(
                self.pending_tasks.items(),
                key=lambda x: (-x[1].priority, x[1].timestamp)
            )
            
            for task_id, task in sorted_tasks[:len(available_nodes)]:
                # 負荷分散戦略に基づいてノードを選択
                selected_node = self._select_node_for_task(task, available_nodes)
                
                if selected_node:
                    task.assigned_node = selected_node.node_id
                    # タスクを実行（実際の実装では分散実行）
                    success = await self._execute_task_on_node(task, selected_node)
                    
                    if success:
                        # 完了タスクに移動
                        self.completed_tasks[task_id] = self.pending_tasks.pop(task_id)
                        self.metrics['tasks_processed'] += 1
                    else:
                        # リトライまたは失敗処理
                        task.retries += 1
                        if task.retries >= task.max_retries:
                            self.pending_tasks.pop(task_id)
                            self.metrics['tasks_failed'] += 1
                            self.logger.error(f"タスク {task_id} が最大リトライ回数に達して失敗")

    def _select_node_for_task(self, task: Task, available_nodes: List[WorkerNode]) -> Optional[WorkerNode]:
        """タスクに適したノードを選択"""
        if not available_nodes:
            return None
        
        if self.load_balance_strategy == 'round_robin':
            # ラウンドロビン（簡略版）
            return available_nodes[len(self.completed_tasks) % len(available_nodes)]
        
        elif self.load_balance_strategy == 'least_loaded':
            # 最少負荷のノードを選択
            return min(available_nodes, key=lambda n: n.load)
        
        elif self.load_balance_strategy == 'capability_aware':
            # タスクの要求する機能を持つノードを選択
            required_capability = task.payload.get('capability', 'general')
            capable_nodes = [
                n for n in available_nodes 
                if required_capability in n.capabilities
            ]
            return min(capable_nodes, key=lambda n: n.load) if capable_nodes else available_nodes[0]
        
        return available_nodes[0]

    async def _execute_task_on_node(self, task: Task, node: WorkerNode) -> bool:
        """ノードでタスクを実行"""
        try:
            # 実際の実装では分散実行
            # ここではシミュレーション
            await asyncio.sleep(0.1)  # 実行時間をシミュレート
            
            # ノードの負荷を更新
            node.load = min(node.load + 0.1, 1.0)
            
            # 実行結果をタスクに設定
            task.payload['result'] = {
                'status': 'completed',
                'executed_by': node.node_id,
                'execution_time': 0.1
            }
            
            self.logger.debug(f"タスク {task.task_id} をノード {node.node_id} で実行完了")
            return True
            
        except Exception as e:
            self.logger.error(f"タスク実行エラー: {e}")
            return False

    async def _rebalance_tasks(self):
        """タスクの負荷再分散"""
        # 新しいノードが追加された際の負荷再分散
        self.logger.info("タスク負荷再分散を実行")
        # 実装は負荷分散戦略に依存

    async def _redistribute_tasks(self, failed_node_id: str):
        """失敗したノードのタスクを再配布"""
        self.logger.info(f"ノード {failed_node_id} のタスクを再配布")
        
        # 失敗したノードに割り当てられていたタスクを見つけて再配布
        for task in list(self.pending_tasks.values()):
            if task.assigned_node == failed_node_id:
                task.assigned_node = None
                task.retries += 1
                self.logger.info(f"タスク {task.task_id} を再配布のためリセット")

    def get_cluster_status(self) -> Dict[str, Any]:
        """クラスター状態を取得"""
        active_nodes = sum(1 for n in self.nodes.values() if n.state == NodeState.ACTIVE)
        total_load = sum(n.load for n in self.nodes.values() if n.state == NodeState.ACTIVE)
        avg_load = total_load / active_nodes if active_nodes > 0 else 0
        
        return {
            'node_id': self.node_id,
            'is_leader': self.is_leader,
            'leader_id': self.leader_id,
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'average_load': avg_load,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'metrics': self.metrics.copy()
        }

class DistributedLoadBalancer:
    """
    分散負荷分散器
    
    負荷に基づいてタスクを適切なワーカーに分散
    """
    
    def __init__(self, coordinator: DistributedCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        # 負荷監視設定
        self.load_check_interval = 5.0
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.min_nodes = 1
        self.max_nodes = 10
        
        self._running = False

    async def start(self):
        """負荷分散器開始"""
        if self._running:
            return
            
        self._running = True
        self.logger.info("分散負荷分散器開始")
        
        # 負荷監視ループ
        asyncio.create_task(self._load_monitoring_loop())

    async def stop(self):
        """負荷分散器停止"""
        self._running = False
        self.logger.info("分散負荷分散器停止")

    async def _load_monitoring_loop(self):
        """負荷監視ループ"""
        while self._running:
            try:
                await self._check_and_scale()
                await asyncio.sleep(self.load_check_interval)
            except Exception as e:
                self.logger.error(f"負荷監視エラー: {e}")

    async def _check_and_scale(self):
        """負荷チェックとスケーリング判定"""
        status = self.coordinator.get_cluster_status()
        avg_load = status['average_load']
        active_nodes = status['active_nodes']
        
        # スケールアップ判定
        if (avg_load > self.scale_up_threshold and 
            active_nodes < self.max_nodes and 
            self.coordinator.is_leader):
            
            await self._scale_up()
            
        # スケールダウン判定
        elif (avg_load < self.scale_down_threshold and 
              active_nodes > self.min_nodes and 
              self.coordinator.is_leader):
            
            await self._scale_down()

    async def _scale_up(self):
        """スケールアップ（ワーカー追加）"""
        self.logger.info("負荷増加によりワーカーを追加")
        
        # 新しいワーカーノードを作成
        new_node_id = f"worker_{int(time.time())}"
        new_node = WorkerNode(
            node_id=new_node_id,
            address=f"node://{new_node_id}",
            capabilities=['general']
        )
        
        await self.coordinator.register_node(new_node)

    async def _scale_down(self):
        """スケールダウン（ワーカー削除）"""
        self.logger.info("負荷減少によりワーカーを削除")
        
        # 最も負荷の少ないノードを削除
        active_nodes = [
            n for n in self.coordinator.nodes.values() 
            if n.state == NodeState.ACTIVE and n.node_id != self.coordinator.node_id
        ]
        
        if active_nodes:
            node_to_remove = min(active_nodes, key=lambda n: n.load)
            await self.coordinator.unregister_node(node_to_remove.node_id)

# Ray分散ワーカー（Rayが利用可能な場合）
if RAY_AVAILABLE:
    @ray.remote
    class DistributedWorkerActor:
        """Ray分散ワーカーアクター"""
        
        def __init__(self, worker_id: str, config: Dict[str, Any]):
            self.worker_id = worker_id
            self.config = config
            self.logger = logging.getLogger(f"DistributedWorker.{worker_id}")
            self.coordinator = DistributedCoordinator(config, worker_id)
            
        async def start(self):
            """ワーカー開始"""
            await self.coordinator.start()
            return True
            
        async def stop(self):
            """ワーカー停止"""
            await self.coordinator.stop()
            return True
            
        async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
            """タスク実行"""
            try:
                # タスク実行ロジック
                result = await self._process_task(task_data)
                return {'status': 'success', 'result': result}
            except Exception as e:
                self.logger.error(f"タスク実行エラー: {e}")
                return {'status': 'error', 'error': str(e)}
        
        async def _process_task(self, task_data: Dict[str, Any]) -> Any:
            """実際のタスク処理"""
            # 実装依存のタスク処理
            await asyncio.sleep(0.1)  # 処理時間をシミュレート
            return {"processed": True, "worker": self.worker_id}
        
        def get_status(self) -> Dict[str, Any]:
            """ワーカー状態取得"""
            return self.coordinator.get_cluster_status()

def create_distributed_coordinator(config: Dict[str, Any], node_id: Optional[str] = None) -> DistributedCoordinator:
    """分散協調システムを作成"""
    return DistributedCoordinator(config, node_id)

def create_load_balancer(coordinator: DistributedCoordinator) -> DistributedLoadBalancer:
    """負荷分散器を作成"""
    return DistributedLoadBalancer(coordinator)
