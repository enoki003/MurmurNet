#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
コンポーネントレジストリ
~~~~~~~~~~~~~~~~~~~
分散システム内のコンポーネント管理とライフサイクル制御

作者: Yuhi Sonoki
"""

import asyncio
import logging
import time
from typing import Dict, List, Set, Optional, Type, Any
from enum import Enum
from dataclasses import dataclass, field

from .interfaces import IComponent, ComponentState

logger = logging.getLogger(__name__)

@dataclass
class ComponentInfo:
    """コンポーネント情報"""
    component_id: str
    component_type: str
    instance: IComponent
    state: ComponentState = ComponentState.CREATED
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComponentRegistry:
    """
    コンポーネントレジストリ
    
    分散システム内のコンポーネントの登録、管理、ライフサイクル制御を行う
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # コンポーネント管理
        self._components: Dict[str, ComponentInfo] = {}
        self._factories: Dict[str, callable] = {}
        
        # 依存関係グラフ
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._reverse_dependency_graph: Dict[str, Set[str]] = {}
        
        # ステート管理
        self._state_change_callbacks: Dict[ComponentState, List[callable]] = {
            state: [] for state in ComponentState
        }
        
        # メトリクス
        self._metrics = {
            'components_registered': 0,
            'components_started': 0,
            'components_stopped': 0,
            'startup_failures': 0
        }
        
        self.logger.info("コンポーネントレジストリ初期化完了")

    def register_factory(self, component_type: str, factory: callable):
        """
        コンポーネントファクトリを登録
        
        Args:
            component_type: コンポーネントタイプ
            factory: ファクトリ関数
        """
        self._factories[component_type] = factory
        self.logger.info(f"ファクトリ登録: {component_type}")

    def register_component(self, 
                          component_id: str,
                          component: IComponent,
                          dependencies: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        コンポーネントを登録
        
        Args:
            component_id: コンポーネントID
            component: コンポーネントインスタンス
            dependencies: 依存するコンポーネントIDのリスト
            metadata: メタデータ
            
        Returns:
            成功時True
        """
        if component_id in self._components:
            self.logger.warning(f"コンポーネント既に登録済み: {component_id}")
            return False
        
        # コンポーネント情報を作成
        info = ComponentInfo(
            component_id=component_id,
            component_type=component.__class__.__name__,
            instance=component,
            dependencies=set(dependencies or []),
            metadata=metadata or {}
        )
        
        # 登録
        self._components[component_id] = info
        self._dependency_graph[component_id] = info.dependencies.copy()
        
        # 逆依存関係を更新
        for dep_id in info.dependencies:
            if dep_id not in self._reverse_dependency_graph:
                self._reverse_dependency_graph[dep_id] = set()
            self._reverse_dependency_graph[dep_id].add(component_id)
            
            # 依存先コンポーネントの依存者リストを更新
            if dep_id in self._components:
                self._components[dep_id].dependents.add(component_id)
        
        self._metrics['components_registered'] += 1
        self.logger.info(f"コンポーネント登録: {component_id} (依存: {info.dependencies})")
        
        return True

    def create_component(self, 
                        component_id: str,
                        component_type: str,
                        config: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        ファクトリを使用してコンポーネントを作成・登録
        
        Args:
            component_id: コンポーネントID
            component_type: コンポーネントタイプ
            config: 設定
            dependencies: 依存するコンポーネントIDのリスト
            metadata: メタデータ
            
        Returns:
            成功時True
        """
        if component_type not in self._factories:
            self.logger.error(f"ファクトリが見つかりません: {component_type}")
            return False
        
        try:
            # ファクトリでコンポーネントを作成
            factory = self._factories[component_type]
            component = factory(config or {})
            
            # 登録
            return self.register_component(
                component_id=component_id,
                component=component,
                dependencies=dependencies,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"コンポーネント作成エラー ({component_type}): {e}")
            return False

    def unregister_component(self, component_id: str) -> bool:
        """
        コンポーネントの登録解除
        
        Args:
            component_id: コンポーネントID
            
        Returns:
            成功時True
        """
        if component_id not in self._components:
            return False
        
        info = self._components[component_id]
        
        # 停止していない場合は停止
        if info.state not in [ComponentState.STOPPED, ComponentState.ERROR]:
            asyncio.create_task(self.stop_component(component_id))
        
        # 依存関係を削除
        for dep_id in info.dependencies:
            if dep_id in self._components:
                self._components[dep_id].dependents.discard(component_id)
            if dep_id in self._reverse_dependency_graph:
                self._reverse_dependency_graph[dep_id].discard(component_id)
        
        # 依存者に通知
        for dependent_id in info.dependents:
            if dependent_id in self._components:
                self._components[dependent_id].dependencies.discard(component_id)
        
        # 削除
        del self._components[component_id]
        if component_id in self._dependency_graph:
            del self._dependency_graph[component_id]
        if component_id in self._reverse_dependency_graph:
            del self._reverse_dependency_graph[component_id]
        
        self.logger.info(f"コンポーネント登録解除: {component_id}")
        return True

    async def start_component(self, component_id: str) -> bool:
        """
        コンポーネントを開始
        
        Args:
            component_id: コンポーネントID
            
        Returns:
            成功時True
        """
        if component_id not in self._components:
            self.logger.error(f"コンポーネントが見つかりません: {component_id}")
            return False
        
        info = self._components[component_id]
        
        if info.state == ComponentState.RUNNING:
            self.logger.warning(f"コンポーネント既に実行中: {component_id}")
            return True
        
        # 依存関係を確認・開始
        for dep_id in info.dependencies:
            if dep_id not in self._components:
                self.logger.error(f"依存コンポーネントが見つかりません: {dep_id}")
                return False
            
            dep_info = self._components[dep_id]
            if dep_info.state != ComponentState.RUNNING:
                # 依存コンポーネントを先に開始
                if not await self.start_component(dep_id):
                    self.logger.error(f"依存コンポーネントの開始に失敗: {dep_id}")
                    return False
        
        try:
            # コンポーネントを開始
            self._set_component_state(component_id, ComponentState.STARTING)
            
            await info.instance.start()
            
            info.started_at = time.time()
            self._set_component_state(component_id, ComponentState.RUNNING)
            self._metrics['components_started'] += 1
            
            self.logger.info(f"コンポーネント開始: {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"コンポーネント開始エラー ({component_id}): {e}")
            self._set_component_state(component_id, ComponentState.ERROR)
            self._metrics['startup_failures'] += 1
            return False

    async def stop_component(self, component_id: str) -> bool:
        """
        コンポーネントを停止
        
        Args:
            component_id: コンポーネントID
            
        Returns:
            成功時True
        """
        if component_id not in self._components:
            self.logger.error(f"コンポーネントが見つかりません: {component_id}")
            return False
        
        info = self._components[component_id]
        
        if info.state in [ComponentState.STOPPED, ComponentState.STOPPING]:
            return True
        
        # 依存者を先に停止
        for dependent_id in info.dependents:
            if dependent_id in self._components:
                dep_info = self._components[dependent_id]
                if dep_info.state == ComponentState.RUNNING:
                    await self.stop_component(dependent_id)
        
        try:
            self._set_component_state(component_id, ComponentState.STOPPING)
            
            await info.instance.stop()
            
            self._set_component_state(component_id, ComponentState.STOPPED)
            self._metrics['components_stopped'] += 1
            
            self.logger.info(f"コンポーネント停止: {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"コンポーネント停止エラー ({component_id}): {e}")
            self._set_component_state(component_id, ComponentState.ERROR)
            return False

    async def start_all(self) -> bool:
        """
        全コンポーネントを依存関係順に開始
        
        Returns:
            成功時True
        """
        self.logger.info("全コンポーネント開始")
        
        # トポロジカルソートで開始順序を決定
        start_order = self._topological_sort()
        
        for component_id in start_order:
            if not await self.start_component(component_id):
                self.logger.error(f"コンポーネント開始失敗により中断: {component_id}")
                return False
        
        self.logger.info("全コンポーネント開始完了")
        return True

    async def stop_all(self) -> bool:
        """
        全コンポーネントを逆依存関係順に停止
        
        Returns:
            成功時True
        """
        self.logger.info("全コンポーネント停止")
        
        # 逆順で停止
        start_order = self._topological_sort()
        stop_order = list(reversed(start_order))
        
        for component_id in stop_order:
            await self.stop_component(component_id)
        
        self.logger.info("全コンポーネント停止完了")
        return True

    def get_component(self, component_id: str) -> Optional[IComponent]:
        """コンポーネントインスタンスを取得"""
        if component_id in self._components:
            return self._components[component_id].instance
        return None

    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """コンポーネント情報を取得"""
        return self._components.get(component_id)

    def list_components(self, state: Optional[ComponentState] = None) -> List[str]:
        """コンポーネント一覧を取得"""
        if state is None:
            return list(self._components.keys())
        
        return [
            comp_id for comp_id, info in self._components.items()
            if info.state == state
        ]

    def get_status(self) -> Dict[str, Any]:
        """システムステータスを取得"""
        states = {}
        for state in ComponentState:
            states[state.value] = len([
                info for info in self._components.values()
                if info.state == state
            ])
        
        return {
            'total_components': len(self._components),
            'component_states': states,
            'metrics': self._metrics.copy()
        }

    def _set_component_state(self, component_id: str, state: ComponentState):
        """コンポーネントの状態を変更"""
        if component_id in self._components:
            old_state = self._components[component_id].state
            self._components[component_id].state = state
            
            # コールバックを実行
            for callback in self._state_change_callbacks[state]:
                try:
                    callback(component_id, old_state, state)
                except Exception as e:
                    self.logger.error(f"状態変更コールバックエラー: {e}")

    def add_state_change_callback(self, state: ComponentState, callback: callable):
        """状態変更コールバックを追加"""
        self._state_change_callbacks[state].append(callback)

    def _topological_sort(self) -> List[str]:
        """依存関係のトポロジカルソート"""
        # Kahn's algorithm
        in_degree = {comp_id: len(deps) for comp_id, deps in self._dependency_graph.items()}
        queue = [comp_id for comp_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 依存者の入次数を減らす
            for dependent in self._reverse_dependency_graph.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self._components):
            self.logger.warning("循環依存が検出されました")
        
        return result
