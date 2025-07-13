#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ローカルRayシステム
~~~~~~~~~~~~~~~~~
Rayを使わずにローカルで分散処理風の機能を提供

作者: Yuhi Sonoki
"""

import threading
import time
import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import queue
import pickle
import uuid

logger = logging.getLogger(__name__)

@dataclass
class LocalTask:
    """ローカルタスククラス"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: str = "pending"  # pending, running, done, failed
    result: Any = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class LocalRay:
    """Ray風のローカル分散処理システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 設定
        self.max_workers = config.get('max_workers', multiprocessing.cpu_count())
        self.use_processes = config.get('use_processes', False)
        
        # タスク管理
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.RLock()
        
        # ワーカー管理
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 統計
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_tasks': 0
        }
        
        self._shutdown = False
        self.logger.info(f"ローカルRayシステムを初期化しました (workers: {self.max_workers}, processes: {self.use_processes})")
    
    def remote(self, func: Callable):
        """リモート関数デコレータ"""
        def wrapper(*args, **kwargs):
            return self.submit_task(func, *args, **kwargs)
        return wrapper
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """タスク投入"""
        try:
            task_id = str(uuid.uuid4())
            task = LocalTask(
                id=task_id,
                func=func,
                args=args,
                kwargs=kwargs
            )
            
            with self.lock:
                self.tasks[task_id] = task
                self.stats['total_tasks'] += 1
                self.stats['active_tasks'] += 1
            
            # 実行者に投入
            future = self.executor.submit(self._execute_task, task_id)
            
            self.logger.debug(f"タスク投入: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"タスク投入エラー: {e}")
            return None
    
    def _execute_task(self, task_id: str):
        """タスク実行"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return
                
                task.status = "running"
                task.started_at = time.time()
            
            # 関数実行
            result = task.func(*task.args, **task.kwargs)
            
            with self.lock:
                task.status = "done"
                task.result = result
                task.completed_at = time.time()
                self.stats['completed_tasks'] += 1
                self.stats['active_tasks'] -= 1
            
            self.logger.debug(f"タスク完了: {task_id}")
            
        except Exception as e:
            with self.lock:
                task = self.tasks.get(task_id)
                if task:
                    task.status = "failed"
                    task.error = e
                    task.completed_at = time.time()
                    self.stats['failed_tasks'] += 1
                    self.stats['active_tasks'] -= 1
            
            self.logger.error(f"タスク実行エラー {task_id}: {e}")
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """タスク結果取得"""
        start_time = time.time()
        
        while True:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    raise ValueError(f"タスクが見つかりません: {task_id}")
                
                if task.status == "done":
                    return task.result
                elif task.status == "failed":
                    raise task.error
            
            # タイムアウトチェック
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"タスクタイムアウト: {task_id}")
            
            time.sleep(0.01)  # 短いポーリング間隔
    
    def wait(self, task_ids: List[str], timeout: Optional[float] = None) -> List[Any]:
        """複数タスクの完了待機"""
        results = []
        for task_id in task_ids:
            try:
                result = self.get_result(task_id, timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"タスク待機エラー {task_id}: {e}")
                results.append(None)
        return results
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """タスク状態取得"""
        with self.lock:
            task = self.tasks.get(task_id)
            return task.status if task else None
    
    def cancel_task(self, task_id: str) -> bool:
        """タスクキャンセル"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task or task.status in ["done", "failed"]:
                    return False
                
                task.status = "cancelled"
                self.stats['active_tasks'] -= 1
            
            return True
        except Exception as e:
            self.logger.error(f"タスクキャンセルエラー: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self.lock:
            current_stats = self.stats.copy()
            current_stats['worker_count'] = self.max_workers
            current_stats['use_processes'] = self.use_processes
            current_stats['task_count'] = len(self.tasks)
            return current_stats
    
    def cleanup_completed_tasks(self, max_age: float = 3600):
        """完了タスクのクリーンアップ"""
        current_time = time.time()
        cleanup_count = 0
        
        with self.lock:
            task_ids_to_remove = []
            for task_id, task in self.tasks.items():
                if (task.status in ["done", "failed", "cancelled"] and 
                    task.completed_at and 
                    (current_time - task.completed_at) > max_age):
                    task_ids_to_remove.append(task_id)
            
            for task_id in task_ids_to_remove:
                del self.tasks[task_id]
                cleanup_count += 1
        
        if cleanup_count > 0:
            self.logger.info(f"完了タスクをクリーンアップしました: {cleanup_count}個")
    
    def shutdown(self, wait: bool = True):
        """システムシャットダウン"""
        self._shutdown = True
        self.executor.shutdown(wait=wait)
        self.logger.info("ローカルRayシステムをシャットダウンしました")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

class LocalActor:
    """Actor風のローカルオブジェクト"""
    
    def __init__(self, cls, *args, **kwargs):
        self.instance = cls(*args, **kwargs)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def __getattr__(self, name):
        """メソッド呼び出しの代理"""
        if hasattr(self.instance, name):
            attr = getattr(self.instance, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    with self.lock:
                        return attr(*args, **kwargs)
                return wrapper
            return attr
        raise AttributeError(f"'{type(self.instance).__name__}' object has no attribute '{name}'")

# グローバルRayインスタンス
_local_ray_instance = None

def init(config: Dict[str, Any] = None):
    """ローカルRay初期化"""
    global _local_ray_instance
    if _local_ray_instance is None:
        _local_ray_instance = LocalRay(config or {})
    return _local_ray_instance

def shutdown():
    """ローカルRayシャットダウン"""
    global _local_ray_instance
    if _local_ray_instance:
        _local_ray_instance.shutdown()
        _local_ray_instance = None

def remote(func: Callable):
    """リモート関数デコレータ"""
    global _local_ray_instance
    if _local_ray_instance is None:
        init()
    return _local_ray_instance.remote(func)

def put(value: Any) -> str:
    """オブジェクトストア風の実装"""
    # 単純にグローバル辞書に保存
    if not hasattr(put, '_store'):
        put._store = {}
    
    obj_id = str(uuid.uuid4())
    put._store[obj_id] = value
    return obj_id

def get(obj_id: str) -> Any:
    """オブジェクト取得"""
    if hasattr(put, '_store') and obj_id in put._store:
        return put._store[obj_id]
    raise ValueError(f"オブジェクトが見つかりません: {obj_id}")

def wait(task_ids: List[str], timeout: Optional[float] = None) -> List[Any]:
    """タスク完了待機"""
    global _local_ray_instance
    if _local_ray_instance is None:
        init()
    return _local_ray_instance.wait(task_ids, timeout)

def get_result(task_id: str, timeout: Optional[float] = None) -> Any:
    """タスク結果取得"""
    global _local_ray_instance
    if _local_ray_instance is None:
        init()
    return _local_ray_instance.get_result(task_id, timeout)

class ActorHandle:
    """Actor参照"""
    def __init__(self, actor: LocalActor):
        self.actor = actor
    
    def __getattr__(self, name):
        return getattr(self.actor, name)

def create_actor(cls, *args, **kwargs) -> ActorHandle:
    """Actor作成"""
    actor = LocalActor(cls, *args, **kwargs)
    return ActorHandle(actor)
