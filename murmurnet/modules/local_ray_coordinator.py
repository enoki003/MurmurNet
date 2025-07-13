#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local Ray Coordinator モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ローカル分散処理用のRay最適化設定
同一マシン内でのマルチプロセス・マルチコア処理に特化

機能:
- シングルノードRayクラスター自動設定
- CPU/メモリ最適化
- ローカルプロセス間通信最適化
- 高速タスクスケジューリング

作者: Yuhi Sonoki
"""

import logging
import os
import psutil
import time
from typing import Dict, Any, Optional
import multiprocessing

logger = logging.getLogger(__name__)

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    raise ImportError("Ray is required for local distributed processing. Install with: pip install ray")

class LocalRayCoordinator:
    """
    ローカル分散処理用Rayコーディネーター
    
    同一マシン内でのRayクラスター最適化設定を提供
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        LocalRayCoordinator初期化
        
        Args:
            config: 設定辞書
                - num_cpus: 使用CPU数 (default: auto-detect)
                - memory_gb: 使用メモリ量GB (default: auto-detect)
                - object_store_memory: オブジェクトストアメモリ (default: auto-detect)
                - dashboard_port: ダッシュボードポート (default: 8265)
        """
        self.config = config or {}
        
        # システム情報取得
        self.total_cpus = multiprocessing.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Ray設定
        self.num_cpus = self.config.get('num_cpus', max(1, self.total_cpus - 1))  # 1コア予約
        self.memory_gb = self.config.get('memory_gb', max(1, int(self.total_memory_gb * 0.8)))  # 80%使用
        self.object_store_memory = self.config.get('object_store_memory', int(self.memory_gb * 0.3 * 1024**3))  # 30%をオブジェクトストア
        self.dashboard_port = self.config.get('dashboard_port', 8265)
        
        self.cluster_initialized = False
        
        logger.info(f"LocalRayCoordinator初期化: CPU={self.num_cpus}/{self.total_cpus}, Memory={self.memory_gb}GB/{self.total_memory_gb:.1f}GB")
    
    def init_cluster(self) -> bool:
        """
        ローカルRayクラスターを初期化
        
        Returns:
            bool: 初期化成功フラグ
        """
        if self.cluster_initialized:
            logger.info("Rayクラスターは既に初期化済みです")
            return True
        
        try:
            # 既存クラスターがあれば停止
            if ray.is_initialized():
                logger.info("既存Rayクラスターを停止中...")
                ray.shutdown()
                time.sleep(1)
            
            # ローカル最適化設定
            ray_config = {
                'num_cpus': self.num_cpus,
                'object_store_memory': self.object_store_memory,
                'dashboard_port': self.dashboard_port,
                'include_dashboard': True,
                'ignore_reinit_error': True,
                'log_to_driver': False,  # ログ量削減
                '_temp_dir': os.path.join(os.path.expanduser("~"), ".ray", "temp")
            }
            
            # Rayクラスター開始
            logger.info("ローカルRayクラスターを開始中...")
            ray.init(**ray_config)
            
            # 初期化確認
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            logger.info(f"✅ Rayクラスター初期化完了:")
            logger.info(f"   - CPU: {cluster_resources.get('CPU', 0)}")
            logger.info(f"   - Memory: {cluster_resources.get('memory', 0) / 1024**3:.1f}GB")
            logger.info(f"   - Object Store: {cluster_resources.get('object_store_memory', 0) / 1024**3:.1f}GB")
            logger.info(f"   - Dashboard: http://localhost:{self.dashboard_port}")
            
            self.cluster_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Rayクラスター初期化エラー: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        クラスター情報を取得
        
        Returns:
            Dict: クラスター情報
        """
        if not ray.is_initialized():
            return {"error": "Ray not initialized"}
        
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            info = {
                "initialized": True,
                "cluster_resources": dict(cluster_resources),
                "available_resources": dict(available_resources),
                "nodes": len(ray.nodes()),
                "dashboard_url": f"http://localhost:{self.dashboard_port}",
                "config": {
                    "num_cpus": self.num_cpus,
                    "memory_gb": self.memory_gb,
                    "object_store_memory_gb": self.object_store_memory / 1024**3
                }
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def shutdown(self) -> None:
        """Rayクラスターをシャットダウン"""
        try:
            if ray.is_initialized():
                logger.info("Rayクラスターをシャットダウン中...")
                ray.shutdown()
                self.cluster_initialized = False
                logger.info("Rayクラスターシャットダウン完了")
            
        except Exception as e:
            logger.error(f"Rayシャットダウンエラー: {e}")
    
    def is_healthy(self) -> bool:
        """クラスターヘルスチェック"""
        try:
            return ray.is_initialized() and len(ray.nodes()) > 0
        except:
            return False

# Ray タスク最適化デコレーター
def ray_task(num_cpus: float = 1.0, memory: Optional[int] = None):
    """
    ローカル分散処理用Rayタスクデコレーター
    
    Args:
        num_cpus: 使用CPU数
        memory: 使用メモリ量（バイト）
    """
    def decorator(func):
        if HAS_RAY:
            task_config = {'num_cpus': num_cpus}
            if memory:
                task_config['memory'] = memory
            return ray.remote(**task_config)(func)
        else:
            # Ray未使用時は通常関数として実行
            return func
    return decorator

def create_local_ray_coordinator(config: Dict[str, Any] = None) -> LocalRayCoordinator:
    """
    ローカルRayコーディネーターを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        LocalRayCoordinator: ローカルRayコーディネーター
    """
    if config is None:
        config = {}
    
    return LocalRayCoordinator(config)
