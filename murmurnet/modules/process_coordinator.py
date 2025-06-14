#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process Coordinator モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
単一責任: プロセス管理とライフサイクル制御

設計原則:
- 単一責任: プロセスの生成、管理、終了のみ
- KISS: シンプルなプロセス管理
- 分離: ビジネスロジックは含まない

作者: Yuhi Sonoki
"""

import logging
import multiprocessing as mp
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from MurmurNet.modules.process_agent_worker import AgentTask, AgentResult, worker_process_entry


class ProcessCoordinator:
    """
    単一責任: プロセス管理とライフサイクル制御
    
    設計原則:
    - プロセス生成と終了のみを管理
    - ビジネスロジックは含まない
    - シンプルなインターフェース
    """
    
    def __init__(self, num_processes: int = None):
        """
        プロセスコーディネーターの初期化
        
        引数:
            num_processes: プロセス数（None時は自動設定）
        """
        self.num_processes = num_processes or min(mp.cpu_count(), 4)
        self.logger = logging.getLogger('ProcessCoordinator')
        
        # プロセス間通信用キュー（KISS原則：シンプルなキュー）
        self.task_queue = mp.Queue(maxsize=self.num_processes * 2)
        self.result_queue = mp.Queue()
        
        # プロセス管理
        self.processes: List[mp.Process] = []
        self.is_running = False
        
        self.logger.info(f"ProcessCoordinator initialized with {self.num_processes} processes")
    
    def start(self) -> bool:
        """
        ワーカープロセスを開始
        
        戻り値:
            開始成功フラグ
        """
        if self.is_running:
            self.logger.warning("Processes already running")
            return True
        
        try:
            # ワーカープロセスを起動
            for i in range(self.num_processes):
                process = mp.Process(
                    target=worker_process_entry,
                    args=(self.task_queue, self.result_queue, i)
                )
                process.start()
                self.processes.append(process)
                self.logger.debug(f"Started worker process {i}: PID {process.pid}")
            
            self.is_running = True
            self.logger.info(f"Started {self.num_processes} worker processes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start processes: {e}")
            self.stop()
            return False
    
    def stop(self) -> bool:
        """
        ワーカープロセスを停止
        
        戻り値:
            停止成功フラグ
        """
        if not self.is_running:
            return True
        
        try:
            # 終了シグナルを送信
            for _ in self.processes:
                self.task_queue.put(None)
            
            # プロセス終了を待機（タイムアウト付き）
            for i, process in enumerate(self.processes):
                process.join(timeout=5.0)
                if process.is_alive():
                    self.logger.warning(f"Force terminating process {i}")
                    process.terminate()
                    process.join(timeout=2.0)
                    if process.is_alive():
                        self.logger.error(f"Failed to terminate process {i}")
            
            self.processes.clear()
            self.is_running = False
            self.logger.info("All worker processes stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping processes: {e}")
            return False
    
    def submit_task(self, task: AgentTask) -> bool:
        """
        タスクを送信
        
        引数:
            task: エージェントタスク
            
        戻り値:
            送信成功フラグ
        """
        if not self.is_running:
            self.logger.error("Processes not running")
            return False
        
        try:
            self.task_queue.put(task, timeout=1.0)
            return True
        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[AgentResult]:
        """
        結果を取得
        
        引数:
            timeout: タイムアウト秒数
            
        戻り値:
            エージェント結果（タイムアウト時はNone）
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None
    
    def get_all_results(self, expected_count: int, timeout: float = 30.0) -> List[AgentResult]:
        """
        全結果を取得
        
        引数:
            expected_count: 期待する結果数
            timeout: 総タイムアウト秒数
            
        戻り値:
            結果リスト
        """
        results = []
        start_time = time.time()
        
        while len(results) < expected_count:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                self.logger.warning(f"Timeout: got {len(results)}/{expected_count} results")
                break
            
            result = self.get_result(min(remaining_time, 1.0))
            if result:
                results.append(result)
        
        return results
    
    def __enter__(self):
        """コンテキストマネージャー開始"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        self.stop()
