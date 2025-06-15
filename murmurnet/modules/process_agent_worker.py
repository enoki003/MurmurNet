#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process Agent Worker モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
単一責任: 独立プロセスでエージェントを実行する

設計原則:
- 単一責任: エージェント実行のみ
- KISS: シンプルなデータ構造とインターフェース
- 分離: プロセス間通信を明確に分離

作者: Yuhi Sonoki
"""

import logging
import multiprocessing as mp
import pickle
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentTask:
    """エージェントタスク（ピクル可能なシンプルなデータ構造）"""
    agent_id: int
    prompt: str
    role: str
    config: Dict[str, Any]
    blackboard_data: Dict[str, Any]
    
    def __post_init__(self):
        """データ検証（KISS原則）"""
        if not isinstance(self.agent_id, int) or self.agent_id < 0:
            raise ValueError("agent_id must be a non-negative integer")
        if not self.prompt:
            raise ValueError("prompt cannot be empty")


@dataclass
class AgentResult:
    """エージェント結果（ピクル可能なシンプルなデータ構造）"""
    agent_id: int
    response: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        """データ検証（KISS原則）"""
        if not isinstance(self.agent_id, int) or self.agent_id < 0:
            raise ValueError("agent_id must be a non-negative integer")


class ProcessAgentWorker:
    """
    単一責任: 独立プロセスでエージェント実行
    
    設計原則:
    - 各プロセスが独自のモデルインスタンスを持つ
    - シンプルなピクル可能なデータのみを使用
    - エラーハンドリングを明確に分離
    """    @staticmethod
    def run_agent(task: AgentTask) -> AgentResult:
        """
        エージェントを実行（静的メソッド：シンプルなインターフェース）
        
        引数:
            task: エージェントタスク
            
        戻り値:
            実行結果
        """
        start_time = time.time()
        
        try:
            # デバッグログを追加
            print(f"[DEBUG] Starting agent {task.agent_id} with prompt: {task.prompt[:50]}...")
            
            # プロセス内で独立したモデルインスタンスを作成
            from MurmurNet.modules.model_factory import ModelFactory
            from MurmurNet.modules.config_manager import get_config
            
            print(f"[DEBUG] Creating config manager for agent {task.agent_id}")
            
            # 新しいConfigManagerインスタンスを作成（プロセス独立）
            config_manager = get_config()
            
            print(f"[DEBUG] Creating model for agent {task.agent_id}")
            
            # プロセス独立のモデルインスタンスを作成
            model = ModelFactory.create_model(task.config)
            
            print(f"[DEBUG] Checking model availability for agent {task.agent_id}")
            
            if not model.is_available():
                print(f"[DEBUG] Model not available for agent {task.agent_id}")
                return AgentResult(
                    agent_id=task.agent_id,
                    response="",
                    success=False,
                    error_message="Model not available in worker process",
                    execution_time=time.time() - start_time
                )
            
            print(f"[DEBUG] Building role prompt for agent {task.agent_id}")
            
            # ロールベースのプロンプト構築（シンプルな文字列操作）
            role_prompt = ProcessAgentWorker._build_role_prompt(task.role, task.prompt)
            
            print(f"[DEBUG] Generating response for agent {task.agent_id}")
            
            # モデル実行（各プロセスが独立したモデルを使用）
            response = model.generate(role_prompt)
            
            print(f"[DEBUG] Agent {task.agent_id} completed successfully")
            
            return AgentResult(
                agent_id=task.agent_id,
                response=response,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"[DEBUG] Agent {task.agent_id} failed with error: {str(e)}")
            return AgentResult(
                agent_id=task.agent_id,
                response="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    @staticmethod
    def _build_role_prompt(role: str, prompt: str) -> str:
        """
        ロールベースプロンプト構築（KISS原則：シンプルな文字列操作）
        """
        role_templates = {
            "researcher": "あなたは研究者です。客観的で論理的な分析を行ってください。",
            "critic": "あなたは批評家です。批判的な視点で検討してください。", 
            "synthesizer": "あなたは統合者です。様々な視点をまとめてください。",
            "default": "あなたは専門的なアシスタントです。"
        }
        
        role_instruction = role_templates.get(role, role_templates["default"])
        return f"{role_instruction}\n\n{prompt}"


def worker_process_entry(task_queue: mp.Queue, result_queue: mp.Queue, worker_id: int):
    """
    ワーカープロセスのエントリーポイント（KISS原則：シンプルなループ）
    
    引数:
        task_queue: タスクキュー
        result_queue: 結果キュー  
        worker_id: ワーカーID
    """
    print(f"[DEBUG] Worker process {worker_id} entry point called")
    
    # プロセス独立のロガー設定
    logger = logging.getLogger(f'ProcessWorker-{worker_id}')
    logger.info(f"Worker process {worker_id} started")
    print(f"[DEBUG] Worker process {worker_id} started")
    
    try:
        print(f"[DEBUG] Worker {worker_id} entering main loop")
        while True:
            try:
                print(f"[DEBUG] Worker {worker_id} waiting for task...")
                # タスクを取得（タイムアウト付き）
                task = task_queue.get(timeout=1.0)
                
                if task is None:  # 終了シグナル
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    print(f"[DEBUG] Worker {worker_id} received shutdown signal")
                    break
                
                print(f"[DEBUG] Worker {worker_id} received task for agent {task.agent_id}")
                
                # エージェント実行
                result = ProcessAgentWorker.run_agent(task)
                
                print(f"[DEBUG] Worker {worker_id} completed task for agent {task.agent_id}")
                
                # 結果を送信
                result_queue.put(result)
                print(f"[DEBUG] Worker {worker_id} sent result for agent {task.agent_id}")
                
            except mp.TimeoutError:
                # タイムアウトは正常（継続）
                print(f"[DEBUG] Worker {worker_id} timeout, continuing...")
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                print(f"[DEBUG] Worker {worker_id} error: {e}")
                # エラー結果を送信
                error_result = AgentResult(
                    agent_id=-1,
                    response="",
                    success=False,
                    error_message=f"Worker process error: {str(e)}"
                )
                result_queue.put(error_result)
                
    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")
        print(f"[DEBUG] Worker {worker_id} fatal error: {e}")
    finally:
        logger.info(f"Worker process {worker_id} terminated")
        print(f"[DEBUG] Worker process {worker_id} terminated")
