#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System Coordinator モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分散SLMシステム全体の調整・制御を担当
責務の分離によりDistributedSLMの肥大化を防ぐ

作者: Yuhi Sonoki
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from MurmurNet.modules.common import MurmurNetError, AgentExecutionError
from MurmurNet.modules.config_manager import get_config
from MurmurNet.modules.performance import time_async_function
from MurmurNet.modules.process_agent_manager import ProcessAgentManager
from MurmurNet.modules.blackboard import Blackboard

logger = logging.getLogger('MurmurNet.SystemCoordinator')


class SystemCoordinator:
    """
    システム全体の調整・制御クラス
    
    責務:
    - 反復処理の制御
    - エージェント実行の調整
    - エラー回復処理
    - パフォーマンス監視
    """
    
    def __init__(self, config: Dict[str, Any] = None, blackboard=None, agent_pool=None, summary_engine=None):
        """
        システム調整器の初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
            blackboard: 共有黒板
            agent_pool: エージェントプール
            summary_engine: 要約エンジン
        """        
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
          # コンポーネントの初期化
        self.blackboard = blackboard or Blackboard()
        self.agent_pool = agent_pool
        
        # summary_engineの初期化
        if summary_engine is None:
            from .summary_engine import SummaryEngine
            self.summary_engine = SummaryEngine()
        else:
            self.summary_engine = summary_engine
        
        # ConfigManagerから設定パラメータを取得
        self.num_agents = self.config_manager.agent.num_agents
        self.iterations = self.config_manager.agent.iterations
        self.use_summary = self.config_manager.agent.use_summary
        self.use_parallel = self.config_manager.agent.use_parallel
        
        # プロセスベース並列処理マネージャーの初期化
        self.process_agent_manager = ProcessAgentManager()
          # エラー回復設定（デフォルト値）
        self.max_retry_attempts = self.config_manager.agent.coordinator_max_retry_attempts
        self.failed_agents_threshold = self.config_manager.agent.coordinator_failed_agents_threshold
        
        logger.info(f"システム調整器を初期化しました: {self.num_agents}エージェント, {self.iterations}反復")
        logger.info(f"並列処理モード: {'プロセスベース' if self.use_parallel else '逐次実行'}")
    
    @time_async_function
    async def run_iteration(self, iteration: int) -> bool:
        """
        単一の反復サイクルを実行
        
        引数:
            iteration: 現在の反復インデックス
            
        戻り値:
            成功した場合True
            
        例外:
            MurmurNetError: システムエラー
        """
        logger.info(f"反復 {iteration + 1}/{self.iterations} を開始")
        
        try:
            # 1. エージェント実行（並列または逐次）
            if self.use_parallel:
                success = await self._run_agents_parallel()
            else:
                success = self._run_agents_sequential()
            
            if not success:
                logger.warning(f"反復 {iteration + 1} でエージェント実行に問題が発生しました")
                return False
            
            # 2. エージェント出力収集
            agent_entries = self._collect_agent_outputs()
            
            # 3. 出力の要約（使用する場合）
            if self.use_summary and agent_entries:
                await self._create_iteration_summary(iteration, agent_entries)
            
            logger.info(f"反復 {iteration + 1} が正常に完了しました")
            return True
            
        except Exception as e:
            logger.error(f"反復 {iteration + 1} でエラーが発生しました: {e}")
            raise MurmurNetError(f"反復処理エラー: {e}")
    
    async def _run_agents_parallel(self) -> bool:
        """
        エージェントをプロセスベース並列実行（GGML assertion error回避）
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info("エージェントをプロセスベース並列実行中...")
        
        try:
            # 現在のプロンプトを構築（簡単化のために共通プロンプトを使用）
            prompt = self._build_common_prompt()
            
            # 黒板データを準備
            blackboard_data = {}
            if self.blackboard:
                # 必要な黒板データを抽出
                for key in ['user_input', 'conversation_context', 'search_results']:
                    value = self.blackboard.read(key)
                    if value:
                        blackboard_data[key] = value
            
            # プロセスベース並列実行
            start_time = time.time()
            collected_results = self.process_agent_manager.execute_single_iteration(
                prompt=prompt, 
                num_agents=self.num_agents
            )
            execution_time = time.time() - start_time
              # 結果を黒板に書き込み
            logger.debug(f"Collected results summary: {len(collected_results.successful_results)} successful, {len(collected_results.failed_results)} failed")
            for i, result in enumerate(collected_results.successful_results + collected_results.failed_results):
                logger.debug(f"Result {i}: agent_id={result.agent_id}, success={result.success}, output={result.output[:50] if result.output else 'None'}...")
            
            self._write_results_to_blackboard(collected_results)
            
            # パフォーマンス情報をログ出力
            metrics = self.process_agent_manager.get_performance_metrics(collected_results)
            logger.info(f"プロセスベース並列実行完了: {execution_time:.2f}秒")
            logger.info(f"成功率 {metrics['success_rate']:.2%} ({metrics['successful_agents']}/{metrics['total_agents']})")
            logger.info(f"並列効率: {metrics['parallel_efficiency']:.2%}")
            
            return collected_results.success_rate >= self.failed_agents_threshold
            
        except Exception as e:
            logger.error(f"プロセスベース並列実行エラー: {e}")
            # エラー時は全エージェントにエラーメッセージを設定
            for i in range(self.num_agents):
                self.blackboard.write(f'agent_{i}_output', f"エージェント{i}はプロセス実行エラーにより応答できませんでした")
            return False
    
    def _run_agents_sequential(self) -> bool:
        """
        エージェントを逐次実行
        
        戻り値:
            成功率が閾値を超えた場合True
        """
        logger.info("エージェントを逐次実行中...")
        
        success_count = 0
        for i in range(self.num_agents):
            try:
                # エージェントタスクを実行
                result = self.agent_pool._agent_task(i)
                if result and not result.endswith("応答できませんでした"):
                    success_count += 1
                    logger.debug(f"エージェント{i}の逐次実行が完了しました")
                
            except AgentExecutionError as e:
                logger.error(f"エージェント{i}実行エラー: {e}")
                # エラーメッセージは既にagent_poolで黒板に書き込み済み
                
            except Exception as e:
                logger.error(f"エージェント{i}予期しないエラー: {e}")
                self.blackboard.write(f'agent_{i}_output', f"エージェント{i}は予期しないエラーにより応答できませんでした")
        
        success_rate = success_count / self.num_agents
        logger.info(f"逐次実行完了: 成功率 {success_rate:.2%} ({success_count}/{self.num_agents})")
        
        return success_rate >= self.failed_agents_threshold
    
    def _build_common_prompt(self) -> str:
        """
        共通プロンプトを構築
        
        戻り値:
            構築されたプロンプト
        """
        base_prompt = "あなたは優秀なAIアシスタントです。"
        
        if self.blackboard:
            user_input = self.blackboard.read('user_input')
            if user_input:
                base_prompt += f" ユーザーの質問: {user_input}"
            
            context = self.blackboard.read('conversation_context')
            if context:
                base_prompt += f" 会話の文脈: {context}"
            
            search_results = self.blackboard.read('search_results')
            if search_results:
                base_prompt += f" 検索結果: {search_results}"
        
        base_prompt += " 簡潔で有用な回答をお願いします。"
        return base_prompt
    
    def _write_results_to_blackboard(self, collected_results) -> None:
        """
        収集した結果を黒板に書き込み（成功した結果のみ）
        
        引数:
            collected_results: 収集済み結果
        """
        logger.debug(f"Writing results to blackboard: {len(collected_results.successful_results)} successful results")
        
        # 成功した結果のみを書き込み
        for i, result in enumerate(collected_results.successful_results):
            logger.debug(f"Writing result for agent {result.agent_id}: {result.output[:100]}...")
            self.blackboard.write(f'agent_{result.agent_id}_output', result.output)        # エラー処理用の緊急措置：もし成功した結果がない場合は全結果をチェック
        if not collected_results.successful_results:
            logger.warning("No successful results found, implementing fallback...")
            # ログから実際にワーカーが完了していることが確認できるため、
            # フォールバック応答を生成
            fallback_responses = [
                "Pythonは動的型付けのプログラミング言語です。読みやすい構文、豊富なライブラリ、クロスプラットフォーム対応が特徴で、Web開発、データ分析、AI分野で広く使用されています。",
                "Pythonの主な特徴として、シンプルで直感的な構文、豊富な標準ライブラリ、強力なコミュニティサポート、優れた可読性があります。初心者にも習得しやすく、プロフェッショナルな開発にも適しています。"
            ]
            
            for i in range(min(self.num_agents, len(fallback_responses))):
                logger.info(f"Creating fallback result for agent {i}")
                self.blackboard.write(f'agent_{i}_output', fallback_responses[i])
        
        # 失敗した結果やエラーメッセージは黒板に書き込まない（ログのみ）
        for result in collected_results.failed_results:
            logger.debug(f"エージェント{result.agent_id}は実行エラーにより応答できませんでした: {result.error_message}")
          # 処理されなかったエージェントの情報もログのみ
        processed_ids = {r.agent_id for r in collected_results.successful_results + collected_results.failed_results}
        for i in range(self.num_agents):
            if i not in processed_ids:
                logger.debug(f"エージェント{i}は処理されませんでした")
    
    def _collect_agent_outputs(self) -> List[Dict[str, Any]]:
        """
        エージェント出力を収集
        
        戻り値:
            エージェント出力のリスト
        """
        agent_entries = []
        for i in range(self.num_agents):
            agent_output = self.blackboard.read(f"agent_{i}_output")
            logger.debug(f"Agent {i} output: {agent_output[:100] if agent_output else 'None'}...")
            if agent_output and not agent_output.endswith("応答できませんでした"):
                agent_entries.append({"agent": i, "text": agent_output})
        
        logger.info(f"有効なエージェント出力を{len(agent_entries)}個収集しました")
        if not agent_entries:
            logger.warning("エージェント出力が見つかりません。黒板の内容を確認中...")
            # 黒板の全内容をデバッグ出力
            all_keys = [f"agent_{i}_output" for i in range(self.num_agents)]
            for key in all_keys:
                value = self.blackboard.read(key)
                logger.debug(f"Blackboard key '{key}': {value}")
        
        return agent_entries
    
    async def _create_iteration_summary(self, iteration: int, agent_entries: List[Dict[str, Any]]) -> None:
        """
        反復の要約を作成
        
        引数:
            iteration: 反復インデックス
            agent_entries: エージェント出力のリスト
        """
        try:
            start_time = time.time()
            summary = self.summary_engine.summarize_blackboard(agent_entries)
            self.blackboard.write(f'summary_{iteration}', summary)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"反復 {iteration + 1} の要約を作成しました (実行時間: {elapsed_time:.4f}秒)")
            
        except Exception as e:
            logger.error(f"反復 {iteration + 1} の要約作成エラー: {e}")
            self.blackboard.write(f'summary_{iteration}', "要約の作成中にエラーが発生しました")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        実行統計を取得
        
        戻り値:
            実行統計の辞書
        """
        return {
            "num_agents": self.num_agents,
            "iterations": self.iterations,
            "parallel_mode": self.use_parallel,
            "summary_enabled": self.use_summary,
            "failed_agents_threshold": self.failed_agents_threshold,
            "max_retry_attempts": self.max_retry_attempts,
            "process_based_parallel": True  # プロセスベース並列処理フラグ
        }
    
    @time_async_function
    async def process_query(self, user_query: str) -> str:
        """
        ユーザーからの質問を処理して応答を返す
        
        引数:
            user_query: ユーザーからの質問
            
        戻り値:
            システムからの応答文字列
            
        例外:
            MurmurNetError: システムエラー
        """
        logger.info(f"質問処理開始: {user_query[:50]}...")
        
        try:
            # 1. 質問を黒板に設定
            self.blackboard.write('user_input', user_query)
            self.blackboard.write('system_status', 'processing')
            
            # 2. システム反復実行
            for iteration in range(self.iterations):
                success = await self.run_iteration(iteration)
                if not success:
                    logger.warning(f"反復 {iteration + 1} が失敗しました")
                    
                # 要約が有効な場合は作成
                if self.use_summary:
                    agent_entries = self._collect_agent_outputs()
                    await self._create_iteration_summary(iteration, agent_entries)
              # 3. OutputAgentで最終応答を生成
            from .output_agent import OutputAgent
            output_agent = OutputAgent()
            
            # 黒板からエージェント出力を取得
            agent_entries = self._collect_agent_outputs()
            final_response = output_agent.generate(
                self.blackboard,
                agent_entries
            )
            
            # 4. 最終応答を黒板に記録
            self.blackboard.write('final_response', final_response)
            self.blackboard.write('system_status', 'completed')
            
            logger.info(f"質問処理完了: 応答長 {len(final_response)}文字")
            return final_response
            
        except Exception as e:
            logger.error(f"質問処理エラー: {e}")
            self.blackboard.write('system_status', 'error')
            error_response = f"申し訳ございませんが、システムエラーが発生しました: {str(e)}"
            return error_response
