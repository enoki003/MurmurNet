#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Pool モジュール
~~~~~~~~~~~~~~~~~~~
複数のエージェントを管理し、並列/逐次実行を制御
各エージェントの生成や実行を統合的に管理

作者: Yuhi Sonoki
"""

import logging
import os
import json
import re
import threading
import asyncio
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Tuple, Callable
from MurmurNet.modules.model_factory import ModelFactory
from MurmurNet.modules.performance import cpu_profile
from MurmurNet.modules.prompt_manager import get_prompt_manager

logger = logging.getLogger('MurmurNet.AgentPool')

# クラス外にグローバルロックを定義
_global_llama_lock = threading.Lock()

class AgentPoolManager:
    """
    分散SLMにおけるエージェントプールの管理
    
    責務:
    - 複数エージェントの生成と実行管理
    - 役割ベースの分担処理
    - 並列/逐次実行の制御
    - CPU最適化された実行
    
    属性:
        config: 設定辞書
        blackboard: 共有黒板
        num_agents: エージェント数
        roles: 役割リスト
        performance: パフォーマンスモニター
    """
    def __init__(self, config: Dict[str, Any], blackboard):
        """
        エージェントプールの初期化
        
        引数:
            config: 設定辞書
            blackboard: 共有黒板インスタンス
        """
        self.config = config
        self.blackboard = blackboard
        self.debug = config.get('debug', False)
        self.num_agents = config.get('num_agents', 2)
        
        # パフォーマンスモニターへの参照を取得
        self.performance = getattr(blackboard, 'performance', None)
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
          
        # CPU最適化のための設定        self.cpu_count = psutil.cpu_count(logical=True) or 8
        self.cpu_count_physical = psutil.cpu_count(logical=False) or 4
        
        # 並列処理の設定
        self.parallel_mode = config.get('use_parallel', False)
        self.optimal_threads = self._calculate_optimal_threads()
        
        # ModelFactoryからモデルを取得（共有インスタンス）
        self.llm = ModelFactory.get_shared_model(self.config)
        
        self._load_role_templates()
        self._load_roles()
        
        # 並列モードの場合の設定
        if self.parallel_mode:
            # CPU最適化されたスレッドプールを作成
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.optimal_threads,
                thread_name_prefix="MurmurNet-Agent"
            )
            logger.info(f"並列処理モードを初期化: {self.optimal_threads}スレッド")
        
        logger.info(f"エージェントプールを初期化 (エージェント数: {self.num_agents}, CPU最適化: {self.optimal_threads}スレッド)")

    def _calculate_optimal_threads(self) -> int:
        """
        CPU最適化のための最適スレッド数を計算
        
        戻り値:
            最適なスレッド数
        """
        # エージェント数とCPUコア数を考慮
        base_threads = min(self.num_agents, self.cpu_count_physical)
        
        # システム負荷に基づく調整
        try:
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.5
            if load_avg > self.cpu_count_physical * 0.8:
                # 高負荷時は控えめに
                optimal = max(base_threads // 2, 2)
            else:
                # 低負荷時は積極的に並列化
                optimal = min(base_threads * 2, self.cpu_count)
        except:
            optimal = base_threads
        
        return max(optimal, 2)  # 最低2スレッド

    def _load_role_templates(self) -> None:
        """
        役割テンプレートの初期化（内部メソッド）
        
        質問タイプ別の役割テンプレートを定義
        """
        # 質問タイプ別の役割テンプレート定義
        self.role_templates = {
            # 議論型質問用の役割
            "discussion": [
                {"role": "多角的視点AI", "system": "あなたは多角的思考のスペシャリストです。論点を多面的に分析して議論の全体像を示してください。", "temperature": 0.7},
                {"role": "批判的思考AI", "system": "あなたは批判的思考の専門家です。前提や論理に疑問を投げかけ、新たな視点を提供してください。", "temperature": 0.8},
                {"role": "実証主義AI", "system": "あなたはデータと証拠を重視する科学者です。事実に基づいた分析と検証可能な情報を提供してください。", "temperature": 0.6},
                {"role": "倫理的視点AI", "system": "あなたは倫理学者です。道徳的・倫理的観点から議論を分析し、価値判断の視点を提供してください。", "temperature": 0.7}
            ],
            
            # 計画・構想型質問用の役割
            "planning": [
                {"role": "実用主義AI", "system": "あなたは実用主義の専門家です。実行可能で具体的なアプローチを提案してください。", "temperature": 0.7},
                {"role": "創造的思考AI", "system": "あなたは創造的思考のスペシャリストです。革新的なアイデアと可能性を探索してください。", "temperature": 0.9},
                {"role": "戦略的視点AI", "system": "あなたは戦略家です。長期的な視点と全体像を考慮した計画を立案してください。", "temperature": 0.7},
                {"role": "リスク分析AI", "system": "あなたはリスク管理専門家です。潜在的な問題点と対策を特定してください。", "temperature": 0.6}
            ],
            
            # 情報提供型質問用の役割
            "informational": [
                {"role": "事実提供AI", "system": "あなたは情報の専門家です。正確で検証可能な事実情報を簡潔に提供してください。", "temperature": 0.5},
                {"role": "教育的視点AI", "system": "あなたは教育者です。わかりやすく体系的に情報を整理して説明してください。", "temperature": 0.6},
                {"role": "比較分析AI", "system": "あなたは比較分析の専門家です。異なる視点や選択肢を公平に比較してください。", "temperature": 0.7}
            ],
            
            # 一般会話型質問用の役割
            "conversational": [
                {"role": "共感的リスナーAI", "system": "あなたは共感的なリスナーです。相手の感情や意図を理解し、温かみのある応答をしてください。", "temperature": 0.8},
                {"role": "実用アドバイザーAI", "system": "あなたは日常の実用知識に詳しいアドバイザーです。役立つ情報や提案を提供してください。", "temperature": 0.7}
            ],
            
            # デフォルト役割（どのタイプにも当てはまらない場合）
            "default": [
                {"role": "バランス型AI", "system": "あなたは総合的な分析ができるバランス型AIです。公平で多面的な視点から回答してください。", "temperature": 0.7},
                {"role": "専門知識AI", "system": "あなたは幅広い知識を持つ専門家です。正確でわかりやすい情報を提供してください。", "temperature": 0.6}
            ]
        }

    def _load_roles(self) -> None:
        """役割の割り当て（内部メソッド）"""
        # 役割の選択（設定またはランダム）
        self.roles = []
        
        # 設定から役割タイプを取得
        role_type = self.config.get('role_type', 'default')
        if role_type not in self.role_templates:
            role_type = 'default'
            
        # 利用可能な役割テンプレート
        available_roles = self.role_templates[role_type]
        
        # エージェント数に合わせて役割を割り当て
        for i in range(self.num_agents):
            role_index = i % len(available_roles)  # 循環させる
            self.roles.append(available_roles[role_index])
            
        if self.debug:
            roles_info = ", ".join(role["role"] for role in self.roles)
            logger.debug(f"割り当てられた役割: {roles_info}")

    @cpu_profile
    def run_agents(self, blackboard) -> None:
        """
        すべてのエージェントを逐次実行
        
        引数:
            blackboard: 共有黒板
        """
        logger.info("エージェントを逐次実行中...")
        
        # パフォーマンス統計の記録
        if self.performance:
            self.performance.record_parallel_execution('sequential')
        
        # 各エージェントを順番に実行
        for i in range(self.num_agents):
            try:
                result = self._agent_task(i)
                blackboard.write(f'agent_{i}_output', result)
                
                if self.debug:
                    logger.debug(f"エージェント{i}の実行が完了しました")
                    
            except Exception as e:
                error_msg = f"エージェント{i}の実行エラー: {str(e)}"
                logger.error(error_msg)
                blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
                
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())

    @cpu_profile
    async def run_agents_parallel(self, blackboard) -> None:
        """
        エージェントを並列実行（CPU最適化版）
        
        引数:
            blackboard: 共有黒板
        """
        logger.info(f"エージェントを並列実行中: {self.optimal_threads}スレッド")
        
        # パフォーマンス統計の記録
        if self.performance:
            self.performance.record_parallel_execution('parallel')
            self.performance.record_parallel_execution('thread_pool')
        
        # スレッドプール内でエージェントタスクを実行するラッパー
        def run_agent_task(agent_id: int) -> Tuple[int, str]:
            """エージェントタスクのスレッドプールラッパー"""
            try:
                # CPU最適化されたエージェント実行
                result = self._agent_task_optimized(agent_id)
                return agent_id, result
            except Exception as e:
                logger.error(f"エージェント {agent_id} 実行エラー: {str(e)}")
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
                return agent_id, f"エージェント{agent_id}は応答できませんでした"
        
        # イベントループを取得
        loop = asyncio.get_event_loop()
        
        try:
            # 並列実行：すべてのエージェントを同時に実行
            tasks = []
            for i in range(self.num_agents):
                # 各エージェントのタスクを作成
                task = loop.run_in_executor(self.thread_pool, run_agent_task, i)
                tasks.append(task)
            
            # すべてのタスクを同時に実行して結果を取得
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を黒板に書き込み
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"エージェント処理エラー: {str(result)}")
                    # エラーの場合、エージェントIDを特定できないため、順番に処理
                    continue
                elif isinstance(result, tuple) and len(result) == 2:
                    agent_id, output = result
                    # 出力を必ず文字列に変換
                    output_str = str(output) if output is not None else ""
                    
                    # 空応答や無効応答のフィルタリングを強化
                    if output_str and output_str.strip() and len(output_str.strip()) > 2:
                        # エラーメッセージや無効応答の検出
                        if not any(pattern in output_str.lower() for pattern in [
                            '応答できませんでした', 'エラー', '申し訳', '生成できませんでした'
                        ]):
                            blackboard.write(f'agent_{agent_id}_output', output_str.strip())
                        else:
                            logger.warning(f"エージェント{agent_id}からエラー応答: {output_str[:50]}")
                            blackboard.write(f'agent_{agent_id}_output', f"エージェント{agent_id}は適切な応答を生成できませんでした")
                    else:
                        logger.warning(f"エージェント{agent_id}から空応答: '{output_str}'")
                        blackboard.write(f'agent_{agent_id}_output', f"エージェント{agent_id}は空の応答を返しました")
                        
        except Exception as e:
            logger.error(f"並列実行エラー: {str(e)}")
            # エラーが発生した場合は逐次実行にフォールバック
            logger.info("逐次実行にフォールバックします")
            self.run_agents(blackboard)

    def _agent_task_optimized(self, agent_id: int) -> str:
        """
        CPU最適化されたエージェントタスク実行
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        # プロンプトの構築
        prompt = self._format_prompt(agent_id)
        
        # エージェントの役割と設定
        role = self.roles[agent_id]
        temperature = role.get('temperature', 0.7)
        
        try:
            # CPU最適化されたモデル出力の生成
            with _global_llama_lock:
                # 短いトークン数でCPU負荷を軽減
                resp = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,  # CPUパフォーマンスのため短めに
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40,  # CPU最適化
                    repeat_penalty=1.1,  # 繰り返し防止
                    mirostat=0,  # CPU最適化のため無効化
                )
            
            # レスポンスの形式によって適切にアクセス
            if isinstance(resp, dict):
                output = resp['choices'][0]['message']['content']
            else:
                output = resp.choices[0].message.content
            
            # 必ず文字列に変換
            output = str(output).strip() if output else ""
                  
            # 出力を制限（200文字以内、CPU負荷軽減）
            if len(output) > 200:
                output = output[:200] + "..."
                
            return output
            
        except Exception as e:
            logger.error(f"エージェント{agent_id}のタスク実行エラー: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return f"エージェント{agent_id}は応答できませんでした"

    def _format_prompt(self, agent_id: int) -> str:
        """
        エージェント用のプロンプトをフォーマット（プロンプトマネージャー使用）
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            フォーマットされたプロンプト
        """
        # 入力情報の取得
        input_data = self.blackboard.read('input')
        if isinstance(input_data, dict) and 'normalized' in input_data:
            input_text = input_data['normalized']
        else:
            input_text = str(input_data)
            
        # 空の入力チェック（「お問い合わせスパム」対策）
        if not input_text or input_text.strip() == "":
            logger.warning(f"エージェント{agent_id}: 空の入力が検出されました")
            return "適切な質問を入力してください。"
            
        # RAG情報の取得
        rag_info = self.blackboard.read('rag')
        rag_text = str(rag_info) if rag_info else ""
        
        # 会話コンテキストの取得
        conversation_context = self.blackboard.read('conversation_context')
        context_text = str(conversation_context) if conversation_context else ""
        
        # エージェントの役割情報
        role = self.roles[agent_id]
        role_name = role.get('role', f"agent_{agent_id}")
        
        # 他のエージェントの出力を収集（簡潔版）
        other_agents_output = []
        for i in range(self.num_agents):
            if i != agent_id:  # 自分以外のエージェント
                output = self.blackboard.read(f'agent_{i}_output')
                if output and len(output.strip()) > 0:
                    other_role = self.roles[i].get('role', f"agent_{i}")
                    # 出力を短縮
                    short_output = output[:100] + "..." if len(output) > 100 else output
                    other_agents_output.append(f"{other_role}: {short_output}")
                    
        # コンテキスト情報を構築
        context_parts = []
        
        if rag_text and rag_text.strip() and rag_text != "RAG情報なし":
            context_parts.append(f"参考情報: {rag_text[:200]}")
        
        if context_text and context_text != "過去の会話はありません。":
            context_parts.append(f"会話履歴: {context_text[:150]}")
            
        if other_agents_output:
            context_parts.append(f"他エージェント意見: {'; '.join(other_agents_output[:2])}")
            
        # プロンプトマネージャーを使用してプロンプト構築
        try:
            from MurmurNet.modules.prompt_manager import get_prompt_manager
            
            # モデルタイプとモデル名を取得
            model_type = self.config.get('model_type', 'llama')
            model_name = self.config.get('huggingface_model_name', '') if model_type == 'huggingface' else ''
            
            prompt_manager = get_prompt_manager(model_type, model_name)
            
            # 統合コンテキスト
            full_context = "\n".join(context_parts) if context_parts else ""
            
            # プロンプト生成（役割別システムプロンプト付き）
            prompt = prompt_manager.build_prompt(
                user_text=input_text,
                role=role_name,
                context=full_context
            )
            
            logger.debug(f"エージェント{agent_id}({role_name}): プロンプト生成完了 {len(prompt)}文字")
            return prompt
            
        except Exception as e:
            logger.error(f"プロンプトマネージャーエラー (エージェント{agent_id}): {str(e)}")
            
            # フォールバック: 最小限のプロンプト（空回答防止）
            if not input_text.strip():
                return "具体的な質問をしてください。"
            
            return f"質問: {input_text}\n\n{role_name}として回答してください:"
            context_parts.append(f"会話履歴: {context_text[:150]}")
            
        if other_agents_output:
            context_parts.append(f"他の観点: {' | '.join(other_agents_output[:2])}")  # 最大2つの他意見
        
        context_for_prompt = "\n\n".join(context_parts)
        
        # プロンプトマネージャーを使用してプロンプト生成
        model_type = self.config.get('model_type', 'llama')
        prompt_manager = get_prompt_manager(model_type)
        
        # ロール名をプロンプトマネージャーの標準ロールにマッピング
        role_mapping = {
            "researcher": "researcher",
            "分析者": "researcher", 
            "研究者": "researcher",
            "critic": "critic",
            "批判者": "critic",
            "評論家": "critic",
            "writer": "writer", 
            "文書家": "writer",
            "記者": "writer",
            "judge": "judge",
            "判断者": "judge",
            "調整者": "judge",
        }
        
        # デフォルトロールの決定
        standard_role = role_mapping.get(role_name.lower(), "assistant")
        
        # プロンプト生成
        prompt = prompt_manager.build_prompt(
            user_text=input_text,
            role=standard_role,
            context=context_for_prompt
        )
        
        logger.debug(f"エージェント{agent_id}({role_name} → {standard_role}): プロンプト生成完了 ({len(prompt)}文字)")
        
        return prompt

    def _agent_task(self, agent_id: int) -> str:
        """
        単一エージェントのタスク実行（内部メソッド）
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        # 最適化版があれば優先して使用
        if hasattr(self, '_agent_task_optimized'):
            return self._agent_task_optimized(agent_id)
        
        # プロンプトの構築
        prompt = self._format_prompt(agent_id)
        
        # エージェントの役割と設定
        role = self.roles[agent_id]
        temperature = role.get('temperature', 0.7)
        
        try:
            # モデル出力の生成（グローバルロックで保護）
            with _global_llama_lock:
                resp = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,  # 話し言葉に適したトークン数
                    temperature=temperature,
                    top_p=0.9
                )
            
            # レスポンスの形式によって適切にアクセス
            if isinstance(resp, dict):
                output = resp['choices'][0]['message']['content'].strip()
            else:
                output = resp.choices[0].message.content.strip()
                  
            # 出力を制限（250文字以内、話し言葉に適したサイズ）
            if len(output) > 250:
                output = output[:250]
                
            return output
            
        except Exception as e:
            logger.error(f"エージェント{agent_id}のタスク実行エラー: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return f"エージェント{agent_id}は応答できませんでした"

    def get_agent_info(self, agent_id: int) -> Dict[str, Any]:
        """
        エージェントの情報を取得
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェント情報の辞書
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            return {"error": "無効なエージェントID"}
            
        role = self.roles[agent_id]
        
        return {
            "id": agent_id,
            "role": role.get("role", f"エージェント{agent_id}"),
            "description": role.get("system", "情報なし"),
            "temperature": role.get("temperature", 0.7)
        }

    def update_roles_based_on_question(self, question: str) -> None:
        """
        質問の内容に基づいて適切な役割を動的に更新
        
        引数:
            question: 入力された質問
        """
        # 質問の分析
        question_type = self._analyze_question_type(question)
        
        # 役割の再割り当て
        if question_type in self.role_templates:
            available_roles = self.role_templates[question_type]
            self.roles = []
            
            for i in range(self.num_agents):
                role_index = i % len(available_roles)
                self.roles.append(available_roles[role_index])
                
            logger.info(f"質問タイプ '{question_type}' に基づいて役割を更新しました")
            
            if self.debug:
                roles_info = ", ".join(role["role"] for role in self.roles)
                logger.debug(f"新しい役割: {roles_info}")

    def _analyze_question_type(self, question: str) -> str:
        """
        質問のタイプを分析
        
        引数:
            question: 質問文
            
        戻り値:
            質問タイプ
        """
        question_lower = question.lower()
        
        # 議論型質問の判定
        discussion_keywords = ['なぜ', 'どう思う', '議論', '意見', '考え', '賛成', '反対', '問題', '課題']
        if any(keyword in question_lower for keyword in discussion_keywords):
            return 'discussion'
        
        # 計画型質問の判定
        planning_keywords = ['どうやって', 'どのように', '方法', '手順', '計画', '戦略', '進め方', 'アプローチ']
        if any(keyword in question_lower for keyword in planning_keywords):
            return 'planning'
        
        # 情報提供型質問の判定
        info_keywords = ['何', '誰', 'いつ', 'どこ', '教えて', '説明', '情報', '知りたい', 'について']
        if any(keyword in question_lower for keyword in info_keywords):
            return 'informational'
        
        # 会話型質問の判定
        conv_keywords = ['こんにちは', 'ありがとう', 'お疲れ', '元気', '調子', '気分']
        if any(keyword in question_lower for keyword in conv_keywords):
            return 'conversational'
        
        return 'default'

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        エージェントプールのパフォーマンス統計を取得
        
        戻り値:
            パフォーマンス統計
        """
        stats = {
            'num_agents': self.num_agents,
            'parallel_mode': self.parallel_mode,
            'optimal_threads': self.optimal_threads,
            'cpu_count_logical': self.cpu_count,
            'cpu_count_physical': self.cpu_count_physical,
            'roles_count': len(self.roles)
        }
        
        if self.performance:
            stats.update(self.performance.get_performance_summary())
        
        return stats

    def __del__(self):
        """デストラクタ：リソースのクリーンアップ"""
        if hasattr(self, 'thread_pool'):
            try:
                self.thread_pool.shutdown(wait=False)
            except:
                pass

    def shutdown(self):
        """
        AgentPoolManagerの完全なシャットダウン処理
        
        スレッドプールと全てのリソースを適切に終了する
        """
        logger.info("AgentPoolManagerシャットダウン開始")
        
        try:
            # 1. スレッドプールのシャットダウン
            if hasattr(self, 'thread_pool') and self.thread_pool:
                logger.info("スレッドプールをシャットダウン中...")
                try:
                    # 進行中のタスクの完了を待つ（最大3秒）
                    self.thread_pool.shutdown(wait=True, timeout=3.0)
                    logger.debug("スレッドプールシャットダウン完了")
                except Exception as e:
                    logger.warning(f"スレッドプール強制終了: {e}")
                    # 強制終了
                    try:
                        self.thread_pool.shutdown(wait=False)
                    except:
                        pass
                finally:
                    self.thread_pool = None
            
            # 2. モデル参照のクリア
            if hasattr(self, 'llm'):
                logger.debug("モデル参照をクリア中...")
                self.llm = None
            
            # 3. 役割テンプレートとロールのクリア
            if hasattr(self, 'role_templates'):
                self.role_templates.clear()
            if hasattr(self, 'roles'):
                self.roles.clear()
            
            # 4. パフォーマンス参照のクリア
            self.performance = None
            self.blackboard = None
            
            logger.info("AgentPoolManagerシャットダウン完了")
            
        except Exception as e:
            logger.error(f"AgentPoolManagerシャットダウンエラー: {e}")
            # エラーが発生してもシャットダウンを継続
            import traceback
            logger.debug(traceback.format_exc())
