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
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any, List, Optional, Tuple, Callable
from MurmurNet.modules.model_factory import get_shared_model
from MurmurNet.modules.common import AgentExecutionError, ThreadSafetyError
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.AgentPool')

# エージェント毎の個別ロック管理
class AgentLockManager:
    """エージェント毎の個別ロック管理"""
    
    def __init__(self, num_agents: int):
        self._locks = {i: threading.Lock() for i in range(num_agents)}
        self._global_lock = threading.RLock()  # 管理用のロック
    
    def get_agent_lock(self, agent_id: int) -> threading.Lock:
        """指定エージェントのロックを取得"""
        with self._global_lock:
            if agent_id not in self._locks:
                self._locks[agent_id] = threading.Lock()
            return self._locks[agent_id]
    
    def cleanup(self):
        """リソースクリーンアップ"""
        with self._global_lock:
            self._locks.clear()

class AgentPoolManager:
    """
    分散SLMにおけるエージェントプールの管理
    
    責務:
    - 複数エージェントの生成と実行管理    - 役割ベースの分担処理
    - 並列/逐次実行の制御
    - メモリ最適化エージェントパターン
    
    属性:
        config: 設定辞書
        blackboard: 共有黒板
        num_agents: エージェント数
        roles: 役割リスト
        lock_manager: エージェント毎のロック管理
    """
    def __init__(self, config: Dict[str, Any] = None, blackboard=None):
        """
        エージェントプールの初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
            blackboard: 共有黒板インスタンス
        """
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
        self.blackboard = blackboard
        
        # ConfigManagerから直接設定値を取得
        self.debug = self.config_manager.logging.debug
        self.num_agents = self.config_manager.agent.num_agents
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
          # ConfigManagerから並列モード設定を取得
        self.parallel_mode = self.config_manager.agent.use_parallel
          # モデルプールを使用（真の並列処理）
        if self.parallel_mode:
            from MurmurNet.modules.model_pool import get_model_pool
            self.model_pool = get_model_pool()
            
            # モデルプールの事前初期化を実行（エラーを早期発見）
            try:
                logger.info("並列モード: モデルプールの事前初期化を開始します")
                self.model_pool._initialize_pool()
                
                # 初期化結果を確認
                stats = self.model_pool.get_stats()
                if stats['available_models'] == 0:
                    logger.error("モデルプールに利用可能なモデルがありません！")
                    raise RuntimeError("モデルプールの初期化に失敗しました")
                
                logger.info(f"モデルプール初期化成功: {stats['available_models']}個のモデルが利用可能")
                logger.info("モデルプールを使用した真の並列処理モードを初期化しました")
                
            except Exception as e:
                logger.error(f"モデルプールの初期化に失敗: {e}")
                # フォールバックとして通常モードに切り替え
                logger.warning("フォールバック: 通常モードに切り替えます")
                self.parallel_mode = False
                self.llm = get_shared_model(self.config)
        else:
            # 従来のシングルトンモデル
            self.llm = get_shared_model(self.config)
        
        # エージェント毎の個別ロック管理
        self.lock_manager = AgentLockManager(self.num_agents)
        self._load_role_templates()
        self._load_roles()
        
        # 並列モードの場合の設定
        if self.parallel_mode:
            logger.info("並列処理モードを初期化しました（個別ロック使用）")
        
        logger.info(f"エージェントプールを初期化しました (エージェント数: {self.num_agents})")

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
        role_type = self.config_manager.agent.role_type
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
    
    def run_agents(self, blackboard) -> None:
        """
        すべてのエージェントを逐次実行
        
        引数:
            blackboard: 共有黒板
        """
        logger.info(f"エージェントを逐次実行中... (エージェント数: {self.num_agents})")
        
        # 各エージェントを順番に実行
        for i in range(self.num_agents):
            try:
                result = self._agent_task(i)
                blackboard.write(f'agent_{i}_output', result)
                
                if self.debug:                logger.debug(f"エージェント{i}の実行が完了しました")
                    
            except AgentExecutionError as e:
                # エージェント実行エラーの適切な処理
                logger.error(f"エージェント実行エラー: {e}")
                blackboard.write(f'agent_{e.agent_id}_output', f"エージェント{e.agent_id}は実行エラーにより応答できませんでした")
                
            except TimeoutError as e:
                # タイムアウトエラーの処理
                logger.error(f"エージェント{i}がタイムアウトしました: {e}")
                blackboard.write(f'agent_{i}_output', f"エージェント{i}は処理時間制限により応答できませんでした")
                
            except Exception as e:
                # その他の予期しないエラー
                error_msg = f"エージェント{i}の予期しないエラー: {str(e)}"
                logger.error(error_msg)
                blackboard.write(f'agent_{i}_output', f"エージェント{i}は技術的な問題により応答できませんでした")
                
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        logger.info(f"エージェント実行完了 (実行数: {self.num_agents})")

    def _format_prompt(self, agent_id: int) -> str:
        """
        エージェント用のプロンプトをフォーマット（内部メソッド）
        
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
            
        # RAG情報の取得
        rag_info = self.blackboard.read('rag')
        rag_text = str(rag_info) if rag_info else "関連情報はありません。"
        
        # 会話コンテキストの取得
        conversation_context = self.blackboard.read('conversation_context')
        context_text = str(conversation_context) if conversation_context else "過去の会話はありません。"
        
        # エージェントの役割情報
        role = self.roles[agent_id]
        role_name = role.get('role', f"エージェント{agent_id}")
        role_desc = role.get('system', "あなたは質問に答えるAIアシスタントです。")
        
        # 他のエージェントの出力を収集
        other_agents_output = []
        for i in range(self.num_agents):
            if i != agent_id:  # 自分以外のエージェント
                output = self.blackboard.read(f'agent_{i}_output')
                if output:
                    other_role = self.roles[i].get('role', f"エージェント{i}")
                    other_agents_output.append(f"{other_role}の回答: {output[:200]}...")
                    
        other_agents_text = "\n\n".join(other_agents_output) if other_agents_output else "他のエージェントの出力はまだありません。"
                  # プロンプトの構築（話し言葉重視）
        prompt = f"""こんにちは！私は「{role_name}」だよ。

{role_desc}

質問: {input_text}

参考になりそうな情報: {rag_text}

これまでの会話: {context_text}

仲間たちの意見:
{other_agents_text}

お願い:
- 私らしい視点で話すね
- わかりやすく具体的に説明するよ
- みんなの意見も参考にして、より良い答えを考えてみる
- 150〜250文字くらいで話し言葉でお答えするね

それじゃあ、{role_name}として答えるよ:"""

        return prompt
    
    def _agent_task(self, agent_id: int) -> str:
        """
        単一エージェントのタスク実行（内部メソッド）
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        try:
            # エージェントIDの検証
            if agent_id < 0 or agent_id >= len(self.roles):
                logger.error(f"無効なエージェントID: {agent_id}")
                return f"エージェント{agent_id}は設定されていません"
            
            # プロンプトの構築
            prompt = self._format_prompt(agent_id)
            if not prompt:
                logger.error(f"エージェント{agent_id}のプロンプト生成に失敗")
                return f"エージェント{agent_id}はプロンプトを生成できませんでした"
              # エージェントの役割と設定
            role = self.roles[agent_id]
            temperature = max(0.1, min(1.0, role.get('temperature', 0.7)))
            role_name = role.get('role', f'エージェント{agent_id}')
            logger.debug(f"エージェント{agent_id} ({role_name}) タスク開始")
              # モデル出力の生成（個別エージェントロックで保護、タイムアウト対応）
            import time
            start_time = time.time()
            
            # エージェント毎の個別ロックを使用
            agent_lock = self.lock_manager.get_agent_lock(agent_id)
            with agent_lock:
                # 並列モードかどうかでモデルアクセス方法を変更
                if self.parallel_mode and hasattr(self, 'model_pool'):
                    # 並列モード：model_poolから専用モデルを取得（コンテキストマネージャー使用）
                    try:
                        # モデルプールの初期化を確実に実行
                        if not self.model_pool._initialized:
                            logger.info(f"エージェント{agent_id}: モデルプールの初期化を開始")
                            self.model_pool._initialize_pool()
                        
                        # プール統計情報をログ出力
                        stats = self.model_pool.get_stats()
                        logger.debug(f"エージェント{agent_id}: プール統計 - 利用可能: {stats['available_models']}, アクティブ: {stats['active_models']}")
                        
                        with self.model_pool.get_model(timeout=30.0) as model:
                            # モデルインスタンスの詳細チェック
                            if model is None:
                                logger.error(f"エージェント{agent_id}: モデルプールからNoneが返されました")
                                return f"{role_name}はモデルの取得に失敗しました"
                            
                            # モデルの初期化状態を確認
                            if not hasattr(model, '_llm') or model._llm is None:
                                logger.error(f"エージェント{agent_id}: モデルの内部インスタンス(_llm)が初期化されていません")
                                return f"{role_name}のモデルが正しく初期化されていません"
                            
                            # モデルの可用性チェック
                            if not model.is_available():
                                logger.error(f"エージェント{agent_id}: モデルが利用不可能な状態です")
                                return f"{role_name}のモデルが利用できません"
                            
                            # メソッドの存在チェック
                            if not hasattr(model, 'create_chat_completion') and not hasattr(model, 'generate'):
                                logger.error(f"エージェント{agent_id}: モデルに必要なメソッドがありません")
                                return f"{role_name}はモデルメソッドの問題で応答できませんでした"
                            
                            # チャット完了を試行
                            if hasattr(model, 'create_chat_completion'):
                                resp = model.create_chat_completion(
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=300,  # 話し言葉に適したトークン数
                                    temperature=temperature,
                                    top_p=0.9,
                                    stop=["。", ".", "\n\n"]
                                )
                                if resp and 'choices' in resp and len(resp['choices']) > 0:
                                    content = resp['choices'][0]['message']['content'].strip()
                                    logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                                    return content
                                else:
                                    logger.warning(f"エージェント{agent_id}の応答が空でした")
                                    return f"{role_name}は適切な応答を生成できませんでした"
                            elif hasattr(model, 'generate'):
                                resp = model.generate(prompt, max_tokens=300, temperature=temperature)
                                if resp:
                                    logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                                    return resp.strip()
                                else:
                                    logger.warning(f"エージェント{agent_id}の応答が空でした")
                                    return f"{role_name}は適切な応答を生成できませんでした"
                    except Exception as e:
                        logger.error(f"エージェント{agent_id} モデルプールエラー: {type(e).__name__}: {e}")
                        import traceback
                        logger.debug(f"エージェント{agent_id} モデルプールエラー詳細:\n{traceback.format_exc()}")
                        return f"{role_name}はモデルプールエラーで応答できませんでした"
                else:
                    # 通常モード：共有llmを使用
                    model = self.llm
                    
                    # モデルの可用性チェック
                    if model is None:
                        logger.error(f"エージェント{agent_id}: 共有モデルが初期化されていません")
                        return f"{role_name}は共有モデルの問題で応答できませんでした"
                    
                    if not hasattr(model, 'create_chat_completion') and not hasattr(model, 'generate'):
                        logger.error(f"エージェント{agent_id}: 共有モデルに必要なメソッドがありません")
                        return f"{role_name}は共有モデルのメソッド問題で応答できませんでした"
                    
                    # チャット完了を試行
                    if hasattr(model, 'create_chat_completion'):
                        resp = model.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,  # 話し言葉に適したトークン数
                            temperature=temperature,
                            top_p=0.9,
                            stop=["。", ".", "\n\n"]
                        )
                        if resp and 'choices' in resp and len(resp['choices']) > 0:
                            content = resp['choices'][0]['message']['content'].strip()
                            logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                            return content
                        else:
                            logger.warning(f"エージェント{agent_id}の応答が空でした")
                            return f"{role_name}は適切な応答を生成できませんでした"
                    else:
                        # フォールバック：generate メソッドを使用
                        resp_text = model.generate(
                            prompt=prompt,
                            max_tokens=300,
                            temperature=temperature
                        )
                        if resp_text:
                            logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                            return resp_text.strip()
                        else:
                            logger.warning(f"エージェント{agent_id}の応答が空でした")
                            return f"{role_name}は適切な応答を生成できませんでした"
            
        except TimeoutError as e:
            # タイムアウト専用処理
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は処理時間の制限により応答できませんでした"
            logger.error(f"エージェント{agent_id} タイムアウト: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            raise AgentExecutionError(agent_id, "タイムアウト", e)
            
        except MemoryError as e:
            # メモリ不足専用処理
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}はメモリ不足のため応答できませんでした"
            logger.error(f"エージェント{agent_id} メモリ不足: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            raise AgentExecutionError(agent_id, "メモリ不足", e)
            
        except Exception as e:
            # その他の例外
            error_type = type(e).__name__
            logger.error(f"エージェント{agent_id} 実行エラー ({error_type}): {str(e)}")
            
            # エラータイプ別の適切な応答
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            
            if "connection" in str(e).lower() or "network" in str(e).lower():
                error_msg = f"{role_name}は接続の問題により応答できませんでした"
            else:
                error_msg = f"{role_name}は技術的な問題により応答できませんでした"
            
            # 黒板にエラー情報を記録
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            
            # カスタム例外として再発生
            raise AgentExecutionError(agent_id, str(e), e)
            return error_msg

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
        # 質問タイプを判定
        question_type = self._classify_question_type(question)
        
        # 黒板に質問タイプを書き込み
        self.blackboard.write('question_type', question_type)
        
        # 質問タイプに基づいて役割を更新
        if question_type in self.role_templates:
            available_roles = self.role_templates[question_type]
        else:
            available_roles = self.role_templates['default']
        
        # エージェント数に合わせて役割を再割り当て
        self.roles = []
        for i in range(self.num_agents):
            role_index = i % len(available_roles)
            self.roles.append(available_roles[role_index])
        
        # agent_rolesという属性も設定（テストコードで使用されている）
        self.agent_roles = self.roles
        
        if self.debug:
            roles_info = ", ".join(role["role"] for role in self.roles)
            logger.debug(f"質問タイプ '{question_type}' に基づいて役割を更新: {roles_info}")

    def _classify_question_type(self, question: str) -> str:
        """
        質問のタイプを分類（内部メソッド）
        
        引数:
            question: 分析する質問
            
        戻り値:
            質問タイプ ('discussion', 'planning', 'informational', 'conversational', 'default')
        """
        question_lower = question.lower()
        
        # 議論型のキーワード
        discussion_keywords = [
            'について議論', '議論して', '賛成', '反対', '問題', '課題', '影響',
            'どう思う', 'どう考える', 'メリット', 'デメリット', '比較',
            'discuss', 'debate', 'argue', 'opinion', 'think', 'pros', 'cons'
        ]
        
        # 計画・構想型のキーワード
        planning_keywords = [
            'どうすれば', 'どのように', '方法', '手順', '計画', '戦略',
            '実現', '達成', '解決', '改善', '対策', '案', '提案',
            'how to', 'plan', 'strategy', 'solve', 'achieve', 'implement'
        ]
        
        # 情報提供型のキーワード
        informational_keywords = [
            'とは', 'なに', '何', '定義', '説明', '詳細', '仕組み',
            '原因', '理由', 'なぜ', '歴史', '背景', '特徴',
            'what is', 'define', 'explain', 'why', 'how', 'cause', 'reason'
        ]
        
        # 会話型のキーワード
        conversational_keywords = [
            'こんにちは', 'はじめまして', 'おはよう', 'こんばんは',
            'ありがとう', 'すみません', 'お疲れ', '元気',
            'hello', 'hi', 'good morning', 'thank you', 'sorry'
        ]
        
        # キーワードマッチング
        if any(keyword in question_lower for keyword in discussion_keywords):
            return 'discussion'
        elif any(keyword in question_lower for keyword in planning_keywords):
            return 'planning'
        elif any(keyword in question_lower for keyword in informational_keywords):
            return 'informational'
        elif any(keyword in question_lower for keyword in conversational_keywords):            return 'conversational'
        else:
            return 'default'
    
    # 追加メソッド: 並列実行最適化
    async def run_agent_parallel(self, agent_id: int) -> str:
        """
        エージェントを並列実行に最適化された方法で実行
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        try:
            # 非同期で並列実行（ロックなし）
            loop = asyncio.get_event_loop()
            
            # 専用のモデルインスタンスまたは並列対応実行
            result = await loop.run_in_executor(
                None, 
                self._agent_task_parallel_safe, 
                agent_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"エージェント{agent_id}並列実行エラー: {e}")
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は並列実行エラーにより応答できませんでした"
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            return error_msg

    def _agent_task_parallel_safe(self, agent_id: int) -> str:
        """
        並列実行に安全なエージェントタスク（ロックなし版）
        
        引数:
            agent_id: エージェントID
            
        戻り値:
            エージェントの応答テキスト
        """
        try:
            # エージェントIDの検証
            if agent_id < 0 or agent_id >= len(self.roles):
                logger.error(f"無効なエージェントID: {agent_id}")
                return f"エージェント{agent_id}は設定されていません"
            
            # プロンプトの構築
            prompt = self._format_prompt(agent_id)
            if not prompt:
                logger.error(f"エージェント{agent_id}のプロンプト生成に失敗")
                return f"エージェント{agent_id}はプロンプトを生成できませんでした"
            
            # エージェントの役割と設定
            role = self.roles[agent_id]
            temperature = max(0.1, min(1.0, role.get('temperature', 0.7)))
            role_name = role.get('role', f'エージェント{agent_id}')
            
            logger.debug(f"エージェント{agent_id} ({role_name}) 並列タスク開始")
            
            # 並列実行用のモデル呼び出し（ロックなし）
            import time
            start_time = time.time()
            
            try:
                # 並列モードかどうかでモデルアクセス方法を変更
                if self.parallel_mode and hasattr(self, 'model_pool'):
                    # 並列モード：model_poolから専用モデルを取得（コンテキストマネージャー使用）
                    with self.model_pool.get_model(timeout=30.0) as model:
                        if model is None:
                            logger.error(f"エージェント{agent_id}: モデルプールからNoneが返されました")
                            return f"{role_name}はモデルの取得に失敗しました"
                        
                        # モデルの初期化状態を確認
                        if not hasattr(model, '_llm') or model._llm is None:
                            logger.error(f"エージェント{agent_id}: モデルの内部インスタンス(_llm)が初期化されていません")
                            return f"{role_name}のモデルが正しく初期化されていません"
                        
                        # モデルの可用性チェック
                        if not model.is_available():
                            logger.error(f"エージェント{agent_id}: モデルが利用不可能な状態です")
                            return f"{role_name}のモデルが利用できません"
                        
                        # メソッドの存在チェック
                        if not hasattr(model, 'create_chat_completion') and not hasattr(model, 'generate'):
                            logger.error(f"エージェント{agent_id}: モデルに必要なメソッドがありません")
                            return f"{role_name}はモデルメソッドの問題で応答できませんでした"
                        
                        # チャット完了を試行
                        if hasattr(model, 'create_chat_completion'):
                            resp = model.create_chat_completion(
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=300,
                                temperature=temperature,
                                top_p=0.9,
                                stop=["。", ".", "\n\n"]
                            )
                            if resp and 'choices' in resp and len(resp['choices']) > 0:
                                content = resp['choices'][0]['message']['content'].strip()
                                logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                                return content
                            else:
                                logger.warning(f"エージェント{agent_id}の応答が空でした")
                                return f"{role_name}は適切な応答を生成できませんでした"
                        elif hasattr(model, 'generate'):
                            resp = model.generate(prompt, max_tokens=300, temperature=temperature)
                            if resp:
                                logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                                return resp.strip()
                            else:
                                logger.warning(f"エージェント{agent_id}の応答が空でした")
                                return f"{role_name}は適切な応答を生成できませんでした"
                else:
                    # 通常モード：共有llmを使用
                    model = self.llm
                    
                    # モデルの可用性チェック
                    if model is None:
                        logger.error(f"エージェント{agent_id}: 共有モデルが初期化されていません")
                        return f"{role_name}は共有モデルの問題で応答できませんでした"
                    
                    if not hasattr(model, 'create_chat_completion') and not hasattr(model, 'generate'):
                        logger.error(f"エージェント{agent_id}: 共有モデルに必要なメソッドがありません")
                        return f"{role_name}は共有モデルのメソッド問題で応答できませんでした"
                    
                    # チャット完了を試行
                    if hasattr(model, 'create_chat_completion'):
                        resp = model.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            temperature=temperature,
                            top_p=0.9,
                            stop=["。", ".", "\n\n"]
                        )
                        if resp and 'choices' in resp and len(resp['choices']) > 0:
                            content = resp['choices'][0]['message']['content'].strip()
                            logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                            return content
                        else:
                            logger.warning(f"エージェント{agent_id}の応答が空でした")
                            return f"{role_name}は適切な応答を生成できませんでした"
                    else:
                        # フォールバック：generate メソッドを使用
                        resp_text = model.generate(
                            prompt=prompt,
                            max_tokens=300,
                            temperature=temperature
                        )
                        if resp_text:
                            logger.debug(f"エージェント{agent_id} ({role_name})の応答生成成功")
                            return resp_text.strip()
                        else:
                            logger.warning(f"エージェント{agent_id}の応答が空でした")
                            return f"{role_name}は適切な応答を生成できませんでした"
                
            except Exception as model_error:
                logger.error(f"エージェント{agent_id} モデル実行エラー: {type(model_error).__name__}: {model_error}")
                import traceback
                logger.debug(f"エージェント{agent_id} モデル実行エラー詳細:\n{traceback.format_exc()}")
                return f"{role_name}はモデル実行エラーにより応答できませんでした"
            
        except Exception as e:
            # 全般的なエラーハンドリング
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            error_msg = f"{role_name}は並列実行エラーにより応答できませんでした"
            logger.error(f"エージェント{agent_id} 並列実行エラー: {e}")
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
            return error_msg