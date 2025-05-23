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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple, Callable
from MurmurNet.modules.model_factory import get_shared_model

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
    - メモリ最適化エージェントパターン
    
    属性:
        config: 設定辞書
        blackboard: 共有黒板
        num_agents: エージェント数
        roles: 役割リスト
        _memory_usage_tracker: メモリ使用量追跡
        _dynamic_pool_size: 動的プールサイズ
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
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
          # 安全のため、デフォルトでは並列モードを無効化
        self.parallel_mode = config.get('parallel_mode', False)
          # 共有モデルインスタンスを取得（シングルトンパターン）
        self.llm = get_shared_model(self.config)
        
        self._load_role_templates()
        self._load_roles()
          # 並列モードの場合の設定
        if self.parallel_mode:
            # スレッド間の同期のためにロックを作成
            self.model_lock = threading.Lock()
            logger.info("並列処理モードを初期化しました")
        
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
                
                if self.debug:
                    logger.debug(f"エージェント{i}の実行が完了しました")
                    
            except Exception as e:
                error_msg = f"エージェント{i}の実行エラー: {str(e)}"
                logger.error(error_msg)
                blackboard.write(f'agent_{i}_output', f"エージェント{i}は応答できませんでした")
                
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
            
            # モデル出力の生成（グローバルロックで保護、タイムアウト対応）
            import time
            start_time = time.time()
            
            with _global_llama_lock:
                # モデルの可用性チェック
                if not hasattr(self.llm, 'create_chat_completion') and not hasattr(self.llm, 'generate'):
                    logger.error(f"モデルインスタンスが正しく初期化されていません")
                    return f"{role_name}はモデルの問題で応答できませんでした"
                
                # チャット完了を試行
                if hasattr(self.llm, 'create_chat_completion'):
                    resp = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,  # 話し言葉に適したトークン数
                        temperature=temperature,
                        top_p=0.9,
                        stop=["。", ".", "\n\n"]
                    )
                else:
                    # フォールバック：generate メソッドを使用
                    resp_text = self.llm.generate(
                        prompt=prompt,
                        max_tokens=300,
                        temperature=temperature
                    )
                    # 辞書形式に変換
                    resp = {"choices": [{"message": {"content": resp_text}}]}
            
            generation_time = time.time() - start_time
            if generation_time > 20:  # 20秒以上の場合
                logger.warning(f"エージェント{agent_id}の応答生成に時間がかかりました: {generation_time:.2f}秒")
            
            # レスポンスの解析（堅牢化）
            try:
                if isinstance(resp, dict) and 'choices' in resp and resp['choices']:
                    response_text = resp['choices'][0]['message']['content'].strip()
                elif hasattr(resp, 'choices') and resp.choices:
                    response_text = resp.choices[0].message.content.strip()
                else:
                    logger.error(f"予期しないレスポンス形式: {type(resp)}")
                    response_text = str(resp).strip()
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"レスポンス解析エラー: {e}")
                return f"{role_name}は応答の解析に失敗しました"
            
            # 出力の検証とクリーニング
            if not response_text:
                logger.warning(f"エージェント{agent_id}が空の応答を生成")
                return f"{role_name}は適切な応答を生成できませんでした"
            
            # 出力長の制限
            if len(response_text) > 500:
                response_text = response_text[:500]
                logger.debug(f"エージェント{agent_id}の応答を切り詰めました")
            
            # 黒板への書き込み
            self.blackboard.write(f'agent_{agent_id}_output', response_text)
            
            logger.debug(f"エージェント{agent_id} ({role_name}) タスク完了: {len(response_text)}文字")
            return response_text
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"エージェント{agent_id} 実行エラー ({error_type}): {str(e)}")
              # エラータイプ別の適切な応答
            role_name = self.roles[agent_id].get('role', f'エージェント{agent_id}') if agent_id < len(self.roles) else f'エージェント{agent_id}'
            
            if "timeout" in str(e).lower():
                error_msg = f"{role_name}は処理時間の制限により応答できませんでした"
            elif "memory" in str(e).lower():
                error_msg = f"{role_name}はメモリ不足のため応答できませんでした"
            elif "connection" in str(e).lower() or "network" in str(e).lower():
                error_msg = f"{role_name}は接続の問題により応答できませんでした"
            else:
                error_msg = f"{role_name}は技術的な問題により応答できませんでした"
            
            # 黒板にエラー情報を記録
            self.blackboard.write(f'agent_{agent_id}_output', error_msg)
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