#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Pool モジュール
~~~~~~~~~~~~~~~~~~~
複数のエージェントを管理し、並列/逐次実行を制御
各エージェントの生成や実行を統合的に管理

作者: Yuhi Sonoki
"""

from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
import os
import json
from typing import Dict, Any, List
import re
import threading

# クラス外にグローバルロックを定義
_global_llama_lock = threading.Lock()

class AgentPoolManager:
    """
    分散SLMにおけるエージェントプールの管理
    - 複数エージェントの生成と実行管理
    - 役割ベースの分担処理
    """
    def __init__(self, config: Dict[str, Any], blackboard):
        self.config = config
        self.blackboard = blackboard
        self.debug = config.get('debug', False)
        self.num_agents = config.get('num_agents', 2)
        self.model_lock = None  # 並列処理用のローカルロック
        
        # 安全のため、デフォルトでは並列モードを無効化
        self.parallel_mode = config.get('parallel_mode', False)
        
        self._load_model()
        self._load_role_templates()
        self._load_roles()
        
        # 並列モードの場合の設定
        if self.parallel_mode:
            try:
                # スレッド間の同期のためにロックを作成
                self.model_lock = threading.Lock()
                
                if self.debug:
                    print("並列処理モードを初期化しています...")
                    
                # 現在の実装では並列モードでも1つのモデルインスタンスを共有
                # 各リクエストはグローバルロックで保護され、1つずつ処理される
                self.agent_models = []
            except Exception as e:
                if self.debug:
                    print(f"並列モード初期化エラー: {e}")
                self.parallel_mode = False
                self.agent_models = []

    def _load_model(self):
        """モデルの初期化（内部メソッド）"""
        model_path = self.config.get('model_path')
        if not model_path:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                        '../../models/gemma-3-1b-it-q4_0.gguf'))
            
        chat_template = self.config.get('chat_template')
        params_path = self.config.get('params')
        
        # モデルパラメータのロード
        llama_kwargs = {
            'model_path': model_path,
            'n_ctx': self.config.get('n_ctx', 1024),  # 親設定から受け取る
            'n_threads': self.config.get('n_threads', 4),  # 親設定から受け取る
            'use_mmap': True,
            'use_mlock': False,
            'n_gpu_layers': 0,
            'seed': 42,
            'chat_format': "gemma",
            'verbose': False  # ログ出力抑制
        }
        
        # JSONパラメータがあれば追加ロード
        if params_path and os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    llama_kwargs.update(params)
            except Exception as e:
                if self.debug:
                    print(f"パラメータロードエラー: {e}")
        
        # チャットテンプレートがあれば設定
        if chat_template and os.path.exists(chat_template):
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    llama_kwargs['chat_template'] = f.read()
            except Exception as e:
                if self.debug:
                    print(f"テンプレートロードエラー: {e}")
        
        # デバッグログ
        if self.debug:
            print(f"エージェントプール: モデル設定 n_ctx={llama_kwargs['n_ctx']}, n_threads={llama_kwargs['n_threads']}")
                    
        self.llm = Llama(**llama_kwargs)

    def _create_model_instance(self, agent_id=0):
        """各エージェント用の独立したモデルインスタンスを作成（内部メソッド）"""
        model_path = self.config.get('model_path')
        if not model_path:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                      '../../models/gemma-3-1b-it-q4_0.gguf'))
            
        chat_template = self.config.get('chat_template')
        params_path = self.config.get('params')
        
        # モデルパラメータのロード - エージェントごとに異なるシードを使用
        llama_kwargs = {
            'model_path': model_path,
            'n_ctx': self.config.get('n_ctx', 1024),
            'n_threads': max(1, self.config.get('n_threads', 4) // self.num_agents),  # スレッド数を分散
            'use_mmap': True,
            'use_mlock': False,
            'n_gpu_layers': 0,
            'seed': 42 + agent_id,  # 各エージェントに固有のシード
            'chat_format': "gemma",
            'verbose': False
        }
        
        # JSONパラメータがあれば追加ロード
        if params_path and os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    llama_kwargs.update(params)
                    # シードは上書きしない
                    llama_kwargs['seed'] = 42 + agent_id
            except Exception as e:
                if self.debug:
                    print(f"エージェント{agent_id}のパラメータロードエラー: {e}")
        
        # チャットテンプレートがあれば設定
        if chat_template and os.path.exists(chat_template):
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    llama_kwargs['chat_template'] = f.read()
            except Exception as e:
                if self.debug:
                    print(f"エージェント{agent_id}のテンプレートロードエラー: {e}")
                
        try:
            # モデルインスタンスを作成
            return Llama(**llama_kwargs)
        except Exception as e:
            if self.debug:
                print(f"エージェント{agent_id}のモデル初期化エラー: {e}")
            return None

    def _load_role_templates(self):
        """役割テンプレートの初期化（内部メソッド）"""
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

    def _load_roles(self):
        """エージェント役割の初期化（内部メソッド）"""
        # デフォルトの役割定義
        self.agent_roles = [
            {"role": "レポートまとめAI", "system": "あなたは経済の専門家です。経済的観点から手短に意見を述べてください。", "temperature": 0.7},
            {"role": "批判的評価AI", "system": "あなたは倫理の専門家です。倫理的観点から手短に意見を述べてください。", "temperature": 0.9},
            {"role": "技術視点", "system": "あなたは技術の専門家です。技術的観点から簡潔に答えてください。", "temperature": 0.6},
            {"role": "社会視点", "system": "あなたは社会学の専門家です。社会的観点から簡潔に答えてください。", "temperature": 0.8}
        ]
        
        # カスタム役割があれば上書き
        custom_roles = self.config.get('agent_roles')
        if custom_roles:
            self.agent_roles = custom_roles
    
    def _classify_question(self, question: str) -> str:
        """
        質問内容を分類し、適切なタイプを判断する（内部メソッド）
        
        引数:
            question: 分類する質問文字列
            
        戻り値:
            質問タイプ: "discussion", "planning", "informational", "conversational", "default"のいずれか
        """
        try:
            # 入力文が短すぎる場合は一般会話とみなす
            if len(question) < 5:
                return "conversational"
                
            # 挨拶や単純な問いかけは会話型として分類
            simple_greetings = ["こんにちは", "おはよう", "こんばんは", "hello", "hi", "hey", "よろしく", "お願いします"]
            for greeting in simple_greetings:
                if greeting in question.lower():
                    return "conversational"
                    
            # 特定のキーワードによる事前分類
            philosophical_keywords = ["哲学", "存在", "意識", "不条理", "アイデンティティ", "倫理", "道徳", "真理", "存在論"]
            for keyword in philosophical_keywords:
                if keyword in question:
                    return "discussion"
                    
            planning_keywords = ["計画", "アイデア", "未来", "予測", "どうやって", "どうすれば", "どのように", "方法"]
            for keyword in planning_keywords:
                if keyword in question:
                    return "planning"
                    
            info_keywords = ["とは", "意味", "定義", "説明", "概念", "歴史", "何ですか", "何ですか？", "何ですか?", "何ですか。"]
            for keyword in info_keywords:
                if keyword in question:
                    return "informational"
            
            # LLMを使用して質問タイプを分類
            prompt = f"""
あなたは質問分類AIです。以下の質問を分析し、最も合致するタイプを1つだけ選んでください。

質問タイプの定義:
- 議論型(discussion): 主観的な見解、価値判断、哲学的考察、倫理的議論、多角的な視点が必要な質問
- 計画・構想型(planning): 未来志向や実用的なアイデア、行動計画、改善策、方法論に関する質問
- 情報提供型(informational): 客観的な事実、定義、解説、情報収集が主な目的の質問
- 一般会話型(conversational): 挨拶、雑談、個人的な意見、好みを尋ねる質問

質問文: {question}

質問タイプを以下のいずれかの単語で回答してください: discussion、planning、informational、conversational
回答:
"""

            # 温度を低く設定して決定論的な分類を促進
            with _global_llama_lock:  # グローバルロックで保護
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.1,
                    stop=["\n", " ", "。", "."]
                )
            
            if isinstance(response, dict):
                classification = response['choices'][0]['message']['content'].strip().lower()
            else:
                classification = response.choices[0].message.content.strip().lower()
            
            # 分類結果の正規化
            if "discussion" in classification:
                result = "discussion"
            elif "planning" in classification:
                result = "planning"
            elif "informational" in classification:
                result = "informational"
            elif "conversational" in classification:
                result = "conversational"
            else:
                # キーワードに応じたフォールバック
                if any(keyword in question for keyword in philosophical_keywords):
                    result = "discussion"
                elif any(keyword in question for keyword in planning_keywords):
                    result = "planning"
                elif any(keyword in question for keyword in info_keywords):
                    result = "informational"
                else:
                    result = "default"
                
            if self.debug:
                print(f"質問分類: {result} (元出力: {classification})")
                
            return result
            
        except Exception as e:
            if self.debug:
                print(f"質問分類エラー: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # エラー時でもキーワードベースの分類を試みる
            if any(keyword in question for keyword in ["哲学", "存在", "意識", "不条理", "道徳"]):
                return "discussion"
            elif "計画" in question or "方法" in question or "どのように" in question:
                return "planning"
            elif "とは" in question or "何ですか" in question:
                return "informational"
            elif len(question) < 10:
                return "conversational"
                
            return "default"  # それでも分類できない場合はデフォルト
    
    def update_roles_based_on_question(self, question: str) -> None:
        """
        質問内容に基づいて最適な役割セットを選択
        
        引数:
            question: ユーザーの質問文字列
        """
        # 質問を分類
        question_type = self._classify_question(question)
        
        # 質問タイプに基づいて役割テンプレートを選択
        selected_roles = self.role_templates.get(question_type, self.role_templates["default"])
        
        # エージェント数に合わせて役割を調整
        if len(selected_roles) > self.num_agents:
            # 役割が多すぎる場合は先頭から必要な数だけ使用
            self.agent_roles = selected_roles[:self.num_agents]
        elif len(selected_roles) < self.num_agents:
            # 役割が足りない場合はデフォルト役割で補完
            additional_roles = self.role_templates["default"]
            needed = self.num_agents - len(selected_roles)
            self.agent_roles = selected_roles + additional_roles[:needed]
        else:
            self.agent_roles = selected_roles
            
        # 黒板に選択された役割タイプを記録
        self.blackboard.write('question_type', question_type)
        
        if self.debug:
            print(f"役割更新: 質問タイプ={question_type}, 役割数={len(self.agent_roles)}")
            for i, role in enumerate(self.agent_roles):
                print(f"  エージェント{i}: {role['role']}")

    def _detect_language(self, text: str) -> str:
        """入力テキストの言語を判定（内部メソッド）"""
        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text):
            return 'ja'
        elif re.search(r'[a-zA-Z]', text):
            return 'en' 
        return 'ja'  # デフォルトは日本語

    def _agent_task(self, agent_id: int) -> str:
        """単一エージェントの処理（内部メソッド）"""
        try:
            input_data = self.blackboard.read('input')
            if not input_data:
                return ""
                
            # エージェント役割の選択
            role_idx = agent_id % len(self.agent_roles)
            role = self.agent_roles[role_idx]
            
            # 入力データから正規化テキストのみを抽出（埋め込みを除外）
            normalized_text = ""
            if isinstance(input_data, dict) and 'normalized' in input_data:
                normalized_text = input_data['normalized']
            elif isinstance(input_data, str):
                normalized_text = input_data
            
            # 会話コンテキストがあれば追加
            context = ""
            if isinstance(input_data, dict) and 'context' in input_data:
                context = f"これまでの会話の要約: {input_data['context']}\n\n"
            
            # 入力言語に合わせた応答言語設定
            lang = self._detect_language(normalized_text)
            system_prompt = role['system']
            if lang == 'ja':
                system_prompt += " 必ず日本語で100〜200字で簡潔に返答してください。"
            else:
                system_prompt += " Always respond briefly in English, around 30-50 words maximum."
                
            # プロンプト生成 - トークン数を明示的に制限
            prompt = f"{system_prompt}\n\n{context}問い: {normalized_text[:200]}"  # 入力も制限
            
            # エージェント入力の保存（埋め込みは除外）
            self.blackboard.write(f'agent_{agent_id}_input', normalized_text[:200])
            self.blackboard.write(f'agent_{agent_id}_role', role['role'])
            
            # LLMの実行 - グローバルロックで保護
            messages = [{"role": "user", "content": prompt}]
            temperature = role.get('temperature', 0.7)
            max_tokens = min(self.config.get('max_tokens', 256), 256)  # トークン生成数制限
            
            # グローバルロックを使用して、LLaMA.cppの同時アクセスを防ぐ
            with _global_llama_lock:
                response = self.llm.create_chat_completion(
                    messages=messages, 
                    max_tokens=max_tokens, 
                    temperature=temperature,
                    stop=["。", ".", "\n\n"]  # 早めに停止
                )
            
            # 応答の取得と保存
            if isinstance(response, dict):
                output = response["choices"][0]["message"]["content"].strip()
            else:
                output = response.choices[0].message.content.strip()
                
            # 出力を制限（長すぎる応答を防止）
            output = output[:300]  # 最大300文字に制限
            
            self.blackboard.write(f'agent_{agent_id}_output', output)
            
            if self.debug:
                print(f"エージェント{agent_id}({role['role']}): 出力長={len(output)}")
            
            return output
            
        except Exception as e:
            error_msg = f"エージェント{agent_id}エラー: {str(e)}"
            if self.debug:
                print(error_msg)
                import traceback
                traceback.print_exc()
            self.blackboard.write(f'agent_{agent_id}_error', error_msg)
            return f"エージェント{agent_id}は応答できませんでした。"

    def _parallel_agent_task(self, agent_id: int) -> str:
        """並列処理モード用の単一エージェント処理（内部メソッド）"""
        try:
            input_data = self.blackboard.read('input')
            if not input_data:
                return ""
                
            # エージェント役割の選択
            role_idx = agent_id % len(self.agent_roles)
            role = self.agent_roles[role_idx]
            model = self.agent_models[agent_id]  # 各エージェント専用のモデルを使用
            
            # 入力データから正規化テキストのみを抽出
            normalized_text = ""
            if isinstance(input_data, dict) and 'normalized' in input_data:
                normalized_text = input_data['normalized']
            elif isinstance(input_data, str):
                normalized_text = input_data
            
            # 会話コンテキストがあれば追加
            context = ""
            if isinstance(input_data, dict) and 'context' in input_data:
                context = f"これまでの会話の要約: {input_data['context']}\n\n"
            
            # 入力言語に合わせた応答言語設定
            lang = self._detect_language(normalized_text)
            system_prompt = role['system']
            if lang == 'ja':
                system_prompt += " 必ず日本語で100〜200字で簡潔に返答してください。"
            else:
                system_prompt += " Always respond briefly in English, around 30-50 words maximum."
                
            # プロンプト生成 - トークン数を制限
            prompt = f"{system_prompt}\n\n{context}問い: {normalized_text[:200]}"
            
            # エージェント入力の保存
            self.blackboard.write(f'agent_{agent_id}_input', normalized_text[:200])
            self.blackboard.write(f'agent_{agent_id}_role', role['role'])
            
            # 専用モデルインスタンスでLLMを実行
            messages = [{"role": "user", "content": prompt}]
            temperature = role.get('temperature', 0.7)
            max_tokens = min(self.config.get('max_tokens', 256), 256)
            
            with _global_llama_lock:  # グローバルロックで保護
                response = model.create_chat_completion(
                    messages=messages, 
                    max_tokens=max_tokens, 
                    temperature=temperature,
                    stop=["。", ".", "\n\n"]
                )
            
            # 応答の取得と保存
            if isinstance(response, dict):
                output = response["choices"][0]["message"]["content"].strip()
            else:
                output.choices[0].message.content.strip()
                
            # 出力を制限
            output = output[:300]
            
            self.blackboard.write(f'agent_{agent_id}_output', output)
            
            if self.debug:
                print(f"エージェント{agent_id}({role['role']}): 出力長={len(output)}")
            
            return output
            
        except Exception as e:
            error_msg = f"並列エージェント{agent_id}エラー: {str(e)}"
            if self.debug:
                print(error_msg)
                import traceback
                traceback.print_exc()
            self.blackboard.write(f'agent_{agent_id}_error', error_msg)
            return f"エージェント{agent_id}は応答できませんでした。"

    def run_agents(self, blackboard) -> List[str]:
        """すべてのエージェントを実行し、結果を取得"""
        self.blackboard = blackboard  # 黒板参照を更新
        
        # 質問に基づいて役割を更新
        input_data = self.blackboard.read('input')
        if input_data:
            question = input_data.get('normalized') if isinstance(input_data, dict) else str(input_data)
            self.update_roles_based_on_question(question)
        
        results = []
        
        # 並列モード実行の試行
        if self.parallel_mode and len(self.agent_models) >= self.num_agents:
            if self.debug:
                print(f"並列処理モードでエージェントを実行します")
                
            try:
                if self.debug:
                    print("並列処理開始")
                
                # 同時実行するワーカー数を制限（CPUコア数に基づいて調整）
                import multiprocessing
                max_workers = min(self.num_agents, max(1, multiprocessing.cpu_count() - 1))
                
                # 並列タスク定義 - 例外はタスク内で処理
                def parallel_agent_task(agent_id):
                    try:
                        # 各エージェントに専用のモデルを使用
                        result = self._parallel_agent_task(agent_id)
                        return result or f"エージェント{agent_id}の応答を取得できませんでした"
                    except Exception as e:
                        error_msg = f"並列エージェント{agent_id}エラー: {str(e)}"
                        if self.debug:
                            print(error_msg)
                            import traceback
                            traceback.print_exc()
                        # エラーが発生しても空文字ではなくエラーメッセージを返す
                        return f"エージェント{agent_id}は応答できませんでした：{str(e)[:100]}"
                
                # ThreadPoolExecutorで並列実行
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 各エージェントのタスクを投入
                    futures = [executor.submit(parallel_agent_task, i) for i in range(self.num_agents)]
                    
                    # 結果を収集（例外は各タスク内で処理）
                    results = []
                    for i, future in enumerate(futures):
                        try:
                            # タイムアウト設定（30秒）
                            result = future.result(timeout=30)
                            results.append(result)
                        except Exception as e:
                            # タイムアウトやその他の例外
                            error_msg = f"並列エージェント{i}の実行に失敗: {str(e)}"
                            if self.debug:
                                print(error_msg)
                            results.append(f"エージェント{i}は時間内に応答できませんでした")
                
                # すべてのエージェントが結果を返したか確認
                if len(results) == self.num_agents and all(results):
                    if self.debug:
                        print("並列処理で全エージェントが応答しました")
                    return results
                else:
                    if self.debug:
                        print(f"一部のエージェントが応答していません（{len(results)}/{self.num_agents}）")
                    # 結果が不完全な場合は逐次実行にフォールバック
                    self.blackboard.write('parallel_mode_failed', "応答が不完全なため逐次実行にフォールバックします")
                    results = []  # リセット
                    
            except Exception as e:
                # 並列実行全体の例外
                if self.debug:
                    print(f"並列処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                self.blackboard.write('parallel_mode_error', str(e))
                results = []  # リセット
        
        # 並列モードが無効または失敗した場合は逐次実行
        if not results:
            if self.debug:
                print("逐次処理モードでエージェントを実行します")
            
            # 逐次実行でエラーを慎重に処理
            for agent_id in range(self.num_agents):
                try:
                    if self.model_lock:  # ロックがある場合は使用
                        with self.model_lock:
                            result = self._agent_task(agent_id)
                    else:
                        result = self._agent_task(agent_id)
                    
                    results.append(result or f"エージェント{agent_id}は空の応答を返しました")
                except Exception as e:
                    error_msg = f"エージェント{agent_id}実行エラー: {str(e)}"
                    if self.debug:
                        print(error_msg)
                    results.append(f"エージェント{agent_id}は応答できませんでした")
        
        # 結果の長さがエージェント数と一致するか確認
        while len(results) < self.num_agents:
            results.append("エージェントは応答しませんでした")
            
        return results