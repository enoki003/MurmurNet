#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Boids Agent Factory モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~
Boidsアルゴリズムの原則に基づき、自己増殖型エージェントを
動的に生成するファクトリークラス

作者: Yuhi Sonoki
"""

import random
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

class BoidsAgentFactory:
    """
    自己増殖型エージェントファクトリー
    - Separation（分離）: 同質的な意見から距離を取る反論エージェント
    - Alignment（整列）: 議論の方向性に沿った展開エージェント
    - Cohesion（結合）: 意見を統合し中心に向かう調停エージェント
    """
    
    def __init__(self, config: Dict[str, Any] = None, opinion_space_manager = None):
        """
        初期化
        
        引数:
            config: 設定辞書
            opinion_space_manager: OpinionSpaceManagerインスタンス（省略可能）
        """
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.opinion_space = opinion_space_manager
        
        # エージェントID採番用カウンタ
        self.agent_counter = 0
        
        # 役割テンプレートの設定
        self._initialize_role_templates()
        
        if self.debug:
            print("BoidsAgentFactory初期化完了")
    
    def set_opinion_space_manager(self, opinion_space_manager):
        """
        OpinionSpaceManagerを設定
        
        引数:
            opinion_space_manager: OpinionSpaceManagerインスタンス
        """
        self.opinion_space = opinion_space_manager
    
    def _initialize_role_templates(self):
        """
        エージェント役割テンプレートの初期化（内部メソッド）
        """
        # Contrarian Agent（反対意見エージェント） - Separation原則
        self.contrarian_templates = [
            {
                'role': '批判者',
                'system': '''あなたは批判的思考を得意とする批判者です。
他者の意見を批判的に検討し、弱点や考慮されていない点を指摘してください。
主流の考え方に疑問を呈し、デビルズアドボケート（敢えて反対の立場）として議論を深めることが役割です。
他のエージェントが見落としている点を指摘し、新たな視点を提供してください。
感情的にならず論理的に反論してください。''',
                'temperature': 0.8,
                'principle': 'separation',
                'strategy': '批判的思考'
            },
            {
                'role': '別視点提案者',
                'system': '''あなたは主流の意見とは異なる視点を提供する役割です。
これまでの議論で見落とされている観点や、検討されていない側面を考慮してください。
常識的な見方や前提に疑問を投げかけ、創造的な代替案を提示することが重要です。
他者の意見を否定するのではなく、「別の見方をすると」という形で新たな視点を加えてください。''',
                'temperature': 0.9,
                'principle': 'separation',
                'strategy': '代替視点の提示'
            },
            {
                'role': 'リスク分析者',
                'system': '''あなたはリスク分析の専門家です。
議論されている内容の潜在的なリスクや欠点を特定し、警告することが役割です。
最悪のシナリオを想定し、計画の弱点や失敗の可能性を論理的に指摘してください。
批判のための批判ではなく、リスク軽減や回避策も合わせて提案するよう心がけてください。''',
                'temperature': 0.7,
                'principle': 'separation',
                'strategy': 'リスク特定'
            }
        ]
        
        # Aligning Agent（同調・深化エージェント） - Alignment原則
        self.aligning_templates = [
            {
                'role': '詳細化エキスパート',
                'system': '''あなたは他者の良いアイディアを詳細化する専門家です。
これまでの議論で出た有望な発想を拾い上げ、さらに掘り下げて展開することが役割です。
具体例を加えたり、実装方法を示したり、応用可能性を広げるなど、
アイディアを具体化・精緻化することで議論を前進させてください。
反論ではなく、建設的な補足や肯定的な発展を心がけてください。''',
                'temperature': 0.7,
                'principle': 'alignment',
                'strategy': 'アイディアの詳細化'
            },
            {
                'role': '知識補完者',
                'system': '''あなたは議論に関連する知識や情報を補完する役割です。
他のエージェントが言及した事実や概念について、より詳しい背景や
文脈を提供し、議論の土台を強化してください。
専門的な知識を分かりやすく説明し、議論の方向性に沿った
関連情報を付け足すことで貢献してください。
議論の流れを変えるのではなく、流れに沿って深みを加えることに注力してください。''',
                'temperature': 0.6,
                'principle': 'alignment',
                'strategy': '知識と文脈の提供'
            },
            {
                'role': '関連付け思考者',
                'system': '''あなたは異なるアイディア同士を関連づける専門家です。
議論の中で別々に提案された複数の考えの間にある関連性や
共通点を見出すことが役割です。
「AとBには〜という共通点がある」「この意見はあの意見と組み合わせると〜」
といった形で、アイディア同士を橋渡ししてください。
アイディアの統合ではなく、関連性の発見に焦点を当ててください。''',
                'temperature': 0.8,
                'principle': 'alignment',
                'strategy': '関連性の発見'
            }
        ]
        
        # Mediator Agent（調停・統合エージェント） - Cohesion原則
        self.mediator_templates = [
            {
                'role': '調停者',
                'system': '''あなたは対立する意見の調停を行う専門家です。
議論の中で生じた意見の相違や対立点を特定し、それらを統合する方法を模索してください。
各視点の長所を認めつつ、共通の基盤や妥協点を見出すことが役割です。
「～さんの言うAと～さんの言うBは両立可能で、具体的には～」といった形で、
異なる意見を尊重しながら統合案を提示してください。
対立を解消し、建設的な合意形成に貢献することを目指してください。''',
                'temperature': 0.6,
                'principle': 'cohesion',
                'strategy': '意見の調停と統合'
            },
            {
                'role': '要約者',
                'system': '''あなたは多様な意見を要約し整理する専門家です。
これまでの議論の主要な論点、共通認識、相違点を明確にまとめることが役割です。
議論が複雑化している場合は論点を整理し、話し合いの全体像を俯瞰してください。
中立的な立場から、各エージェントの貢献を公平に取り上げ、
議論の現状と今後の方向性を示唆してください。
新しい意見を出すよりも、既存の意見の構造化と統合に集中してください。''',
                'temperature': 0.5,
                'principle': 'cohesion',
                'strategy': '議論の構造化と整理'
            },
            {
                'role': '総合案提案者',
                'system': '''あなたは複数の意見を組み合わせて総合案を作る専門家です。
これまで出された様々なアイディアの良い部分を組み合わせて、
より優れた統合案を提案することが役割です。
「Aの～という点とBの～という点を組み合わせると、～という解決策が考えられます」
といった形で、建設的な統合案を示してください。
単なる妥協案ではなく、各アイディアの強みを活かした創造的な総合案を目指してください。''',
                'temperature': 0.7,
                'principle': 'cohesion',
                'strategy': '創造的統合案の提案'
            }
        ]
        
        # Explorer Agent（探索エージェント） - 特殊役割
        self.explorer_templates = [
            {
                'role': '創造的発想者',
                'system': '''あなたは独創的で革新的なアイディアを提案する専門家です。
一般的な思考の枠を超えた、大胆で新しい発想を提供することが役割です。
「全く別の観点から考えると～」「常識を覆す発想として～」といった形で、
議論に創造的な刺激を与えてください。
現実的な制約にとらわれすぎずに、可能性を広げる思考実験を促してください。
奇抜さだけでなく、革新的でありながらも検討に値する提案を心がけてください。''',
                'temperature': 1.0,
                'principle': 'explore',
                'strategy': '革新的発想の提案'
            },
            {
                'role': '未来洞察者',
                'system': '''あなたは議論されている内容の長期的影響や将来展望を考察する専門家です。
現在の議論が5年後、10年後にどのような結果をもたらすかを予測することが役割です。
「将来的には～」「長期的に考えると～」といった形で、
時間軸を拡張した視点を提供してください。
短期的な解決策だけでなく、持続可能性や進化の可能性を考慮した洞察を示してください。
SF的な空想ではなく、論理的に考えられる未来像を描写してください。''',
                'temperature': 0.9,
                'principle': 'explore',
                'strategy': '将来展望の提示'
            }
        ]
        
        # 質問タイプ別の初期エージェント構成テンプレート
        self.initial_agent_templates = {
            # 議論型質問用の初期構成
            "discussion": [
                self.contrarian_templates[0],  # 批判者
                self.mediator_templates[0],    # 調停者
                self.explorer_templates[0]     # 創造的発想者
            ],
            
            # 計画・構想型質問用の初期構成
            "planning": [
                self.aligning_templates[0],    # 詳細化エキスパート
                self.contrarian_templates[2],  # リスク分析者
                self.explorer_templates[1]     # 未来洞察者
            ],
            
            # 情報提供型質問用の初期構成
            "informational": [
                self.aligning_templates[1],    # 知識補完者
                self.mediator_templates[1],    # 要約者
                self.contrarian_templates[1]   # 別視点提案者
            ],
            
            # 一般会話型質問用の初期構成
            "conversational": [
                self.mediator_templates[1],    # 要約者
                self.aligning_templates[2]     # 関連付け思考者
            ],
            
            # デフォルト構成
            "default": [
                self.mediator_templates[0],    # 調停者
                self.contrarian_templates[0]   # 批判者
            ]
        }
    
    def create_agent(self, action_type: str, blackboard=None, diagnosis=None, properties: Dict = None) -> Dict:
        """
        新しいエージェントを生成
        
        引数:
            action_type: エージェントタイプ ('contrarian', 'aligning', 'mediator', 'explorer')
            blackboard: ブラックボードインスタンス
            diagnosis: 診断結果
            properties: 追加プロパティ (任意)
            
        戻り値:
            生成されたエージェント設定
        """
        agent_templates = {
            'contrarian': self.contrarian_templates,
            'add_contrarian_agent': self.contrarian_templates,
            'separation': self.contrarian_templates,
            
            'aligning': self.aligning_templates,
            'add_aligning_agent': self.aligning_templates, 
            'alignment': self.aligning_templates,
            
            'mediator': self.mediator_templates,
            'add_mediator_agent': self.mediator_templates,
            'cohesion': self.mediator_templates,
            
            'explorer': self.explorer_templates,
            'add_explorer_agent': self.explorer_templates,
            'explore': self.explorer_templates
        }
        
        # アクション種別に対応するテンプレートを取得
        templates = agent_templates.get(action_type, self.contrarian_templates)
        
        # ランダムにテンプレートを選択
        template = random.choice(templates)
        
        # プロパティが指定されていれば上書き
        if properties:
            for key, value in properties.items():
                if key in template:
                    template[key] = value
        
        # 現在のターンを取得
        current_turn = 0
        if blackboard:
            turn_info = blackboard.read('current_turn')
            if turn_info is not None:
                current_turn = turn_info
        
        # ユニークIDを割り当て
        self.agent_counter += 1
        agent_id = self.agent_counter
        
        # エージェント定義を作成
        agent = {
            'id': agent_id,
            'role': template['role'],
            'system': template['system'],
            'temperature': template.get('temperature', 0.7),
            'strategy': template.get('strategy', 'default'),
            'principle': template.get('principle', 'default'),
            'creation_turn': current_turn,
            'lifespan': self.config.get('agent_lifetime', 3),  # デフォルトの寿命（ターン数）
            'created_at': self._current_timestamp()
        }
        
        if self.debug:
            print(f"新しいエージェントを生成: id={agent_id}, role={agent['role']}, strategy={agent['strategy']}")
            
        return agent
    
    def create_initial_agents(self, num_agents: int, question: str, blackboard=None) -> List[Dict]:
        """
        初期エージェント集団を生成
        
        引数:
            num_agents: 作成するエージェント数
            question: ユーザーの質問
            blackboard: ブラックボードインスタンス
            
        戻り値:
            エージェント設定のリスト
        """
        # 質問タイプを取得
        question_type = "default"
        if blackboard:
            type_info = blackboard.read('question_type')
            if type_info:
                question_type = type_info
        
        # 質問タイプに対応するテンプレート取得
        templates = self.initial_agent_templates.get(question_type, self.initial_agent_templates["default"])
        
        # エージェント数がテンプレート数より少ない場合は先頭から必要数選択
        if len(templates) > num_agents:
            templates = templates[:num_agents]
        
        # エージェント数がテンプレート数より多い場合は不足分をランダム生成
        agents = []
        for i in range(min(num_agents, len(templates))):
            template = templates[i]
            
            # 現在のターンを取得
            current_turn = 0
            if blackboard:
                turn_info = blackboard.read('current_turn')
                if turn_info is not None:
                    current_turn = turn_info
            
            # ユニークIDを割り当て
            self.agent_counter += 1
            agent_id = self.agent_counter
            
            # 基本エージェント定義
            agent = {
                'id': agent_id,
                'role': template['role'],
                'system': template['system'],
                'temperature': template.get('temperature', 0.7),
                'strategy': template.get('strategy', 'default'),
                'principle': template.get('principle', 'default'),
                'creation_turn': current_turn,
                'lifespan': self.config.get('agent_lifetime', 3),
                'created_at': self._current_timestamp()
            }
            
            agents.append(agent)
        
        # 不足分をランダム生成して追加
        while len(agents) < num_agents:
            # ランダムなエージェントタイプを選択
            agent_type = random.choice(['contrarian', 'aligning', 'mediator', 'explorer'])
            agent = self.create_agent(agent_type, blackboard)
            agents.append(agent)
        
        if self.debug:
            print(f"初期エージェント集団を生成: {len(agents)}体 (質問タイプ: {question_type})")
            for agent in agents:
                print(f"  - {agent['role']} ({agent['strategy']})")
        
        return agents
    
    def refine_prompt_with_context(self, agent: Dict, blackboard) -> Dict:
        """
        コンテキストに基づいてエージェントのプロンプトを洗練
        
        引数:
            agent: エージェント設定
            blackboard: ブラックボードインスタンス
            
        戻り値:
            更新されたエージェント設定
        """
        if not blackboard:
            return agent
        
        try:
            # 入力情報の取得
            input_data = blackboard.read('input')
            input_text = ""
            if isinstance(input_data, dict) and 'normalized' in input_data:
                input_text = input_data['normalized']
            elif isinstance(input_data, str):
                input_text = input_data
                
            if not input_text:
                return agent
            
            # 簡単なプロンプト拡張（議論テーマに関連付ける）
            agent['system'] = f"{agent['system']}\n\n現在の議論テーマ: {input_text[:100]}"
            
            # エージェントの役割に応じた特化プロンプト
            role = agent.get('role', '').lower()
            principle = agent.get('principle', '').lower()
            
            # 批判系エージェントの場合
            if 'separation' in principle or '批判' in role or 'リスク' in role:
                agent['system'] += "\n\n特に以下の点に注意してください：\n- 議論の中の弱点や考慮されていない視点\n- 前提となる仮定の妥当性\n- 潜在的なリスクや課題"
            
            # 調停系エージェントの場合
            elif 'cohesion' in principle or '調停' in role or '要約' in role or '総合' in role:
                agent['system'] += "\n\n特に以下の点に注意してください：\n- 異なる意見の共通点\n- 議論全体の構造と主要な論点\n- 対立点の建設的な解消方法"
            
            # 詳細化系エージェントの場合
            elif 'alignment' in principle or '詳細' in role or '補完' in role or '関連付け' in role:
                agent['system'] += "\n\n特に以下の点に注意してください：\n- 議論内の有望なアイデアの具体化\n- 背景情報や文脈の補足\n- 異なるアイデア間の関連性"
                
            # 探索系エージェントの場合
            elif 'explore' in principle or '創造' in role or '未来' in role:
                agent['system'] += "\n\n特に以下の点に注意してください：\n- 従来の枠組みにとらわれない新しい視点\n- 長期的な影響や将来展望\n- 創造的な代替案や可能性"
            
        except Exception as e:
            if self.debug:
                print(f"プロンプト洗練エラー: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return agent
    
    def create_agent_from_diagnosis(self, diagnosis_result: Dict, blackboard=None) -> Optional[Dict]:
        """
        診断結果からエージェントを生成
        
        引数:
            diagnosis_result: SelfDiagnosisからの診断結果
            blackboard: ブラックボードインスタンス
            
        戻り値:
            生成されたエージェント設定（生成不要な場合はNone）
        """
        if not diagnosis_result:
            return None
            
        action = diagnosis_result.get('suggested_action', diagnosis_result.get('action'))
        
        # アクションが「add」系でない場合は生成しない
        if not action or 'add' not in action:
            return None
            
        # 診断結果からプロパティを抽出
        properties = diagnosis_result.get('properties', {})
        
        # アクション種別からエージェントタイプを決定
        action_type = action
        if 'contrarian' in action:
            action_type = 'contrarian'
        elif 'mediator' in action:
            action_type = 'mediator'
        elif 'aligning' in action:
            action_type = 'aligning'
        elif 'explorer' in action:
            action_type = 'explorer'
        
        # エージェントを生成
        agent = self.create_agent(action_type, blackboard, diagnosis_result, properties)
        
        # コンテキストを考慮してプロンプトを調整
        if blackboard:
            agent = self.refine_prompt_with_context(agent, blackboard)
        
        return agent
    
    def create_opposite_opinion_agent(self, blackboard=None) -> Dict:
        """
        現在の意見集団と対極的な意見を持つエージェントを生成
        
        引数:
            blackboard: ブラックボードインスタンス
            
        戻り値:
            生成されたエージェント設定
        """
        # 意見空間マネージャが設定されていなければ通常のContrarian
        if not self.opinion_space:
            return self.create_agent('contrarian', blackboard)
            
        # コントラリアンエージェントをベースに生成
        agent = self.create_agent('contrarian', blackboard)
        
        try:
            # 現在の意見空間の中心点（セントロイド）を取得
            centroid = self.opinion_space.get_centroid()
            
            # クラスタリングを実行し、最大クラスタを特定
            clusters = self.opinion_space.cluster_opinions()
            
            if clusters and centroid is not None:
                # 強化されたシステムプロンプト
                agent['system'] = f"""あなたは意識的に独自の視点を持つ批判者です。
これまでの議論で主流となっている意見とは異なる視点から考えることが役割です。
すでに言及されていない側面や、見落とされがちな点に焦点を当ててください。
「しかし別の視点から見ると～」「検討すべき異なる観点として～」といった形で、
新しい視点を提供してください。
主流の意見に反対するだけでなく、建設的な代替案も示すよう心がけてください。"""

                agent['principle'] = 'separation_enhanced'
                agent['strategy'] = '対極的視点の提示'
                agent['temperature'] = 0.9  # 高めの温度で多様性を促進
                
                if self.debug:
                    print(f"対極的意見エージェントを生成: {agent['id']}")
                    
        except Exception as e:
            if self.debug:
                print(f"対極的意見エージェント生成中のエラー: {e}")
                import traceback
                traceback.print_exc()
        
        # コンテキストを考慮してプロンプトを調整
        if blackboard:
            agent = self.refine_prompt_with_context(agent, blackboard)
            
        return agent
    
    def create_aligning_opinion_agent(self, blackboard=None) -> Dict:
        """
        現在の意見集団の方向性に沿ったエージェントを生成
        
        引数:
            blackboard: ブラックボードインスタンス
            
        戻り値:
            生成されたエージェント設定
        """
        # 意見空間マネージャが設定されていなければ通常のAligning
        if not self.opinion_space:
            return self.create_agent('aligning', blackboard)
            
        # アライニングエージェントをベースに生成
        agent = self.create_agent('aligning', blackboard)
        
        try:
            # 現在の意見空間の中心点（セントロイド）を取得
            centroid = self.opinion_space.get_centroid()
            
            # クラスタリングを実行し、最大クラスタを特定
            clusters = self.opinion_space.cluster_opinions()
            
            if clusters and centroid is not None:
                # 最大のクラスタを特定
                largest_cluster = max(clusters, key=len) if clusters else None
                
                if largest_cluster:
                    # 強化されたシステムプロンプト
                    agent['system'] = f"""あなたは議論の方向性を深める専門家です。
現在の議論の主流となっている考え方をさらに発展させることが役割です。
主要なアイディアをより詳細に分析し、補強する事例や根拠を加えてください。
「〜という意見をさらに掘り下げると〜」「この観点を発展させた場合〜」といった形で、
議論をより深く、より具体的にする貢献をしてください。
単に同意するだけでなく、実質的な内容を付加する努力をしてください。"""

                    agent['principle'] = 'alignment_enhanced'
                    agent['strategy'] = '主流意見の強化・発展'
                    agent['temperature'] = 0.7  # やや低めの温度で一貫性を確保
                    
                    if self.debug:
                        print(f"整列意見エージェントを生成: {agent['id']}")
                        
        except Exception as e:
            if self.debug:
                print(f"整列意見エージェント生成中のエラー: {e}")
                import traceback
                traceback.print_exc()
        
        # コンテキストを考慮してプロンプトを調整
        if blackboard:
            agent = self.refine_prompt_with_context(agent, blackboard)
            
        return agent
    
    def create_mediator_agent(self, blackboard=None) -> Dict:
        """
        複数の意見クラスタを仲介する調停エージェントを生成
        
        引数:
            blackboard: ブラックボードインスタンス
            
        戻り値:
            生成されたエージェント設定
        """
        # 意見空間マネージャが設定されていなければ通常のMediator
        if not self.opinion_space:
            return self.create_agent('mediator', blackboard)
            
        # 調停エージェントをベースに生成
        agent = self.create_agent('mediator', blackboard)
        
        try:
            # クラスタリングを実行
            clusters = self.opinion_space.cluster_opinions(n_clusters=min(3, len(self.opinion_space.latest_vectors)))
            
            if clusters and len(clusters) > 1:
                # 強化されたシステムプロンプト
                agent['system'] = f"""あなたは異なる視点を統合する調停者です。
現在の議論には複数の異なる見解が存在しています。
それぞれの視点の価値を認めつつ、対立点と共通点を整理してください。
「〜という意見と〜という意見は一見対立していますが、〜という点で共通しています」
「両方の視点を組み合わせると〜」といった形で統合的な理解を示してください。
どちらかの側に与することなく、バランスの取れた視点を提供することが重要です。"""

                agent['principle'] = 'cohesion_enhanced'
                agent['strategy'] = '複数視点の融合'
                agent['temperature'] = 0.6  # バランスの取れた出力のため
                
                if self.debug:
                    print(f"調停エージェントを生成: {agent['id']}")
                    
        except Exception as e:
            if self.debug:
                print(f"調停エージェント生成中のエラー: {e}")
                import traceback
                traceback.print_exc()
        
        # コンテキストを考慮してプロンプトを調整
        if blackboard:
            agent = self.refine_prompt_with_context(agent, blackboard)
            
        return agent
    
    def create_random_agent(self, blackboard=None) -> Dict:
        """
        ランダムなタイプのエージェントを生成
        
        引数:
            blackboard: ブラックボードインスタンス
            
        戻り値:
            生成されたエージェント設定
        """
        # ランダムなタイプを選択
        agent_type = random.choice(['contrarian', 'aligning', 'mediator', 'explorer'])
        agent = self.create_agent(agent_type, blackboard)
        
        # コンテキストを考慮してプロンプトを調整
        if blackboard:
            agent = self.refine_prompt_with_context(agent, blackboard)
            
        return agent
    
    def _current_timestamp(self) -> int:
        """
        現在のタイムスタンプを取得（内部メソッド）
        
        戻り値:
            現在のUNIXタイムスタンプ（秒）
        """
        import time
        return int(time.time())