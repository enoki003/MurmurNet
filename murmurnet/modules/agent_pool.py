from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
import os
import json
from typing import Dict, Any, List
import re

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
        self._load_model()
        self._load_roles()

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
            'n_ctx': 1024,
            'n_threads': 6,
            'use_mmap': True,
            'use_mlock': False,
            'n_gpu_layers': 0,
            'seed': 42,
            'chat_format': "gemma"
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
                    
        self.llm = Llama(**llama_kwargs)

    def _load_roles(self):
        """エージェント役割の初期化（内部メソッド）"""
        # デフォルトの役割定義
        self.agent_roles = [
            {"role": "レポートまとめAI", "system": "あなたは経済の専門家です。経済的観点から具体的な意見を述べてください。", "temperature": 0.7},
            {"role": "批判的評価AI", "system": "あなたは倫理の専門家です。倫理的観点から具体的な意見を述べてください。", "temperature": 0.9},
            {"role": "技術視点", "system": "あなたは技術の専門家です。技術的観点から答えてください。", "temperature": 0.6},
            {"role": "社会視点", "system": "あなたは社会学の専門家です。社会的観点から答えてください。", "temperature": 0.8}
        ]
        
        # カスタム役割があれば上書き
        custom_roles = self.config.get('agent_roles')
        if custom_roles:
            self.agent_roles = custom_roles

    def _detect_language(self, text: str) -> str:
        """入力テキストの言語を判定（内部メソッド）"""
        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text):
            return 'ja'
        elif re.search(r'[a-zA-Z]', text):
            return 'en' 
        return 'ja'  # デフォルトは日本語

    def _agent_task(self, agent_id: int) -> str:
        """単一エージェントの処理（内部メソッド）"""
        input_data = self.blackboard.read('input')
        if not input_data:
            return ""
            
        # エージェント役割の選択
        role = self.agent_roles[agent_id % len(self.agent_roles)]
        
        # 入力データから正規化テキストのみを抽出（埋め込みを除外）
        normalized_text = ""
        if isinstance(input_data, dict) and 'normalized' in input_data:
            normalized_text = input_data['normalized']
        elif isinstance(input_data, str):
            normalized_text = input_data
        
        # 入力言語に合わせた応答言語設定
        lang = self._detect_language(normalized_text)
        system_prompt = role['system']
        if lang == 'ja':
            system_prompt += " 必ず日本語で返答してください。"
        else:
            system_prompt += " Always respond in English."
            
        # プロンプト生成
        prompt = f"{system_prompt}\n\n問い: {normalized_text}"
        
        # エージェント入力の保存（埋め込みは除外）
        self.blackboard.write(f'agent_{agent_id}_input', normalized_text)
        
        # LLMの実行
        messages = [{"role": "user", "content": prompt}]
        temperature = role.get('temperature', 0.7)
        max_tokens = self.config.get('max_tokens', 512)
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            output = response["choices"][0]["message"]["content"].strip()
            self.blackboard.write(f'agent_{agent_id}_output', output)
            return output
        except Exception as e:
            if self.debug:
                print(f"エージェント{agent_id}エラー: {e}")
            self.blackboard.write(f'agent_{agent_id}_error', str(e))
            return ""

    def run_agents(self, blackboard) -> List[str]:
        """すべてのエージェントを実行し、結果を取得"""
        self.blackboard = blackboard  # 黒板参照を更新
        results = []
        
        # シーケンシャル実行（簡単にするため）
        for agent_id in range(self.num_agents):
            result = self._agent_task(agent_id)
            results.append(result)
            
        return results
        
        # 並列実行（オプション、コメントアウト）
        # with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
        #    futures = [executor.submit(self._agent_task, i) for i in range(self.num_agents)]
        #    results = [f.result() for f in futures]
        # return results