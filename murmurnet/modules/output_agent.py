#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output Agent モジュール
~~~~~~~~~~~~~~~~~~~~
最終応答生成を担当するエージェント
黒板の要約情報に基づき一貫性のある回答を作成

作者: Yuhi Sonoki
"""

# output_agent.py
from llama_cpp import Llama
import os
import re
from typing import Dict, Any, List

class OutputAgent:
    def __init__(self, config: dict = None):
        config = config or {}
        model_path = config.get('model_path') or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')
        )
        
        # 親設定から値を取得（もしくはデフォルト値を使用）
        llama_kwargs = dict(
            model_path=model_path,
            n_ctx=config.get('n_ctx', 2048),  # 親設定から受け取る  
            n_threads=config.get('n_threads', 4),  # 親設定から受け取る
            use_mmap=True,
            use_mlock=False,
            n_gpu_layers=0,
            seed=42,
            chat_format="gemma",
            verbose=False  # ログ出力抑制
        )
        
        chat_template = config.get('chat_template')
        if chat_template and os.path.exists(chat_template):
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    llama_kwargs['chat_template'] = f.read()
            except Exception as e:
                if config.get('debug'):
                    print(f"テンプレートロードエラー: {e}")

        self.llm = Llama(**llama_kwargs)
        self.config = config
        self.debug = config.get('debug', False)
        self.max_output_tokens = config.get('max_output_tokens', 512)  # 最終出力の最大トークン数
        
        if self.debug:
            print(f"出力エージェント: モデル設定 n_ctx={llama_kwargs['n_ctx']}, n_threads={llama_kwargs['n_threads']}")

    def generate(self, blackboard, entries: List[Dict[str, Any]]) -> str:
        """
        黒板の情報と提供されたエントリからユーザー質問への最終回答を生成
        
        引数:
            blackboard: 共有黒板
            entries: 様々なタイプの入力エントリのリスト
                     各エントリは {"type": "summary"|"agent", ...} の形式
        
        戻り値:
            生成された最終応答テキスト
        """
        try:
            # 1) 入力と RAG を取得
            inp = blackboard.read('input')
            user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
            user_input = user_input[:200]  # 入力を制限
            
            rag = blackboard.read('rag')
            if rag and isinstance(rag, str):
                rag = rag[:300]  # RAG情報も制限

            # 2) 言語検出
            lang = self._detect_language(user_input)
            
            # 3) 要約とエージェント出力の整理
            summaries = []
            agent_outputs = []
            
            for entry in entries:
                entry_type = entry.get('type', 'agent')  # デフォルトはagent
                
                if entry_type == 'summary':
                    iteration = entry.get('iteration', 0)
                    text = entry.get('text', '')[:200]  # テキスト制限
                    summaries.append(f"要約 {iteration+1}: {text}")
                else:  # agent
                    agent_id = entry.get('agent', 0)
                    text = entry.get('text', '')[:200]  # テキスト制限
                    agent_outputs.append(f"エージェント {agent_id+1}: {text}")
            
            # 4) システムプロンプト作成
            if lang == 'ja':
                sys_prompt = "あなたは日本語話者向けの多言語アシスタントです。入力された情報を元に、300文字以内で簡潔に回答してください。"
            else:
                sys_prompt = "You are a general-purpose AI assistant. Answer concisely in English, maximum 50 words."
            
            # 5) プロンプト内容作成 - 制限を加える
            prompt_content = f"問い: {user_input}\n\n"
            
            if rag:
                prompt_content += f"参考情報: {rag}\n\n"
            
            # 入力量が多すぎる場合は選択
            total_length = 0
            max_prompt_length = 1000  # プロンプトの最大長さ
            
            # 要約情報（優先）
            if summaries:
                summary_text = "要約情報:\n" + "\n\n".join(summaries) + "\n\n"
                if total_length + len(summary_text) <= max_prompt_length:
                    prompt_content += summary_text
                    total_length += len(summary_text)
            
            # エージェント出力（2番目に優先）
            if agent_outputs and total_length < max_prompt_length:
                # 残りの長さに合わせて調整
                if len(agent_outputs) > 2 and total_length + len("\n\n".join(agent_outputs)) > max_prompt_length:
                    # 重要なエージェントだけ選択（前半2つ）
                    agent_text = "エージェントの意見:\n" + "\n\n".join(agent_outputs[:2]) + "\n\n"
                else:
                    agent_text = "エージェントの意見:\n" + "\n\n".join(agent_outputs) + "\n\n"
                
                if total_length + len(agent_text) <= max_prompt_length:
                    prompt_content += agent_text
            
            prompt_content += "上記の情報を総合的に考慮して、簡潔に回答してください。"
            
            # 6) プロンプト設定とAPI呼び出し
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_content}
            ]
            
            if self.debug:
                print(f"[Debug] 出力エージェント: プロンプト長={len(prompt_content)}文字")
            
            resp = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_output_tokens,
                temperature=0.7
            )
            
            # レスポンス取得
            if isinstance(resp, dict):
                answer = resp['choices'][0]['message']['content'].strip()
            else:
                answer = resp.choices[0].message.content.strip()
                
            if self.debug:
                print(f"[Debug] 最終応答: {len(answer)}文字")

            return answer
            
        except Exception as e:
            error_msg = f"出力生成エラー: {str(e)}"
            if self.debug:
                print(error_msg)
                import traceback
                traceback.print_exc()
            return "申し訳ありません、応答の生成中にエラーが発生しました。"
        
    def _detect_language(self, text: str) -> str:
        """
        テキストの言語を検出する内部メソッド
        
        引数:
            text: 検出対象テキスト
            
        戻り値:
            言語コード（'ja'または'en'）
        """
        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text):
            return 'ja'
        if re.search(r'[A-Za-z]', text):
            return 'en'
        return 'ja'  # デフォルトは日本語
