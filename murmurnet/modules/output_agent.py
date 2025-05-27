#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output Agent モジュール
~~~~~~~~~~~~~~~~~~~~
最終応答生成を担当するエージェント
黒板の要約情報に基づき一貫性のある回答を作成

作者: Yuhi Sonoki
"""

import logging
import re
from typing import Dict, Any, List, Optional
from MurmurNet.modules.model_factory import get_shared_model
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.OutputAgent')

class OutputAgent:
    """
    最終応答を生成するエージェント
    
    責務:
    - 黒板情報の統合
    - 要約と個別エージェント出力の統合
    - 一貫性のある最終応答の生成
    - 言語検出と応答言語の適応
      属性:
        config: 設定辞書
        max_output_tokens: 最終出力の最大トークン数
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        出力エージェントの初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
        """
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
        
        # ConfigManagerから直接設定値を取得
        self.debug = self.config_manager.logging.debug
        self.max_output_tokens = self.config_manager.model.max_tokens  # モデル設定から取得
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            
        # 共有モデルインスタンスを取得（シングルトンパターン）
        self.llm = get_shared_model(self.config)
        
        logger.info("出力エージェントを初期化しました")

    def _detect_language(self, text: str) -> str:
        """
        テキストの言語を検出する（内部メソッド）
        
        引数:
            text: 言語を検出するテキスト
            
        戻り値:
            検出された言語コード ('ja', 'en' など)
        """
        # 日本語の文字が含まれているかチェック
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
            return 'ja'
        # 英語の文字が主に含まれているかチェック
        elif re.search(r'[a-zA-Z]', text) and not re.search(r'[^\x00-\x7F]', text):
            return 'en'
        # その他の言語（デフォルトは英語）
        return 'en'

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
            logger.debug(f"検出された言語: {lang}")
              # 3) 要約とエージェント出力の整理
            summaries = []
            agent_contents = []  # エージェント内容を直接収集（ラベルなし）
            
            for entry in entries:
                entry_type = entry.get('type', 'agent')  # デフォルトはagent
                text = entry.get('text', '')[:300]  # テキスト制限を拡張
                
                if entry_type == 'summary':
                    iteration = entry.get('iteration', 0)
                    summaries.append(f"要約 {iteration+1}: {text}")
                else:  # agent
                    # エージェント出力はラベルなしで内容のみ収集
                    if text and not text.endswith("応答できませんでした"):
                        agent_contents.append(text)              # 4) システムプロンプト作成 - 統合・合成重視の改善
            if lang == 'ja':
                sys_prompt = (
                    "あなたは優秀な日本語アシスタントです。複数の情報源を統合して一つの完璧な回答を作成してください：\n"
                    "1. 質問に対して直接的で包括的な回答を提供してください。\n"
                    "2. 提供された情報を統合・合成し、一貫した説明にまとめてください。\n"
                    "3. 自然な日本語で、話し言葉で回答してください。\n"
                    "4. 複数の視点や情報を組み合わせて、バランスの取れた答えを作成してください。\n"
                    "5. 情報源がある場合は適切に引用してください。\n"
                    "6. 不確実な情報は「〜と考えられます」「〜の可能性があります」として表現してください。\n"
                    "7. 必要に応じて段落分けや構造化で読みやすくしてください。\n"
                    "8. 200〜400文字程度で、完結で分かりやすい回答を作成してください。"
                )            
            else:
                sys_prompt = (
                    "You are an excellent English assistant. Please synthesize multiple sources of information into one comprehensive answer:\n"
                    "1. Provide a direct and comprehensive answer to the question.\n"
                    "2. Integrate and synthesize the provided information into a coherent explanation.\n"
                    "3. Respond in natural, conversational English.\n"
                    "4. Combine multiple perspectives and information to create a balanced answer.\n"
                    "5. Properly cite sources when information is provided.\n"
                    "6. Express uncertain information using phrases like 'it is believed that' or 'it is possible that'.\n"
                    "7. Structure your response with paragraphs when appropriate for readability.\n"
                    "8. Create a concise and clear answer in approximately 200-400 characters."
                )
            
            # 5) ユーザープロンプト作成（統合された情報を使用）
            user_prompt = f"質問: {user_input}\n\n"
            
            # RAG情報があれば追加
            if rag:
                user_prompt += f"参考情報: {rag}\n\n"
                
            # 要約情報があれば追加
            if summaries:
                user_prompt += "要約情報:\n" + "\n".join(summaries) + "\n\n"
                
            # エージェント出力があれば統合して追加
            if agent_contents:
                user_prompt += "専門家の見解:\n" + "\n\n".join(agent_contents) + "\n\n"
                
            user_prompt += "以上の情報を統合・合成して、質問に対する包括的で一貫した最終回答を生成してください。複数の視点を組み合わせ、バランスの取れた統一された回答にしてください。"
            
            # 6) 出力生成
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_output_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # レスポンスの形式によって適切にアクセス
            if isinstance(resp, dict):
                final_output = resp['choices'][0]['message']['content'].strip()
            else:
                final_output = resp.choices[0].message.content.strip()
                
            if self.debug:
                logger.debug(f"最終出力生成: {len(final_output)}文字")
                
            return final_output
            
        except Exception as e:
            error_msg = f"出力生成エラー: {str(e)}"
            logger.error(error_msg)
            
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
                
            # エラー時のフォールバック
            if lang == 'ja':
                return "申し訳ありませんが、応答の生成中にエラーが発生しました。後でもう一度お試しください。"
            else:
                return "I apologize, but an error occurred while generating the response. Please try again later."
