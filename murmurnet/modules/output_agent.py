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
import time
from typing import Dict, Any, List, Optional
from MurmurNet.modules.model_factory import ModelFactory

logger = logging.getLogger('MurmurNet.OutputAgent')

class OutputAgent:
    """
    最終応答を生成するエージェント
    
    責務:
    - 黒板情報の統合    - 要約と個別エージェント出力の統合
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
            config: 設定辞書（省略時は空の辞書）
        """        
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.max_output_tokens = self.config.get('max_output_tokens', 250)  # より短い制限
        
        # デバッグモードを強制的に有効にする
        logger.setLevel(logging.DEBUG)
        
        # ModelFactoryからモデルを取得
        self.llm = ModelFactory.create_model(self.config)
        
        logger.info("出力エージェントを初期化しました")
        logger.debug(f"OutputAgent初期化: debug={self.debug}, max_tokens={self.max_output_tokens}")

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
            return 'ja'        # 英語の文字が主に含まれているかチェック
        elif re.search(r'[a-zA-Z]', text) and not re.search(r'[^\x00-\x7F]', text):
            return 'en'        # その他の言語（デフォルトは英語）
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
        # パフォーマンス監視開始
        start_time = time.time()
        logger.debug("=== OutputAgent.generate() 呼び出し開始 ===")
        logger.info(f"OutputAgent: エントリ数={len(entries)}で最終応答生成を開始")
        
        try:
            # 1) 入力と RAG を取得
            logger.info("Step 1: 入力とRAG情報を取得中...")
            inp = blackboard.read('input')
            user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
            user_input = user_input[:200]  # 入力を制限
            logger.info(f"ユーザー入力: {user_input}")
            
            rag = blackboard.read('rag')
            if rag and isinstance(rag, str):
                rag = rag[:300]  # RAG情報も制限
            logger.info(f"RAG情報: {rag}")

            # 2) 言語検出
            lang = self._detect_language(user_input)
            logger.debug(f"検出された言語: {lang}")            # 3) 要約とエージェント出力の整理
            summaries = []
            agent_outputs = []
            
            if self.debug:
                logger.debug(f"OutputAgent: 受信したエントリ数: {len(entries)}")            
            for entry in entries:
                entry_type = entry.get('type', 'agent')  # デフォルトはagent
                text = entry.get('text', '')[:300]  # テキスト制限を300文字に拡大
                
                if self.debug:
                    logger.debug(f"OutputAgent: エントリタイプ={entry_type}, テキスト長={len(text)}")
                
                if entry_type == 'summary':
                    iteration = entry.get('iteration', 0)
                    summaries.append(f"要約 {iteration+1}: {text}")
                else:  # agent
                    agent_id = entry.get('agent', 0)
                    agent_outputs.append(f"エージェント {agent_id+1}: {text}")
            
            if self.debug:
                logger.debug(f"OutputAgent: 要約数={len(summaries)}, エージェント出力数={len(agent_outputs)}")
                  # 4) システムプロンプト作成 - 研究に基づく最適化
            if lang == 'ja':
                sys_prompt = (
                    "あなたは親しみやすい日本語アシスタントです。\n\n"
                    "【タスク】\n"
                    "複数のエージェントの意見を統合して、ユーザーの質問に対する自然で有用な回答を作成してください。\n\n"
                    "【重要な指針】\n"
                    "1. エージェントの生の意見を最重視し、各エージェントの異なる視点を活かす\n"
                    "2. 要約情報は補助的な参考として使用する\n"
                    "3. 自然な話し言葉で親しみやすく回答する\n"
                    "4. 250-300文字程度で簡潔かつ完全にまとめる\n"
                    "5. マークダウンや特殊記号は使わず、読みやすい文章にする\n"
                    "6. エージェント間の意見の違いがあれば、バランスよく統合する\n\n"
                    "【出力形式】\n"
                    "- 一つの段落で完結した回答\n"
                    "- 句読点を適切に使用\n"
                    "- 文章の途中で終わらせない"
                )
            else:
                sys_prompt = (
                    "You are a friendly English assistant.\n\n"
                    "【Task】\n"
                    "Integrate multiple agents' opinions to create a natural and helpful response to the user's question.\n\n"
                    "【Key Guidelines】\n"
                    "1. Prioritize agents' raw opinions and leverage different perspectives\n"
                    "2. Use summary information as supplementary reference\n"
                    "3. Respond in natural, conversational language\n"
                    "4. Keep response around 250-300 characters, concise but complete\n"
                    "5. Avoid markdown or special symbols, use readable text\n"
                    "6. If agents have different opinions, integrate them in a balanced way\n\n"
                    "【Output Format】\n"
                    "- Single paragraph with complete response\n"
                    "- Use proper punctuation\n"
                    "- Do not end mid-sentence"
                )            # 5) ユーザープロンプト作成（構造化された研究ベースの形式）
            user_prompt = f"【ユーザーの質問】\n{user_input}\n\n"
            
            # RAG情報があれば追加（明確にラベル付け）
            if rag:
                user_prompt += f"【参考情報】\n{rag}\n\n"
            
            # エージェント出力を最初に配置（最も重要な情報源）
            if agent_outputs:
                user_prompt += "【エージェントの意見】\n"
                for i, output in enumerate(agent_outputs, 1):
                    user_prompt += f"{i}. {output}\n"
                user_prompt += "\n"
            
            # 要約情報は参考として配置
            if summaries:
                user_prompt += "【要約情報（参考）】\n"
                for i, summary in enumerate(summaries, 1):
                    user_prompt += f"{i}. {summary}\n"
                user_prompt += "\n"
            
            # 明確な指示を追加（研究で重要とされる明示的な指示）
            if lang == 'ja':
                user_prompt += (
                    "【回答作成の指示】\n"
                    "上記のエージェントの意見を最重視して、ユーザーの質問に対する統合された回答を作成してください。\n"
                    "・各エージェントの個性的な表現や異なる視点を活かしてください\n"
                    "・要約情報は補助的な参考情報として扱ってください\n"
                    "・自然で読みやすい一つの段落にまとめてください\n"
                    "・文章は完結させ、途中で終わらせないでください"
                )
            else:
                user_prompt += (
                    "【Response Instructions】\n"
                    "Create an integrated response prioritizing the agents' opinions above.\n"
                    "・Leverage each agent's unique expressions and perspectives\n"
                    "・Use summary information as supplementary reference\n"
                    "・Compose in a natural, readable single paragraph\n"
                    "・Ensure the response is complete and doesn't end mid-sentence"
                )
            
            if self.debug:
                logger.debug(f"OutputAgent: システムプロンプト: {sys_prompt[:100]}...")
                logger.debug(f"OutputAgent: ユーザープロンプト: {user_prompt[:300]}...")
              # 6) 出力生成（研究に基づく最適化パラメータ）
            logger.info("Step 6: LLMによる出力生成を開始...")
            logger.info(f"システムプロンプト: {sys_prompt[:100]}...")
            logger.info(f"ユーザープロンプト: {user_prompt[:300]}...")
            
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_output_tokens,
                temperature=0.7,  # 研究推奨: 0.7-0.8の範囲、創造性と一貫性のバランス
                top_p=0.95,  # 研究推奨: 0.9-0.95、多様性を保ちつつ品質維持
                repeat_penalty=1.2  # 研究推奨: 1.1-1.3、繰り返し抑制で出力崩壊防止
            )
            logger.info("LLM応答を受信しました")
              # レスポンスの形式によって適切にアクセス
            if isinstance(resp, dict):
                final_output = resp['choices'][0]['message']['content'].strip()
            else:
                final_output = resp.choices[0].message.content.strip()
                
            # 出力後処理：繰り返しパターンや不安定な出力を除去
            final_output = self._clean_output(final_output)
                
            logger.info(f"最終出力: {final_output}")
            logger.info(f"OutputAgent: 最終応答生成完了 ({len(final_output)}文字)")
            logger.info("=== OutputAgent.generate() 呼び出し終了 ===")
            
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

    def _clean_output(self, text: str) -> str:
        """
        出力後処理：研究に基づく出力崩壊パターンの除去
        
        引数:
            text: 生成されたテキスト
        
        戻り値:
            クリーンアップされたテキスト
        """
        import re
        
        # 1. 典型的な出力崩壊パターンを除去
        # 連続する特殊記号（**、##、--、==等）
        text = re.sub(r'\*{2,}', '', text)  # **以上の連続する*
        text = re.sub(r'#{2,}', '', text)   # ##以上の連続する#
        text = re.sub(r'-{3,}', '', text)   # ---以上の連続する-
        text = re.sub(r'={3,}', '', text)   # ===以上の連続する=
        text = re.sub(r'_{3,}', '', text)   # ___以上の連続する_
        
        # 2. 繰り返し文字パターンを除去
        # 同じ文字が5回以上連続する場合（aaaaaなど）
        text = re.sub(r'(.)\1{4,}', r'\1', text)
        
        # 3. 連続する改行や空白を制限
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {3,}', ' ', text)
        
        # 4. 繰り返し単語/フレーズパターンを除去
        # 同じ単語が3回以上連続
        text = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', text)
        # 同じフレーズが2回以上連続（日本語も考慮）
        text = re.sub(r'([^\.\n]{8,})\1{1,}', r'\1', text)
        
        # 5. 不完全な文章の処理（研究で重要とされる出力品質の改善）
        sentences = text.split('。')
        if len(sentences) > 1:
            # 最後の文が不完全（短すぎる、または特殊文字のみ）の場合除去
            if sentences[-1].strip() == '' or len(sentences[-1].strip()) < 5:
                sentences = sentences[:-1]
            # 最後の文が完全でない場合（句読点で終わらない）
            elif not sentences[-1].strip().endswith(('。', '！', '？', '.', '!', '?')):
                sentences = sentences[:-1]
            text = '。'.join(sentences)
            if text and not text.endswith('。'):
                text += '。'
        
        # 6. マークダウン記号や特殊記号の除去
        text = re.sub(r'^#+\s*', '', text)  # 先頭の#記号
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # リンク記法
        text = re.sub(r'`([^`]+)`', r'\1', text)  # インラインコード
        
        # 7. 異常な文字パターンの除去
        # 制御文字の除去
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # 8. 最終的な整形
        text = text.strip()
        
        # 9. 空の場合のフォールバック
        if not text or len(text.strip()) < 10:
            return "申し訳ございませんが、適切な回答を生成できませんでした。"
        
        return text
