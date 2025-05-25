#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conversation Memory モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~
会話履歴を管理し、関連する記憶を要約・保持する
過去の対話から文脈情報を活用できるようにする
黒板アーキテクチャと統合して分散エージェント間で記憶を共有

作者: Yuhi Sonoki
"""

import logging
import time
import json
import re
from typing import List, Dict, Any, Optional
from MurmurNet.modules.model_factory import ModelFactory
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.ConversationMemory')

class ConversationMemory:
    """
    会話の履歴を管理し、要約・圧縮して保持するクラス
    
    責務:
    - 過去の会話を保存
    - 会話履歴を効率的に要約
    - 長期記憶と直近の会話を区別して管理
    - 重要な情報を抽出して記憶
    
    属性:
        config: 設定辞書
        conversation_history: 会話履歴のリスト
        key_facts: 抽出された重要な情報を保持する辞書
    """
    def __init__(self, config: Dict[str, Any] = None, blackboard=None):
        """
        会話記憶モジュールの初期化
        
        引数:
            config: 設定辞書（オプション、使用されない場合はConfigManagerから取得）
            blackboard: 黒板インスタンス（省略可）
        """
        # ConfigManagerから設定を取得
        self.config_manager = get_config()
        self.config = config or self.config_manager.to_dict()  # 後方互換性のため
        
        # ConfigManagerから直接設定値を取得
        self.debug = self.config_manager.logging.debug
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # 黒板インスタンス
        self.blackboard = blackboard
        
        # 会話履歴の保存
        self.conversation_history: List[Dict[str, Any]] = []
        self.history_summary: Optional[str] = None
        
        # 重要な情報の保存（汎用的な構造）
        self.key_facts: Dict[str, Any] = {
            "entities": [],     # 抽出された重要な実体
            "topics": [],       # 話題や分野
            "context": {}       # 文脈情報（キーバリューペア）
        }
        
        # ConfigManagerから履歴管理設定を取得
        self.max_history_entries = self.config_manager.memory.max_history_entries
        self.max_summary_tokens = self.config_manager.memory.max_summary_tokens
        
        # ModelFactoryからモデルを取得
        self.llm = ModelFactory.create_model(self.config)
        
        # ConfigManagerからembedding model設定を取得
        self.embedding_model_name = self.config_manager.rag.embedding_model
        
        # 黒板から会話履歴を読み込む（存在する場合）
        if self.blackboard:
            self._load_from_blackboard()
            
        logger.info("会話記憶モジュールを初期化しました")
    
    def _load_from_blackboard(self) -> None:
        """
        黒板から会話履歴と要約を読み込む（内部メソッド）
        
        黒板に保存されている会話履歴があれば読み込む
        """
        if not self.blackboard:
            return
            
        # 会話履歴の読み込み
        conversation_history = self.blackboard.read('conversation_history')
        if conversation_history:
            self.conversation_history = conversation_history
            logger.debug(f"黒板から会話履歴を読み込みました ({len(conversation_history)}件)")
            
        # 要約の読み込み
        history_summary = self.blackboard.read('history_summary')
        if history_summary:
            self.history_summary = history_summary
            logger.debug("黒板から履歴要約を読み込みました")
            
        # 重要情報の読み込み
        key_facts = self.blackboard.read('key_facts')
        if key_facts:
            self.key_facts = key_facts
            logger.debug("黒板から重要情報を読み込みました")
    
    def add_conversation_entry(self, user_input: str, system_response: str) -> None:
        """
        会話エントリを追加
        
        引数:
            user_input: ユーザーの入力
            system_response: システムの応答
        """
        if not user_input or not system_response:
            logger.warning("無効な会話エントリ（入力または応答が空）")
            return
            
        # タイムスタンプ付きエントリを作成
        entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'system_response': system_response
        }
        
        # 履歴に追加
        self.conversation_history.append(entry)
        
        # 上限を超えた場合は古いエントリを削除
        if len(self.conversation_history) > self.max_history_entries:
            self.conversation_history = self.conversation_history[-self.max_history_entries:]
            logger.debug(f"履歴を制限: {self.max_history_entries}件に切り詰め")
            
        # 定期的に要約を更新
        if len(self.conversation_history) % 3 == 0:  # 3回ごとに更新
            self._update_summary()
            self._extract_key_facts()
            
        # 黒板があれば会話履歴を保存
        if self.blackboard:
            self.blackboard.write('conversation_history', self.conversation_history)
            if self.history_summary:
                self.blackboard.write('history_summary', self.history_summary)
            if self.key_facts:
                self.blackboard.write('key_facts', self.key_facts)
                logger.debug(f"会話エントリを追加しました (履歴数: {len(self.conversation_history)}件)")
        
    def _update_summary(self) -> None:
        """
        会話履歴の要約を更新（内部メソッド）
        
        現在の会話履歴に基づいて要約を生成・更新する
        """
        if not self.conversation_history:
            self.history_summary = None
            return
            
        try:
            # 直近5つの会話を使用
            recent_history = self.conversation_history[-5:]
            
            # 要約用の入力テキストを作成
            history_text = ""
            for i, entry in enumerate(recent_history):
                user = entry.get('user_input', '')[:100]  # 長さ制限
                system = entry.get('system_response', '')[:200]  # 長さ制限
                history_text += f"会話{i+1}:\nユーザー: {user}\nシステム: {system}\n\n"
                
            # 要約プロンプト
            prompt = (
                "以下の会話履歴を要約してください。ユーザーの名前、興味、趣味、個人情報などの重要な詳細を含めてください。"
                "200文字以内で簡潔にまとめてください。\n\n" + history_text
            )
            
            # ModelFactoryのモデルを使用して要約生成
            if hasattr(self.llm, 'create_chat_completion'):
                resp = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_summary_tokens,
                    temperature=0.5,  # 要約は低温で一貫性を重視
                    top_p=0.9
                )
                
                # レスポンスの形式によって適切にアクセス
                if isinstance(resp, dict) and 'choices' in resp:
                    summary = resp['choices'][0]['message']['content'].strip()
                else:
                    summary = str(resp).strip()
            else:
                # generate メソッドを使用
                summary = self.llm.generate(prompt, max_tokens=self.max_summary_tokens, temperature=0.5)
                
            # 要約を更新
            self.history_summary = summary
            logger.debug(f"会話履歴要約を更新しました ({len(summary)}文字)")
            
        except Exception as e:
            logger.error(f"履歴要約エラー: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
    
    def _extract_key_facts(self) -> None:
        """
        会話から重要な情報を抽出（内部メソッド）
        
        ユーザーに関する重要な情報（名前、好み、個人情報など）を
        抽出して構造化された形式で保存する
        """
        if not self.conversation_history:
            return
            
        try:
            # 直近の会話を使用
            recent_history = self.conversation_history[-3:]
            
            # 抽出用の入力テキストを作成
            history_text = ""
            for i, entry in enumerate(recent_history):
                user = entry.get('user_input', '')[:100]  # 長さ制限
                system = entry.get('system_response', '')[:200]  # 長さ制限
                history_text += f"会話{i+1}:\nユーザー: {user}\nシステム: {system}\n\n"
                
            # 抽出プロンプト
            prompt = (
                "以下の会話からユーザーに関する重要な情報を抽出してください。"
                "名前、興味、趣味、好きなもの、嫌いなもの、個人情報などを特定してください。"
                "JSONフォーマットで返してください:\n"
                "{\n"
                "  \"名前\": \"抽出された名前または不明\",\n"
                "  \"興味\": [\"興味1\", \"興味2\", ...],\n"
                "  \"好きなもの\": [\"項目1\", \"項目2\", ...],\n"
                "  \"その他の情報\": {\"キー1\": \"値1\", ...}\n"
                "}\n\n" + history_text
            )
              # 情報抽出
            if hasattr(self.llm, 'create_chat_completion'):
                resp = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.3,  # 抽出は低温で一貫性を重視
                    top_p=0.9
                )
                
                # レスポンスの形式によって適切にアクセス
                if isinstance(resp, dict) and 'choices' in resp:
                    content = resp['choices'][0]['message']['content'].strip()
                else:
                    content = str(resp).strip()
            else:
                # generate メソッドを使用
                content = self.llm.generate(prompt, max_tokens=512, temperature=0.3)
                
            # JSON部分を抽出
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    extracted = json.loads(json_str)
                    
                    # 既存の情報と統合
                    if "名前" in extracted and extracted["名前"] != "不明":
                        self.key_facts["context"]["名前"] = extracted["名前"]
                        
                    if "興味" in extracted:
                        for interest in extracted["興味"]:
                            if interest not in self.key_facts["topics"]:
                                self.key_facts["topics"].append(interest)
                                
                    if "好きなもの" in extracted:
                        if "好きなもの" not in self.key_facts["context"]:
                            self.key_facts["context"]["好きなもの"] = []
                        for item in extracted["好きなもの"]:
                            if item not in self.key_facts["context"]["好きなもの"]:
                                self.key_facts["context"]["好きなもの"].append(item)
                                
                    if "その他の情報" in extracted:
                        for key, value in extracted["その他の情報"].items():
                            self.key_facts["context"][key] = value
                            
                    logger.debug("重要情報を抽出しました")
                    
                except json.JSONDecodeError:
                    logger.error("JSON解析エラー")
                    
        except Exception as e:
            logger.error(f"情報抽出エラー: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
    
    def get_conversation_context(self) -> str:
        """
        現在の会話コンテキストを取得
        
        戻り値:
            会話コンテキスト文字列
        """
        # 履歴がない場合
        if not self.conversation_history:
            return "過去の会話はありません。"
            
        # 要約があればそれを使用
        if self.history_summary:
            return self.history_summary
            
        # なければ直近の会話を表示
        recent = self.conversation_history[-1]
        user = recent.get('user_input', '')[:50]
        system = recent.get('system_response', '')[:50]
        return f"直前の会話 - ユーザー: {user}... システム: {system}..."
    
    def get_answer_for_question(self, question: str) -> Optional[str]:
        """
        記憶に基づいた質問への回答を取得
        
        引数:
            question: 質問テキスト
            
        戻り値:
            回答文字列（該当なしの場合はNone）
        """
        # 質問マッチング用キーワードを抽出
        keywords = question.lower().split()
        
        # 個人情報関連の質問パターン
        name_patterns = [
            r"(私|僕|俺|わたし|ぼく|おれ)の名前(は|を)(なに|何|なん)",
            r"(私|僕|俺|わたし|ぼく|おれ)(の|は|を)(なに|何|なん)と(言い|いい|呼び|よび|よん)",
            r"(私|僕|俺|わたし|ぼく|おれ)の名前(は|を)(覚え|おぼえ)"
        ]
        
        hobby_patterns = [
            r"(私|僕|俺|わたし|ぼく|おれ)の(趣味|しゅみ)(は|を)(なに|何|なん)",
            r"(私|僕|俺|わたし|ぼく|おれ)(は|が)(なに|何|なん)(が好き|を好きと言いました)"
        ]
        
        # 名前を尋ねる質問
        for pattern in name_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                if "名前" in self.key_facts["context"]:
                    name = self.key_facts["context"]["名前"]
                    return f"あなたの名前は{name}ですね。覚えています。"
                else:
                    return "すみません、まだあなたのお名前をうかがっていないようです。"
        
        # 趣味を尋ねる質問
        for pattern in hobby_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                if "好きなもの" in self.key_facts["context"] and self.key_facts["context"]["好きなもの"]:
                    likes = "、".join(self.key_facts["context"]["好きなもの"][:3])
                    return f"あなたは{likes}が好きだとおっしゃっていましたね。"
                elif self.key_facts["topics"]:
                    topics = "、".join(self.key_facts["topics"][:3])
                    return f"あなたは{topics}に興味があるようですね。"
                else:
                    return "すみません、まだあなたの趣味や好きなものについて話していないようです。"
        
        # 一般的な「覚えてる？」系の質問
        if "覚え" in question or "おぼえ" in question:
            if self.history_summary:
                return f"はい、覚えています。{self.history_summary}"
            elif self.conversation_history:
                return "はい、会話は覚えていますが、何について具体的に知りたいですか？"
            else:
                return "まだ十分な会話をしていないので、特に覚えているものはありません。"
        
        # キーワードベースの検索
        for entry in reversed(self.conversation_history):  # 新しい順に検索
            user_input = entry.get('user_input', '').lower()
            system_response = entry.get('system_response', '')
            
            # キーワードマッチング
            match_score = 0
            for keyword in keywords:
                if len(keyword) > 2 and keyword in user_input:  # 3文字以上のキーワードのみ
                    match_score += 1
                    
            if match_score >= 2:  # 2つ以上のキーワードがマッチした場合
                return f"以前、あなたが「{entry['user_input'][:30]}...」と言ったとき、私は「{system_response[:50]}...」とお答えしました。"
        
        # マッチする記憶が見つからない場合
        return None
        
    def clear_memory(self) -> None:
        """会話履歴をリセット"""
        self.conversation_history = []
        self.history_summary = None
        self.key_facts = {
            "entities": [],
            "topics": [],
            "context": {}
        }
        
        # 黒板があれば情報をクリア
        if self.blackboard:
            self.blackboard.write('conversation_history', [])
            self.blackboard.write('history_summary', None)
            self.blackboard.write('key_facts', self.key_facts)
            
        logger.info("会話記憶をクリアしました")
