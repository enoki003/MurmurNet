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

# conversation_memory.py
from llama_cpp import Llama
import os
import time
import json
from typing import List, Dict, Any, Optional

class ConversationMemory:
    """
    会話の履歴を管理し、要約・圧縮して保持するクラス
    - 過去の会話を保存
    - 会話履歴を効率的に要約
    - 長期記憶と直近の会話を区別して管理
    - 重要な情報を抽出して記憶
    - 黒板アーキテクチャと統合して分散エージェント間で記憶を共有
    """
    
    def __init__(self, config: dict = None, blackboard=None):
        config = config or {}
        self.config = config
        self.debug = config.get('debug', False)
        
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
        
        # 履歴管理設定
        self.max_history_entries = config.get('max_history_entries', 10)  # 保持する会話数の上限
        self.max_summary_tokens = config.get('max_summary_tokens', 256)  # 要約の最大トークン数
        
        # モデルの初期化
        self._init_model()
        
        # 黒板から会話履歴を読み込む（存在する場合）
        if self.blackboard:
            self._load_from_blackboard()
    
    def _init_model(self):
        """モデルの初期化（内部メソッド）"""
        model_path = self.config.get('model_path') or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../models/gemma-3-1b-it-q4_0.gguf')
        )
        
        # 親設定から値を取得（もしくはデフォルト値を使用）
        llama_kwargs = dict(
            model_path=model_path,
            n_ctx=self.config.get('n_ctx', 2048),  # 親設定から受け取る
            n_threads=self.config.get('n_threads', 4),  # 親設定から受け取る
            use_mmap=True,
            use_mlock=False,
            n_gpu_layers=0,
            seed=42,
            chat_format="gemma",
            verbose=False  # ログ出力抑制
        )
        
        chat_template = self.config.get('chat_template')
        if chat_template and os.path.exists(chat_template):
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    llama_kwargs['chat_template'] = f.read()
            except Exception as e:
                if self.debug:
                    print(f"テンプレートロードエラー: {e}")
        
        if self.debug:
            print(f"会話記憶: モデル設定 n_ctx={llama_kwargs['n_ctx']}, n_threads={llama_kwargs['n_threads']}")
        
        self.llm = Llama(**llama_kwargs)
    
    def _load_from_blackboard(self):
        """黒板から会話履歴と要約を読み込む（内部メソッド）"""
        if not self.blackboard:
            return
            
        # 会話履歴の読み込み
        conversation_history = self.blackboard.read('conversation_history')
        if conversation_history:
            self.conversation_history = conversation_history
            
        # 会話要約の読み込み
        history_summary = self.blackboard.read('conversation_summary')
        if history_summary:
            self.history_summary = history_summary
            
        # 重要情報の読み込み
        key_facts = self.blackboard.read('conversation_key_facts')
        if key_facts:
            # 互換性のために古い形式から新しい形式に変換
            if "names" in key_facts or "locations" in key_facts:
                self._migrate_key_facts(key_facts)
            else:
                self.key_facts = key_facts
            
        if self.debug:
            print(f"黒板から読み込み: 履歴={len(self.conversation_history)}件, 要約={self.history_summary is not None}")
    
    def _migrate_key_facts(self, old_key_facts):
        """古い形式のkey_factsを新しい形式に移行（内部メソッド）"""
        # 新しい構造を作成
        new_key_facts = {
            "entities": [],
            "topics": [],
            "context": {}
        }
        
        # 名前を実体として追加
        if "names" in old_key_facts and old_key_facts["names"]:
            for name in old_key_facts["names"]:
                new_key_facts["entities"].append({
                    "value": name,
                    "type": "person",
                    "confidence": 1.0
                })
        
        # 場所を実体として追加
        if "locations" in old_key_facts and old_key_facts["locations"]:
            for location in old_key_facts["locations"]:
                new_key_facts["entities"].append({
                    "value": location,
                    "type": "location",
                    "confidence": 1.0
                })
        
        # 興味・趣味をトピックとして追加
        if "interests" in old_key_facts and old_key_facts["interests"]:
            for interest in old_key_facts["interests"]:
                new_key_facts["topics"].append({
                    "value": interest,
                    "relevance": 0.8
                })
        
        # その他の事実をコンテキストとして追加
        if "facts" in old_key_facts and old_key_facts["facts"]:
            for i, fact in enumerate(old_key_facts["facts"]):
                new_key_facts["context"][f"fact_{i}"] = fact
        
        self.key_facts = new_key_facts
        
    def _save_to_blackboard(self):
        """会話履歴と要約を黒板に保存する（内部メソッド）"""
        if not self.blackboard:
            return
            
        # 会話履歴の保存
        self.blackboard.write('conversation_history', self.conversation_history)
        
        # 会話要約の保存
        if self.history_summary:
            self.blackboard.write('conversation_summary', self.history_summary)
            
        # 重要情報の保存
        self.blackboard.write('conversation_key_facts', self.key_facts)
        
        # 会話コンテキスト（最新の会話要約＋直近の会話）を保存
        context = self._get_conversation_context()
        self.blackboard.write('conversation_context', context)
        
        if self.debug:
            print(f"黒板に保存: 履歴={len(self.conversation_history)}件, コンテキスト長={len(context)}")
        
    def add_conversation_entry(self, user_input: str, system_response: str) -> None:
        """
        会話履歴に新しいエントリを追加
        
        引数:
            user_input: ユーザーの入力
            system_response: システムの応答
        """
        entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'system_response': system_response
        }
        
        # 履歴に追加
        self.conversation_history.append(entry)
        
        # 重要な情報を抽出
        self._extract_key_information(user_input, system_response)
        
        # 上限を超えたら古いものを削除
        if len(self.conversation_history) > self.max_history_entries:
            # 古い会話を削除する前に要約に組み込む
            old_entries = self.conversation_history[:len(self.conversation_history) - self.max_history_entries]
            self._condense_old_entries(old_entries)
            
            # 古い会話を削除
            self.conversation_history = self.conversation_history[-self.max_history_entries:]
            
        # 黒板に保存
        if self.blackboard:
            self._save_to_blackboard()
    
    def _extract_key_information(self, user_input: str, system_response: str) -> None:
        """ユーザー入力とシステム応答から重要な情報を抽出（内部メソッド）"""
        # 会話テキストの準備
        combined_text = f"User: {user_input}\nSystem: {system_response}"
        
        # LLMを使用した情報抽出のプロンプト
        prompt = """
        会話から重要な情報を抽出してください。フォーマットに従って出力してください。

        会話:
        {text}
        
        # 抽出すべき情報
        1. 実体: 会話で言及された重要な人、物、場所、組織など
        2. トピック: 会話で触れられた主要な話題や分野
        3. コンテキスト: キーと値のペアで表現できる会話の重要な情報
        
        JSON形式で出力:
        """
        
        # 長い会話の場合は短縮
        if len(combined_text) > 800:
            combined_text = combined_text[:800] + "..."
        
        try:
            # LLMで情報抽出する代わりに、シンプルな処理で抽出
            # 実際の実装ではLLMを使うことで汎用性を高めるが、テストのための簡易実装
            
            # 1. トピックの抽出
            potential_topics = self._extract_potential_topics(user_input, system_response)
            for topic in potential_topics:
                if topic and not any(existing["value"] == topic for existing in self.key_facts["topics"]):
                    self.key_facts["topics"].append({
                        "value": topic,
                        "relevance": 0.8
                    })
            
            # 2. 文脈情報の更新
            self._update_context_from_conversation(user_input, system_response)
            
        except Exception as e:
            if self.debug:
                print(f"情報抽出エラー: {e}")
    
    def _extract_potential_topics(self, user_input, system_response):
        """会話からトピックを推測する簡易実装（内部メソッド）"""
        # これは簡易実装。実際はLLMを使ってより賢く抽出
        topics = []
        
        # テキストから名詞を抽出したり、頻出単語を分析したりするロジック
        # 簡易版として単語の出現頻度を考慮
        combined_text = f"{user_input} {system_response}"
        words = [w for w in combined_text.split() if len(w) > 1]
        
        # 頻出ワードの集計（実際はもっと高度な処理が必要）
        word_count = {}
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        
        # 頻度上位の単語をトピックとして抽出
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:3]:  # 上位3つのみ
            if count > 1 and len(word) > 2:  # 複数回出現かつ長さ3文字以上
                topics.append(word)
        
        return topics
    
    def _update_context_from_conversation(self, user_input, system_response):
        """会話から文脈情報を更新（内部メソッド）"""
        # これも簡易実装。実際はLLMを使用してキーバリューペアを抽出
        
        # 質問と回答からキーバリューペアを抽出する簡易ロジック
        if "について" in user_input and "は" in system_response:
            parts = user_input.split("について")
            if len(parts) > 1:
                potential_key = parts[0].strip()
                
                # 回答から値を抽出
                response_parts = system_response.split("は", 1)
                if len(response_parts) > 1:
                    potential_value = response_parts[1].split("。")[0].strip()
                    
                    # 十分な長さがあれば追加
                    if len(potential_key) > 1 and len(potential_value) > 2:
                        self.key_facts["context"][potential_key] = potential_value
    
    def _condense_old_entries(self, old_entries: List[Dict[str, Any]]) -> None:
        """古い会話エントリを要約して長期記憶に統合（内部メソッド）"""
        if not old_entries:
            return
            
        # 会話を整形
        conversation_text = ""
        for entry in old_entries:
            conversation_text += f"ユーザー: {entry['user_input']}\n"
            conversation_text += f"システム: {entry['system_response']}\n\n"
            
        # 現在の要約と結合
        if self.history_summary:
            combined_text = f"{self.history_summary}\n\n{conversation_text}"
        else:
            combined_text = conversation_text
            
        # 要約が長すぎる場合は、LLMで要約
        if len(combined_text) > 1000:
            summarized = self._summarize_with_llm(combined_text)
            if summarized:
                self.history_summary = summarized
        else:
            self.history_summary = combined_text
    
    def _summarize_with_llm(self, text: str) -> str:
        """LLMを使用してテキストを要約（内部メソッド）"""
        # テキストが長すぎる場合は事前に切り詰める
        if len(text) > 4000:  # 安全のため最大文字数を制限
            text = text[:4000] + "..."
        
        try:
            # チャットモデル形式でのプロンプト
            messages = [
                {"role": "system", "content": "あなたは会話履歴を要約する専門家です。重要なポイントを簡潔にまとめてください。"},
                {"role": "user", "content": f"次の会話履歴を要約して、重要なポイントだけを200単語以内でまとめてください。\n\n{text}"}
            ]
            
            # LLM呼び出し - チャット形式
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_summary_tokens,
                temperature=0.3,
                top_p=0.9,
                stop=["会話履歴:"],
            )
            
            # 結果の取得
            summary = response['choices'][0]['message']['content'].strip()
            return summary
        except Exception as e:
            if self.debug:
                print(f"要約生成エラー: {e}")
                
            # エラー時は元のテキストを安全に切り詰めて返す
            return text[:1000] + "..."  # エラー時は単純に切り詰め
    
    def _get_conversation_context(self) -> str:
        """
        会話コンテキスト文字列を生成
        要約と直近の会話を組み合わせる
        """
        context = ""
        
        # 要約があれば追加
        if (self.history_summary):
            context += f"過去の会話要約:\n{self.history_summary}\n\n"
            
        # 直近の会話を追加
        context += "最近の会話:\n"
        recent_entries = self.conversation_history[-min(3, len(self.conversation_history)):]
        for entry in recent_entries:
            context += f"ユーザー: {entry['user_input']}\n"
            context += f"システム: {entry['system_response']}\n\n"
        
        # 抽出された重要情報の追加
        if self.key_facts["entities"] or self.key_facts["topics"] or self.key_facts["context"]:
            context += "会話から抽出された情報:\n"
            
            if self.key_facts["entities"]:
                entities_str = ", ".join([entity["value"] for entity in self.key_facts["entities"]])
                context += f"関連する実体: {entities_str}\n"
            
            if self.key_facts["topics"]:
                topics_str = ", ".join([topic["value"] for topic in self.key_facts["topics"]])
                context += f"話題: {topics_str}\n"
            
            if self.key_facts["context"]:
                context += "コンテキスト情報:\n"
                for key, value in self.key_facts["context"].items():
                    context += f"- {key}: {value}\n"
        
        return context
    
    def get_conversation_context(self) -> str:
        """
        外部から呼び出し可能な会話コンテキスト取得メソッド
        """
        return self._get_conversation_context()
    
    def get_answer_for_question(self, question: str) -> str:
        """
        与えられた質問に対する回答を会話履歴や抽出情報から検索する
        
        Args:
            question: 検索する質問
        
        Returns:
            質問に対する回答（見つからない場合は空文字列）
        """
        # まず、コンテキスト情報から回答を探す
        context_answer = self._search_context_for_answer(question)
        if context_answer:
            return context_answer
        
        # 次に、会話履歴から類似する対話を探す
        conversation_answer = self._search_conversation_for_answer(question)
        if conversation_answer:
            return conversation_answer
        
        # LLMを使って回答を生成するのがベストだが、簡易実装として空文字を返す
        return ""
    
    def _search_context_for_answer(self, question: str) -> str:
        """コンテキスト情報から質問に関連する回答を探す（内部メソッド）"""
        # 質問の語彙を分解
        question_words = set(question.lower().split())
        
        # コンテキスト情報を検索
        for key, value in self.key_facts["context"].items():
            # キーと質問の単語が一致するか確認
            key_words = set(key.lower().split())
            if any(word in question_words for word in key_words):
                return f"{key}は{value}です。"
        
        # 実体の検索
        for entity in self.key_facts["entities"]:
            if entity["value"].lower() in question.lower():
                if entity["type"] == "person":
                    return f"{entity['value']}という人物について言及がありました。"
                elif entity["type"] == "location":
                    return f"{entity['value']}という場所について言及がありました。"
                else:
                    return f"{entity['value']}について言及がありました。"
        
        return ""
    
    def _search_conversation_for_answer(self, question: str) -> str:
        """会話履歴から質問に関連する回答を探す（内部メソッド）"""
        # 質問の語彙を分解
        question_words = set(question.lower().split())
        
        best_match = None
        highest_score = 0
        
        # すべての会話履歴をループして類似度を計算
        for entry in self.conversation_history:
            user_input = entry["user_input"].lower()
            user_words = set(user_input.split())
            
            # 共通単語に基づくシンプルな類似度計算
            # 実際の実装では、埋め込みベクトルの類似度などを使うべき
            common_words = question_words.intersection(user_words)
            score = len(common_words) / max(len(question_words), 1)
            
            if score > highest_score:
                highest_score = score
                best_match = entry
        
        # 十分な類似度があれば回答を返す
        if (best_match and highest_score > 0.3):  # 類似度閾値
            return best_match["system_response"]
        
        return ""
        
    def clear_memory(self):
        """会話履歴と要約を完全にクリア"""
        self.conversation_history = []
        self.history_summary = None
        self.key_facts = {
            "entities": [],     # 抽出された重要な実体
            "topics": [],       # 話題や分野
            "context": {}       # 文脈情報（キーバリューペア）
        }
        
        # 黒板上のデータもクリア
        if self.blackboard:
            self.blackboard.delete('conversation_history')
            self.blackboard.delete('conversation_summary')
            self.blackboard.delete('conversation_key_facts')
            self.blackboard.delete('conversation_context')
            
        if self.debug:
            print("会話記憶がクリアされました")
    
    def get_key_facts(self) -> Dict[str, Any]:
        """
        抽出された重要な情報を取得
        """
        return self.key_facts