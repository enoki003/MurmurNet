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
            # 1) 入力と RAG を取得（長さ制限を導入）
            inp = blackboard.read('input')
            user_input = inp.get('normalized') if isinstance(inp, dict) else str(inp)
            user_input = user_input[:200]  # 入力を200文字に制限
            
            rag = blackboard.read('rag')
            if rag and isinstance(rag, str):
                rag = rag[:300]  # RAG情報を300文字に制限

            # 2) 言語検出
            lang = self._detect_language(user_input)
            
            # 3) 要約とエージェント出力の整理
            summaries = []
            agent_outputs = []
            
            # サンプル数の制限
            max_summaries = 2  # 最大要約数
            max_agent_outputs = 3  # 最大エージェント出力数
            
            # 最新のものを優先的に選択
            for entry in entries[-max(5, len(entries)):]:  # 最新のエントリのみ処理
                entry_type = entry.get('type', 'agent')
                
                if entry_type == 'summary' and len(summaries) < max_summaries:
                    iteration = entry.get('iteration', 0)
                    text = entry.get('text', '')[:150]  # 要約は150文字に制限
                    summaries.append(f"要約 {iteration+1}: {text}")
                elif len(agent_outputs) < max_agent_outputs:
                    agent_id = entry.get('agent', 0)
                    text = entry.get('text', '')[:150]  # エージェント出力も150文字に制限
                    agent_outputs.append(f"エージェント {agent_id+1}: {text}")
            
            # 4) システムプロンプト作成 - より具体的な指示でLLMを誘導
            if lang == 'ja':
                sys_prompt = (
                    "あなたは高品質な回答を生成する専門家です。以下の指示に従って回答を作成してください：\n"
                    "1. 質問に直接答え、質問の主題に集中し、余計な内容は書かない\n"
                    "2. 短く簡潔な応答を生成する（200-300文字程度）\n"
                    "3. 情報を適切に統合し、話題から外れない\n"
                    "4. 応答が繰り返しになったり、崩壊したりしないよう注意する\n"
                    "5. 質問がない場合は、簡潔に会話を促す"
                )
            else:
                sys_prompt = (
                    "You are an expert in generating high-quality responses. Please follow these guidelines:\n"
                    "1. Answer directly to the question, focus on the topic, and avoid unnecessary content\n"
                    "2. Generate concise responses (about 40-60 words)\n"
                    "3. Integrate information properly and stay on topic\n"
                    "4. Make sure your response is not repetitive or malformed\n"
                    "5. If there's no question, briefly encourage conversation"
                )
            
            # 5) プロンプト内容作成 - 質問を明確化
            # 質問を最も目立つ位置に
            prompt_content = f"ユーザーの質問: 「{user_input}」\n\n"
            
            # 入力量が多すぎる場合は選択
            total_length = len(prompt_content)
            max_prompt_length = 800  # プロンプトの最大長さを短縮（以前は1000）
            
            # 要約情報（優先）
            if summaries:
                summary_text = "要約情報:\n" + "\n".join(summaries) + "\n\n"
                if total_length + len(summary_text) <= max_prompt_length:
                    prompt_content += summary_text
                    total_length += len(summary_text)
            
            # エージェント出力（2番目に優先）
            if agent_outputs and total_length < max_prompt_length:
                # 残りの長さに合わせて調整（常に最も新しいエージェント出力を優先）
                remaining_length = max_prompt_length - total_length
                selected_outputs = []
                
                for output in reversed(agent_outputs):  # 最新のものから
                    if len("\n".join(selected_outputs + [output])) <= remaining_length:
                        selected_outputs.insert(0, output)  # 元の順序を保持
                    else:
                        break
                        
                if selected_outputs:
                    agent_text = "エージェントの意見:\n" + "\n".join(selected_outputs) + "\n\n"
                    prompt_content += agent_text
                    
            # RAGがあれば追加（優先度を下げる）
            if rag and (len(prompt_content) + len(rag) + 30) <= max_prompt_length:
                prompt_content += f"参考情報: {rag}\n\n"
            
            # 明確な指示を追加
            if lang == 'ja':
                prompt_content += (
                    "上記の情報を統合して、以下の点に注意して回答を生成してください：\n"
                    "1. 質問「" + user_input + "」に対する明確で簡潔な回答\n"
                    "2. 要点を押さえた論理的な応答\n"
                    "3. 300文字以内の簡潔さ\n"
                    "4. 繰り返しや冗長な表現を避ける\n"
                    "回答："
                )
            else:
                prompt_content += (
                    "Integrate the information above and generate a response that:\n"
                    "1. Clearly and concisely answers the question: \"" + user_input + "\"\n"
                    "2. Captures the key points logically\n"
                    "3. Is brief (under 60 words)\n"
                    "4. Avoids repetition or redundancy\n"
                    "Response:"
                )
            
            # 6) プロンプト設定とAPI呼び出し
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_content}
            ]
            
            if self.debug:
                print(f"[Debug] 出力エージェント: プロンプト長={len(prompt_content)}文字")
            
            # 出力パラメータを調整して繰り返しを防止
            resp = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_output_tokens,
                temperature=0.5,  # 低めの温度で安定性を向上（以前は0.7）
                top_p=0.9,
                frequency_penalty=1.0,  # 繰り返しを防止
                presence_penalty=0.6    # 繰り返しの単語を減らす
            )
            
            # レスポンス取得
            if isinstance(resp, dict):
                answer = resp['choices'][0]['message']['content'].strip()
            else:
                answer = resp.choices[0].message.content.strip()
                
            # 出力の整形と繰り返し検出 
            answer = self._clean_response(answer, lang)
                
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
    
    def _clean_response(self, text: str, lang: str = 'ja') -> str:
        """
        出力を整形し、応答の品質をチェックする
        
        引数:
            text: 生成された応答テキスト
            lang: 言語コード ('ja'または'en')
            
        戻り値:
            整形された応答テキスト
        """
        # 1. 不要な接頭辞を削除
        prefixes_to_remove = [
            "回答：", "Response:", "答え：", "最終回答：", 
            "Final answer:", "回答:", "Response:", "A:", "回答: "
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # 2. 繰り返しパターンの検出と修正
        # 同じ文や段落が繰り返される場合
        lines = text.split('\n')
        if len(lines) > 3:
            # 行の類似性をチェック
            unique_lines = []
            for line in lines:
                if line and not any(self._is_similar(line, uline) for uline in unique_lines):
                    unique_lines.append(line)
            
            # 繰り返しが検出された場合
            if len(unique_lines) < len(lines) * 0.7:
                text = '\n'.join(unique_lines)
        
        # 3. 同じフレーズの繰り返しを検出（文字列内の繰り返し）
        if len(text) > 50:
            chunks = [text[i:i+20] for i in range(0, len(text)-20, 10)]
            repeats = 0
            
            for i in range(len(chunks)):
                for j in range(i+1, len(chunks)):
                    if self._is_similar(chunks[i], chunks[j], threshold=0.8):
                        repeats += 1
            
            # 繰り返しが多すぎる場合は切り詰め
            if repeats > len(chunks) // 3:
                # テキストの前半部分のみを使用
                if lang == 'ja':
                    text = text[:100] + "..."
                else:
                    text = text[:200] + "..."
        
        # 4. 長さ制限
        max_length = 300 if lang == 'ja' else 500
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def _is_similar(self, str1: str, str2: str, threshold: float = 0.7) -> bool:
        """
        2つの文字列が類似しているかをチェック
        
        引数:
            str1: 比較する文字列1
            str2: 比較する文字列2
            threshold: 類似と判断する閾値
            
        戻り値:
            類似していればTrue
        """
        # 短すぎる文字列はスキップ
        if len(str1) < 5 or len(str2) < 5:
            return False
            
        # 全く同じ場合
        if str1 == str2:
            return True
            
        # 簡易的な類似度計算（Jaccard係数の変形）
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return False
            
        similarity = len(intersection) / len(union)
        return similarity > threshold

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
