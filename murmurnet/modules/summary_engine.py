#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary Engine モジュール
~~~~~~~~~~~~~~~~~~~~~
黒板上の情報を要約するエンジン
長いコンテキストを簡潔にまとめる機能を提供

作者: Yuhi Sonoki
"""

# summary_engine.py
from llama_cpp import Llama
import os

class SummaryEngine:
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
        
        self.debug = config.get('debug', False)
        if self.debug:
            print(f"要約エンジン: モデル設定 n_ctx={llama_kwargs['n_ctx']}, n_threads={llama_kwargs['n_threads']}")
        
        self.llm = Llama(**llama_kwargs)
        self.max_summary_tokens = config.get('max_summary_tokens', 256)  # 要約の最大トークン数

    def summarize_blackboard(self, entries: list) -> str:
        """
        黒板のエントリを要約する
        
        引数:
            entries: 要約するエントリのリスト。各エントリは{'agent': id, 'text': str}の形式
        
        戻り値:
            要約されたテキスト
        """
        try:
            if not entries:
                return "要約するエントリがありません。"
                
            # 入力テキストを制限
            combined = "\n\n".join(e['text'][:200] for e in entries)  # 各エントリを200文字に制限
            
            # 要約用のプロンプト - 簡潔な出力を指示
            prompt = (
                "以下の各エージェントの出力を簡潔に統合・要約してください。200文字以内で要点をまとめてください。\n\n" + 
                combined
            )
            
            resp = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_tokens,
                temperature=0.3,  # 要約は低温ほうが一貫性が高い
                stop=["。", ".", "\n\n"]  # 早めに停止
            )
            
            # レスポンスの形式によって適切にアクセス
            if isinstance(resp, dict):
                summary = resp['choices'][0]['message']['content'].strip()
            else:
                summary = resp.choices[0].message.content.strip()
            
            # 出力を制限（300文字以内）
            summary = summary[:300]
            
            if self.debug:
                print(f"要約生成: 入力={len(combined)}文字, 出力={len(summary)}文字")
                
            return summary
            
        except Exception as e:
            error_msg = f"要約エラー: {str(e)}"
            if self.debug:
                print(error_msg)
                import traceback
                traceback.print_exc()
            return "要約の生成中にエラーが発生しました。"
