#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary Engine モジュール
~~~~~~~~~~~~~~~~~~~~~
黒板上の情報を要約するエンジン
長いコンテキストを簡潔にまとめる機能を提供

作者: Yuhi Sonoki
"""

import logging
from typing import Dict, Any, List, Optional
from MurmurNet.modules.model_factory import ModelFactory

logger = logging.getLogger('MurmurNet.SummaryEngine')

class SummaryEngine:
    """
    エージェント出力の要約を行うエンジン
    
    責務:
    - 複数エージェントの出力を統合
    - 長いテキストを簡潔に要約
    - 一貫性のある出力生成
    
    属性:
        config: 設定辞書
        max_summary_tokens: 要約の最大トークン数
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        要約エンジンの初期化
        
        引数:
            config: 設定辞書（省略時は空の辞書）
        """
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.max_summary_tokens = self.config.get('max_summary_tokens', 200)  # 話し言葉に適した要約の最大トークン数
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # ModelFactoryからモデルを取得
        self.llm = ModelFactory.create_model(self.config)
        
        logger.info("要約エンジンを初期化しました")

    def summarize_blackboard(self, entries: List[Dict[str, Any]]) -> str:
        """
        黒板のエントリを要約する
        
        引数:
            entries: 要約するエントリのリスト。各エントリは{'agent': id, 'text': str}の形式
        
        戻り値:
            要約されたテキスト
        """
        try:
            if not entries:
                logger.warning("要約するエントリがありません")
                return "要約するエントリがありません。"
                
            # 入力テキストを制限
            combined_entries = []
            for entry in entries:
                text = entry.get('text', '')
                if len(text) > 200:
                    text = text[:200] + "..."  # 各エントリを200文字に制限
                combined_entries.append(text)
                
            combined = "\n\n".join(combined_entries)
              # 要約用のプロンプト - 話し言葉重視
            prompt = (
                "こんにちは！みんなの意見をまとめて、分かりやすく要約してほしいんだ。"
                "話し言葉で自然に、150〜200文字くらいでポイントをまとめてね。\n\n" + 
                combined
            )
            
            if self.debug:
                logger.debug(f"要約入力: {len(combined)}文字")
            
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
              # 出力を制限（250文字以内、話し言葉に適したサイズ）
            if len(summary) > 250:
                summary = summary[:250]
            
            if self.debug:
                logger.debug(f"要約生成: 入力={len(combined)}文字, 出力={len(summary)}文字")
                
            return summary
            
        except Exception as e:
            error_msg = f"要約エラー: {str(e)}"
            logger.error(error_msg)
            
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
                
            return "要約の生成中にエラーが発生しました。"
