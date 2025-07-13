#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Template Builder
~~~~~~~~~~~~~~~
テンプレート破損防止用の安全なテンプレート生成

機能:
- 完全な<start_of_turn>形式テンプレートの生成
- 各Slotは断片を書かず、dict形式でメッセージを渡すだけ
- テンプレート生成の一元化による破損防止

作者: Yuhi Sonoki
"""

import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TemplateBuilder:
    """
    テンプレート破損防止用ビルダー
    外部テンプレートに依存せず、自前で完全なテンプレートを構築
    """
    START = "<start_of_turn>"
    END = "<end_of_turn>"
    VALID_ROLES = {"user", "model", "system"}
    
    @staticmethod
    def build(messages: List[Dict[str, str]], add_generation: bool = True) -> str:
        """
        メッセージリストから安全なテンプレートを構築
        
        Args:
            messages: [{"role": "user", "content": "..."}, ...]形式のメッセージ
            add_generation: 生成プロンプトを追加するかどうか
            
        Returns:
            完成したテンプレート文字列
        """
        if not messages:
            return f"<bos>\n{TemplateBuilder.START}model\n" if add_generation else "<bos>"
        
        buf = ["<bos>"]
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            
            # roleの正規化 - Gemma特化
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Gemmaでは、systemロールを特別扱い
                if i == 0:
                    # 最初のメッセージがsystemの場合、専用フォーマット
                    buf.append(f"{TemplateBuilder.START}system\n{content}{TemplateBuilder.END}")
                    continue
                else:
                    # 途中のsystemメッセージはuserとして扱う
                    role = "user"
                    content = f"System: {content}"
            elif role not in TemplateBuilder.VALID_ROLES:
                role = "user"
            
            # contentの安全化
            content = TemplateBuilder._clean_content(content)
            
            # テンプレート追加
            buf.append(f"{TemplateBuilder.START}{role}\n{content}{TemplateBuilder.END}")
        
        # 生成プロンプトの追加
        if add_generation:
            buf.append(f"{TemplateBuilder.START}model\n")
        
        result = "\n".join(buf)
        logger.debug(f"TemplateBuilder: 構築完了 ({len(messages)}メッセージ, {len(result)}文字)")
        return result
    
    @staticmethod
    def _clean_content(content: str) -> str:
        """
        コンテンツの安全化
        テンプレート断片や制御文字を除去
        """
        if not content:
            return ""
        
        # 1. 既存のテンプレート断片を除去
        content = re.sub(r"<start_of_turn>.*?<end_of_turn>", "", content, flags=re.DOTALL)
        content = re.sub(r"<\|.*?\|>", "", content, flags=re.DOTALL)
        content = re.sub(r"</?s>", "", content)
        content = re.sub(r"<bos>", "", content)
        
        # 2. 連続する制御文字を除去
        content = re.sub(r"[|=]{2,}", "", content)
        content = re.sub(r"[\n\r\t]{3,}", "\n\n", content)
        
        # 3. HTML的な要素を除去
        content = re.sub(r"<[^>]*>", "", content)
        
        # 4. 空白の正規化
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n\s*\n", "\n\n", content)
        
        return content.strip()
    
    @staticmethod
    def slots_to_messages(slot_entries: List[Any]) -> List[Dict[str, str]]:
        """
        Slotエントリをメッセージ形式に変換
        
        Args:
            slot_entries: SlotEntryのリスト
            
        Returns:
            メッセージ形式のリスト
        """
        messages = []
        
        for entry in slot_entries:
            # Slotの出力をクリーンアップ
            clean_text = TemplateBuilder._clean_content(entry.text)
            
            # 空の場合はスキップ
            if not clean_text:
                continue
            
            # メッセージ形式に変換
            messages.append({
                "role": "user",
                "content": f"{entry.slot_name}: {clean_text}"
            })
        
        return messages


class TemplateValidator:
    """
    テンプレート破損検出・修復システム
    """
    PATTERN = re.compile(r"<start_of_turn>(\w+)\n([\s\S]*?)<end_of_turn>")
    VALID_ROLES = {"user", "model", "system"}
    
    @staticmethod
    def is_valid_response(text: str) -> bool:
        """
        応答の妥当性を検証
        
        Args:
            text: 検証するテキスト
            
        Returns:
            妥当性（True/False）
        """
        if not text or len(text.strip()) == 0:
            return False
        
        # 1. テンプレート構造の検証
        if not TemplateValidator.PATTERN.search(text):
            logger.warning("テンプレート構造が見つかりません")
            return False
        
        # 2. 異常な文字パターンの検出
        if re.search(r"[|]{3,}", text):
            logger.warning("縦棒洪水を検出")
            return False
        
        if re.search(r"[=]{5,}", text):
            logger.warning("等号洪水を検出")
            return False
        
        # 3. 破損したテンプレートトークンの検出
        if re.search(r"<\|[^>]*$", text):
            logger.warning("破損したテンプレートトークンを検出")
            return False
        
        if re.search(r"of_of_of|end_of_turn\|", text):
            logger.warning("テンプレート破損パターンを検出")
            return False
        
        # 4. 意味のあるコンテンツの検証
        japanese_chars = len(re.findall(r"[ひらがなカタカナ一-龯]", text))
        if japanese_chars < 3:
            logger.warning("日本語コンテンツが不足")
            return False
        
        return True
    
    @staticmethod
    def parse_chat(text: str) -> List[Dict[str, str]]:
        """
        チャットテンプレートをパース
        
        Args:
            text: パースするテキスト
            
        Returns:
            パース結果のメッセージリスト（失敗時は空リスト）
        """
        try:
            messages = []
            
            for role, content in TemplateValidator.PATTERN.findall(text):
                if role not in TemplateValidator.VALID_ROLES:
                    logger.warning(f"無効なロール: {role}")
                    return []
                
                messages.append({
                    "role": role,
                    "content": content.strip()
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"テンプレートパースエラー: {e}")
            return []


class SafeTemplateGenerator:
    """
    安全なテンプレート生成器
    
    破損検出・自動修復機能付き
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_retry = config.get('template_max_retry', 1)
        self.enable_recovery = config.get('template_enable_recovery', True)
    
    def generate_safe_template(self, messages: List[Dict[str, str]]) -> str:
        """
        安全なテンプレート生成
        
        Args:
            messages: メッセージリスト
            
        Returns:
            安全なテンプレート文字列
        """
        try:
            # 1. 基本テンプレート生成
            template = TemplateBuilder.build(messages)
            
            # 2. 妥当性検証
            if TemplateValidator.is_valid_response(template):
                return template
            
            # 3. 修復試行
            if self.enable_recovery:
                logger.warning("テンプレート修復を試行します")
                return self._repair_template(messages)
            
            # 4. フォールバック
            return self._create_fallback_template(messages)
            
        except Exception as e:
            logger.error(f"テンプレート生成エラー: {e}")
            return self._create_fallback_template(messages)
    
    def _repair_template(self, messages: List[Dict[str, str]]) -> str:
        """
        破損したテンプレートの修復
        """
        # より厳密なクリーニングでリトライ
        cleaned_messages = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # 徹底的なクリーニング
            content = re.sub(r"[|=]{1,}", "", content)
            content = re.sub(r"<[^>]*>", "", content)
            content = re.sub(r"[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF。、！？]", "", content)
            
            if content.strip():
                cleaned_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content.strip()
                })
        
        if cleaned_messages:
            return TemplateBuilder.build(cleaned_messages)
        else:
            return self._create_fallback_template(messages)
    
    def _create_fallback_template(self, messages: List[Dict[str, str]]) -> str:
        """
        フォールバックテンプレートの生成
        """
        fallback_content = "申し訳ございません。テンプレートの生成に問題が発生しました。"
        
        fallback_messages = [
            {"role": "user", "content": fallback_content}
        ]
        
        return TemplateBuilder.build(fallback_messages)
