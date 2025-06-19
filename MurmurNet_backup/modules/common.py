#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 共通ユーティリティモジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
プロジェクト全体で使用される共通関数とクラス

作者: Yuhi Sonoki
"""

import logging
import os
import sys
from typing import Dict, Any, Optional


class PerformanceError(Exception):
    """パフォーマンス監視関連のエラー"""
    pass


def setup_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    プロジェクト用のロガーを設定
    
    引数:
        config: 設定辞書（ログレベルやファイル出力設定など）
        
    戻り値:
        設定されたロガーインスタンス
    """
    config = config or {}
    
    # ログレベルの設定
    log_level = config.get('log_level', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    
    # ロガーの作成
    logger = logging.getLogger('MurmurNet')
    
    # 既存のハンドラーがある場合はクリア
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # フォーマッターの設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（設定されている場合）
    log_file = config.get('log_file')
    if log_file:
        try:
            # ログディレクトリの作成
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"ログファイルの設定に失敗しました: {e}")
    
    # プロパゲーションを無効にして重複を防ぐ
    logger.propagate = False
    
    return logger


def validate_config(config: Dict[str, Any], required_keys: list = None) -> bool:
    """
    設定辞書の妥当性を検証
    
    引数:
        config: 検証する設定辞書
        required_keys: 必須キーのリスト
        
    戻り値:
        妥当性検証の結果
    """
    if not isinstance(config, dict):
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in config:
                return False
    
    return True


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    辞書から安全に値を取得
    
    引数:
        dictionary: 検索対象の辞書
        key: 取得するキー
        default: キーが存在しない場合のデフォルト値
        
    戻り値:
        取得した値またはデフォルト値
    """
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default


def format_memory_size(bytes_size: int) -> str:
    """
    バイト数を人間が読みやすい形式に変換
    
    引数:
        bytes_size: バイト数
        
    戻り値:
        フォーマットされた文字列
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    テキストを指定した長さで切り詰める
    
    引数:
        text: 切り詰め対象のテキスト
        max_length: 最大長
        suffix: 切り詰め時に追加する接尾辞
        
    戻り値:
        切り詰められたテキスト
    """
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix