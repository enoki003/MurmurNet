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
    
    # ログレベルの設定（デバッグモードの考慮）
    log_level = config.get('log_level', 'INFO').upper()
    debug_mode = config.get('debug_mode', False)
    if debug_mode and log_level == 'INFO':
        log_level = 'DEBUG'
    
    level = getattr(logging, log_level, logging.INFO)
    
    # ロガーの作成
    logger = logging.getLogger('MurmurNet')
    
    # 既存のハンドラーがある場合はクリア
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # フォーマッターの設定（詳細レベルに応じて調整）
    if level == logging.DEBUG:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
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
            
            logger.debug(f"ログファイル設定完了: {log_file}")
        except Exception as e:
            logger.warning(f"ログファイルの設定に失敗しました: {e}")
    
    # エラー専用ハンドラー（WARNING以上のレベル用）
    if config.get('error_log_file'):
        try:
            error_handler = logging.FileHandler(config['error_log_file'], encoding='utf-8')
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
            logger.debug(f"エラーログファイル設定完了: {config['error_log_file']}")
        except Exception as e:
            logger.warning(f"エラーログファイルの設定に失敗しました: {e}")
    
    # プロパゲーションを無効にして重複を防ぐ
    logger.propagate = False
    
    logger.debug(f"ロガー設定完了: レベル={log_level}, ハンドラー数={len(logger.handlers)}")
    
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


def log_agent_performance(logger: logging.Logger, agent_id: str, 
                         execution_time: float, output_length: int,
                         error_info: Optional[str] = None) -> None:
    """
    エージェントのパフォーマンス情報をログ出力
    
    引数:
        logger: ロガーインスタンス
        agent_id: エージェントID
        execution_time: 実行時間
        output_length: 出力の長さ
        error_info: エラー情報（任意）
    """
    if error_info:
        logger.warning(f"エージェント{agent_id}: エラー発生 - {error_info} (実行時間: {execution_time:.3f}秒)")
    elif output_length == 0:
        logger.warning(f"エージェント{agent_id}: 空の応答 (実行時間: {execution_time:.3f}秒)")
    else:
        logger.debug(f"エージェント{agent_id}: 正常完了 - {output_length}文字 (実行時間: {execution_time:.3f}秒)")


def log_system_state(logger: logging.Logger, component: str, 
                    state_info: Dict[str, Any]) -> None:
    """
    システム状態をログ出力
    
    引数:
        logger: ロガーインスタンス
        component: コンポーネント名
        state_info: 状態情報辞書
    """
    if logger.isEnabledFor(logging.DEBUG):
        state_str = ", ".join([f"{k}={v}" for k, v in state_info.items()])
        logger.debug(f"[{component}] 状態: {state_str}")


def log_execution_metrics(logger: logging.Logger, process_name: str,
                         start_time: float, end_time: float,
                         success: bool = True, metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    実行メトリクスをログ出力
    
    引数:
        logger: ロガーインスタンス
        process_name: プロセス名
        start_time: 開始時間
        end_time: 終了時間
        success: 成功フラグ
        metrics: 追加メトリクス情報
    """
    execution_time = end_time - start_time
    status = "成功" if success else "失敗"
    
    if metrics:
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"[メトリクス] {process_name}: {status} ({execution_time:.3f}秒) - {metrics_str}")
    else:
        logger.info(f"[メトリクス] {process_name}: {status} ({execution_time:.3f}秒)")


def log_memory_usage(logger: logging.Logger, component: str, memory_mb: float) -> None:
    """
    メモリ使用量をログ出力
    
    引数:
        logger: ロガーインスタンス
        component: コンポーネント名
        memory_mb: メモリ使用量（MB）
    """
    formatted_memory = format_memory_size(int(memory_mb * 1024 * 1024))
    logger.debug(f"[メモリ] {component}: {formatted_memory}")


def create_debug_summary(blackboard_data: Dict[str, Any]) -> str:
    """
    デバッグ用のブラックボードサマリーを作成
    
    引数:
        blackboard_data: ブラックボードデータ
        
    戻り値:
        デバッグサマリー文字列
    """
    summary_parts = []
    
    # 基本情報
    if 'input' in blackboard_data:
        input_data = blackboard_data['input']
        if isinstance(input_data, dict) and 'normalized' in input_data:
            input_text = input_data['normalized']
            summary_parts.append(f"入力: {truncate_text(input_text, 50)}")
    
    # エージェント出力の要約
    agent_outputs = []
    for key, value in blackboard_data.items():
        if key.startswith('agent_') and key.endswith('_output'):
            agent_num = key.split('_')[1]
            output_length = len(str(value)) if value else 0
            if output_length == 0:
                agent_outputs.append(f"Agent{agent_num}:空")
            else:
                agent_outputs.append(f"Agent{agent_num}:{output_length}文字")
    
    if agent_outputs:
        summary_parts.append(f"エージェント: {', '.join(agent_outputs)}")
    
    # 最終応答
    if 'final_response' in blackboard_data:
        response = blackboard_data['final_response']
        if response:
            response_length = len(str(response))
            summary_parts.append(f"最終応答: {response_length}文字")
        else:
            summary_parts.append("最終応答: 空")
    
    return " | ".join(summary_parts) if summary_parts else "データなし"


def validate_agent_output(output: Any, agent_id: str, logger: logging.Logger) -> bool:
    """
    エージェント出力の妥当性を検証
    
    引数:
        output: エージェント出力
        agent_id: エージェントID
        logger: ロガーインスタンス
        
    戻り値:
        妥当性検証の結果
    """
    if output is None:
        logger.warning(f"エージェント{agent_id}: 出力がNoneです")
        return False
    
    if isinstance(output, str) and len(output.strip()) == 0:
        logger.warning(f"エージェント{agent_id}: 空文字列の出力です")
        return False
    
    if hasattr(output, '__len__') and len(output) == 0:
        logger.warning(f"エージェント{agent_id}: 空のコレクションです")
        return False
    
    return True


def log_configuration(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    設定情報をログ出力（機密情報をマスク）
    
    引数:
        logger: ロガーインスタンス
        config: 設定辞書
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    
    # 機密情報をマスクするキー
    sensitive_keys = {'password', 'token', 'key', 'secret', 'api_key'}
    
    masked_config = {}
    for key, value in config.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            masked_config[key] = "***masked***"
        else:
            masked_config[key] = value
    
    logger.debug(f"設定情報: {masked_config}")