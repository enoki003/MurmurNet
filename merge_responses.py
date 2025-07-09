#!/usr/bin/env python3
"""
merge_responses.py
エージェント応答のマージロジック（空レス排除対応）
"""

import logging

logger = logging.getLogger(__name__)

def merge_agent_responses(responses: list, max_responses: int = 3) -> str:
    """
    エージェント応答をマージする（空レス排除版）
    
    Args:
        responses: エージェント応答のリスト
        max_responses: マージする最大応答数
        
    Returns:
        マージされた応答文字列
    """
    if not responses:
        return "応答がありませんでした。"
    
    # 空レス排除フィルタ
    valid_responses = []
    for response in responses:
        response_str = str(response).strip()
        
        # 空レス判定（長さ、内容、エラーメッセージチェック）
        if (response_str and 
            len(response_str) > 3 and  # 最小文字数
            response_str not in ["...", "。", "、", ""] and  # 無意味な応答
            not any(pattern in response_str.lower() for pattern in [
                "エラー", "応答できません", "生成できません", "申し訳", "わかりません"
            ])):
            valid_responses.append(response_str)
    
    if not valid_responses:
        logger.warning("有効な応答がありませんでした")
        return "有効な応答を生成できませんでした。"
    
    # 最大数制限
    selected_responses = valid_responses[:max_responses]
    
    # マージ処理
    if len(selected_responses) == 1:
        return selected_responses[0]
    
    # 複数応答をまとめる
    merged = ""
    for i, response in enumerate(selected_responses, 1):
        if len(selected_responses) > 1:
            merged += f"観点{i}: {response}\n\n"
        else:
            merged += response
    
    return merged.strip()


def filter_valid_responses(blackboard_data: dict) -> list:
    """
    黒板データから有効な応答のみをフィルタリング
    
    Args:
        blackboard_data: 黒板データ
        
    Returns:
        有効な応答のリスト
    """
    valid_responses = []
    
    for key, value in blackboard_data.items():
        if key.startswith('agent_') and key.endswith('_output'):
            response_str = str(value).strip()
            
            # 空レス排除フィルタ（merge_agent_responsesと同じロジック）
            if (response_str and 
                len(response_str) > 3 and
                response_str not in ["...", "。", "、", ""] and
                not any(pattern in response_str.lower() for pattern in [
                    "エラー", "応答できません", "生成できません", "申し訳", "わかりません",
                    "空の応答", "適切な応答を生成できません"
                ])):
                valid_responses.append(response_str)
            else:
                logger.debug(f"無効な応答を排除: {key} = '{response_str[:50]}'")
    
    return valid_responses


def smart_response_merge(responses: list, merge_strategy: str = "diverse") -> str:
    """
    スマート応答マージ（重複排除、多様性重視）
    
    Args:
        responses: 応答リスト
        merge_strategy: マージ戦略（"diverse", "consensus", "best"）
        
    Returns:
        マージされた応答
    """
    if not responses:
        return "応答がありませんでした。"
    
    # 有効応答のフィルタリング
    valid_responses = []
    for response in responses:
        response_str = str(response).strip()
        if len(response_str) > 10:  # より厳しい基準
            valid_responses.append(response_str)
    
    if not valid_responses:
        return "有効な応答を生成できませんでした。"
    
    # 重複除去（部分一致も考慮）
    unique_responses = []
    for response in valid_responses:
        is_duplicate = False
        for existing in unique_responses:
            # 80%以上類似していたら重複とみなす
            if len(set(response.split()) & set(existing.split())) / max(len(response.split()), len(existing.split())) > 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_responses.append(response)
    
    # 戦略別マージ
    if merge_strategy == "best" and unique_responses:
        # 最長の応答を選択
        return max(unique_responses, key=len)
    elif merge_strategy == "consensus":
        # 共通キーワードで要約
        return _create_consensus_response(unique_responses)
    else:  # "diverse"
        # 多様性を重視して組み合わせ
        return merge_agent_responses(unique_responses, max_responses=3)


def _create_consensus_response(responses: list) -> str:
    """コンセンサス応答の作成"""
    if not responses:
        return "コンセンサスを形成できませんでした。"
    
    # 共通キーワードの抽出
    all_words = []
    for response in responses:
        all_words.extend(response.split())
    
    from collections import Counter
    common_words = [word for word, count in Counter(all_words).items() if count >= 2]
    
    if common_words:
        return f"共通観点: {' '.join(common_words[:10])} に基づく総合的な回答として、{responses[0][:100]}..."
    else:
        return responses[0] if responses else "コンセンサスを形成できませんでした。"
