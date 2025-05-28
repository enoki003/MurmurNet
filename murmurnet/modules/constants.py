#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet Constants Module
~~~~~~~~~~~~~~~~~~~~~~~~~~
Centralized location for internal constants, default strings, and fixed values
not intended for user configuration via config.yaml.
"""

# Default Error Messages
DEFAULT_AGENT_ERROR_RESPONSE = "エージェントは技術的な問題により応答を生成できませんでした。"
AGENT_EMPTY_RESPONSE = "エージェントは空の応答を生成しました。" # Specific for empty
AGENT_NOT_FOUND_ERROR = "要求されたエージェント {agent_id} が見つかりません。" # Added placeholder
PROMPT_GENERATION_ERROR = "エージェントはプロンプトを生成できませんでした。"
LLM_CLIENT_ERROR = "LLMクライアントが正しく設定されていないか、互換性がありません。"
TIMEOUT_ERROR_MSG_TEMPLATE = "{role_name}は処理時間の制限により応答できませんでした。"
MEMORY_ERROR_MSG_TEMPLATE = "{role_name}はメモリ不足のため応答できませんでした。"
CONNECTION_ERROR_MSG_TEMPLATE = "{role_name}は接続の問題により応答できませんでした。"
GENERIC_LLM_ERROR_MSG_TEMPLATE = "{role_name}はモデルの問題で応答できませんでした。"
UNEXPECTED_ERROR_MSG_TEMPLATE = "{role_name}で予期せぬエラーが発生しました: {error}"
AGENT_EXECUTION_ERROR_MSG_TEMPLATE = "{role_name}はエラーにより応答できませんでした: {error_message}"


# Default Agent/Role Configuration Values (if not overridden by specific roles or global config)
DEFAULT_AGENT_ROLE_NAME = "汎用エージェント"
DEFAULT_SYSTEM_PROMPT = "あなたは役立つAIアシスタントです。提供された情報に基づいて回答してください。"
DEFAULT_FALLBACK_USER_QUERY = "ユーザーからの具体的な質問はありませんが、一般的な情報提供をお願いします。"
DEFAULT_NO_OTHER_AGENT_OUTPUTS = "他のエージェントからの意見はまだありません。"
DEFAULT_NO_RAG_INFO = "関連する追加情報はありません。"
DEFAULT_NO_CONVERSATION_CONTEXT = "過去の会話の記録はありません。"
DEFAULT_NO_PREVIOUS_SUMMARY = "前回の議論の要約はありません。"


# Default Operational Parameters (internal, not typically user-tuned, but good to have as constants)
DEFAULT_MAX_RESPONSE_SNIPPET_LOG_LENGTH = 100 # For logging agent responses
DEFAULT_MAX_PROMPT_SNIPPET_LOG_LENGTH = 200 # For logging prompts

# Question Classification Types (if used as constants, though also in AgentPoolManager)
# These might be better if AgentPoolManager imports them from here.
QTYPE_DISCUSSION = "discussion"
QTYPE_PLANNING = "planning"
QTYPE_INFORMATIONAL = "informational"
QTYPE_CONVERSATIONAL = "conversational"
QTYPE_DEFAULT = "default"

# Role names (examples, actual roles are in templates)
ROLE_FALLBACK_ERROR = "ErrorAgent"
ROLE_SYSTEM_ERROR = "SystemError" # Used in SystemCoordinator

# Blackboard related constants (Sources, ContentTypes) are primarily defined in blackboard.py
# as they form its "schema". If any are used more broadly and less tied to BB structure,
# they could be considered here.

# Specific Blackboard Keys not fitting general content_type/source pattern
BLACKBOARD_KEY_PERFORMANCE_ENABLED = "performance_enabled"
BLACKBOARD_KEY_QUESTION_TYPE = "question_type" # Used in AgentPoolManager's old write

# OutputAgent related constants
OUTPUT_AGENT_ENTRY_TYPE_SUMMARY = "summary"
OUTPUT_AGENT_ENTRY_TYPE_AGENT = "agent"


# --- Fin ---
