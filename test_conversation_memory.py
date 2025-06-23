#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ConversationMemoryのテスト
"""

import sys
sys.path.append('.')

from MurmurNet.modules.conversation_memory import OptimizedConversationMemory, ConversationMemory
import inspect

print('=== OptimizedConversationMemory メソッド一覧 ===')
methods = [method for method in dir(OptimizedConversationMemory) if not method.startswith('_')]
print(methods)

print('\n=== ConversationMemory エイリアス確認 ===')
print(f'ConversationMemory is OptimizedConversationMemory: {ConversationMemory is OptimizedConversationMemory}')

print('\n=== インスタンス作成テスト ===')
try:
    # 設定とblackboardが必要なので、ダミーを作成
    config = {'conversation_memory': {}}
    
    class DummyBlackboard:
        def read(self, key, default=None):
            return default
        def write(self, key, value):
            pass
    
    blackboard = DummyBlackboard()
    instance = ConversationMemory(config, blackboard)
    print('インスタンス作成成功')
    
    print(f'hasattr(instance, "add_conversation_entry"): {hasattr(instance, "add_conversation_entry")}')
    print(f'callable(getattr(instance, "add_conversation_entry", None)): {callable(getattr(instance, "add_conversation_entry", None))}')
    
    if hasattr(instance, 'add_conversation_entry'):
        sig = inspect.signature(instance.add_conversation_entry)
        print(f'add_conversation_entry signature: {sig}')
        
        # 実際にメソッドを呼び出してみる
        print('\n=== メソッド呼び出しテスト ===')
        instance.add_conversation_entry("テスト入力", "テスト応答")
        print('add_conversation_entry呼び出し成功')
    else:
        print('add_conversation_entryメソッドが見つかりません')
        print('利用可能なメソッド:', [m for m in dir(instance) if not m.startswith('_')])
        
except Exception as e:
    print(f'エラー: {e}')
    import traceback
    traceback.print_exc()
