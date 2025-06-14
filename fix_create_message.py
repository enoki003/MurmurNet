#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_message呼び出しを修正するスクリプト
"""
import re

def fix_create_message_calls():
    """create_message呼び出しを修正"""
    file_path = 'MurmurNet/modules/module_adapters.py'
    
    # ファイルを読み込み
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # パターン1: MessageType.AGENT_RESPONSE の場合
    content = re.sub(
        r'create_message\(\s*MessageType\.AGENT_RESPONSE,\s*\{',
        'create_message(MessageType.AGENT_RESPONSE, "agent_adapter", {',
        content
    )
    
    # パターン2: MessageType.AGENT_ERROR の場合  
    content = re.sub(
        r'create_message\(\s*MessageType\.AGENT_ERROR,\s*\{',
        'create_message(MessageType.AGENT_ERROR, "agent_adapter", {',
        content
    )
    
    # パターン3: MessageType.SUMMARY の場合
    content = re.sub(
        r'create_message\(\s*MessageType\.SUMMARY,\s*\{',
        'create_message(MessageType.SUMMARY, "summary_adapter", {',
        content
    )
    
    # パターン4: その他のメッセージタイプも修正
    content = re.sub(
        r'create_message\(\s*MessageType\.([A-Z_]+),\s*\{',
        r'create_message(MessageType.\1, "adapter", {',
        content
    )
    
    # ファイルに書き戻し
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print('修正完了')
    print('修正されたcreate_message呼び出し数:', len(re.findall(r'create_message\(', content)))

if __name__ == "__main__":
    fix_create_message_calls()
