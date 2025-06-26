#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llama-cpp-python単体動作確認スクリプト
"""

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python import成功")
    
    model_path = r"C:\Users\admin\Desktop\課題研究\models\gemma-3-1b-it-q4_0.gguf"
    print(f"📁 モデルパス: {model_path}")
    
    m = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    print("✅ Llamaモデル初期化成功")
    
    response = m.create_chat_completion(
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=4
    )
    print("✅ create_chat_completion成功")
    print("📤 レスポンス:", response["choices"][0]["message"]["content"])
    
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
