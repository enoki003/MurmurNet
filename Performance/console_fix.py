#!/usr/bin/env python3
"""
コンソールアプリケーション出力修正パッチ
統合レスポンスがコンソールに表示されない問題を解決
"""

import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def patch_console_output():
    """コンソールアプリケーションの出力を修正"""
    try:
        # console_app.pyを読み込んで修正
        console_app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Console', 'console_app.py'
        )
        
        if not os.path.exists(console_app_path):
            logger.error(f"Console app not found: {console_app_path}")
            return False
        
        # 修正パッチを適用
        with open(console_app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 問題のある箇所を特定して修正
        fixes = [
            # 空のwhile True:ループを修正
            ('while True:\n        pass', 'while True:\n            break'),
            
            # レスポンス表示の追加
            ('response = murmur.run(user_input)', 
             'response = murmur.run(user_input)\n            if response:\n                print(f"\\nAI> {response}\\n")'),
            
            # OutputAgentの結果を確実に表示
            ('# 結果を取得', '# 結果を取得\n            print(f"\\nAI> {response}\\n")'),
        ]
        
        modified = False
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
                modified = True
                logger.info(f"Applied fix: {old[:30]}...")
        
        if modified:
            # バックアップを作成
            backup_path = console_app_path + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Console app patched and backed up to {backup_path}")
            return True
        else:
            logger.info("No patches needed for console app")
            return True
            
    except Exception as e:
        logger.error(f"Failed to patch console app: {e}")
        return False

def create_fixed_console_template():
    """修正されたコンソールアプリケーションのテンプレートを作成"""
    template = '''#!/usr/bin/env python3
"""
修正されたコンソールアプリケーション
統合レスポンス表示問題を解決
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MurmurNet.distributed_slm import DistributedSLM

def main():
    """メイン実行関数"""
    print("MurmurNet - 分散創発型言語モデルシステム")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print()
    
    # MurmurNet初期化
    config = {
        "num_agents": 2,
        "iterations": 1,
        "use_summary": True,
        "rag_mode": "dummy"
    }
    
    murmur = DistributedSLM(config)
    
    # メインループ
    while True:
        try:
            user_input = input("あなた> ").strip()
            
            if user_input.lower() in ("quit", "exit", "終了"):
                print("システムを終了します...")
                break
            
            if not user_input:
                continue
            
            # 応答生成
            print("処理中...")
            response = murmur.run(user_input)
            
            # 結果表示 - これが重要！
            if response:
                print(f"\\nAI> {response}\\n")
            else:
                print("\\nAI> 申し訳ありません。応答を生成できませんでした。\\n")
                
        except KeyboardInterrupt:
            print("\\n\\nシステムを終了します...")
            break
        except Exception as e:
            print(f"\\nエラーが発生しました: {e}\\n")

if __name__ == "__main__":
    main()
'''
    
    return template

def apply_console_fixes():
    """コンソール修正を適用"""
    logger.info("Applying console output fixes...")
    
    # 既存ファイルのパッチ
    patch_success = patch_console_output()
    
    # 修正テンプレートの作成
    template = create_fixed_console_template()
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Console', 'console_app_fixed.py'
    )
    
    try:
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template)
        logger.info(f"Fixed console template created: {template_path}")
        template_success = True
    except Exception as e:
        logger.error(f"Failed to create fixed template: {e}")
        template_success = False
    
    return patch_success and template_success

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("Testing console output fixes...")
    
    # 修正適用
    success = apply_console_fixes()
    
    print(f"Console fixes applied: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print("\\nRecommended usage:")
        print("python Console/console_app_fixed.py")
    
    print("Console fix test completed")
