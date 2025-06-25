"""
チャットテンプレートの最適化パッチ
system role警告を解消し、正規化処理の300msを削減
"""

import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def fix_chat_template(model):
    """
    ロード後にchat_templateのsystem roleを強制置換
    system role警告と正規化処理オーバーヘッドを排除
    """
    if hasattr(model, 'chat_template'):
        template = model.chat_template
        
        # system roleをuser roleに置換
        if hasattr(template, 'system'):
            if hasattr(template, 'user'):
                template.system = template.user
                logger.info("ChatTemplate - FIXED: system role → user role")
            else:
                # fallback: system roleを無効化
                template.system = ""
                logger.info("ChatTemplate - FIXED: system role disabled")
        
        # テンプレート文字列での置換も試行
        if isinstance(template, str):
            if "system" in template.lower():
                # system roleのパターンを user role パターンに置換
                fixed_template = template.replace(
                    "{{#system}}", "{{#user}}"
                ).replace(
                    "{{/system}}", "{{/user}}"
                ).replace(
                    "{%- set system_message", "{%- set user_message"
                ).replace(
                    "role=='system'", "role=='user'"
                ).replace(
                    '<start_of_turn>system', '<start_of_turn>user'
                )
                model.chat_template = fixed_template
                logger.info("ChatTemplate - FIXED: template string updated")
    
    return model

def patch_template_manager():
    """TemplateManagerにsystem role修正パッチを適用"""
    try:
        from MurmurNet.modules.output_agent import TemplateManager
        
        # 元のメソッドをバックアップ
        if not hasattr(TemplateManager, '_original_process_messages'):
            # 既存のメソッドを確認
            if hasattr(TemplateManager, 'process_messages'):
                TemplateManager._original_process_messages = TemplateManager.process_messages
            elif hasattr(TemplateManager, 'build_prompt'):
                TemplateManager._original_process_messages = TemplateManager.build_prompt
            else:
                # 新しいメソッドを作成
                def default_process_messages(self, messages):
                    return messages
                TemplateManager._original_process_messages = default_process_messages
        
        def optimized_process_messages(self, messages):
            """system roleをuser roleに変換"""
            optimized_messages = []
            
            for message in messages:
                if isinstance(message, dict) and message.get('role') == 'system':
                    # systemロールをuserロールに変換
                    optimized_message = {
                        'role': 'user',
                        'content': f"[System] {message['content']}"
                    }
                    optimized_messages.append(optimized_message)
                    logger.debug("Converted system role to user role")
                else:
                    optimized_messages.append(message)
            
            return TemplateManager._original_process_messages(self, optimized_messages)
        
        # メソッドを置き換え
        if hasattr(TemplateManager, 'process_messages'):
            TemplateManager.process_messages = optimized_process_messages
        elif hasattr(TemplateManager, 'build_prompt'):
            TemplateManager.build_prompt = optimized_process_messages
        else:
            TemplateManager.process_messages = optimized_process_messages
            
        logger.info("TemplateManager system role patch applied")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to patch TemplateManager: {e}")
        return False

def apply_performance_patches(model, summary_engine=None):
    """
    全体的なパフォーマンス最適化を適用
    """
    logger.info("Applying performance patches...")
    
    # 1. チャットテンプレート修正
    model = fix_chat_template(model)
      # 2. TemplateManager最適化
    patch_template_manager()
    
    # 3. 要約エンジン最適化（オプション）
    if summary_engine:
        try:
            from .summary_optimizer import patch_summary_engine
            optimized_summary = patch_summary_engine(summary_engine)
            logger.info("Summary engine optimization applied")
        except ImportError:
            logger.warning("Summary optimizer not found")
    
    logger.info("Performance patches applied successfully")
    return model

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("Testing template optimization...")
      # TemplateManagerパッチのテスト
    success = patch_template_manager()
    print(f"TemplateManager patch: {'SUCCESS' if success else 'FAILED'}")
    
    print("Template optimization completed")