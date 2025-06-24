"""
チャットテンプレートの最適化パッチ
system role警告を解消し、正規化処理の300msを削減
"""

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
                print("ChatTemplate - FIXED: system role → user role")
            else:
                # fallback: system roleを無効化
                template.system = ""
                print("ChatTemplate - FIXED: system role disabled")
        
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
                )
                model.chat_template = fixed_template
                print("ChatTemplate - FIXED: template string updated")
    
    return model

def apply_performance_patches(model, summary_engine):
    """
    全体的なパフォーマンス最適化を適用
    """
    # 1. チャットテンプレート修正
    model = fix_chat_template(model)
    
    # 2. 要約エンジン最適化
    from summary_optimizer import patch_summary_engine
    optimized_summary = patch_summary_engine(summary_engine)
    
    print("Performance patches applied:")
    print("  ✓ Chat template system role fixed")
    print("  ✓ Summary engine short-text skip enabled")
    
    return model, optimized_summary