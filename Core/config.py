"""
設定ファイル処理モジュール
"""
import os
try:
    import yaml
except ImportError:
    print("警告: PyYAMLがインストールされていません。pip install PyYAMLを実行してください。")
    yaml = None

def get_config(config_path=None):
    """
    設定ファイルを読み込んで設定辞書を返す
    
    Args:
        config_path (str, optional): 設定ファイルのパス
        
    Returns:
        dict: 設定辞書
    """
    try:
        if config_path is None:
            # デフォルトの設定ファイルパスを使用
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        if yaml and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        else:
            if not yaml:
                print("PyYAMLがないため、デフォルト設定を使用します")
            else:
                print(f"設定ファイルが見つかりません: {config_path}")
            return get_default_config()
            
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        return get_default_config()

def get_default_config():
    """
    デフォルト設定を返す
    
    Returns:
        dict: デフォルト設定辞書
    """
    return {
        'system': {
            'max_workers': 4,
            'timeout': 30,
            'retry_count': 3
        },
        'model': {
            'name': 'default',
            'max_length': 512,
            'temperature': 0.7
        },
        'storage': {
            'cache_size': 1000,
            'persist': True
        }
    }