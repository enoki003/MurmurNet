import logging
import time
from typing import Any, Callable, Dict, Optional

class ErrorRecoveryManager:
    """エラー回復とシステム復旧を管理するクラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retry_attempts = {}
        self.max_retries = 3
        self.backoff_factor = 1.5
        
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """指数バックオフでリトライ実行"""
        func_name = func.__name__
        attempt = self.retry_attempts.get(func_name, 0)
        
        for i in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                # 成功した場合はリトライカウントをリセット
                self.retry_attempts[func_name] = 0
                return result
                
            except Exception as e:
                attempt += 1
                self.retry_attempts[func_name] = attempt
                
                if attempt >= self.max_retries:
                    self.logger.error(f"{func_name} が {self.max_retries} 回失敗しました: {e}")
                    raise
                
                wait_time = (self.backoff_factor ** attempt)
                self.logger.warning(f"{func_name} 失敗 (試行 {attempt}/{self.max_retries}), {wait_time:.1f}秒後にリトライ: {e}")
                time.sleep(wait_time)
        
        return None
    
    def safe_execute(self, func: Callable, default_return=None, *args, **kwargs) -> Any:
        """安全な実行（エラーでもシステムを停止しない）"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"{func.__name__} でエラーが発生しましたが処理を継続します: {e}")
            return default_return
    
    def check_system_health(self) -> Dict[str, bool]:
        """システムの健全性チェック"""
        health_status = {
            'model_pool': True,
            'agent_pool': True,
            'rag_retriever': True,
            'communication': True
        }
        
        # 各コンポーネントの健全性をチェック
        # 実装は具体的なコンポーネントに依存
        
        return health_status
