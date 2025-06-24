"""
シャットダウン管理モジュール

MurmurNetシステムの統合的なシャットダウン管理を提供します。
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional, Callable, List
from threading import Event, Lock
import weakref

logger = logging.getLogger(__name__)

class ShutdownManager:   
    """
    分散SLMシステムの統合シャットダウン管理クラス
    """
    
    def __init__(self):
        self.shutdown_event = Event()
        self.shutdown_callbacks: List[Callable] = []
        self.shutdown_lock = Lock()
        self.is_shutting_down = False
        self.registered_components: Dict[str, Dict[str, Any]] = {}  # {name: {component: weakref, priority: int}}
        
    def register_component(self, name: str, component: Any, priority: int = 50):
        """
        シャットダウン対象コンポーネントを登録
        
        Args:
            name: コンポーネント名
            component: コンポーネントオブジェクト
            priority: シャットダウン優先度（数値が大きいほど先に実行）
        """
        with self.shutdown_lock:
            self.registered_components[name] = {
                'component': weakref.ref(component),
                'priority': priority
            }
            logger.info(f"Registered component for shutdown: {name} (priority: {priority})")
    
    def register_callback(self, callback: Callable):
        """
        シャットダウン時のコールバック関数を登録
        
        Args:
            callback: シャットダウン時に実行する関数
        """
        with self.shutdown_lock:
            self.shutdown_callbacks.append(callback)
            logger.info(f"Registered shutdown callback: {callback.__name__}")
    
    def is_shutdown_requested(self) -> bool:
        """
        シャットダウンが要求されているかチェック
        
        Returns:
            bool: シャットダウンが要求されている場合True
        """
        return self.shutdown_event.is_set()
    
    async def shutdown_system(self, reason: str = "Manual shutdown"):
        """
        システム全体のシャットダウンを実行
        
        Args:
            reason: シャットダウンの理由
        """
        with self.shutdown_lock:
            if self.is_shutting_down:
                logger.warning("Shutdown already in progress")
                return
            self.is_shutting_down = True
        
        logger.info(f"Starting system shutdown: {reason}")
          # シャットダウンイベントを設定
        self.shutdown_event.set()
        
        try:
            # 登録されたコンポーネントを優先度順でシャットダウン
            sorted_components = sorted(
                self.registered_components.items(),
                key=lambda x: x[1]['priority'],
                reverse=True  # 高い優先度から実行
            )
            
            for name, component_info in sorted_components:
                component = component_info['component']()
                if component is not None:
                    try:
                        if hasattr(component, 'shutdown'):
                            if asyncio.iscoroutinefunction(component.shutdown):
                                await component.shutdown()
                            else:
                                component.shutdown()
                            logger.info(f"Shutdown completed for component: {name} (priority: {component_info['priority']})")
                    except Exception as e:
                        logger.error(f"Error shutting down component {name}: {e}")
            
            # 登録されたコールバック関数の実行
            for callback in self.shutdown_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                    logger.info(f"Executed shutdown callback: {callback.__name__}")
                except Exception as e:
                    logger.error(f"Error executing shutdown callback {callback.__name__}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
        finally:
            logger.info("System shutdown completed")

# グローバルシャットダウンマネージャーインスタンス
_shutdown_manager: Optional[ShutdownManager] = None
_shutdown_manager_lock = Lock()

def get_shutdown_manager() -> ShutdownManager:
    """
    グローバルShutdownManagerインスタンスを取得（シングルトン）
    
    Returns:
        ShutdownManager: グローバルインスタンス
    """
    global _shutdown_manager
    
    if _shutdown_manager is None:
        with _shutdown_manager_lock:
            if _shutdown_manager is None:
                _shutdown_manager = ShutdownManager()
                logger.info("Created global shutdown manager")
    
    return _shutdown_manager

def register_for_shutdown(component: Any, name: str = None, priority: int = 50):
    """
    コンポーネントをシャットダウン対象として登録
    
    Args:
        component: 登録するコンポーネント
        name: コンポーネント名（省略時はクラス名を使用）
        priority: シャットダウン優先度（数値が大きいほど先に実行、デフォルト50）
    """
    if name is None:
        name = component.__class__.__name__
    
    manager = get_shutdown_manager()
    manager.register_component(name, component, priority)

def register_shutdown_callback(callback: Callable):
    """
    シャットダウン時のコールバック関数を登録
    
    Args:
        callback: シャットダウン時に実行する関数
    """
    manager = get_shutdown_manager()
    manager.register_callback(callback)

async def shutdown_system(reason: str = "Manual shutdown"):
    """
    システム全体のシャットダウンを実行
    
    Args:
        reason: シャットダウンの理由
    """
    manager = get_shutdown_manager()
    await manager.shutdown_system(reason)

def is_shutdown_requested() -> bool:
    """
    シャットダウンが要求されているかチェック
    
    Returns:
        bool: シャットダウンが要求されている場合True
    """
    manager = get_shutdown_manager()
    return manager.is_shutdown_requested()

def setup_signal_handlers():
    """
    シグナルハンドリングを設定
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # イベントループが存在しない場合は新しく作成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # ループが実行中の場合はタスクとして追加
            loop.create_task(shutdown_system(f"Signal {signum}"))
        else:
            # ループが停止中の場合は実行
            loop.run_until_complete(shutdown_system(f"Signal {signum}"))
    
    # Windows環境でも利用可能なシグナルのみ設定
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers set up successfully")
    except Exception as e:
        logger.warning(f"Failed to set up signal handlers: {e}")

# モジュール初期化時にシグナルハンドラーを設定
if __name__ != "__main__":
    setup_signal_handlers()
