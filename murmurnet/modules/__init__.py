"""
MurmurNet modules package

分散SLMシステムのコアモジュール群
"""

from .shutdown_manager import (
    ShutdownManager,
    get_shutdown_manager,
    register_for_shutdown,
    register_shutdown_callback,
    shutdown_system,
    is_shutdown_requested,
    setup_signal_handlers
)

__all__ = [
    'ShutdownManager',
    'get_shutdown_manager', 
    'register_for_shutdown',
    'register_shutdown_callback',
    'shutdown_system',
    'is_shutdown_requested',
    'setup_signal_handlers'
]
