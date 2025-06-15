"""
MurmurNet Database Module
データベース関連の機能
"""

try:
    from .zim_reader import ZIMReader
    __all__ = ['ZIMReader']
except ImportError:
    __all__ = []
