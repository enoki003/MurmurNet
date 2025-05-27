# -*- coding: utf-8 -*-
"""
MurmurNet - 分散創発型言語モデルシステム
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

分散SLM (Small Language Model) アーキテクチャを実装したパッケージ

作者: Yuhi Sonoki
"""

__version__ = "1.0.0"
__author__ = "Yuhi Sonoki"

from .distributed_slm import DistributedSLM

__all__ = ["DistributedSLM"]
