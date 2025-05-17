#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Manager モジュール
~~~~~~~~~~~~~~~~~~~~~
モデルとリソースを共有管理するための中央モジュール
- 埋め込みモデル
- ZIMアーカイブ
- その他の共有リソース

作者: Yuhi Sonoki
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# モデル依存関係
# ───────────────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    SentenceTransformer = None
    logger.warning("sentence-transformers not found: %s", e)

try:
    from libzim.reader import Archive
    from libzim.search import Searcher, Query
    HAS_LIBZIM = True
except ImportError:
    HAS_LIBZIM = False
    logger.warning("libzim not found, ZIM機能は無効です")

# ───────────────────────────────────────────────────────────────────────────
# 共有モデル管理クラス
# ───────────────────────────────────────────────────────────────────────────

class ModelManager:
    """モデルとリソースの共有管理クラス"""
    
    # クラス変数 - 共有リソース
    _embedding_models: Dict[str, Any] = {}
    _zim_archives: Dict[str, Any] = {}
    _is_loading: bool = False
    _lock = None
    
    @classmethod
    def _get_lock(cls):
        """スレッドセーフなロックを取得"""
        if cls._lock is None:
            import threading
            cls._lock = threading.RLock()
        return cls._lock
    
    @classmethod
    def get_embedding_model(cls, model_name: str, debug: bool = False) -> Optional[Any]:
        """
        埋め込みモデルを取得（なければロード）
        
        引数:
            model_name: モデル名
            debug: デバッグモード
            
        戻り値:
            ロードされたモデル（失敗時はNone）
        """
        # クラスレベルのロックを取得
        with cls._get_lock():
            # 既にロード済みの場合はそれを返す
            if model_name in cls._embedding_models:
                if debug:
                    logger.debug(f"既存の埋め込みモデルを使用: {model_name}")
                return cls._embedding_models[model_name]
                
            # モジュールが利用できない場合
            if SentenceTransformer is None:
                logger.warning("sentence-transformers がインストールされていません")
                return None
                
            # 多重ロード防止
            if cls._is_loading:
                if debug:
                    logger.debug("別のモデルをロード中のため待機")
                return None
                
            # ロード処理
            cls._is_loading = True
            
        # ロックの外でモデルをロード（時間のかかる処理）
        start_time = time.time()
        model = None
        
        try:
            model = SentenceTransformer(model_name)
            if debug:
                elapsed = time.time() - start_time
                logger.debug(f"埋め込みモデルをロードしました: {model_name} ({elapsed:.2f}秒)")
                
        except Exception as e:
            logger.error(f"埋め込みモデルロードエラー: {e}")
            model = None
            
        finally:
            # ロックを再取得してモデルを保存
            with cls._get_lock():
                if model is not None:
                    cls._embedding_models[model_name] = model
                cls._is_loading = False
                
        return model
    
    @classmethod
    def get_zim_archive(cls, path: str, debug: bool = False) -> Optional[Any]:
        """
        ZIMアーカイブを取得（なければロード）
        
        引数:
            path: ZIMファイルへのパス
            debug: デバッグモード
            
        戻り値:
            ロードされたZIMアーカイブ（失敗時はNone）
        """
        # クラスレベルのロックを取得
        with cls._get_lock():
            # 既にロード済みの場合はそれを返す
            if path in cls._zim_archives:
                if debug:
                    logger.debug(f"既存のZIMアーカイブを使用: {path}")
                return cls._zim_archives[path]
                
            # モジュールが利用できない場合
            if not HAS_LIBZIM:
                logger.warning("libzim がインストールされていません")
                return None
                
            # ファイルが存在しない場合
            if not os.path.exists(path):
                logger.error(f"ZIMファイルが見つかりません: {path}")
                return None
                
            # 多重ロード防止
            if cls._is_loading:
                if debug:
                    logger.debug("別のアーカイブをロード中のため待機")
                return None
                
            # ロード状態をセット
            cls._is_loading = True
        
        # ロックの外でアーカイブをロード（時間のかかる処理）
        start_time = time.time()
        archive = None
        
        try:
            archive = Archive(path)
            if debug:
                elapsed = time.time() - start_time
                logger.debug(f"ZIMアーカイブをロードしました: {path} ({elapsed:.2f}秒)")
                
        except Exception as e:
            logger.error(f"ZIMアーカイブロードエラー: {e}")
            archive = None
            
        finally:
            # ロックを再取得してアーカイブを保存
            with cls._get_lock():
                if archive is not None:
                    cls._zim_archives[path] = archive
                cls._is_loading = False
                
        return archive
    
    @classmethod
    def get_model_status(cls) -> Dict[str, Any]:
        """
        ロード済みモデルの状態を取得
        
        戻り値:
            Dict: ロード済みモデルの情報
        """
        return {
            'embedding_models': list(cls._embedding_models.keys()),
            'zim_archives': list(cls._zim_archives.keys()),
            'total_models': len(cls._embedding_models) + len(cls._zim_archives)
        }
