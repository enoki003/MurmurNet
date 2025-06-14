#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Pool モジュール
~~~~~~~~~~~~~~~~~~~
複数のモデルインスタンスをプール管理して真の並列処理を実現

作者: Yuhi Sonoki
"""

import logging
import queue
import threading
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from MurmurNet.modules.model_factory import LlamaModel
from MurmurNet.modules.config_manager import get_config

logger = logging.getLogger('MurmurNet.ModelPool')


class ModelPoolManager:
    """
    モデルプール管理クラス
    
    複数のモデルインスタンスを管理し、真の並列処理を実現
    各エージェントが独立したモデルインスタンスにアクセス可能
    """
    
    def __init__(self, pool_size: Optional[int] = None):
        """
        モデルプールの初期化
        
        引数:
            pool_size: プールサイズ（デフォルト：エージェント数と同じ）
        """
        config_manager = get_config()
        self.config = config_manager.to_dict()
        
        # プールサイズをエージェント数に合わせる
        self.pool_size = pool_size or config_manager.agent.num_agents
        
        # 利用可能なモデルインスタンスのキュー
        self.available_models = queue.Queue(maxsize=self.pool_size)
        
        # 統計情報
        self.total_requests = 0
        self.active_models = 0
        self.initialization_time = 0
        
        # 初期化用のロック
        self._init_lock = threading.Lock()
        self._initialized = False
        
        logger.info(f"モデルプールマネージャーを初期化しました (プールサイズ: {self.pool_size})")
    def _initialize_pool(self):
        """モデルプールの遅延初期化"""
        if self._initialized:
            return
            
        with self._init_lock:
            if self._initialized:
                return
                
            logger.info(f"モデルプールを初期化中... (サイズ: {self.pool_size})")
            start_time = time.time()
            
            successful_models = 0
            failed_models = 0
            
            # モデルインスタンスを作成してプールに追加
            for i in range(self.pool_size):
                try:
                    logger.info(f"モデルインスタンス {i+1}/{self.pool_size} を作成中...")
                    model = LlamaModel(self.config)
                    
                    # モデルの初期化を確実に実行
                    model._ensure_initialized()
                    
                    # 初期化の結果を詳細チェック
                    if model._llm is None:
                        logger.error(f"モデルインスタンス {i+1}: 内部モデル(_llm)の初期化に失敗しました")
                        if model._initialization_error:
                            logger.error(f"モデルインスタンス {i+1}: 初期化エラー詳細: {model._initialization_error}")
                        failed_models += 1
                        continue
                    
                    if not model.is_available():
                        logger.error(f"モデルインスタンス {i+1}: is_available()がFalseです")
                        failed_models += 1
                        continue
                    
                    # メソッドの存在確認
                    if not hasattr(model, 'create_chat_completion') and not hasattr(model, 'generate'):
                        logger.error(f"モデルインスタンス {i+1}: 必要なメソッドが実装されていません")
                        failed_models += 1
                        continue
                    
                    # 簡単なテスト実行
                    try:
                        test_prompt = "テスト"
                        if hasattr(model, 'create_chat_completion'):
                            test_resp = model.create_chat_completion(
                                messages=[{"role": "user", "content": test_prompt}],
                                max_tokens=5,
                                temperature=0.1
                            )
                            if not test_resp or 'choices' not in test_resp:
                                logger.error(f"モデルインスタンス {i+1}: テスト実行に失敗しました（無効なレスポンス）")
                                failed_models += 1
                                continue
                        elif hasattr(model, 'generate'):
                            test_resp = model.generate(test_prompt, max_tokens=5, temperature=0.1)
                            if not test_resp:
                                logger.error(f"モデルインスタンス {i+1}: テスト実行に失敗しました（空のレスポンス）")
                                failed_models += 1
                                continue
                        
                        logger.info(f"モデルインスタンス {i+1}: テスト実行に成功しました")
                    except Exception as test_e:
                        logger.error(f"モデルインスタンス {i+1}: テスト実行中のエラー: {test_e}")
                        failed_models += 1
                        continue
                    
                    # 全てのチェックをパスしたモデルをプールに追加
                    self.available_models.put(model)
                    successful_models += 1
                    logger.info(f"モデルインスタンス {i+1} を正常にプールに追加しました")
                        
                except Exception as e:
                    logger.error(f"モデルインスタンス {i+1} の作成エラー: {type(e).__name__}: {e}")
                    import traceback
                    logger.debug(f"モデルインスタンス {i+1} 作成エラー詳細:\n{traceback.format_exc()}")
                    failed_models += 1
            
            self.initialization_time = time.time() - start_time
            self._initialized = True
            actual_pool_size = self.available_models.qsize()
            
            if actual_pool_size == 0:
                logger.error(f"モデルプールの初期化に完全に失敗しました！利用可能なモデルがありません")
                logger.error(f"成功: {successful_models}, 失敗: {failed_models}")
                raise RuntimeError(f"モデルプールの初期化に失敗しました。{failed_models}個のモデルで初期化エラーが発生しました")
            elif successful_models < self.pool_size:
                logger.warning(f"モデルプール初期化部分成功: {successful_models}/{self.pool_size} インスタンス ({self.initialization_time:.2f}秒)")
                logger.warning(f"失敗したモデル数: {failed_models}")
            else:
                logger.info(f"モデルプール初期化完了: {actual_pool_size}/{self.pool_size} インスタンス ({self.initialization_time:.2f}秒)")
    
    @contextmanager
    def get_model(self, timeout: float = 30.0):
        """
        モデルインスタンスを取得（コンテキストマネージャー）
        
        引数:
            timeout: タイムアウト時間（秒）
            
        戻り値:
            モデルインスタンス
        """
        # 遅延初期化
        if not self._initialized:
            self._initialize_pool()
        
        model = None
        start_time = time.time()
        
        try:
            # 利用可能なモデルを取得（ブロッキング）
            model = self.available_models.get(timeout=timeout)
            self.active_models += 1
            self.total_requests += 1
            
            wait_time = time.time() - start_time
            if wait_time > 1.0:
                logger.warning(f"モデル取得に時間がかかりました: {wait_time:.2f}秒")
            
            logger.debug(f"モデルインスタンスを取得 (アクティブ: {self.active_models})")
            yield model
            
        except queue.Empty:
            logger.error(f"モデル取得タイムアウト ({timeout}秒)")
            raise TimeoutError(f"モデルプールからのインスタンス取得がタイムアウトしました ({timeout}秒)")
            
        finally:
            # モデルをプールに返却
            if model is not None:
                self.available_models.put(model)
                self.active_models -= 1
                logger.debug(f"モデルインスタンスを返却 (アクティブ: {self.active_models})")
    
    def get_stats(self) -> Dict[str, Any]:
        """プールの統計情報を取得"""
        return {
            'pool_size': self.pool_size,
            'available_models': self.available_models.qsize(),
            'active_models': self.active_models,
            'total_requests': self.total_requests,
            'initialization_time': self.initialization_time,
            'initialized': self._initialized
        }
    
    def shutdown(self):
        """プールをシャットダウン"""
        logger.info("モデルプールをシャットダウン中...")
        
        # 全てのモデルインスタンスをクリア
        while not self.available_models.empty():
            try:
                model = self.available_models.get_nowait()
                # モデルのクリーンアップ処理があれば実行
                del model
            except queue.Empty:
                break
        
        logger.info("モデルプールのシャットダウンが完了しました")


# グローバルモデルプールインスタンス
_model_pool = None
_pool_lock = threading.Lock()


def get_model_pool() -> ModelPoolManager:
    """グローバルモデルプールインスタンスを取得"""
    global _model_pool
    
    if _model_pool is None:
        with _pool_lock:
            if _model_pool is None:
                _model_pool = ModelPoolManager()
    
    return _model_pool


def reset_model_pool():
    """モデルプールをリセット（テスト用）"""
    global _model_pool
    
    with _pool_lock:
        if _model_pool is not None:
            _model_pool.shutdown()
            _model_pool = None


# 便利な関数
@contextmanager
def use_model(timeout: float = 30.0):
    """モデルインスタンスを使用（便利関数）"""
    pool = get_model_pool()
    with pool.get_model(timeout=timeout) as model:
        yield model
