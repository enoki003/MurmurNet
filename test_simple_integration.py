#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡単な統合テスト
~~~~~~~~~~~~~~~~
DistributedSLMの基本的な動作を確認するテスト

作者: Yuhi Sonoki
"""

import sys
import os
from pathlib import Path
import pytest

# プロジェクトのパスを追加
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

def test_import_distributed_slm():
    """DistributedSLMのインポートテスト"""
    try:
        from MurmurNet.distributed_slm import DistributedSLM
        assert DistributedSLM is not None
        print("✅ DistributedSLMのインポートに成功")
    except ImportError as e:
        pytest.fail(f"DistributedSLMのインポートに失敗: {e}")

def test_create_instance():
    """DistributedSLMインスタンスの作成テスト"""
    try:
        from MurmurNet.distributed_slm import DistributedSLM
        
        # 最小限の設定でインスタンス作成
        config = {
            "model_path": "dummy_path",  # 実際のモデルファイルは不要
            "num_agents": 1,
            "iterations": 1,
            "use_summary": False,
            "use_parallel": False,
            "use_memory": False
        }
        
        slm = DistributedSLM(config)
        assert slm is not None
        assert slm.num_agents == 1
        assert slm.iterations == 1
        print("✅ DistributedSLMインスタンスの作成に成功")
        
    except Exception as e:
        pytest.fail(f"DistributedSLMインスタンスの作成に失敗: {e}")

def test_get_stats():
    """統計情報取得テスト"""
    try:
        from MurmurNet.distributed_slm import DistributedSLM
        
        config = {
            "model_path": "dummy_path",
            "num_agents": 2,
            "iterations": 1,
            "use_summary": True,
            "use_parallel": False,
            "use_memory": True
        }
        
        slm = DistributedSLM(config)
        stats = slm.get_stats()
        
        assert isinstance(stats, dict)
        assert "agents" in stats
        assert "iterations" in stats
        assert stats["agents"] == 2
        assert stats["iterations"] == 1
        print("✅ 統計情報の取得に成功")
        
    except Exception as e:
        pytest.fail(f"統計情報の取得に失敗: {e}")

def test_system_status():
    """システム状態取得テスト"""
    try:
        from MurmurNet.distributed_slm import DistributedSLM
        
        config = {
            "model_path": "dummy_path",
            "num_agents": 1,
            "iterations": 1
        }
        
        slm = DistributedSLM(config)
        status = slm.get_system_status()
        
        assert isinstance(status, dict)
        assert "compatibility_mode" in status
        assert "modules_initialized" in status
        assert "agent_config" in status
        print("✅ システム状態の取得に成功")
        
    except Exception as e:
        pytest.fail(f"システム状態の取得に失敗: {e}")

def test_communication_stats():
    """通信統計取得テスト"""
    try:
        from MurmurNet.distributed_slm import DistributedSLM
        
        config = {
            "model_path": "dummy_path",
            "num_agents": 1,
            "iterations": 1
        }
        
        slm = DistributedSLM(config)
        comm_stats = slm.get_communication_stats()
        
        assert isinstance(comm_stats, dict)
        print("✅ 通信統計の取得に成功")
        
    except Exception as e:
        pytest.fail(f"通信統計の取得に失敗: {e}")

if __name__ == "__main__":
    # 直接実行時は各テストを順番に実行
    print("=== 簡単な統合テスト開始 ===")
    
    test_import_distributed_slm()
    test_create_instance()
    test_get_stats()
    test_system_status()
    test_communication_stats()
    
    print("=== すべてのテストが完了しました ===")
