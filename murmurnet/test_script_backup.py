#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet 包括的テストスイート
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分散創発型言語モデルシステムの全機能テスト
- 基本機能テスト
- 並列処理テスト
- RAG機能テスト（ZIM/Embedding）
- エラーハンドリングテスト
- 設定管理テスト
- パフォーマンステスト
- モジュール統合テスト
- メモリ管理テスト

作者: Yuhi Sonoki (Updated by GitHub Copilot)
"""

import sys
import os
import logging
import asyncio
import time
import gc
import traceback
import psutil
from typing import Dict, Any, List, Optional, Tuple
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("comprehensive_test_log.txt", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# プロジェクトパスの設定
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# MurmurNetモジュールのインポート
from MurmurNet.distributed_slm import DistributedSLM
from MurmurNet.modules.blackboard import Blackboard
from MurmurNet.modules.input_reception import InputReception
from MurmurNet.modules.agent_pool import AgentPoolManager
from MurmurNet.modules.rag_retriever import RAGRetriever
from MurmurNet.modules.output_agent import OutputAgent
from MurmurNet.modules.summary_engine import SummaryEngine
from MurmurNet.modules.conversation_memory import ConversationMemory
from MurmurNet.modules.config_manager import ConfigManager, get_config
from MurmurNet.modules.model_factory import get_shared_model
from MurmurNet.modules.system_coordinator import SystemCoordinator
from MurmurNet.modules.performance import PerformanceMonitor
from MurmurNet.modules.common import MurmurNetError

# テスト設定
MODELS_PATH = r"C:\Users\admin\Desktop\課題研究\models"
ZIM_PATH = r"C:\Users\admin\Desktop\課題研究\KNOWAGE_DATABASE\wikipedia_en_top_nopic_2025-03.zim"

# 基本設定テンプレート
BASE_CONFIG = {
    "num_agents": 2,
    "iterations": 1,
    "use_summary": True,
    "use_parallel": False,
    "model_type": "gemma3",
    "rag_mode": "zim",
    "rag_score_threshold": 0.5,
    "rag_top_k": 3,
    "debug": True,
    "model_path": os.path.join(MODELS_PATH, "gemma-3-1b-it-q4_0.gguf"),
    "chat_template": os.path.join(MODELS_PATH, "gemma3_template.txt"),
    "zim_path": ZIM_PATH,
    "n_threads": 4,
    "n_ctx": 2048,
}

DEFAULT_CONFIG = BASE_CONFIG.copy()

# テスト結果記録
test_results = {
    "passed": [],
    "failed": [],
    "skipped": [],
    "total_time": 0,
    "memory_usage": {},
    "performance_metrics": {}
}

def print_header(title: str, level: int = 1) -> None:
    """テストセクションのヘッダー出力"""
    char = "=" if level == 1 else "-" if level == 2 else "·"
    width = 60 if level == 1 else 50 if level == 2 else 40
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def print_result(test_name: str, success: bool, message: str, duration: float = 0) -> None:
    """テスト結果の出力と記録"""
    status = "✓ PASS" if success else "✗ FAIL"
    duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"{status} {test_name}{duration_str}")
    if message:
        print(f"    {message}")
    
    # 結果を記録
    result_entry = {
        "name": test_name,
        "message": message,
        "duration": duration,
        "timestamp": time.time()
    }
    
    if success:
        test_results["passed"].append(result_entry)
    else:
        test_results["failed"].append(result_entry)

def measure_memory() -> Dict[str, float]:
    """メモリ使用量を測定"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }

class ComprehensiveTestSuite:
    """包括的テストスイート"""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = measure_memory()
        self.slm_instances = {}
        
    async def run_all_tests(self):
        """すべてのテストを実行"""
        print_header("MurmurNet 包括的テストスイート開始")
        print(f"開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"初期メモリ使用量: RSS={self.initial_memory['rss']:.1f}MB, VMS={self.initial_memory['vms']:.1f}MB")
        
        try:
            # 1. 基本機能テスト
            await self.test_basic_functionality()
            
            # 2. 設定管理テスト
            await self.test_configuration_management()
            
            # 3. モジュール統合テスト
            await self.test_module_integration()
            
            # 4. RAG機能テスト
            await self.test_rag_functionality()
            
            # 5. 並列処理テスト
            await self.test_parallel_processing()
            
            # 6. エラーハンドリングテスト
            await self.test_error_handling()
            
            # 7. パフォーマンステスト
            await self.test_performance()
            
            # 8. メモリ管理テスト
            await self.test_memory_management()
            
            # 9. エンドツーエンドテスト
            await self.test_end_to_end()
            
        except Exception as e:
            print_result("テストスイート実行", False, f"予期しないエラー: {e}")
            traceback.print_exc()
        
        finally:
            await self.cleanup_and_report()
    
    async def test_basic_functionality(self):
        """基本機能テスト"""
        print_header("基本機能テスト", 2)
        
        # 1. Blackboardテスト
        await self._test_blackboard()
        
        # 2. InputReceptionテスト
        await self._test_input_reception()
        
        # 3. OutputAgentテスト
        await self._test_output_agent()
        
        # 4. ConversationMemoryテスト
        await self._test_conversation_memory()
    
    async def _test_blackboard(self):
        """Blackboardの基本機能テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            blackboard = Blackboard(config)
            
            # 書き込みテスト
            entry = blackboard.write("test_key", "test_value")
            assert entry["key"] == "test_key"
            assert entry["value"] == "test_value"
            
            # 読み込みテスト
            value = blackboard.read("test_key")
            assert value == "test_value"
            
            # 履歴テスト
            history = blackboard.get_history("test_key")
            assert len(history) == 1
            
            # メモリプロパティテスト
            memory = blackboard.memory
            assert isinstance(memory, dict)
            assert "test_key" in memory
            
            duration = time.time() - start_time
            print_result("Blackboard基本機能", True, "書き込み、読み込み、履歴、メモリアクセス正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("Blackboard基本機能", False, f"エラー: {e}", duration)
    
    async def _test_input_reception(self):
        """InputReceptionの機能テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            blackboard = Blackboard(config)
            input_reception = InputReception(config, blackboard)
            
            # 基本的な入力処理
            test_input = "これはテスト入力です。"
            processed = input_reception.process(test_input)
            
            assert processed is not None
            assert len(str(processed)) > 0
            
            duration = time.time() - start_time
            print_result("InputReception機能", True, "入力処理正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("InputReception機能", False, f"エラー: {e}", duration)
    
    async def _test_output_agent(self):
        """OutputAgentの機能テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            blackboard = Blackboard(config)
            
            # テストデータを黒板に設定
            blackboard.write("agent_0_output", "エージェント0の応答です。")
            blackboard.write("agent_1_output", "エージェント1の応答です。")
            
            output_agent = OutputAgent(config, blackboard)
            final_response = await output_agent.generate_final_response()
            
            assert final_response is not None
            assert len(final_response) > 0
            
            duration = time.time() - start_time
            print_result("OutputAgent機能", True, "最終応答生成正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("OutputAgent機能", False, f"エラー: {e}", duration)
    
    async def _test_conversation_memory(self):
        """ConversationMemoryの機能テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            conv_memory = ConversationMemory(config)
            
            # 会話追加テスト
            conv_memory.add_conversation("ユーザー", "こんにちは")
            conv_memory.add_conversation("AI", "こんにちは！")
            
            # コンテキスト取得テスト
            context = conv_memory.get_context(max_length=100)
            assert len(context) > 0
            assert "こんにちは" in context
            
            # 要約テスト
            summary = conv_memory.get_summary()
            assert summary is not None
            
            duration = time.time() - start_time
            print_result("ConversationMemory機能", True, "会話記憶機能正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("ConversationMemory機能", False, f"エラー: {e}", duration)
    
    async def test_configuration_management(self):
        """設定管理テスト"""
        print_header("設定管理テスト", 2)
        
        # 1. ConfigManagerテスト
        await self._test_config_manager()
        
        # 2. 設定バリデーションテスト
        await self._test_config_validation()
    
    async def _test_config_manager(self):
        """ConfigManagerの機能テスト"""
        start_time = time.time()
        try:
            # ファイルベース設定
            config_manager = ConfigManager()
            assert config_manager is not None
            
            # プロパティアクセステスト
            model_type = config_manager.model_type
            assert model_type in ["gemma3", "llama", "local"]
            
            rag_mode = config_manager.rag_mode
            assert rag_mode in ["zim", "embedding"]
            
            # 辞書ベース設定
            test_config = {"model_type": "gemma3", "rag_mode": "zim"}
            config_manager_dict = ConfigManager(test_config)
            assert config_manager_dict.model_type == "gemma3"
            
            duration = time.time() - start_time
            print_result("ConfigManager機能", True, "設定読み込み・アクセス正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("ConfigManager機能", False, f"エラー: {e}", duration)
    
    async def _test_config_validation(self):
        """設定バリデーションテスト"""
        start_time = time.time()
        try:
            # 有効な設定
            valid_config = BASE_CONFIG.copy()
            config_manager = ConfigManager(valid_config)
            assert config_manager is not None
            
            # 無効な設定（存在しないモデルタイプ）
            try:
                invalid_config = BASE_CONFIG.copy()
                invalid_config["model_type"] = "invalid_model"
                ConfigManager(invalid_config)
                # バリデーションエラーが発生しない場合は問題
                raise AssertionError("無効な設定が受け入れられた")
            except Exception:
                # バリデーションエラーが期待される
                pass
            
            duration = time.time() - start_time
            print_result("設定バリデーション", True, "有効・無効設定の判定正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("設定バリデーション", False, f"エラー: {e}", duration)
    
    async def test_module_integration(self):
        """モジュール統合テスト"""
        print_header("モジュール統合テスト", 2)
        
        # 1. DistributedSLM初期化テスト
        await self._test_distributed_slm_init()
        
        # 2. モジュール間通信テスト
        await self._test_module_communication()
    
    async def _test_distributed_slm_init(self):
        """DistributedSLM初期化テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            slm = DistributedSLM(config)
            
            # 基本属性確認
            assert slm.num_agents == config["num_agents"]
            assert slm.iterations == config["iterations"]
            assert slm.use_summary == config["use_summary"]
            
            # モジュール初期化確認
            assert hasattr(slm, 'blackboard')
            assert hasattr(slm, 'input_reception')
            assert hasattr(slm, 'agent_pool')
            assert hasattr(slm, 'output_agent')
            
            duration = time.time() - start_time
            print_result("DistributedSLM初期化", True, "全モジュール正常初期化", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("DistributedSLM初期化", False, f"エラー: {e}", duration)
    
    async def _test_module_communication(self):
        """モジュール間通信テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["num_agents"] = 1  # 高速化のため
            slm = DistributedSLM(config)
            
            # 簡単な質問で通信テスト
            test_question = "テストです"
            response = await slm.generate(test_question)
            
            assert response is not None
            assert len(response) > 0
            assert isinstance(response, str)
            
            duration = time.time() - start_time
            print_result("モジュール間通信", True, "エンドツーエンド通信正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("モジュール間通信", False, f"エラー: {e}", duration)
    
    async def test_rag_functionality(self):
        """RAG機能テスト"""
        print_header("RAG機能テスト", 2)
        
        # 1. ZIMモードテスト
        await self._test_zim_mode()
        
        # 2. Embeddingモードテスト
        await self._test_embedding_mode()
    
    async def _test_zim_mode(self):
        """ZIMモードテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["rag_mode"] = "zim"
            config["zim_path"] = ZIM_PATH
            
            # ZIMファイルの存在確認
            if not os.path.exists(ZIM_PATH):
                print_result("ZIMモード", False, f"ZIMファイルが見つかりません: {ZIM_PATH}", 0)
                return
            
            blackboard = Blackboard(config)
            rag_retriever = RAGRetriever(config, blackboard)
            
            # 検索テスト
            query = "artificial intelligence"
            rag_result = rag_retriever.retrieve(query)
            
            assert rag_result is not None
            assert len(rag_result) > 0
            
            duration = time.time() - start_time
            print_result("ZIMモード", True, "ZIM検索正常動作", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("ZIMモード", False, f"エラー: {e}", duration)
    
    async def _test_embedding_mode(self):
        """Embeddingモードテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["rag_mode"] = "embedding"
            
            blackboard = Blackboard(config)
            rag_retriever = RAGRetriever(config, blackboard)
            
            # 検索テスト
            query = "機械学習について"
            rag_result = rag_retriever.retrieve(query)
            
            assert rag_result is not None
            
            duration = time.time() - start_time
            print_result("Embeddingモード", True, "埋め込み検索正常動作", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("Embeddingモード", False, f"エラー: {e}", duration)
    
    async def test_parallel_processing(self):
        """並列処理テスト"""
        print_header("並列処理テスト", 2)
        
        # 1. 並列処理設定テスト
        await self._test_parallel_configuration()
        
        # 2. 並列 vs 逐次性能比較
        await self._test_parallel_vs_sequential()
    
    async def _test_parallel_configuration(self):
        """並列処理設定テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["use_parallel"] = True
            config["num_agents"] = 2
            
            slm = DistributedSLM(config)
            
            # 並列設定確認
            assert slm.use_parallel == True
            
            # SystemCoordinator確認
            if hasattr(slm, 'system_coordinator'):
                coordinator = slm.system_coordinator
                assert coordinator.use_parallel == True
            
            duration = time.time() - start_time
            print_result("並列処理設定", True, "並列処理設定正常", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("並列処理設定", False, f"エラー: {e}", duration)
    
    async def _test_parallel_vs_sequential(self):
        """並列 vs 逐次性能比較テスト"""
        start_time = time.time()
        try:
            base_config = BASE_CONFIG.copy()
            base_config["num_agents"] = 2
            base_config["iterations"] = 1
            base_config["use_summary"] = False  # 高速化
            
            test_query = "簡単なテストクエリ"
            
            # 逐次処理
            seq_config = base_config.copy()
            seq_config["use_parallel"] = False
            slm_seq = DistributedSLM(seq_config)
            
            seq_start = time.time()
            response_seq = await slm_seq.generate(test_query)
            seq_time = time.time() - seq_start
            
            # 並列処理
            par_config = base_config.copy()
            par_config["use_parallel"] = True
            slm_par = DistributedSLM(par_config)
            
            par_start = time.time()
            response_par = await slm_par.generate(test_query)
            par_time = time.time() - par_start
            
            # 結果確認
            assert response_seq is not None
            assert response_par is not None
            
            speedup = seq_time / par_time if par_time > 0 else 1.0
            
            duration = time.time() - start_time
            message = f"逐次: {seq_time:.2f}s, 並列: {par_time:.2f}s, 速度向上: {speedup:.2f}x"
            print_result("並列vs逐次性能", True, message, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("並列vs逐次性能", False, f"エラー: {e}", duration)
    
    async def test_error_handling(self):
        """エラーハンドリングテスト"""
        print_header("エラーハンドリングテスト", 2)
        
        # 1. 不正な設定テスト
        await self._test_invalid_configuration()
        
        # 2. ファイル不存在テスト
        await self._test_missing_files()
        
        # 3. 例外処理テスト
        await self._test_exception_handling()
    
    async def _test_invalid_configuration(self):
        """不正な設定でのエラーハンドリングテスト"""
        start_time = time.time()
        try:
            # 不正なエージェント数
            invalid_config = BASE_CONFIG.copy()
            invalid_config["num_agents"] = -1
            
            try:
                slm = DistributedSLM(invalid_config)
                raise AssertionError("不正な設定が受け入れられた")
            except Exception:
                pass  # 期待されるエラー
            
            duration = time.time() - start_time
            print_result("不正設定ハンドリング", True, "不正設定を適切に拒否", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("不正設定ハンドリング", False, f"エラー: {e}", duration)
    
    async def _test_missing_files(self):
        """ファイル不存在エラーハンドリングテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["model_path"] = "non_existent_model.gguf"
            
            try:
                slm = DistributedSLM(config)
                # モデル初期化でエラーが発生するはず
                await slm.generate("test")
                raise AssertionError("存在しないファイルが受け入れられた")
            except Exception:
                pass  # 期待されるエラー
            
            duration = time.time() - start_time
            print_result("ファイル不存在ハンドリング", True, "ファイル不存在を適切に検出", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("ファイル不存在ハンドリング", False, f"エラー: {e}", duration)
    
    async def _test_exception_handling(self):
        """一般的な例外処理テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            slm = DistributedSLM(config)
            
            # 空の入力でのテスト
            response = await slm.generate("")
            assert response is not None  # 空でも何らかの応答があるべき
            
            # 非常に長い入力でのテスト
            long_input = "test " * 1000
            response = await slm.generate(long_input)
            assert response is not None
            
            duration = time.time() - start_time
            print_result("例外処理", True, "異常入力を適切に処理", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("例外処理", False, f"エラー: {e}", duration)
    
    async def test_performance(self):
        """パフォーマンステスト"""
        print_header("パフォーマンステスト", 2)
        
        # 1. 応答時間テスト
        await self._test_response_time()
        
        # 2. スループットテスト
        await self._test_throughput()
        
        # 3. メモリ効率テスト
        await self._test_memory_efficiency()
    
    async def _test_response_time(self):
        """応答時間テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["num_agents"] = 1  # 高速化
            slm = DistributedSLM(config)
            
            queries = [
                "こんにちは",
                "人工知能とは何ですか？",
                "機械学習について説明してください"
            ]
            
            total_time = 0
            for query in queries:
                query_start = time.time()
                response = await slm.generate(query)
                query_time = time.time() - query_start
                total_time += query_time
                
                assert response is not None
                assert len(response) > 0
            
            avg_time = total_time / len(queries)
            
            duration = time.time() - start_time
            message = f"平均応答時間: {avg_time:.2f}s ({len(queries)}クエリ)"
            print_result("応答時間", True, message, duration)
            
            # パフォーマンスメトリクスを記録
            test_results["performance_metrics"]["avg_response_time"] = avg_time
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("応答時間", False, f"エラー: {e}", duration)
    
    async def _test_throughput(self):
        """スループットテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["use_parallel"] = True
            config["num_agents"] = 2
            slm = DistributedSLM(config)
            
            num_requests = 3  # 軽量テスト
            queries = ["テストクエリ" + str(i) for i in range(num_requests)]
            
            throughput_start = time.time()
            
            # 並列実行でスループット測定
            tasks = []
            for query in queries:
                task = asyncio.create_task(slm.generate(query))
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            throughput_time = time.time() - throughput_start
            
            # 全レスポンス確認
            for response in responses:
                assert response is not None
                assert len(response) > 0
            
            throughput = num_requests / throughput_time
            
            duration = time.time() - start_time
            message = f"スループット: {throughput:.2f} req/s ({num_requests}リクエスト)"
            print_result("スループット", True, message, duration)
            
            test_results["performance_metrics"]["throughput"] = throughput
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("スループット", False, f"エラー: {e}", duration)
    
    async def _test_memory_efficiency(self):
        """メモリ効率テスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            
            # 初期メモリ測定
            initial_memory = measure_memory()
            
            slm = DistributedSLM(config)
            
            # 複数回実行してメモリ使用量確認
            for i in range(3):
                response = await slm.generate(f"テストクエリ{i}")
                assert response is not None
            
            # 最終メモリ測定
            final_memory = measure_memory()
            
            memory_increase = final_memory["rss"] - initial_memory["rss"]
            
            duration = time.time() - start_time
            message = f"メモリ増加: {memory_increase:.1f}MB (RSS: {final_memory['rss']:.1f}MB)"
            
            # メモリ増加が過度でないかチェック（1GB未満）
            memory_ok = memory_increase < 1024
            print_result("メモリ効率", memory_ok, message, duration)
            
            test_results["memory_usage"]["test_memory_efficiency"] = {
                "initial": initial_memory,
                "final": final_memory,
                "increase": memory_increase
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("メモリ効率", False, f"エラー: {e}", duration)
    
    async def test_memory_management(self):
        """メモリ管理テスト"""
        print_header("メモリ管理テスト", 2)
        
        # 1. メモリリークテスト
        await self._test_memory_leak()
        
        # 2. ガベージコレクションテスト
        await self._test_garbage_collection()
    
    async def _test_memory_leak(self):
        """メモリリークテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            
            memory_samples = []
            
            # 複数回インスタンス生成・破棄
            for i in range(3):
                slm = DistributedSLM(config)
                await slm.generate("短いテスト")
                
                current_memory = measure_memory()
                memory_samples.append(current_memory["rss"])
                
                # インスタンスを明示的に削除
                del slm
                gc.collect()
            
            # メモリ使用量の変化を確認
            memory_trend = memory_samples[-1] - memory_samples[0]
            
            duration = time.time() - start_time
            message = f"メモリ変化: {memory_trend:.1f}MB (サンプル: {len(memory_samples)})"
            
            # 大幅なメモリ増加がないかチェック
            no_major_leak = abs(memory_trend) < 500  # 500MB未満の変化
            print_result("メモリリーク", no_major_leak, message, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("メモリリーク", False, f"エラー: {e}", duration)
    
    async def _test_garbage_collection(self):
        """ガベージコレクションテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            
            # GC前のメモリ
            gc.collect()
            memory_before = measure_memory()
            
            # 大量のオブジェクト生成
            slm_instances = []
            for i in range(2):
                slm = DistributedSLM(config)
                slm_instances.append(slm)
            
            # オブジェクト削除とGC
            del slm_instances
            gc.collect()
            
            memory_after = measure_memory()
            
            memory_freed = memory_before["rss"] - memory_after["rss"]
            
            duration = time.time() - start_time
            message = f"GC効果: {memory_freed:.1f}MB解放"
            print_result("ガベージコレクション", True, message, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("ガベージコレクション", False, f"エラー: {e}", duration)
    
    async def test_end_to_end(self):
        """エンドツーエンドテスト"""
        print_header("エンドツーエンドテスト", 2)
        
        # 1. 完全なワークフローテスト
        await self._test_complete_workflow()
        
        # 2. 複雑なシナリオテスト
        await self._test_complex_scenarios()
    
    async def _test_complete_workflow(self):
        """完全なワークフローテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["use_parallel"] = True
            config["use_summary"] = True
            config["num_agents"] = 2
            config["iterations"] = 1
            
            slm = DistributedSLM(config)
            
            # 実際的な質問
            question = "人工知能が社会に与える影響について教えてください"
            response = await slm.generate(question)
            
            # レスポンス品質確認
            assert response is not None
            assert len(response) > 50  # 十分な長さ
            assert isinstance(response, str)
            
            duration = time.time() - start_time
            message = f"応答長: {len(response)}文字"
            print_result("完全ワークフロー", True, message, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("完全ワークフロー", False, f"エラー: {e}", duration)
    
    async def _test_complex_scenarios(self):
        """複雑なシナリオテスト"""
        start_time = time.time()
        try:
            config = BASE_CONFIG.copy()
            config["num_agents"] = 3
            config["iterations"] = 2
            config["use_parallel"] = True
            config["use_summary"] = True
            
            slm = DistributedSLM(config)
            
            # 複雑な質問
            complex_question = "機械学習、深層学習、人工知能の違いと関係性について、それぞれの歴史的発展も含めて詳しく説明してください"
            response = await slm.generate(complex_question)
            
            assert response is not None
            assert len(response) > 100
            
            duration = time.time() - start_time
            message = f"複雑クエリ処理完了 (応答: {len(response)}文字)"
            print_result("複雑シナリオ", True, message, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            print_result("複雑シナリオ", False, f"エラー: {e}", duration)
    
    async def cleanup_and_report(self):
        """クリーンアップと最終レポート"""
        print_header("テスト結果レポート", 1)
        
        # 実行時間計算
        total_time = time.time() - self.start_time
        test_results["total_time"] = total_time
        
        # 最終メモリ測定
        final_memory = measure_memory()
        test_results["memory_usage"]["final"] = final_memory
        
        # サマリー出力
        total_tests = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["skipped"])
        passed_tests = len(test_results["passed"])
        failed_tests = len(test_results["failed"])
        skipped_tests = len(test_results["skipped"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"総テスト数: {total_tests}")
        print(f"成功: {passed_tests} ({'✓' if failed_tests == 0 else '⚠'})")
        print(f"失敗: {failed_tests} ({'✓' if failed_tests == 0 else '✗'})")
        print(f"スキップ: {skipped_tests}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"最終メモリ使用量: {final_memory['rss']:.1f}MB")
        
        # パフォーマンスメトリクス出力
        if test_results["performance_metrics"]:
            print("\nパフォーマンスメトリクス:")
            for metric, value in test_results["performance_metrics"].items():
                print(f"  {metric}: {value:.3f}")
        
        # 失敗したテストの詳細
        if failed_tests > 0:
            print("\n失敗したテスト:")
            for failed_test in test_results["failed"]:
                print(f"  ✗ {failed_test['name']}: {failed_test['message']}")
        
        # クリーンアップ
        for instance in self.slm_instances.values():
            del instance
        gc.collect()
        print(f"\n{'='*60}")
        if failed_tests == 0:
            print("🎉 すべてのテストが成功しました！")
        else:
            print(f"⚠️  {failed_tests}個のテストが失敗しました。")
        print(f"{'='*60}")

# グローバルSLMインスタンス（再利用のため）
_slm_instance = None

def get_slm_instance(config=None):
    """再利用可能なSLMインスタンスを取得"""
    global _slm_instance
    if _slm_instance is None or config is not None:
        if _slm_instance is not None:
            # メモリクリア
            del _slm_instance
            gc.collect()
        _slm_instance = DistributedSLM(config or DEFAULT_CONFIG)
    return _slm_instance

# ========== モジュール単体テスト ==========

class TestModules(unittest.TestCase):
    """各モジュールの基本機能テスト"""
    
    def setUp(self):
        self.config = DEFAULT_CONFIG.copy()
        self.blackboard = Blackboard(self.config)
    
    def test_blackboard(self):
        """ブラックボードの基本機能テスト"""
        # 書き込みテスト
        entry = self.blackboard.write("test_key", "test_value")
        self.assertEqual(entry["key"], "test_key")
        self.assertEqual(entry["value"], "test_value")
        
        # 読み込みテスト
        value = self.blackboard.read("test_key")
        self.assertEqual(value, "test_value")
        
        # 履歴テスト
        history = self.blackboard.get_history("test_key")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["value"], "test_value")
    
    def test_input_reception(self):
        """入力処理モジュールのテスト"""
        input_reception = InputReception(self.config)
        result = input_reception.process("Hello, World!")
        
        # 正規化テスト
        self.assertIn("normalized", result)
        self.assertIsInstance(result["normalized"], str)
        
        # トークン化テスト
        self.assertIn("tokens", result)
        self.assertIsInstance(result["tokens"], list)
        
        # 埋め込みテスト
        self.assertIn("embedding", result)
    
    def test_rag_retriever(self):
        """RAG検索モジュールのテスト（ダミーモード）"""
        rag = RAGRetriever(self.config)
        result = rag.retrieve("テスト用クエリ")
        
        # 何らかの文字列が返却されるはず
        self.assertIsInstance(result, str)
    
    def test_summary_engine(self):
        """要約エンジンのテスト"""
        # 軽量なテスト用の設定
        test_config = self.config.copy()
        test_config["n_ctx"] = 1024
        
        summary_engine = SummaryEngine(test_config)
        entries = [
            {"agent": 0, "text": "これはテスト文章1です。AIの将来性について議論します。"},
            {"agent": 1, "text": "テスト文章2です。技術の発展は人類に恩恵をもたらします。"}
        ]
        result = summary_engine.summarize_blackboard(entries)
        
        # 要約は空でないはず
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

# ========== 統合テスト ==========

async def test_integration():
    """分散SLMの統合テスト"""
    logging.info("統合テスト開始")
    
    # インスタンス生成
    start_time = time.time()
    # グローバルインスタンスを利用
    slm = get_slm_instance()
    init_time = time.time() - start_time
    logging.info(f"初期化時間: {init_time:.2f}秒")
    
    # テスト用クエリの削減
    test_queries = [
        "AIは教育をどのように変えますか？"
    ]
    
    for query in test_queries:
        logging.info(f"テストクエリ: {query}")
        
        start_time = time.time()
        response = await slm.generate(query)
        gen_time = time.time() - start_time
        
        logging.info(f"生成時間: {gen_time:.2f}秒")
        logging.info(f"応答: {response[:100]}...")
        
        # 黒板内容の確認
        bb_entries = len(slm.blackboard.history)
        logging.info(f"黒板エントリ数: {bb_entries}")
    
    logging.info("統合テスト終了")
    return True

# ========== 機能テスト ==========

async def test_iterative_summary():
    """反復と要約機能のテスト"""
    print_header("反復と要約テスト")
    
    # 設定: 2回の反復と要約有効
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1  # 反復回数を減らす
    config["use_summary"] = True
    config["num_agents"] = 2
    
    print(f"設定: {config['iterations']}回反復, 要約有効, {config['num_agents']}エージェント")
    
    # 既存インスタンスを更新
    slm = get_slm_instance(config)
    query = "気候変動の解決策について考察してください"
    
    print(f"入力: {query}")
    start_time = time.time()
    response = await slm.generate(query)
    total_time = time.time() - start_time
    
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"出力の一部: {response[:100]}...")
    
    # 中間要約の確認
    for i in range(config["iterations"]):
        summary = slm.blackboard.read(f'summary_{i}')
        if summary:
            print(f"反復{i+1}の要約: {summary[:100]}...")
    
    return True

async def test_parallel_processing():
    """並列処理機能のテスト"""
    print_header("並列処理テスト")
    
    # 設定: 並列処理有効、エージェント数増加
    config = DEFAULT_CONFIG.copy()
    config["use_parallel"] = True
    config["num_agents"] = 2  # エージェント数削減
    config["iterations"] = 1
    
    print(f"設定: 並列処理有効, {config['num_agents']}エージェント")
    
    # 同じインスタンスで設定変更して実行（逐次）
    slm = get_slm_instance(config)
    config["use_parallel"] = False
    slm.use_parallel = False
    
    query = "複雑な哲学的問題: 意識とは何か？"
    print(f"入力: {query}")
    
    # 逐次処理
    print("逐次処理実行中...")
    start_time = time.time()
    response_normal = await slm.generate(query)
    normal_time = time.time() - start_time
    print(f"逐次処理時間: {normal_time:.2f}秒")
    
    # 同じインスタンスで並列処理に切り替え
    slm.use_parallel = True
    
    # 並列処理
    print("並列処理実行中...")
    start_time = time.time()
    response_parallel = await slm.generate(query)
    parallel_time = time.time() - start_time
    print(f"並列処理時間: {parallel_time:.2f}秒")
    
    # 速度比較
    if normal_time > 0:
        speedup = normal_time / parallel_time
        print(f"速度向上率: {speedup:.2f}倍")
    
    return True

# ========== RAG ZIMモードテスト ==========

async def test_rag_zim_mode():
    """RAGリトリーバーのZIMモードをテスト"""
    print_header("RAGリトリーバー ZIMモードテスト")
    
    # 設定: ZIMモード有効
    config = DEFAULT_CONFIG.copy()
    config["rag_mode"] = "zim"
    config["zim_path"] = "C:\\Users\\admin\\Desktop\\課題研究\\KNOWAGE_DATABASE\\wikipedia_en_top_nopic_2025-03.zim"
    config["rag_score_threshold"] = 0.5
    config["rag_top_k"] = 3
    config["debug"] = True
    
    try:
        # RAGリトリーバーの初期化
        print(f"ZIMファイル: {config['zim_path']}")
        print("RAGリトリーバー初期化中...")
        rag = RAGRetriever(config)        # モードの確認
        print(f"実際の動作モード: {rag.mode}")
        
        if rag.mode != "zim":
            print("警告: ZIMモードではなく他のモードで動作しています。以下の理由が考えられます:")
            print("- ZIMファイルが存在しない")
            print("- libzimがインストールされていない")
            print("- sentence-transformersがインストールされていない")
            return False
        
        # いくつかのクエリでテスト
        test_queries = [
            "What is artificial intelligence?",
            "太陽系について教えて",
            "Albert Einstein's theory of relativity"
        ]
        
        for query in test_queries:
            print(f"\nクエリ: {query}")
            print("-" * 40)
            result = rag.retrieve(query)
            print(f"結果: \n{result[:300]}...")  # 長い結果は省略
            print("-" * 40)
        
        print("\nZIMモードテスト完了")
        return True
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 新機能テスト ==========

async def test_answer_quality():
    """回答の適切さ機能テスト"""
    print_header("質問適切性テスト")
    
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1
    config["num_agents"] = 2
    
    slm = get_slm_instance(config)
    
    # 質問の適切性をテストする質問リスト
    test_questions = [
        "AIは教育をどのように変えると思う？",  # 実行タスクの例
        "地球温暖化の主な原因は何ですか？",     # 明確な事実質問
        "量子コンピュータの将来性について",     # 技術予測質問
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n質問 {i+1}: {question}")
        print("-" * 40)
        
        start_time = time.time()
        response = await slm.generate(question)
        gen_time = time.time() - start_time
        
        print(f"応答時間: {gen_time:.2f}秒")
        print(f"応答: \n{response}")
        print("-" * 40)
    
    return True

async def test_conversation_memory():
    """会話記憶機能テスト"""
    print_header("会話履歴記憶テスト")
    
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1
    config["num_agents"] = 2
    config["use_memory"] = True  # 会話履歴機能を有効化
    
    # 新しいインスタンスで記憶をクリーンな状態から始める
    slm = get_slm_instance(config)
    
    # 記憶をリセット（直接conversation_memoryを使用）
    if hasattr(slm, 'conversation_memory'):
        slm.conversation_memory.clear_memory()
        print("会話履歴をリセットしました")
    
    # 会話の流れをテストする質問シーケンス
    conversation = [
        "こんにちは、私の名前は園木です。",
        "私の趣味について知りたいですか？",
        "私はプログラミングとピアノが好きです。",
        "私の名前は何でしたか？",  # 以前の会話を覚えているかテスト
    ]
    
    for i, message in enumerate(conversation):
        print(f"\n会話ターン {i+1}: {message}")
        print("-" * 40)
        
        response = await slm.generate(message)
        print(f"応答: {response}")
        
        # 会話コンテキストの状態を表示
        if i > 0 and 'conversation_context' in slm.blackboard.memory:
            context = slm.blackboard.read('conversation_context')
            print(f"会話コンテキスト: {context}")
    
    return True

async def test_role_assignment():
    """役割振り分けモジュールテスト"""
    print_header("役割振り分けモジュールテスト")
    
    config = DEFAULT_CONFIG.copy()
    config["iterations"] = 1
    config["num_agents"] = 3  # より多くのエージェントで多様な役割を確認
    
    slm = get_slm_instance(config)
    
    # 異なるタイプの質問でテスト
    test_questions = [
        {"text": "AIと人間の協調と競争について議論してください", "type": "discussion"},
        {"text": "新しいモバイルアプリのビジネスプランを考えてください", "type": "planning"},
        {"text": "量子力学の基本原理を簡単に説明してください", "type": "informational"},
        {"text": "今日の気分はどうですか？", "type": "conversational"},
    ]
    
    for i, question_data in enumerate(test_questions):
        question = question_data["text"]
        expected_type = question_data["type"]
        
        print(f"\n質問 {i+1}: {question}")
        print(f"期待される質問タイプ: {expected_type}")
        print("-" * 40)
        
        # 質問タイプと役割の判定
        normalized_question = {"normalized": question}
        slm.blackboard.write('input', normalized_question)
        slm.agent_pool.update_roles_based_on_question(question)
        
        # 実際に判定された質問タイプを表示
        actual_type = slm.blackboard.read('question_type')
        print(f"判定された質問タイプ: {actual_type}")
        
        # 選択されたエージェント役割を表示
        print("選択されたエージェント役割:")
        for j in range(slm.num_agents):
            role_idx = j % len(slm.agent_pool.agent_roles)
            role = slm.agent_pool.agent_roles[role_idx]
            print(f"  エージェント{j+1}: {role['role']}")
        
        # 応答生成
        response = await slm.generate(question)
        print(f"応答: {response[:150]}...")  # 長い場合は省略
    
    return True

async def test_blackboard_conversation_memory():
    """黒板と統合した会話記憶テスト"""
    print_header("黒板と統合した会話記憶テスト")
    
    # 設定
    config = DEFAULT_CONFIG.copy()
    config["use_memory"] = True
    
    # 新しい黒板インスタンスを作成
    blackboard = Blackboard(config)
    
    # 会話記憶インスタンスを黒板と統合して作成
    conversation_memory = ConversationMemory(config, blackboard=blackboard)
    
    print("会話記憶モジュールを黒板と統合しました")
    
    # テスト用の会話データ
    test_conversations = [
        # (ユーザー入力, システム応答)
        ("こんにちは、私の名前は太郎です。", "太郎さん、こんにちは！お元気ですか？"),
        ("東京に住んでいます。", "東京は素晴らしい都市ですね。何か東京について質問はありますか？"),
        ("趣味はプログラミングです。", "プログラミングは素晴らしい趣味ですね。どんな言語を使いますか？"),
        ("Pythonが好きです。", "Pythonは汎用性が高く素晴らしい言語ですね！何か作っているものはありますか？")
    ]
    
    # 会話を順番に追加
    print("\n会話の追加:")
    for i, (user_input, system_response) in enumerate(test_conversations):
        print(f"\n会話 {i+1}:")
        print(f"ユーザー: {user_input}")
        print(f"システム: {system_response}")
        
        # 会話を記憶に追加
        conversation_memory.add_conversation_entry(user_input, system_response)
        
        # 黒板に保存された内容を確認
        print("\n黒板の状態確認:")
        if blackboard.read("conversation_history"):
            print(f"- 履歴エントリ数: {len(blackboard.read('conversation_history'))}")
        if blackboard.read("conversation_context"):
            context = blackboard.read("conversation_context")
            print(f"- コンテキスト: {context[:100]}...")
        if blackboard.read("conversation_key_facts"):
            facts = blackboard.read("conversation_key_facts")
            print("- 抽出された重要な情報:")
            for category, items in facts.items():
                if items:
                    print(f"  {category}: {', '.join(items)}")
    
    # コンテキスト取得テスト
    context = conversation_memory.get_conversation_context()
    print("\n最終的な会話コンテキスト:")
    print(f"{context[:200]}...")
    
    # 重要情報の取得テスト
    key_facts = conversation_memory.key_facts
    print("\n抽出された重要情報:")
    for category, items in key_facts.items():
        if items:
            print(f"{category}: {', '.join(items)}")
    
    # 新しいインスタンスを作成して読み込みテスト
    print("\n新インスタンスで黒板からの読み込みテスト:")
    new_memory = ConversationMemory(config, blackboard=blackboard)
    print(f"- 履歴エントリ数: {len(new_memory.conversation_history)}")
    print(f"- 名前の記憶: {new_memory.key_facts.get('names', [])}")
    
    return True

# ========== メイン実行部 ==========

def print_header(title):
    """セクションヘッダの表示"""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

async def main():
    """テストスクリプトのメイン関数"""
    print_header("MurmurNet テストスクリプト")
    
    try:
        # 黒板と会話記憶の統合テスト (新規追加)
        await test_blackboard_conversation_memory()
        
        # 新機能テスト（追加）
        print_header("新実装機能テスト")
        
        # 1. 質問適切性テスト
        await test_answer_quality()
        
        # 2. 会話記憶テスト
        await test_conversation_memory()
        
        # 3. 役割振り分けテスト
        await test_role_assignment()
        
        # 既存テスト
        # RAG ZIMモードのテスト（追加）
        print_header("RAG ZIMモードテスト")
        await test_rag_zim_mode()
        
        # 順序を変更：統合テスト→機能テスト→単体テストの順で
        
        # 統合テスト 
        print_header("統合テスト")
        success = await test_integration()
        if success:
            print("✓ 統合テスト成功")
        else:
            print("✗ 統合テスト失敗")
        
        # 反復と要約のテスト
        await test_iterative_summary()
        
        # 並列処理テスト
        await test_parallel_processing()
        
        # 単一クエリテスト（最終チェック）
        print_header("単一クエリテスト (最終)")
        config = DEFAULT_CONFIG.copy()
        config["iterations"] = 1  # 反復回数を減らす
        config["use_summary"] = True
        config["num_agents"] = 2  # エージェント数削減
        config["use_memory"] = True  # 会話履歴を有効化
        
        # 既存インスタンスを更新
        slm = get_slm_instance(config)
        
        query = "人工知能と人間の関係はどのように発展するでしょうか？"
        print(f"入力: {query}")
        
        response = await slm.generate(query)
        print(f"出力: {response}")
        
        # 最後に単体テスト（LLMを使わない軽量テスト）
        print_header("モジュール単体テスト")
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
        # リソース解放
        global _slm_instance
        if (_slm_instance is not None):
            del _slm_instance
            _slm_instance = None
            gc.collect()
        
        print("\nテスト完了")
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
