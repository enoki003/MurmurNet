#!/usr/bin/env python3
"""
スレッド・エージェント最適化
4C/8T環境でのスレッド数とエージェント数の最適化
"""

import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class ThreadOptimizer:
    """スレッド・エージェント最適化クラス"""
    
    @staticmethod
    def get_optimal_threads():
        """最適なスレッド数を取得"""
        cpu_count = os.cpu_count()
        
        if cpu_count >= 8:  # 論理8T以上
            return 6  # 6スレッドが最適
        elif cpu_count >= 4:  # 物理4C以上
            return cpu_count - 1  # 1つ余裕を残す
        else:
            return cpu_count
    
    @staticmethod
    def get_optimal_agents():
        """最適なエージェント数を取得"""
        cpu_count = os.cpu_count()
        
        if cpu_count >= 8:  # 論理8T以上
            return 4  # 4エージェント
        elif cpu_count >= 4:  # 物理4C以上
            return 3  # 3エージェント
        else:
            return 2  # デフォルト

def apply_thread_optimizations():
    """スレッド最適化を適用"""
    optimal_threads = ThreadOptimizer.get_optimal_threads()
    optimal_agents = ThreadOptimizer.get_optimal_agents()
    
    # GGML用環境変数設定
    os.environ['GGML_NUM_THREADS'] = str(optimal_threads)
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    
    logger.info(f"Thread optimization applied:")
    logger.info(f"  GGML threads: {optimal_threads}")
    logger.info(f"  Recommended agents: {optimal_agents}")
    
    return optimal_threads, optimal_agents

def patch_agent_pool_config():
    """AgentPoolの設定にスレッド最適化を適用"""
    try:
        from MurmurNet.modules.agent_pool import AgentPoolManager
        
        # 元の__init__メソッドをバックアップ
        original_init = AgentPoolManager.__init__
        
        def optimized_init(self, config):
            """最適化されたAgentPool初期化"""
            # スレッド数を最適化
            optimal_threads, optimal_agents = apply_thread_optimizations()
            
            # configを動的に更新
            if isinstance(config, dict):
                config['num_agents'] = min(config.get('num_agents', 2), optimal_agents)
                config['threads'] = optimal_threads
            
            logger.info(f"AgentPool optimized: {config.get('num_agents', 2)} agents, {optimal_threads} threads")
            return original_init(self, config)
        
        # メソッドを置き換え
        AgentPoolManager.__init__ = optimized_init
        logger.info("Thread optimization patch applied to AgentPool")
        
    except ImportError as e:
        logger.error(f"Failed to apply thread optimization patch: {e}")

def patch_llm_cpp_config():
    """llama-cpp-pythonの設定を最適化"""
    try:
        # llama_cpp_pythonが使用する環境変数
        optimal_threads = ThreadOptimizer.get_optimal_threads()
        
        os.environ['LLAMA_CPP_N_THREADS'] = str(optimal_threads)
        os.environ['LLAMA_CPP_N_BATCH'] = '1024'  # バッチサイズ倍増
        
        logger.info(f"llama-cpp-python optimizations:")
        logger.info(f"  n_threads: {optimal_threads}")
        logger.info(f"  n_batch: 1024")
        
    except Exception as e:
        logger.error(f"Failed to apply llama-cpp optimization: {e}")

def create_optimized_config():
    """最適化された設定ファイルを生成"""
    optimal_threads, optimal_agents = apply_thread_optimizations()
    
    config = {
        # スレッド最適化
        'threads': optimal_threads,
        'agents': optimal_agents,
        
        # GGML最適化
        'n_batch': 1024,
        'n_ctx': 2048,
        'n_seq_max': 2,  # KVキャッシュ共有
        
        # その他最適化
        'use_mlock': True,
        'use_mmap': True,
        'numa': False
    }
    
    return config

def optimize_threading():
    """包括的なスレッド最適化を実行"""
    logger.info("Applying thread and agent optimizations...")
    
    # 基本スレッド最適化
    optimal_threads, optimal_agents = apply_thread_optimizations()
    
    # AgentPoolパッチ適用
    patch_agent_pool_config()
    
    # llama-cpp最適化
    patch_llm_cpp_config()
    
    # RAG・要約最適化
    rag_optimizer = RAGSummaryOptimizer()
    rag_success = rag_optimizer.patch_rag_retriever()
    summary_success = rag_optimizer.patch_summary_engine()
    
    # 最適化設定を返す
    config = create_optimized_config()
    config['rag_optimized'] = rag_success
    config['summary_optimized'] = summary_success
    
    logger.info("Comprehensive optimization completed")
    return config

class RAGSummaryOptimizer:
    """RAG・要約の無駄削減クラス"""
    
    @staticmethod
    def patch_rag_retriever():
        """RAGRetrieverに短文スキップパッチを適用"""
        try:
            from MurmurNet.modules.rag_retriever import RAGRetriever
              # 元のretrieveメソッドをバックアップ
            if not hasattr(RAGRetriever, '_original_retrieve'):
                RAGRetriever._original_retrieve = RAGRetriever.retrieve
            
            def optimized_retrieve(self, query, threshold=64):
                """短い質問時はRAGをスキップ（ヒット率0%を防ぐ）"""
                if len(query.strip()) < threshold:
                    logger.debug(f"RAG skipped: query too short ({len(query)} < {threshold})")
                    return ""  # 空文字列を返す
                
                return RAGRetriever._original_retrieve(self, query)
            
            RAGRetriever.retrieve = optimized_retrieve
            logger.info("RAG short-text skip patch applied")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to patch RAGRetriever: {e}")
            return False
    
    @staticmethod
    def patch_summary_engine():
        """SummaryEngineに短文スキップパッチを適用"""
        try:
            from MurmurNet.modules.summary_engine import SummaryEngine
            
            # 元のsummarize_blackboardメソッドをバックアップ
            if not hasattr(SummaryEngine, '_original_summarize_blackboard'):
                SummaryEngine._original_summarize_blackboard = SummaryEngine.summarize_blackboard
            
            def optimized_summarize_blackboard(self, entries, threshold=64):
                """短いテキストは要約をスキップ（5.24s → 0s）"""
                # エントリ数または総文字数で判定
                if len(entries) < 2:
                    logger.debug("Summary skipped: too few entries")
                    return "短いため要約をスキップしました。"
                
                # 総文字数を計算
                total_chars = sum(len(str(entry.get('content', ''))) for entry in entries)
                if total_chars < threshold:
                    logger.info(f"Summary skipped: text too short ({total_chars} < {threshold})")
                    return "短いため要約をスキップしました。"
                
                return SummaryEngine._original_summarize_blackboard(self, entries)
            
            SummaryEngine.summarize_blackboard = optimized_summarize_blackboard
            logger.info("Summary short-text skip patch applied")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to patch SummaryEngine: {e}")
            return False

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("Testing thread optimization...")
    
    cpu_count = os.cpu_count()
    print(f"Detected CPU cores: {cpu_count}")
    
    # 最適化適用
    config = optimize_threading()
    
    print("Optimized configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("Environment variables:")
    for env_var in ['GGML_NUM_THREADS', 'OMP_NUM_THREADS', 'LLAMA_CPP_N_THREADS']:
        value = os.environ.get(env_var, 'Not set')
        print(f"  {env_var}: {value}")
    
    print("Thread optimization test completed")
