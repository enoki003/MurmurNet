#!/usr/bin/env python3
"""
RAG・要約スキップ最適化
短文時の無駄な処理（RAGヒット率0%、要約5.24s）をスキップ
"""

import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class SkipOptimizer:
    """RAG・要約スキップ最適化クラス"""
    
    # スキップ閾値
    RAG_MIN_TOKENS = 64      # 64トークン未満でRAGスキップ
    SUMMARY_MIN_CHARS = 128  # 128文字未満で要約スキップ
    
    @staticmethod
    def should_skip_rag(input_text, token_count=None):
        """RAGをスキップすべきかチェック"""
        if token_count is None:
            # 簡易トークン数推定（日本語：1文字≈1.5トークン、英語：1単語≈1.3トークン）
            char_count = len(input_text)
            if char_count < 32:  # 32文字未満は確実にスキップ
                return True
            
            # より詳細な推定
            word_count = len(input_text.split())
            estimated_tokens = max(char_count * 0.8, word_count * 1.3)
        else:
            estimated_tokens = token_count
        
        return estimated_tokens < SkipOptimizer.RAG_MIN_TOKENS
    
    @staticmethod
    def should_skip_summary(input_text):
        """要約をスキップすべきかチェック"""
        return len(input_text) < SkipOptimizer.SUMMARY_MIN_CHARS

def apply_rag_skip_patch():
    """RAGスキップパッチを適用"""
    try:
        from MurmurNet.modules.rag_retriever import RAGRetriever
        
        # 元のretrieveメソッドをバックアップ
        original_retrieve = RAGRetriever.retrieve
        
        def smart_retrieve(self, query, top_k=5):
            """スマートRAG検索（短文スキップ機能付き）"""
            # 短文チェック
            if SkipOptimizer.should_skip_rag(query):
                logger.info(f"RAG skipped for short query: {len(query)} chars")
                return []  # 空の結果を返す
            
            # 通常のRAG処理
            return original_retrieve(self, query, top_k)
        
        # メソッドを置き換え
        RAGRetriever.retrieve = smart_retrieve
        logger.info("RAG skip patch applied")
        
    except ImportError as e:
        logger.error(f"Failed to apply RAG skip patch: {e}")

def apply_summary_skip_patch():
    """要約スキップパッチを適用"""
    try:
        from MurmurNet.modules.summary_engine import SummaryEngine
        
        # 元のsummarizeメソッドをバックアップ
        original_summarize = SummaryEngine.summarize
        
        def smart_summarize(self, text, max_length=150):
            """スマート要約（短文スキップ機能付き）"""
            # 短文チェック
            if SkipOptimizer.should_skip_summary(text):
                logger.info(f"Summary skipped for short text: {len(text)} chars")
                return text  # 元のテキストをそのまま返す
            
            # 通常の要約処理
            return original_summarize(self, text, max_length)
        
        # メソッドを置き換え
        SummaryEngine.summarize = smart_summarize
        logger.info("Summary skip patch applied")
        
    except ImportError as e:
        logger.error(f"Failed to apply summary skip patch: {e}")

def apply_distributed_slm_skip_patch():
    """DistributedSLMにスキップロジックを統合"""
    try:
        from MurmurNet.distributed_slm import DistributedSLM
        
        # 元のgenerateメソッドをバックアップ
        original_generate = DistributedSLM.generate
        
        async def smart_generate(self, user_input, **kwargs):
            """スマート生成（統合スキップ機能付き）"""
            # 入力長チェック
            input_length = len(user_input)
            
            # フラグ設定
            skip_rag = SkipOptimizer.should_skip_rag(user_input)
            skip_summary = SkipOptimizer.should_skip_summary(user_input)
            
            if skip_rag or skip_summary:
                logger.info(f"Optimization applied - RAG skip: {skip_rag}, Summary skip: {skip_summary}")
            
            # 設定を動的に更新
            original_rag_mode = getattr(self, 'rag_mode', 'dummy')
            original_use_summary = getattr(self, 'use_summary', True)
            
            if skip_rag:
                self.rag_mode = 'dummy'
            if skip_summary:
                self.use_summary = False
            
            try:
                # 通常の生成処理
                result = await original_generate(self, user_input, **kwargs)
                return result
            finally:
                # 設定を復元
                self.rag_mode = original_rag_mode
                self.use_summary = original_use_summary
        
        # メソッドを置き換え
        DistributedSLM.generate = smart_generate
        logger.info("DistributedSLM skip optimization applied")
        
    except ImportError as e:
        logger.error(f"Failed to apply DistributedSLM skip patch: {e}")

def create_skip_config():
    """スキップ最適化設定を生成"""
    config = {
        'rag_min_tokens': SkipOptimizer.RAG_MIN_TOKENS,
        'summary_min_chars': SkipOptimizer.SUMMARY_MIN_CHARS,
        'skip_optimization': True,
        'smart_rag': True,
        'smart_summary': True
    }
    
    return config

def optimize_skip_logic():
    """包括的なスキップ最適化を実行"""
    logger.info("Applying RAG and summary skip optimizations...")
    
    # RAGスキップパッチ適用
    apply_rag_skip_patch()
    
    # 要約スキップパッチ適用
    apply_summary_skip_patch()
    
    # DistributedSLMスキップパッチ適用
    apply_distributed_slm_skip_patch()
    
    # 設定を返す
    config = create_skip_config()
    
    logger.info("Skip optimization completed")
    return config

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("Testing skip optimization...")
    
    # テストケース
    test_cases = [
        ("短い質問", "こんにちは"),                    # 5文字：両方スキップ
        ("中程度の質問", "今日の天気はどうですか？" * 3),    # 約30文字：要約のみスキップ
        ("長い質問", "AI技術の発展について詳しく教えてください。" * 5)  # 約100文字：スキップなし
    ]
    
    for case_name, test_input in test_cases:
        print(f"\n{case_name}: '{test_input[:20]}...'")
        print(f"  Length: {len(test_input)} chars")
        print(f"  Skip RAG: {SkipOptimizer.should_skip_rag(test_input)}")
        print(f"  Skip Summary: {SkipOptimizer.should_skip_summary(test_input)}")
    
    # 最適化適用
    config = optimize_skip_logic()
    
    print("\nSkip optimization configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("Skip optimization test completed")
