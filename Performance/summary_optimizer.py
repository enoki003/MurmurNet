"""
要約エンジンの最適化パッチ
短文入力時にLLM呼び出しをスキップして処理時間を削減
"""

import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class OptimizedSummaryEngine:
    def __init__(self, original_engine):
        self.original_engine = original_engine
        self.min_length_threshold = 64  # 64文字未満はスキップ
        
    def summarize(self, text, **kwargs):
        """
        短文の場合はLLM呼び出しをスキップ
        12文字で128token生成する無駄を排除
        """
        if len(text.strip()) < self.min_length_threshold:
            print(f"SummaryEngine - SKIP: 入力長 {len(text)} < {self.min_length_threshold}")
            return None  # 要約不要
            
        print(f"SummaryEngine - PROCESS: 入力長 {len(text)}")
        return self.original_engine.summarize(text, **kwargs)

def patch_summary_engine(engine):
    """既存の要約エンジンを最適化版でラップ"""
    return OptimizedSummaryEngine(engine)

def apply_summary_optimization():
    """要約エンジン最適化を適用"""
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
    logging.basicConfig(level=logging.INFO)
    
    print("Testing summary optimization...")
    success = apply_summary_optimization()
    print(f"Summary optimization: {'SUCCESS' if success else 'FAILED'}")

# 使用例:
# summary_engine = patch_summary_engine(original_summary_engine)