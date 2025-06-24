"""
要約エンジンの最適化パッチ
短文入力時にLLM呼び出しをスキップして処理時間を削減
"""

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

# 使用例:
# summary_engine = patch_summary_engine(original_summary_engine)