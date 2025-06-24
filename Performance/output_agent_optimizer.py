"""
OutputAgent推論速度最適化設定
現在の13 t/s → 目標20 t/s への改善
"""

class OutputAgentOptimizer:
    def __init__(self):
        self.optimized_params = {
            # スレッド数増加（論理8コアのうち6を使用）
            "threads": 6,
            
            # バッチサイズ倍増（512 → 1024）
            "n_batch": 1024,
            
            # KVキャッシュ共有でコピーコスト削減
            "n_seq_max": 2,
            
            # コンテキスト長最適化
            "n_ctx": 2048,  # 必要最小限に制限
            
            # GPU使用可能時の設定
            "n_gpu_layers": -1,  # 全レイヤーをGPUへ
        }
    
    def get_optimized_params(self):
        """最適化されたパラメータセットを返す"""
        return self.optimized_params.copy()
    
    def apply_to_model_args(self, model_args):
        """既存のモデル引数に最適化パラメータを適用"""
        optimized = model_args.copy()
        optimized.update(self.optimized_params)
        
        print("OutputAgent最適化パラメータ適用:")
        for key, value in self.optimized_params.items():
            print(f"  {key}: {value}")
        
        return optimized
    
    def estimate_performance_gain(self, baseline_tps=13):
        """
        パフォーマンス向上の推定値を計算
        """
        improvements = {
            "threads": 1.3,      # 6スレッド化で30%向上
            "n_batch": 1.2,      # バッチ倍増で20%向上  
            "n_seq_max": 1.1,    # KV共有で10%向上
        }
        
        total_multiplier = 1.0
        for improvement in improvements.values():
            total_multiplier *= improvement
        
        estimated_tps = baseline_tps * total_multiplier
        
        print(f"推定パフォーマンス向上:")
        print(f"  現在: {baseline_tps} t/s")
        print(f"  推定: {estimated_tps:.1f} t/s")
        print(f"  向上率: {(estimated_tps/baseline_tps-1)*100:.1f}%")
        
        return estimated_tps

# 使用例
optimizer = OutputAgentOptimizer()
optimized_params = optimizer.get_optimized_params()

# 430 token を 7秒で処理する目標（約20 t/s）
target_time_for_430_tokens = 430 / 20  # = 21.5秒 → 目標7秒なので更なる最適化が必要