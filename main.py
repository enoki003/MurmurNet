#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet メインエントリポイント
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Boids対応Slotアーキテクチャの統合実行システム

機能:
- コマンドライン/対話式実行
- Boids有効/無効の切り替え
- 詳細な実行結果と評価レポート
- 設定ファイル駆動制御
- バッチ処理とリアルタイム対話

使用方法:
  python main.py                          # 対話モード
  python main.py --query "質問内容"        # 単発実行
  python main.py --config custom.yaml     # カスタム設定
  python main.py --evaluate               # 評価モード
  python main.py --benchmark              # ベンチマークモード

作者: Yuhi Sonoki
"""

import os
import sys
import argparse
import time
import yaml
import json
from typing import Dict, Any, List, Optional

# MurmurNetモジュールパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'MurmurNet'))

try:
    from modules.slots import SlotRunner
    from modules.slot_blackboard import SlotBlackboard
    from modules.model_factory import ModelFactory
    from modules.embedder import Embedder
    from modules.evaluation import (
        SlotOutputEvaluator, 
        BoidsEvaluator, 
        SystemPerformanceEvaluator,
        EvaluationReporter
    )
    from modules.boids import VectorSpace
    print("✓ MurmurNetモジュール読み込み完了")
    
except ImportError as e:
    print(f"✗ モジュール読み込みエラー: {e}")
    print("MurmurNetの環境設定を確認してください。")
    sys.exit(1)


class MurmurNetCLI:
    """MurmurNet CLI インターフェース"""
    
    def __init__(self, config_path: str = "config_boids_slots.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # コンポーネント（遅延初期化）
        self.model_factory = None
        self.embedder = None
        self.slot_runner = None
        self.evaluators = None
        
        # 実行履歴
        self.session_history = []
        self.session_start_time = time.time()
    
    def load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if not os.path.exists(self.config_path):
            print(f"⚠️  設定ファイルが見つかりません: {self.config_path}")
            print("デフォルト設定を使用します。")
            return self.get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ 設定ファイル読み込み完了: {self.config_path}")
            return config
            
        except Exception as e:
            print(f"✗ 設定ファイル読み込みエラー: {e}")
            print("デフォルト設定を使用します。")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'debug': False,
            'use_boids_synthesizer': True,
            'synthesis_strategy': 'adaptive',
            'max_slot_entries': 50,
            'slot_max_output_length': 200,
            'slot_temperature': 0.8,
            'slot_top_p': 0.9
        }
    
    def initialize_components(self):
        """コンポーネントの初期化"""
        if self.model_factory is not None:
            return  # 既に初期化済み
        
        print("MurmurNet初期化中...")
        
        try:
            # ModelFactory
            self.model_factory = ModelFactory(self.config)
            print("✓ ModelFactory初期化完了")
            
            # Embedder
            self.embedder = Embedder(self.config)
            self.embedder.initialize()
            print("✓ Embedder初期化完了")
            
            # SlotRunner（Boids対応）
            self.slot_runner = SlotRunner(self.config, self.model_factory, self.embedder)
            print(f"✓ SlotRunner初期化完了 (Boids: {self.slot_runner.use_boids_synthesizer})")
            
            # Evaluators
            self.evaluators = {
                'slot': SlotOutputEvaluator(),
                'boids': BoidsEvaluator(self.embedder),
                'system': SystemPerformanceEvaluator(),
                'reporter': EvaluationReporter()
            }
            print("✓ 評価システム初期化完了")
            
        except Exception as e:
            print(f"✗ 初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def process_query(self, query: str, enable_evaluation: bool = True) -> Dict[str, Any]:
        """クエリの処理"""
        self.initialize_components()
        
        print(f"\n🔄 処理開始: {query}")
        start_time = time.time()
        
        try:
            # Blackboard作成
            blackboard = SlotBlackboard()
            
            # Slot実行
            result = self.slot_runner.run_all_slots(blackboard, query, self.embedder)
            
            # 実行時間
            execution_time = time.time() - start_time
            result['total_execution_time'] = execution_time
            
            if result['success']:
                print(f"✅ 処理完了 ({execution_time:.2f}秒)")
                print(f"\n📝 最終応答:")
                print(f"{result['final_response']}")
                print(f"\n📊 統合品質: {result.get('synthesis_quality', 0):.2f}")
                
                # 評価実行
                if enable_evaluation and self.evaluators:
                    evaluation_results = self.run_evaluation(result, query, blackboard)
                    result['evaluation'] = evaluation_results
                
                # セッション履歴に追加
                self.session_history.append({
                    'query': query,
                    'timestamp': time.time(),
                    'result': result,
                    'execution_time': execution_time
                })
                
            else:
                print(f"❌ 処理失敗: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"💥 処理エラー: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'final_response': "処理中にエラーが発生しました。",
                'execution_time': time.time() - start_time
            }
    
    def run_evaluation(self, result: Dict[str, Any], query: str, blackboard: SlotBlackboard) -> Dict[str, Any]:
        """評価の実行"""
        print("\n📈 評価実行中...")
        
        evaluations = {}
        
        try:
            # Slot出力評価
            for slot_name, slot_result in result['slot_results'].items():
                if slot_result.get('text'):
                    eval_result = self.evaluators['slot'].evaluate_output(
                        slot_result['text'], 
                        query,
                        slot_result.get('metadata', {})
                    )
                    evaluations[f"slot_{slot_name}"] = {
                        'score': eval_result.normalized_score,
                        'grade': eval_result.grade,
                        'details': eval_result.details
                    }
            
            # Boids統合評価
            if result.get('boids_enabled'):
                boids_eval = self.evaluators['boids'].evaluate_synthesis(
                    blackboard.get_slot_entries(),
                    result['final_response']
                )
                evaluations['boids_synthesis'] = {
                    'score': boids_eval.normalized_score,
                    'grade': boids_eval.grade,
                    'details': boids_eval.details
                }
            
            # システム性能評価
            system_eval = self.evaluators['system'].evaluate_performance(result)
            evaluations['system_performance'] = {
                'score': system_eval.normalized_score,
                'grade': system_eval.grade,
                'details': system_eval.details
            }
            
            # 評価結果表示
            print("📊 評価結果:")
            for eval_name, eval_data in evaluations.items():
                print(f"  {eval_name}: {eval_data['score']:.2f} ({eval_data['grade']})")
            
            return evaluations
            
        except Exception as e:
            print(f"⚠️  評価エラー: {e}")
            return {'error': str(e)}
    
    def interactive_mode(self):
        """対話モード"""
        print("\n" + "=" * 60)
        print("MurmurNet Boids対応Slotアーキテクチャ - 対話モード")
        print("=" * 60)
        print("'quit', 'exit', 'q' で終了")
        print("'stats' で統計表示")
        print("'config' で設定表示")
        print("'toggle-boids' でBoids有効/無効切り替え")
        print("'help' でヘルプ表示")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n🤖 質問をどうぞ: ").strip()
                
                if not user_input:
                    continue
                
                # 終了コマンド
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 MurmurNetを終了します。")
                    break
                
                # 統計表示
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                    continue
                
                # 設定表示
                elif user_input.lower() == 'config':
                    self.show_config()
                    continue
                
                # Boids切り替え
                elif user_input.lower() == 'toggle-boids':
                    self.toggle_boids()
                    continue
                
                # ヘルプ
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # 通常の質問処理
                else:
                    result = self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 MurmurNetを終了します。")
                break
            except EOFError:
                print("\n\n👋 MurmurNetを終了します。")
                break
    
    def show_statistics(self):
        """統計情報の表示"""
        if not self.slot_runner:
            print("⚠️  統計情報がありません（初期化前）")
            return
        
        stats = self.slot_runner.get_statistics()
        
        print("\n📊 SlotRunner統計:")
        print(f"  総実行回数: {stats['total_runs']}")
        print(f"  成功回数: {stats['successful_runs']}")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  平均品質: {stats['average_quality_score']:.2f}")
        print(f"  Boids有効: {stats['boids_synthesizer_enabled']}")
        
        print("\n📈 セッション統計:")
        session_duration = time.time() - self.session_start_time
        print(f"  セッション時間: {session_duration:.0f}秒")
        print(f"  処理回数: {len(self.session_history)}")
        
        if self.session_history:
            avg_time = sum(h['execution_time'] for h in self.session_history) / len(self.session_history)
            print(f"  平均実行時間: {avg_time:.2f}秒")
    
    def show_config(self):
        """設定情報の表示"""
        print("\n⚙️  現在の設定:")
        key_settings = {
            'use_boids_synthesizer': 'Boids統合',
            'synthesis_strategy': '統合戦略',
            'debug': 'デバッグ',
            'slot_max_output_length': 'Slot最大出力長',
            'slot_temperature': 'Slot温度',
            'max_slot_entries': '最大Slotエントリ'
        }
        
        for key, description in key_settings.items():
            value = self.config.get(key, 'N/A')
            print(f"  {description}: {value}")
    
    def toggle_boids(self):
        """Boids有効/無効の切り替え"""
        if not self.slot_runner:
            print("⚠️  SlotRunnerが初期化されていません")
            return
        
        current_state = self.slot_runner.use_boids_synthesizer
        new_state = not current_state
        
        self.slot_runner.use_boids_synthesizer = new_state
        self.config['use_boids_synthesizer'] = new_state
        
        print(f"🔄 Boids統合: {current_state} → {new_state}")
    
    def show_help(self):
        """ヘルプの表示"""
        print("\n❓ MurmurNet ヘルプ:")
        print("  質問を入力すると、Boids対応Slotシステムが協調して回答します")
        print("\n利用可能なコマンド:")
        print("  stats        - 統計情報表示")
        print("  config       - 設定情報表示")
        print("  toggle-boids - Boids有効/無効切り替え")
        print("  help         - このヘルプ表示")
        print("  quit/exit/q  - 終了")
        print("\nSlot構成:")
        if self.slot_runner:
            for slot_name in self.slot_runner.execution_order:
                print(f"  • {slot_name}")
    
    def batch_mode(self, queries: List[str], output_file: Optional[str] = None):
        """バッチ処理モード"""
        print(f"\n📦 バッチ処理開始 ({len(queries)}件)")
        
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query}")
            result = self.process_query(query, enable_evaluation=True)
            results.append({
                'index': i,
                'query': query,
                'result': result
            })
        
        # 結果保存
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"📁 結果を保存: {output_file}")
            except Exception as e:
                print(f"⚠️  結果保存エラー: {e}")
        
        # 統計表示
        successful = sum(1 for r in results if r['result']['success'])
        print(f"\n📊 バッチ処理完了: {successful}/{len(queries)} 成功")
        
        return results
    
    def benchmark_mode(self):
        """ベンチマークモード"""
        print("\n🏁 ベンチマークモード開始")
        
        benchmark_queries = [
            "短文テスト",
            "Python機械学習の始め方を具体的に教えてください",
            "効率的なチーム開発のベストプラクティスについて、技術的な側面と人的な側面の両方から詳しく説明してください。また、リモートワーク環境での特別な考慮事項も含めて論じてください。"
        ]
        
        return self.batch_mode(benchmark_queries, 'benchmark_results.json')


def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="MurmurNet Boids対応Slotアーキテクチャ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                              # 対話モード
  python main.py --query "質問内容"            # 単発実行
  python main.py --config custom.yaml         # カスタム設定
  python main.py --evaluate                   # 評価重視モード
  python main.py --benchmark                  # ベンチマーク実行
  python main.py --batch queries.txt          # バッチ処理
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='実行する質問（単発モード）'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config_boids_slots.yaml',
        help='設定ファイルパス'
    )
    
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='詳細評価モード'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='ベンチマークモード'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='バッチ処理（ファイルから質問を読み込み）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='結果出力ファイル'
    )
    
    parser.add_argument(
        '--no-boids',
        action='store_true',
        help='Boids統合を無効化'
    )
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    print("🤖 MurmurNet Boids対応Slotアーキテクチャ")
    print("=" * 50)
    
    try:
        # CLI初期化
        cli = MurmurNetCLI(args.config)
        
        # Boids無効化オプション
        if args.no_boids:
            cli.config['use_boids_synthesizer'] = False
            print("⚠️  Boids統合が無効化されました")
        
        # 実行モード選択
        if args.benchmark:
            # ベンチマークモード
            cli.benchmark_mode()
            
        elif args.batch:
            # バッチ処理モード
            if os.path.exists(args.batch):
                with open(args.batch, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f if line.strip()]
                cli.batch_mode(queries, args.output)
            else:
                print(f"✗ バッチファイルが見つかりません: {args.batch}")
                
        elif args.query:
            # 単発実行モード
            result = cli.process_query(args.query, enable_evaluation=args.evaluate)
            
            # 結果保存
            if args.output:
                try:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"📁 結果を保存: {args.output}")
                except Exception as e:
                    print(f"⚠️  結果保存エラー: {e}")
                    
        else:
            # 対話モード（デフォルト）
            cli.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 実行中断されました")
    except Exception as e:
        print(f"\n💥 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
