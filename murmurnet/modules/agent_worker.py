#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MurmurNet エージェントワーカー
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ProcessPoolExecutorによる真の並列エージェント実行
GIL制約を回避し、真の並列処理を実現

作者: Yuhi Sonoki
"""

import logging
import os
import time
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, Callable
import psutil
import re

logger = logging.getLogger('MurmurNet.AgentWorker')

def _clean_response(response: str) -> str:
    """
    モデルの応答を清浄化する
    
    引数:
        response: 生の応答テキスト
        
    戻り値:
        清浄化された応答テキスト
    """
    if not response:
        return ""
    
    # 改行を正規化
    response = response.strip()
    
    # プロンプト指示の除去（緩和版：実際の回答内容を保持）
    prompt_patterns = [
        r'System:\s*.*?\n\nUser:\s*.*?\n\nAssistant:\s*',  # システムプロンプト全体
        r'^回答[：:]\s*',  # 冒頭の「回答:」のみ
        r'^\s*応答[：:]\s*',  # 冒頭の「応答:」のみ
    ]
    
    for pattern in prompt_patterns:
        response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 重複したテキストの除去（同じ文が2回以上続く場合）
    sentences = response.split('。')
    cleaned_sentences = []
    prev_sentence = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence != prev_sentence:
            cleaned_sentences.append(sentence)
            prev_sentence = sentence
    
    response = '。'.join(cleaned_sentences)
    if response and not response.endswith('。'):
        response += '。'
    
    # 長さを制限（1000文字：空レス対策で制限緩和）
    if len(response) > 1000:
        # 文の境界で切断
        sentences = response.split('。')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '。') <= 1000:
                truncated += sentence + '。'
            else:
                break
        response = truncated if truncated else response[:997] + "..."
    
    # 空レス対策：空の場合はデフォルトメッセージ
    if not response or len(response.strip()) < 3:
        return "(no content)"
    
    return response.strip()

def worker_process_function(agent_config: Dict[str, Any],
                          agent_id: int, 
                          input_data: Any,
                          rag_data: Any,
                          context_data: Any) -> Tuple[int, str, Dict[str, Any]]:
    """
    ワーカープロセスで実行されるエージェント関数
    
    各プロセスで独立してLlamaモデルとEmbedderを初期化し、
    真の並列処理を実現
    
    引数:
        agent_config: エージェント設定
        agent_id: エージェントID
        input_data: 入力データ
        rag_data: RAGデータ
        context_data: コンテキストデータ
        
    戻り値:
        (agent_id, response, stats)
    """
    # プロセス内でのログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    worker_logger = logging.getLogger(f'MurmurNet.Worker.{agent_id}')
    
    start_time = time.time()
    process_id = os.getpid()
    thread_id = threading.get_native_id() if hasattr(threading, 'get_native_id') else threading.get_ident()
    
    worker_logger.info(f"🚀 エージェント{agent_id} プロセス{process_id} スレッド{thread_id} 開始")
    
    try:
        # プロセス内でllama-cpp-pythonの可用性を確認
        try:
            from llama_cpp import Llama
            has_llama_cpp = True
        except ImportError as e:
            worker_logger.error(f"❌ プロセス{process_id}でllama-cpp-pythonインポートエラー: {e}")
            has_llama_cpp = False
            
        if not has_llama_cpp:
            return {
                'agent_id': agent_id,
                'success': False,
                'output': "llama-cpp-pythonが利用できません",
                'error': "llama-cpp-pythonインポートエラー",
                'stats': {'process_id': process_id, 'thread_id': thread_id, 'total_time': 0}
            }
        
        # プロセス内でモデルマネージャーを初期化
        from MurmurNet.modules.model_manager import get_singleton_manager
        
        model_manager = get_singleton_manager(agent_config)
        
        # プロンプトの構築
        role = agent_config.get('roles', [{}])[agent_id] if agent_id < len(agent_config.get('roles', [])) else {}
        role_name = role.get('role', f"エージェント{agent_id}")
        
        input_text = str(input_data) if input_data else ""
        rag_text = str(rag_data)[:300] if rag_data else ""
        context_text = str(context_data)[:200] if context_data else ""
        
        # コンテキストの構築
        context = ""
        if rag_text:
            context += f"参考情報: {rag_text}\n"
        if context_text:
            context += f"会話履歴: {context_text}"
        
        # プロンプトマネージャーを使用してHuggingFace形式のプロンプトを構築
        from MurmurNet.modules.prompt_manager import get_prompt_manager
        
        # モデルタイプとモデル名を取得
        model_type = agent_config.get('model_type', 'llama')
        model_name = agent_config.get('huggingface_model_name', '') if model_type == 'huggingface' else ''
        
        if model_type == 'huggingface':
            prompt_manager = get_prompt_manager('huggingface', model_name)
            prompt = prompt_manager.build_prompt(input_text, role_name, context.strip())
        else:
            # Llamaモデルの場合は従来の形式を維持
            system_prompt = f"""あなたは「{role_name}」として行動してください。

以下のルールに従って回答してください：
- 150文字以内で簡潔に回答する
- 質問に直接答える
- 関連情報と会話履歴を参考にする
- 自然で読みやすい日本語で答える"""

            user_prompt = f"""質問: {input_text}

{context}

回答:"""

            prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        # テキスト生成実行
        worker_logger.debug(f"🤖 テキスト生成開始: {len(prompt)}文字")
        generation_start = time.time()
        
        try:
            response = model_manager.generate(
                prompt,
                max_tokens=agent_config.get('max_tokens', 256),
                temperature=agent_config.get('temperature', 0.7)
            )
            generation_time = time.time() - generation_start
            worker_logger.debug(f"✅ テキスト生成完了: {len(response.split())}トークン, {len(response.split())/generation_time:.1f}tok/s")
        except Exception as gen_error:
            generation_time = time.time() - generation_start
            worker_logger.error(f"❌ テキスト生成エラー: {gen_error} ({generation_time:.2f}s)")
            raise gen_error
        
        # 応答の後処理
        response = _clean_response(response)
        
        total_time = time.time() - start_time
        
        # 統計情報
        stats = {
            'process_id': process_id,
            'thread_id': thread_id,
            'total_time': total_time,
            'generation_time': generation_time,
            'response_length': len(response),
            'tokens_estimated': len(response.split()),
            'tokens_per_second': len(response.split()) / generation_time if generation_time > 0 else 0
        }
        
        worker_logger.info(f"✅ エージェント{agent_id} 完了: {total_time:.2f}s, {stats['tokens_per_second']:.1f}tok/s")
        
        return agent_id, response, stats
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
        worker_logger.error(f"❌ エージェント{agent_id} エラー: {error_msg} ({error_time:.2f}s)")
        
        # 詳細なトレースバックを常に出力
        import traceback
        worker_logger.error(f"エラーのトレースバック:\n{traceback.format_exc()}")
        
        stats = {
            'process_id': process_id,
            'thread_id': thread_id,
            'total_time': error_time,
            'generation_time': 0,
            'response_length': 0,
            'tokens_estimated': 0,
            'tokens_per_second': 0,
            'error': error_msg
        }
        
        return agent_id, f"エージェント{agent_id}は応答できませんでした: {e}", stats

def _clean_response(response: str) -> str:
    """
    モデルの応答を清浄化する
    
    引数:
        response: 生の応答テキスト
        
    戻り値:
        清浄化された応答テキスト
    """
    if not response:
        return ""
    
    # 改行を正規化
    response = response.strip()
    
    # プロンプト指示の除去（緩和版：実際の回答内容を保持）
    prompt_patterns = [
        r'System:\s*.*?\n\nUser:\s*.*?\n\nAssistant:\s*',  # システムプロンプト全体
        r'^回答[：:]\s*',  # 冒頭の「回答:」のみ
        r'^\s*応答[：:]\s*',  # 冒頭の「応答:」のみ
    ]
    
    for pattern in prompt_patterns:
        response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 重複したテキストの除去（同じ文が2回以上続く場合）
    sentences = response.split('。')
    cleaned_sentences = []
    prev_sentence = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence != prev_sentence:
            cleaned_sentences.append(sentence)
            prev_sentence = sentence
    
    response = '。'.join(cleaned_sentences)
    if response and not response.endswith('。'):
        response += '。'
    
    # 長さを制限（1000文字：空レス対策で制限緩和）
    if len(response) > 1000:
        # 文の境界で切断
        sentences = response.split('。')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '。') <= 1000:
                truncated += sentence + '。'
            else:
                break
        response = truncated if truncated else response[:997] + "..."

    # 空レス対策：空の場合はデフォルトメッセージ
    if not response or len(response.strip()) < 3:
        return "(no content)"

    return response.strip()

class ProcessParallelAgentWorker:
    """
    ProcessPoolExecutorによる真の並列エージェント実行
    
    ThreadPoolExecutorの代わりにProcessPoolExecutorを使用し、
    GIL制約を回避して真の並列処理を実現
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        ProcessParallelAgentWorkerの初期化
        
        引数:
            config: 設定辞書
        """
        self.config = config
        self.debug = config.get('debug', False)
        self.num_agents = config.get('num_agents', 2)
        
        # CPU最適化設定
        self.cpu_count = psutil.cpu_count(logical=False) or 4  # 物理コア数
        self.max_workers = min(self.num_agents, self.cpu_count)
        
        # ProcessPoolExecutorの初期化
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')  # Windows対応
        )
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"🚀 ProcessParallelAgentWorker初期化: {self.max_workers}プロセス, {self.num_agents}エージェント")
    
    def execute_agents_parallel(self, 
                              input_data: Any,
                              rag_data: Any = None,
                              context_data: Any = None) -> Dict[str, Any]:
        """
        複数エージェントの並列実行
        
        引数:
            input_data: 入力データ
            rag_data: RAGデータ
            context_data: コンテキストデータ
            
        戻り値:
            実行結果辞書
        """
        start_time = time.time()
        
        logger.info(f"🤖 {self.num_agents}エージェント並列実行開始")
        
        # 各エージェントのタスクを送信
        futures = []
        for agent_id in range(self.num_agents):
            future = self.process_pool.submit(
                worker_process_function,
                self.config,
                agent_id,
                input_data,
                rag_data,
                context_data
            )
            futures.append((agent_id, future))        # 結果を収集
        results = {}
        all_stats = []
        
        for agent_id, future in futures:
            try:
                returned_id, response, stats = future.result(timeout=60)  # 60秒タイムアウトに延長
                results[f'agent_{returned_id}_output'] = response
                all_stats.append(stats)
                
                if self.debug:
                    logger.debug(f"✅ エージェント{returned_id} 結果取得: {stats['tokens_per_second']:.1f}tok/s")
                    
            except Exception as e:
                error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
                logger.error(f"❌ エージェント{agent_id} 実行エラー: {error_msg}")
                if self.debug:
                    import traceback
                    logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")
                else:
                    # デバッグモードでない場合も最低限のトレースバックを出力
                    import traceback
                    logger.error(f"エラーのトレースバック: {traceback.format_exc()}")
                    
                results[f'agent_{agent_id}_output'] = f"エージェント{agent_id}は応答できませんでした"
                all_stats.append({
                    'process_id': 0,
                    'thread_id': 0,
                    'total_time': 0,
                    'generation_time': 0,
                    'response_length': 0,
                    'tokens_estimated': 0,
                    'tokens_per_second': 0,
                    'error': error_msg
                })
        
        total_time = time.time() - start_time
        
        # 統計計算
        total_tokens = sum(stat['tokens_estimated'] for stat in all_stats)
        total_generation_time = sum(stat['generation_time'] for stat in all_stats)
        average_tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
        
        # プロセス並列性の確認
        unique_processes = len(set(stat['process_id'] for stat in all_stats if stat['process_id'] > 0))
        
        parallel_stats = {
            'total_execution_time': total_time,
            'total_tokens': total_tokens,
            'average_tokens_per_second': average_tokens_per_second,
            'unique_processes': unique_processes,
            'parallel_efficiency': unique_processes / self.num_agents if self.num_agents > 0 else 0,
            'agent_stats': all_stats
        }
        
        results['parallel_stats'] = parallel_stats
        
        logger.info(f"✅ 並列実行完了: {total_time:.2f}s, {average_tokens_per_second:.1f}tok/s, {unique_processes}プロセス使用")
        
        return results
    
    def execute_agents_sequential(self,
                                input_data: Any,
                                rag_data: Any = None,
                                context_data: Any = None) -> Dict[str, Any]:
        """
        エージェントの逐次実行（比較用）
        
        引数:
            input_data: 入力データ
            rag_data: RAGデータ
            context_data: コンテキストデータ
            
        戻り値:
            実行結果辞書
        """
        start_time = time.time()
        
        logger.info(f"🐌 {self.num_agents}エージェント逐次実行開始")
        
        results = {}
        all_stats = []
        
        for agent_id in range(self.num_agents):
            try:
                returned_id, response, stats = worker_process_function(
                    self.config, agent_id, input_data, rag_data, context_data
                )
                results[f'agent_{returned_id}_output'] = response
                all_stats.append(stats)
                
            except Exception as e:
                logger.error(f"❌ エージェント{agent_id} 逐次実行エラー: {e}")
                results[f'agent_{agent_id}_output'] = f"エージェント{agent_id}は応答できませんでした"
                all_stats.append({
                    'process_id': os.getpid(),
                    'thread_id': threading.get_native_id() if hasattr(threading, 'get_native_id') else threading.get_ident(),
                    'total_time': 0,
                    'generation_time': 0,
                    'response_length': 0,
                    'tokens_estimated': 0,
                    'tokens_per_second': 0,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # 統計計算
        total_tokens = sum(stat['tokens_estimated'] for stat in all_stats)
        total_generation_time = sum(stat['generation_time'] for stat in all_stats)
        average_tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
        
        sequential_stats = {
            'total_execution_time': total_time,
            'total_tokens': total_tokens,
            'average_tokens_per_second': average_tokens_per_second,
            'unique_processes': 1,
            'parallel_efficiency': 0,
            'agent_stats': all_stats
        }
        
        results['sequential_stats'] = sequential_stats
        
        logger.info(f"✅ 逐次実行完了: {total_time:.2f}s, {average_tokens_per_second:.1f}tok/s")
        
        return results
    
    def shutdown(self):
        """
        ProcessPoolExecutorのシャットダウン
        """
        if hasattr(self, 'process_pool') and self.process_pool:
            logger.info("🔄 ProcessPoolExecutorシャットダウン中...")
            try:
                self.process_pool.shutdown(wait=True, timeout=10)
                logger.info("✅ ProcessPoolExecutorシャットダウン完了")
            except Exception as e:
                logger.warning(f"⚠️ ProcessPoolExecutorシャットダウンエラー: {e}")
            finally:
                self.process_pool = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# 便利関数
def create_process_parallel_worker(config: Dict[str, Any]) -> ProcessParallelAgentWorker:
    """
    ProcessParallelAgentWorkerインスタンスを作成
    
    引数:
        config: 設定辞書
        
    戻り値:
        ProcessParallelAgentWorkerインスタンス
    """
    return ProcessParallelAgentWorker(config)

if __name__ == "__main__":
    # テスト用コード
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # テスト設定
    config = {
        'num_agents': 2,
        'model_path': 'path/to/your/model.gguf',  # 実際のパスに変更
        'n_ctx': 2048,
        'n_threads': 4,
        'max_tokens': 100,
        'temperature': 0.7,
        'debug': True,
        'roles': [
            {'role': 'アナリスト', 'system': 'データを分析する専門家です。'},
            {'role': 'アドバイザー', 'system': '助言を提供する専門家です。'}
        ]
    }
    
    print("=== ProcessParallelAgentWorker テスト ===")
    
    with create_process_parallel_worker(config) as worker:
        # 並列実行テスト
        print("\n--- 並列実行 ---")
        parallel_results = worker.execute_agents_parallel(
            input_data="AIについて教えてください",
            rag_data="AIは人工知能の略です",
            context_data="技術的な質問をしています"
        )
        
        print(f"並列実行結果:")
        for key, value in parallel_results.items():
            if 'agent_' in key and 'output' in key:
                print(f"  {key}: {value[:100]}...")
        
        if 'parallel_stats' in parallel_results:
            stats = parallel_results['parallel_stats']
            print(f"  実行時間: {stats['total_execution_time']:.2f}s")
            print(f"  スループット: {stats['average_tokens_per_second']:.1f}tok/s")
            print(f"  使用プロセス数: {stats['unique_processes']}")
            print(f"  並列効率: {stats['parallel_efficiency']:.2f}")
        
        # 逐次実行テスト
        print("\n--- 逐次実行 ---")
        sequential_results = worker.execute_agents_sequential(
            input_data="AIについて教えてください", 
            rag_data="AIは人工知能の略です",
            context_data="技術的な質問をしています"
        )
        
        if 'sequential_stats' in sequential_results:
            stats = sequential_results['sequential_stats']
            print(f"  実行時間: {stats['total_execution_time']:.2f}s")
            print(f"  スループット: {stats['average_tokens_per_second']:.1f}tok/s")
