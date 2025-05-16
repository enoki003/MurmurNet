#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Opinion Vectorizer モジュール
~~~~~~~~~~~~~~~~~~~
エージェントの発言テキストを意見ベクトルとして
数値表現するモジュール

作者: Yuhi Sonoki
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import os
import re

class OpinionVectorizer:
    """
    テキスト意見をベクトル表現に変換するモジュール
    - 事前学習済みエンベディングモデルを使用
    - または簡易キーワード抽出によるフォールバック機能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初期化
        
        引数:
            config: 設定辞書
        """
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        
        # エンベディングモデル
        self.embedding_model = None
        self.embedding_dim = 384  # デフォルト次元数
        
        # 再帰防止フラグ
        self.is_processing = False
        
        # モデルの初期化
        self._initialize_model()
        
        if self.debug:
            print(f"OpinionVectorizer初期化完了: モデル={self.model_type}")
    
    def _initialize_model(self):
        """
        ベクトル化モデルの初期化（内部メソッド）
        """
        self.model_type = self.config.get('vectorizer_type', 'sentence_transformer')
        
        # 高精度モード: sentence-transformersモデル
        if self.model_type == 'sentence_transformer':
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                
                # デバイスの設定（CUDAが利用可能ならGPU、なければCPU）
                device = self.config.get('device', None)
                if device is None:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # モデルの指定
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                
                if self.debug:
                    print(f"Use pytorch device_name: {device}")
                    print(f"Load pretrained SentenceTransformer: {model_name}")
                
                # モデルの読み込み
                self.embedding_model = SentenceTransformer(model_name, device=device)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                
            except ImportError:
                if self.debug:
                    print("警告: sentence-transformers未インストール。簡易ベクトル化を使用します。")
                self.model_type = 'keyword'
                self._initialize_keyword_model()
                
            except Exception as e:
                if self.debug:
                    print(f"モデル読み込みエラー: {e}")
                    import traceback
                    traceback.print_exc()
                self.model_type = 'keyword'
                self._initialize_keyword_model()
                
        # 簡易モード: キーワードベースのベクトル化
        elif self.model_type == 'keyword':
            self._initialize_keyword_model()
            
        else:
            if self.debug:
                print(f"警告: 未知のベクトル化タイプ {self.model_type}。キーワードモードを使用します。")
            self.model_type = 'keyword'
            self._initialize_keyword_model()
    
    def _initialize_keyword_model(self):
        """
        キーワードベースのベクトル化初期化（内部メソッド）
        """
        # 語彙のファイルパス
        vocab_path = self.config.get('vocab_path', 
                                    os.path.join(os.path.dirname(__file__), 
                                                '../../data/vocab.txt'))
        
        # 語彙の読み込み（または生成）
        self.vocabulary = []
        try:
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.vocabulary = [line.strip() for line in f if line.strip()]
            else:
                # デフォルト語彙（日本語と英語の一般的な単語）
                self.vocabulary = ['教育', '技術', '社会', '文化', '歴史', 'AI', '情報', '科学',
                                  '健康', '政治', '経済', '環境', '人間', '芸術', '言語', '宗教',
                                  'education', 'technology', 'society', 'culture', 'history',
                                  'information', 'science', 'health', 'politics', 'economics',
                                  'environment', 'human', 'art', 'language', 'religion']
                # 語彙を保存
                os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.vocabulary))
                    
        except Exception as e:
            if self.debug:
                print(f"語彙ロードエラー: {e}")
                import traceback
                traceback.print_exc()
            
            # 最小限の語彙
            self.vocabulary = ['教育', '技術', '社会', '文化', '歴史', 'AI', '情報', '科学']
        
        # 語彙サイズを設定
        self.embedding_dim = len(self.vocabulary)
        
        if self.debug:
            print(f"キーワードベクトル化初期化: 語彙数={self.embedding_dim}")
    
    def vectorize(self, text: str) -> np.ndarray:
        """
        テキストを意見ベクトルに変換
        
        引数:
            text: 変換するテキスト
            
        戻り値:
            ベクトル表現（numpy配列）
        """
        # 再帰呼び出しチェック
        if self.is_processing:
            # 再帰検出時はフォールバックベクトルを返す
            if self.debug:
                print("警告: ベクトル化の再帰呼び出しを検出。フォールバックベクトルを返します。")
            return np.random.rand(self.embedding_dim)
            
        # 処理開始
        self.is_processing = True
        
        try:
            # テキスト正規化
            text = self._normalize_text(text)
            
            # 高精度モード: sentence-transformers
            if self.model_type == 'sentence_transformer' and self.embedding_model is not None:
                try:
                    # モデルによるエンベディング
                    embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                    return np.array(embedding)
                except Exception as e:
                    if self.debug:
                        print(f"エンベディングエラー: {e}")
                        import traceback
                        traceback.print_exc()
                    # エラー時はキーワードモードにフォールバック
                    return self._vectorize_by_keywords(text)
            
            # 簡易モード: キーワードベース
            return self._vectorize_by_keywords(text)
        finally:
            # 処理完了
            self.is_processing = False
    
    def vectorize_text(self, text: str) -> np.ndarray:
        """
        テキストを意見ベクトルに変換（vectorizeのエイリアス）
        
        引数:
            text: 変換するテキスト
            
        戻り値:
            ベクトル表現（numpy配列）
        """
        # 入力が辞書の場合はテキスト抽出を試みる
        if isinstance(text, dict):
            if 'text' in text:
                text = text['text']
            elif 'content' in text:
                text = text['content']
            else:
                # テキスト部分が見つからない場合は空文字列を使用
                text = str(text)
        
        # 文字列でない場合は文字列に変換
        if not isinstance(text, str):
            text = str(text)
            
        return self.vectorize(text)
    
    def _vectorize_by_keywords(self, text: str) -> np.ndarray:
        """
        キーワードベースのベクトル化（内部メソッド）
        
        引数:
            text: 変換するテキスト
            
        戻り値:
            ベクトル表現（numpy配列）
        """
        # ゼロベクトルで初期化
        vector = np.zeros(self.embedding_dim)
        
        # 各単語の出現をチェック
        for i, word in enumerate(self.vocabulary):
            # 単語が含まれていればその位置を1に
            if word.lower() in text.lower():
                vector[i] = 1.0
        
        # ベクトルの正規化（ゼロベクトルでない場合）
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _normalize_text(self, text: str) -> str:
        """
        テキストの正規化（内部メソッド）
        
        引数:
            text: 元のテキスト
            
        戻り値:
            正規化されたテキスト
        """
        if text is None:
            return ""
            
        # 空白の正規化
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 記号の簡易削除
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        2つのベクトル間の類似度を計算
        
        引数:
            vec1: 1つ目のベクトル
            vec2: 2つ目のベクトル
            
        戻り値:
            コサイン類似度（0～1）
        """
        # コサイン類似度を計算
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # ゼロベクトルチェック
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        2つのベクトル間の距離を計算
        
        引数:
            vec1: 1つ目のベクトル
            vec2: 2つ目のベクトル
            
        戻り値:
            コサイン距離（0～2）
        """
        # 類似度を距離に変換
        similarity = self.compute_similarity(vec1, vec2)
        distance = 1.0 - similarity
        
        return distance
    
    def extend_vocabulary(self, words: List[str]) -> None:
        """
        キーワードモード用の語彙を拡張
        
        引数:
            words: 追加する単語リスト
        """
        if self.model_type != 'keyword':
            return
            
        # 新しい単語を追加
        new_words = [w for w in words if w not in self.vocabulary]
        if not new_words:
            return
            
        # 語彙拡張
        self.vocabulary.extend(new_words)
        old_dim = self.embedding_dim
        self.embedding_dim = len(self.vocabulary)
        
        if self.debug:
            print(f"語彙を拡張: {old_dim} → {self.embedding_dim}")
            print(f"追加単語: {new_words}")
            
        # 語彙ファイルを更新
        vocab_path = self.config.get('vocab_path', 
                                    os.path.join(os.path.dirname(__file__), 
                                                '../../data/vocab.txt'))
        try:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            with open(vocab_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.vocabulary))
        except Exception as e:
            if self.debug:
                print(f"語彙保存エラー: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        埋め込みの次元数を取得
        
        戻り値:
            埋め込みベクトルの次元数
        """
        return self.embedding_dim