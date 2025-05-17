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
                print(f"語彙ファイル処理エラー: {e}")
            # 最小限の語彙セットをメモリ上に作成
            self.vocabulary = ['教育', '技術', '科学', '社会', '文化', '情報', '健康', '環境']
            
        # ベクトル次元数を語彙サイズにセット
        self.embedding_dim = len(self.vocabulary)
        
        # 必要に応じて形態素解析器を初期化
        self.tokenizer = None
        self.use_morphological = self.config.get('use_morphological', False)
        if self.use_morphological:
            try:
                import MeCab
                self.tokenizer = MeCab.Tagger("-Owakati")
                if self.debug:
                    print("MeCab形態素解析器を初期化しました")
            except ImportError:
                if self.debug:
                    print("警告: MeCabがインストールされていません。簡易分割を使用します。")
                self.use_morphological = False
        
        if self.debug:
            print(f"キーワードベクトル化器を初期化: 語彙数={len(self.vocabulary)}, 形態素解析={self.use_morphological}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        テキストからキーワードを抽出（内部メソッド）
        
        引数:
            text: 入力テキスト
            
        戻り値:
            抽出されたキーワードのリスト
        """
        if not text:
            return []
            
        # テキストを正規化
        normalized_text = self._normalize_text(text)
        
        # 形態素解析使用の場合
        if self.use_morphological and self.tokenizer:
            try:
                # MeCabで分かち書き
                tokens = self.tokenizer.parse(normalized_text).split()
                # 名詞・動詞・形容詞を抽出（簡易実装）
                keywords = [token for token in tokens if len(token) > 1]
                if self.debug and len(keywords) < 2:
                    print(f"警告: キーワードが少なすぎます: {keywords}")
                return keywords[:10]  # 最大10キーワードに制限
            except Exception as e:
                if self.debug:
                    print(f"形態素解析エラー: {e}")
                # フォールバック
        
        # 簡易キーワード抽出（形態素解析なしの場合）
        # 単語境界で分割し、頻出単語・ストップワードを除去
        stop_words = set(['は', 'が', 'の', 'に', 'を', 'と', 'です', 'ます', 'た', 'した', 'て',
                          'a', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'])
        
        # 複数の区切り文字で分割
        tokens = re.findall(r'\w+', normalized_text)
        tokens = [t for t in tokens if len(t) > 1 and t.lower() not in stop_words]
        
        # 簡易頻度カウント（文内）
        word_count = {}
        for token in tokens:
            word_count[token] = word_count.get(token, 0) + 1
            
        # 頻度順で上位キーワードを取得
        keywords = sorted(word_count.keys(), key=lambda x: word_count[x], reverse=True)
        return keywords[:10]  # 最大10キーワードに制限
    
    def _normalize_text(self, text: str) -> str:
        """
        テキスト正規化（内部メソッド）
        
        引数:
            text: 入力テキスト
            
        戻り値:
            正規化されたテキスト
        """
        if not text:
            return ""
            
        # 改行・タブを空白に置換
        text = re.sub(r'[\n\t\r]', ' ', text)
        # 連続する空白を1つに
        text = re.sub(r'\s+', ' ', text)
        # 英数字以外の特殊文字を削除（日本語はそのまま）
        text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', text)
        return text.strip()
    
    def vectorize(self, text: str) -> np.ndarray:
        """
        テキストをベクトル化
        
        引数:
            text: 入力テキスト
            
        戻り値:
            ベクトル表現（numpy配列）
        """
        if not text:
            return np.zeros(self.embedding_dim)
            
        # 再帰呼び出し防止
        if self.is_processing:
            if self.debug:
                print("警告: vectorizeの再帰呼び出しを検出。処理をスキップします。")
            return np.zeros(self.embedding_dim)
        
        self.is_processing = True
        try:
            if self.model_type == 'sentence_transformer' and self.embedding_model:
                try:
                    # Sentence Transformerでベクトル化
                    vec = self.embedding_model.encode([text])[0]
                    return vec
                except Exception as e:
                    if self.debug:
                        print(f"SentenceTransformerエラー: {e}")
                    # エラー時はキーワードモードにフォールバック
                    return self._vectorize_with_keywords(text)
            else:
                # キーワードベースのベクトル化
                return self._vectorize_with_keywords(text)
        finally:
            self.is_processing = False
    
    def _vectorize_with_keywords(self, text: str) -> np.ndarray:
        """
        キーワードベースのベクトル化（内部メソッド）
        
        引数:
            text: 入力テキスト
            
        戻り値:
            ベクトル表現（numpy配列）
        """
        # キーワード抽出
        keywords = self._extract_keywords(text)
        
        if not keywords:
            if self.debug:
                print("警告: キーワードが抽出できませんでした")
            return np.zeros(self.embedding_dim)
            
        # キーワードをone-hotエンコーディング
        vector = np.zeros(self.embedding_dim)
        
        # 各キーワードについて
        for keyword in keywords:
            # 完全一致する語彙があればその位置を1に
            if keyword in self.vocabulary:
                idx = self.vocabulary.index(keyword)
                vector[idx] = 1.0
            else:
                # 部分一致する語彙を探す
                for i, vocab in enumerate(self.vocabulary):
                    # 部分文字列マッチングの簡易実装
                    if keyword in vocab or vocab in keyword:
                        vector[i] += 0.5  # 完全一致より弱い重みを付ける
          # ベクトルの正規化（L2ノルム）
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
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
                
        return self.vectorize(text)
    
    def expand_vocabulary(self, new_words: List[str]) -> bool:
        """
        新しい単語を語彙に追加
        
        引数:
            new_words: 追加する単語のリスト
            
        戻り値:
            成功したかどうか
        """
        if not new_words or self.model_type != 'keyword':
            return False
            
        try:
            old_dim = self.embedding_dim
            added = 0
            
            # 新しい単語を語彙に追加（重複を除く）
            for word in new_words:
                if word and word not in self.vocabulary:
                    self.vocabulary.append(word)
                    added += 1
                    
            # 語彙サイズが変わった場合、次元数を更新
            if added > 0:
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
                
                return True
                
            return False
        except Exception as e:
            if self.debug:
                print(f"語彙拡張エラー: {e}")
            return False
    
    def get_embedding_dimension(self) -> int:
        """
        埋め込みの次元数を取得
        
        戻り値:
            埋め込みベクトルの次元数
        """
        return self.embedding_dim