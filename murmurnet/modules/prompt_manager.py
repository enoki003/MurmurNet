#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
プロンプトテンプレートマネージャー
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
外部YAMLテンプレートの読み込みとJinja2レンダリングを提供

作者: Yuhi Sonoki
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

# Jinja2をオプション依存として扱う
try:
    from jinja2 import Environment, FileSystemLoader, Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Environment = None
    FileSystemLoader = None
    Template = None

logger = logging.getLogger('MurmurNet.PromptManager')


class PromptTemplateManager:
    """
    プロンプトテンプレートの管理クラス
    
    責務:
    - YAMLテンプレートファイルの読み込み
    - Jinja2テンプレートのレンダリング
    - プロンプトのバージョン管理
    - キャッシュ機能
    """
    
    def __init__(self, template_dir: str = None, enable_cache: bool = True):
        """
        プロンプトテンプレートマネージャーの初期化
        
        引数:
            template_dir: テンプレートディレクトリのパス
            enable_cache: キャッシュを有効にするかどうか
        """        # デフォルトのテンプレートディレクトリを設定
        if template_dir is None:
            current_dir = Path(__file__).parent
            # MurmurNet/MurmurNet/modules から MurmurNet/prompts/templates へ
            template_dir = current_dir.parent.parent / "prompts" / "templates"
        
        self.template_dir = Path(template_dir)
        self.enable_cache = enable_cache
        self._template_cache = {}
        self._jinja_env = None
        
        # Jinja2環境の初期化
        if HAS_JINJA2:
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            logger.warning("Jinja2が利用できません。基本的な文字列置換を使用します。")
        
        logger.info(f"プロンプトテンプレートマネージャーを初期化しました: {self.template_dir}")
        logger.debug(f"テンプレートディレクトリ存在チェック: {self.template_dir.exists()}")
        
        # ディレクトリが存在しない場合は作成を試行
        if not self.template_dir.exists():
            logger.warning(f"テンプレートディレクトリが存在しません: {self.template_dir}")
            try:
                self.template_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"テンプレートディレクトリを作成しました: {self.template_dir}")
            except Exception as e:
                logger.error(f"テンプレートディレクトリの作成に失敗: {e}")
        else:
            # 既存ファイルをリストアップ
            if self.template_dir.is_dir():
                template_files = list(self.template_dir.glob("*.yaml"))
                logger.debug(f"利用可能なテンプレートファイル: {[f.name for f in template_files]}")
    
    def load_template(self, template_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        YAMLテンプレートファイルを読み込み
        
        引数:
            template_name: テンプレート名（拡張子なし）
            force_reload: キャッシュを無視して強制再読み込み
            
        戻り値:
            テンプレートデータの辞書
        """
        cache_key = template_name
          # キャッシュチェック
        if not force_reload and self.enable_cache and cache_key in self._template_cache:
            logger.debug(f"キャッシュからテンプレートを取得: {template_name}")
            return self._template_cache[cache_key]
        
        # ファイルパスの構築
        template_file = self.template_dir / f"{template_name}.yaml"
        logger.debug(f"テンプレートファイルパス: {template_file}")
        logger.debug(f"テンプレートファイル存在チェック: {template_file.exists()}")
        
        if not template_file.exists():
            raise FileNotFoundError(f"テンプレートファイルが見つかりません: {template_file}")
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            
            # キャッシュに保存
            if self.enable_cache:
                self._template_cache[cache_key] = template_data
            
            logger.debug(f"テンプレートを読み込みました: {template_name}")
            return template_data
            
        except Exception as e:
            logger.error(f"テンプレート読み込みエラー: {template_file} - {e}")
            raise
    
    def render_prompt(self, template_name: str, template_type: str, language: str = 'japanese', **kwargs) -> str:
        """
        プロンプトテンプレートをレンダリング
        
        引数:
            template_name: テンプレート名
            template_type: テンプレートタイプ（system, user）
            language: 言語（japanese, english）
            **kwargs: テンプレート変数
            
        戻り値:
            レンダリングされたプロンプト文字列
        """
        try:
            logger.debug(f"プロンプトレンダリング開始: {template_name}, {template_type}, {language}")
            logger.debug(f"テンプレートディレクトリ: {self.template_dir}")
            
            # テンプレートデータを読み込み
            template_data = self.load_template(template_name)
            logger.debug(f"テンプレートデータ読み込み完了: {list(template_data.keys())}")
            
            # 言語とタイプに対応するテンプレートを取得
            templates = template_data.get('templates', {})
            logger.debug(f"利用可能な言語: {list(templates.keys())}")
            
            lang_templates = templates.get(language, {})
            logger.debug(f"'{language}'のテンプレートタイプ: {list(lang_templates.keys())}")
            
            if template_type not in lang_templates:
                # フォールバック: 他の言語を試す
                for fallback_lang in ['japanese', 'english']:
                    if fallback_lang in templates and template_type in templates[fallback_lang]:
                        lang_templates = templates[fallback_lang]
                        logger.warning(f"言語 '{language}' が見つからないため、'{fallback_lang}' を使用します")
                        break
                else:
                    available_types = []
                    for lang, tmpl in templates.items():
                        available_types.extend([f"{lang}.{t}" for t in tmpl.keys()])
                    raise ValueError(f"テンプレートタイプ '{template_type}' が見つかりません（言語: {language}）。利用可能: {available_types}")
            
            template_str = lang_templates[template_type]
            logger.debug(f"テンプレート文字列取得完了: {len(template_str)} 文字")
            
            # パラメータのマージ（テンプレートファイルのパラメータ + 実行時パラメータ）
            template_params = template_data.get('parameters', {})
            render_vars = {**template_params, **kwargs}
            logger.debug(f"レンダリング変数: {list(render_vars.keys())}")
            
            # Jinja2レンダリング
            if HAS_JINJA2:
                template = Template(template_str)
                rendered = template.render(**render_vars)
            else:
                # 基本的な文字列置換（Jinja2がない場合）
                rendered = template_str
                for key, value in render_vars.items():
                    if isinstance(value, str):
                        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
            
            logger.debug(f"プロンプトをレンダリングしました: {template_name}/{template_type}/{language}")
            return rendered.strip()
            
        except Exception as e:
            logger.error(f"プロンプトレンダリングエラー: {template_name} - {e}")
            logger.error(f"エラー詳細: {type(e).__name__}: {str(e)}")
            # エラー時のフォールバック
            return f"プロンプトの生成に失敗しました。テンプレート: {template_name}"
    
    def get_agent_personality(self, template_name: str, agent_id: int) -> Dict[str, str]:
        """
        エージェントの個性情報を取得
        
        引数:
            template_name: テンプレート名
            agent_id: エージェントID
            
        戻り値:
            エージェントの個性情報
        """
        try:
            template_data = self.load_template(template_name)
            personalities = template_data.get('personalities', {})
            
            agent_key = f"agent_{agent_id}"
            if agent_key in personalities:
                return personalities[agent_key]
            
            # デフォルトの個性を返す
            return {
                'name': f'エージェント{agent_id + 1}',
                'personality': '客観的で分析的な視点を持つ',
                'expertise': '一般的な知識と分析'
            }
            
        except Exception as e:
            logger.error(f"エージェント個性取得エラー: {e}")
            return {
                'name': f'エージェント{agent_id + 1}',
                'personality': '専門的な知識を持つ',
                'expertise': '問題解決'
            }
    
    def list_templates(self) -> List[str]:
        """
        利用可能なテンプレートの一覧を取得
        
        戻り値:
            テンプレート名のリスト
        """
        if not self.template_dir.exists():
            return []
        
        templates = []
        for file_path in self.template_dir.glob("*.yaml"):
            templates.append(file_path.stem)
        
        return sorted(templates)
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        テンプレートの妥当性を検証
        
        引数:
            template_name: テンプレート名
            
        戻り値:
            検証結果の辞書
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            template_data = self.load_template(template_name)
            
            # 必須フィールドの確認
            required_fields = ['version', 'agent_type', 'templates']
            for field in required_fields:
                if field not in template_data:
                    result['errors'].append(f"必須フィールドが不足: {field}")
            
            # テンプレート構造の確認
            templates = template_data.get('templates', {})
            if not templates:
                result['errors'].append("テンプレートが定義されていません")
            
            for lang, lang_templates in templates.items():
                if not isinstance(lang_templates, dict):
                    result['errors'].append(f"言語 '{lang}' のテンプレート形式が不正です")
                    continue
                
                # systemとuserテンプレートの存在確認
                if 'system' not in lang_templates:
                    result['warnings'].append(f"言語 '{lang}' にsystemテンプレートがありません")
                if 'user' not in lang_templates:
                    result['warnings'].append(f"言語 '{lang}' にuserテンプレートがありません")
            
            # バージョン情報
            result['info']['version'] = template_data.get('version', 'unknown')
            result['info']['agent_type'] = template_data.get('agent_type', 'unknown')
            
            # エラーがなければ有効
            result['valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"テンプレート読み込みエラー: {e}")
        
        return result
    
    def clear_cache(self):
        """テンプレートキャッシュをクリア"""
        self._template_cache.clear()
        logger.info("テンプレートキャッシュをクリアしました")


# シングルトンインスタンス
_prompt_manager = None

def get_prompt_manager(template_dir: str = None) -> PromptTemplateManager:
    """
    プロンプトマネージャーのシングルトンインスタンスを取得
    
    引数:
        template_dir: テンプレートディレクトリ（初回のみ使用）
        
    戻り値:
        プロンプトマネージャーインスタンス
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptTemplateManager(template_dir)
    return _prompt_manager
