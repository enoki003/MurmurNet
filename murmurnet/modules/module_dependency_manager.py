#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Dependency Manager モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
モジュール間依存関係の管理と可視化
循環依存の検出と解決支援

設計原則:
- 依存関係の明示化
- 循環依存の防止
- 疎結合の促進
- 依存性注入パターンの活用

作者: Yuhi Sonoki
"""

import logging
import inspect
from typing import Dict, List, Set, Optional, Any, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger('MurmurNet.ModuleDependencyManager')


class DependencyType(Enum):
    """依存関係の種類"""
    COMPOSITION = "composition"          # 合成関係（強い依存）
    AGGREGATION = "aggregation"         # 集約関係（弱い依存）
    ASSOCIATION = "association"         # 関連関係（参照）
    DEPENDENCY = "dependency"           # 依存関係（使用）
    INHERITANCE = "inheritance"         # 継承関係


@dataclass
class ModuleInfo:
    """モジュール情報"""
    name: str
    module_type: str
    file_path: str
    class_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)


@dataclass
class DependencyRelation:
    """依存関係の定義"""
    source: str
    target: str
    dependency_type: DependencyType
    description: str = ""
    is_cyclic: bool = False
    coupling_strength: float = 0.0  # 0.0 (疎結合) ~ 1.0 (密結合)


class ModuleInterface(ABC):
    """モジュールの基底インターフェース"""
    
    @property
    @abstractmethod
    def module_name(self) -> str:
        """モジュール名を返す"""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """依存するモジュール名のリストを返す"""
        pass
    
    @property
    @abstractmethod
    def responsibilities(self) -> List[str]:
        """モジュールの責務リストを返す"""
        pass
    
    @abstractmethod
    def get_interfaces(self) -> List[str]:
        """提供するインターフェースのリストを返す"""
        pass


class DependencyAnalyzer:
    """依存関係の解析と可視化"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependencies: List[DependencyRelation] = []
        self.dependency_graph = nx.DiGraph()
        
    def register_module(self, module_info: ModuleInfo) -> None:
        """モジュール情報を登録"""
        self.modules[module_info.name] = module_info
        self.dependency_graph.add_node(module_info.name)
        
        # 依存関係をグラフに追加
        for dep in module_info.dependencies:
            self.dependency_graph.add_edge(module_info.name, dep)
    
    def add_dependency(self, relation: DependencyRelation) -> None:
        """依存関係を追加"""
        self.dependencies.append(relation)
        
        # グラフを更新
        if relation.source not in self.dependency_graph:
            self.dependency_graph.add_node(relation.source)
        if relation.target not in self.dependency_graph:
            self.dependency_graph.add_node(relation.target)
            
        self.dependency_graph.add_edge(
            relation.source, 
            relation.target,
            weight=relation.coupling_strength,
            type=relation.dependency_type.value
        )
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """循環依存を検出"""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except Exception as e:
            logger.error(f"循環依存検出エラー: {e}")
            return []
    
    def calculate_coupling_metrics(self) -> Dict[str, float]:
        """結合度メトリクスを計算"""
        metrics = {}
        
        for module_name in self.modules.keys():
            # 入次数（このモジュールに依存しているモジュール数）
            in_degree = self.dependency_graph.in_degree(module_name)
            # 出次数（このモジュールが依存しているモジュール数）
            out_degree = self.dependency_graph.out_degree(module_name)
            
            # 結合度スコア（0.0〜1.0）
            total_modules = len(self.modules)
            if total_modules > 1:
                coupling_score = (in_degree + out_degree) / (2 * (total_modules - 1))
            else:
                coupling_score = 0.0
            
            metrics[module_name] = coupling_score
        
        return metrics
    
    def suggest_decoupling_strategies(self) -> List[str]:
        """疎結合化の提案"""
        suggestions = []
        
        # 循環依存の解決
        cycles = self.detect_circular_dependencies()
        if cycles:
            suggestions.append("循環依存が検出されました。以下の解決策を検討してください：")
            for cycle in cycles:
                cycle_str = " -> ".join(cycle + [cycle[0]])
                suggestions.append(f"  - {cycle_str}")
                suggestions.append("    解決策: Dependency Inversion Principle (DIP)の適用")
                suggestions.append("    具体的: インターフェースの導入、依存性注入の活用")
        
        # 高結合度モジュールの特定
        coupling_metrics = self.calculate_coupling_metrics()
        high_coupling_modules = [
            module for module, score in coupling_metrics.items() 
            if score > 0.7
        ]
        
        if high_coupling_modules:
            suggestions.append("\n高結合度モジュールが検出されました：")
            for module in high_coupling_modules:
                suggestions.append(f"  - {module} (結合度: {coupling_metrics[module]:.2f})")
                suggestions.append("    解決策: 責務の分割、Facade パターンの適用")
        
        # BLACKBOARD依存の分析
        blackboard_dependents = [
            module for module, info in self.modules.items()
            if "blackboard" in [dep.lower() for dep in info.dependencies]
        ]
        
        if len(blackboard_dependents) > 3:
            suggestions.append(f"\nBLACKBOARDへの過度な依存が検出されました ({len(blackboard_dependents)}モジュール)：")
            for module in blackboard_dependents:
                suggestions.append(f"  - {module}")
            suggestions.append("    解決策: Communication Interface層の導入")
            suggestions.append("    具体的: Message Broker パターン、Event-driven アーキテクチャ")
        
        return suggestions
    
    def visualize_dependencies(self, output_path: str = "dependency_graph.png") -> None:
        """依存関係グラフを可視化"""
        try:
            plt.figure(figsize=(12, 8))
            
            # レイアウトの計算
            pos = nx.spring_layout(self.dependency_graph, k=2, iterations=50)
            
            # ノードの描画
            node_colors = []
            coupling_metrics = self.calculate_coupling_metrics()
            
            for node in self.dependency_graph.nodes():
                coupling = coupling_metrics.get(node, 0.0)
                if coupling > 0.7:
                    node_colors.append('red')      # 高結合
                elif coupling > 0.4:
                    node_colors.append('orange')   # 中結合
                else:
                    node_colors.append('green')    # 低結合
            
            nx.draw_networkx_nodes(
                self.dependency_graph, pos, 
                node_color=node_colors, 
                node_size=1000,
                alpha=0.8
            )
            
            # エッジの描画
            nx.draw_networkx_edges(
                self.dependency_graph, pos,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                alpha=0.6
            )
            
            # ラベルの描画
            nx.draw_networkx_labels(
                self.dependency_graph, pos,
                font_size=8,
                font_weight='bold'
            )
            
            plt.title("Module Dependency Graph\n(Red: High Coupling, Orange: Medium, Green: Low)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"依存関係グラフを保存しました: {output_path}")
            
        except Exception as e:
            logger.error(f"可視化エラー: {e}")
    
    def generate_report(self) -> str:
        """依存関係分析レポートを生成"""
        report_lines = []
        report_lines.append("=== MurmurNet モジュール依存関係分析レポート ===\n")
        
        # 基本情報
        report_lines.append(f"総モジュール数: {len(self.modules)}")
        report_lines.append(f"依存関係数: {len(self.dependencies)}")
        report_lines.append("")
        
        # 循環依存
        cycles = self.detect_circular_dependencies()
        if cycles:
            report_lines.append("⚠️  循環依存が検出されました:")
            for i, cycle in enumerate(cycles, 1):
                cycle_str = " -> ".join(cycle + [cycle[0]])
                report_lines.append(f"  {i}. {cycle_str}")
        else:
            report_lines.append("✅ 循環依存は検出されませんでした")
        report_lines.append("")
        
        # 結合度メトリクス
        coupling_metrics = self.calculate_coupling_metrics()
        report_lines.append("📊 結合度メトリクス:")
        for module, score in sorted(coupling_metrics.items(), key=lambda x: x[1], reverse=True):
            status = "🔴" if score > 0.7 else "🟡" if score > 0.4 else "🟢"
            report_lines.append(f"  {status} {module}: {score:.3f}")
        report_lines.append("")
        
        # 改善提案
        suggestions = self.suggest_decoupling_strategies()
        if suggestions:
            report_lines.append("💡 改善提案:")
            report_lines.extend([f"  {s}" for s in suggestions])
        report_lines.append("")
        
        # モジュール詳細
        report_lines.append("📋 モジュール詳細:")
        for name, info in self.modules.items():
            report_lines.append(f"\n  📦 {name}")
            if info.responsibilities:
                report_lines.append(f"    責務: {', '.join(info.responsibilities)}")
            if info.dependencies:
                report_lines.append(f"    依存: {', '.join(info.dependencies)}")
            if info.interfaces:
                report_lines.append(f"    IF: {', '.join(info.interfaces)}")
        
        return "\n".join(report_lines)


class DependencyInjectionContainer:
    """依存性注入コンテナ"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._singletons: Dict[str, Any] = {}
        
    def register_service(self, name: str, service: Any) -> None:
        """サービスを登録"""
        self._services[name] = service
        
    def register_factory(self, name: str, factory: callable) -> None:
        """ファクトリーを登録"""
        self._factories[name] = factory
        
    def register_singleton(self, name: str, factory: callable) -> None:
        """シングルトンファクトリーを登録"""
        if name not in self._singletons:
            self._singletons[name] = factory()
            
    def get_service(self, name: str) -> Any:
        """サービスを取得"""
        if name in self._services:
            return self._services[name]
        elif name in self._singletons:
            return self._singletons[name]
        elif name in self._factories:
            return self._factories[name]()
        else:
            raise ValueError(f"Service '{name}' not found")
    
    def inject_dependencies(self, target_class: Type) -> Any:
        """依存関係を注入してインスタンスを作成"""
        init_signature = inspect.signature(target_class.__init__)
        kwargs = {}
        
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
                
            if param_name in self._services or param_name in self._singletons or param_name in self._factories:
                kwargs[param_name] = self.get_service(param_name)
        
        return target_class(**kwargs)


# MurmurNet固有の依存関係定義
def create_murmurnet_dependency_analyzer() -> DependencyAnalyzer:
    """MurmurNet用の依存関係解析器を作成"""
    analyzer = DependencyAnalyzer()
    
    # MurmurNetモジュールの登録
    modules_info = [
        ModuleInfo(
            name="Blackboard",
            module_type="Infrastructure",
            file_path="modules/blackboard.py",
            responsibilities=["共有メモリ管理", "エージェント間通信", "データ永続化"],
            interfaces=["DataStorage", "MessageBroker"]
        ),
        ModuleInfo(
            name="AgentPoolManager",
            module_type="Core",
            file_path="modules/agent_pool.py",
            dependencies=["Blackboard", "ModelFactory", "ConfigManager"],
            responsibilities=["エージェント管理", "並列実行制御", "タスク分散"],
            interfaces=["AgentManager"]
        ),
        ModuleInfo(
            name="SystemCoordinator",
            module_type="Core",
            file_path="modules/system_coordinator.py",
            dependencies=["Blackboard", "AgentPoolManager"],
            responsibilities=["システム調整", "ワークフロー管理", "エラーハンドリング"],
            interfaces=["SystemOrchestrator"]
        ),
        ModuleInfo(
            name="OutputAgent",
            module_type="Core",
            file_path="modules/output_agent.py",
            dependencies=["ModelFactory", "ConfigManager"],
            responsibilities=["最終応答生成", "コンテンツ統合", "品質保証"],
            interfaces=["ResponseGenerator"]
        ),
        ModuleInfo(
            name="RAGRetriever",
            module_type="Service",
            file_path="modules/rag_retriever.py",
            dependencies=["ConfigManager"],
            responsibilities=["知識検索", "コンテキスト拡張", "情報統合"],
            interfaces=["KnowledgeRetriever"]
        ),
        ModuleInfo(
            name="ConfigManager",
            module_type="Infrastructure",
            file_path="modules/config_manager.py",
            responsibilities=["設定管理", "バリデーション", "デフォルト値提供"],
            interfaces=["ConfigProvider"]
        )
    ]
    
    for module_info in modules_info:
        analyzer.register_module(module_info)
    
    # 依存関係の定義
    dependencies = [
        DependencyRelation(
            source="AgentPoolManager",
            target="Blackboard",
            dependency_type=DependencyType.COMPOSITION,
            description="エージェント結果の共有",
            coupling_strength=0.8
        ),
        DependencyRelation(
            source="SystemCoordinator",
            target="Blackboard",
            dependency_type=DependencyType.AGGREGATION,
            description="システム状態の管理",
            coupling_strength=0.7
        ),
        DependencyRelation(
            source="OutputAgent",
            target="ModelFactory",
            dependency_type=DependencyType.DEPENDENCY,
            description="モデルインスタンスの取得",
            coupling_strength=0.5
        )
    ]
    
    for dep in dependencies:
        analyzer.add_dependency(dep)
    
    return analyzer


# 便利な関数
def analyze_murmurnet_dependencies() -> str:
    """MurmurNetの依存関係を分析してレポートを生成"""
    analyzer = create_murmurnet_dependency_analyzer()
    return analyzer.generate_report()


def create_dependency_injection_container() -> DependencyInjectionContainer:
    """MurmurNet用の依存性注入コンテナを作成"""
    container = DependencyInjectionContainer()
    
    # 基本サービスの登録例
    # container.register_singleton('config_manager', lambda: ConfigManager())
    # container.register_factory('blackboard', lambda: Blackboard())
    
    return container
