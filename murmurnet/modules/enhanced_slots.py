#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Slot Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
深い議論と明確な役割分化を実現する改良版Slot実装

主要改善:
- 各Slotの専門性強化
- 構造化Blackboard連携
- RAG統合
- 責任明確な統合

作者: Yuhi Sonoki
"""

import logging
import time
from typing import Any, Dict, List, Optional
from .slots import BaseSlot
from .structured_blackboard import AgentRole
from .slot_blackboard import SlotBlackboard, SlotEntry

logger = logging.getLogger(__name__)

class EnhancedReformulatorSlot(BaseSlot):
    """問題再定式化Slot（分析的思考強化版）"""
    
    def get_role_description(self) -> str:
        return "問題の深層分析・再定式化"
    
    def build_system_prompt(self) -> str:
        return (
            "あなたは問題分析の専門家です。ユーザーの質問を深く分析し、"
            "隠れた前提や多角的な視点を明らかにしてください。"
        )
    
    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        return self._build_prompt(bb, user_input)
    
    def _get_agent_role(self) -> AgentRole:
        return AgentRole.REFORMULATOR
    
    def _build_prompt(self, bb: SlotBlackboard, user_input: str, context: str = "") -> str:
        """分析的な問題再定式化プロンプト"""
        base_prompt = f"""ユーザーの質問を深く分析し、隠れた前提や多角的な視点を明らかにしてください。

【ユーザーの質問】
{user_input}

【分析指針】
1. 質問の背景にある真の問題は何か？
2. どのような前提条件が含まれているか？
3. 他にどのような解釈や視点が可能か？
4. 解決すべき本質的な課題は何か？

【出力形式】
「この質問の核心は...です。特に...の観点から考えると、...という課題があります。また、...という前提を見直すことで、新たな解決策が見えてきます。」

150文字以内で具体的に分析してください："""

        return base_prompt
    
    def _build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> str:
        """他Slotの意見を参照した相互議論プロンプト"""
        opinions_text = ""
        if other_opinions:
            opinions_text = "【他のエージェントの意見】\n"
            for op in other_opinions:
                opinions_text += f"- {op['role']}: {op['content'][:100]}...\n"
        
        cross_ref_prompt = f"""他のエージェントの意見を踏まえ、あなたの分析を深化・修正してください。

【元の質問】
{user_input}

{opinions_text}

【相互参照の指針】
1. 他の意見で見落とされている側面はないか？
2. 既存の分析をどう補強・修正できるか？
3. 異なる視点からの新しい洞察は？
4. 他の意見との統合可能性は？

【出力形式】
「{opinions_text.split(':')[0] if opinions_text else '他のエージェント'}の指摘を受け、...という新たな視点を追加します。特に...の点で、私の分析を...のように修正します。」

他の意見を明示的に参照しながら150文字以内で："""

        return cross_ref_prompt
    
    def execute_cross_reference(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """相互参照モードでの実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_cross_reference_prompt(bb, user_input, other_opinions)
            
            if self.debug:
                print(f"\n🔄 {self.name} (相互参照モード) ---")
                print(f"参照意見数: {len(other_opinions)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # メタデータに相互参照情報を追加
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 2,  # 相互参照フェーズ
                "referenced_opinions": len(other_opinions),
                "cross_reference_mode": True
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

class EnhancedCriticSlot(BaseSlot):
    """批判的分析Slot（建設的批判強化版）"""
    
    def get_role_description(self) -> str:
        return "建設的批判・リスク分析"
    
    def build_system_prompt(self) -> str:
        return (
            "あなたは建設的批判の専門家です。問題や既存の意見に対して、"
            "見落とされがちなリスクや課題を指摘し、改善案を提示してください。"
        )
    
    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        return self._build_prompt(bb, user_input)
    
    def _get_agent_role(self) -> AgentRole:
        return AgentRole.CRITIC
    
    def execute_cross_reference(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """相互参照モードでの実行（CriticSlot版）"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_cross_reference_prompt(bb, user_input, other_opinions)
            
            if self.debug:
                print(f"\n🔄 {self.name} (相互参照モード) ---")
                print(f"参照意見数: {len(other_opinions)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 2,
                "referenced_opinions": len(other_opinions),
                "cross_reference_mode": True
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> str:
        """他Slotの意見を参照した批判的議論プロンプト"""
        opinions_text = ""
        if other_opinions:
            opinions_text = "【他のエージェントの意見】\n"
            for op in other_opinions:
                opinions_text += f"- {op['role']}: {op['content'][:100]}...\n"
        
        cross_ref_prompt = f"""他のエージェントの意見を踏まえ、批判的な視点から議論を深化させてください。

【元の質問】
{user_input}

{opinions_text}

【批判的相互参照の指針】
1. 他の意見の論理的矛盾や見落としは何か？
2. 「楽観的すぎる」部分はないか？
3. 実現可能性の課題は？
4. 他の意見を統合する際のリスクは？

【出力形式】
「{opinions_text.split(':')[0] if opinions_text else '他のエージェント'}の意見について、...という課題があります。特に...の点で、...というリスクを考慮すべきです。より堅実な案として...を提案します。」

他の意見を明示的に引用しながら150文字以内で："""

        return cross_ref_prompt
    
    def _build_prompt(self, bb: SlotBlackboard, user_input: str, context: str = "") -> str:
        """建設的批判プロンプト"""
        
        # 他の意見を取得
        other_opinions = []
        if hasattr(bb, 'get_structured_context'):
            ctx = bb.get_structured_context()
            other_opinions = ctx.get('recent_opinions', [])
        
        opinions_text = ""
        if other_opinions:
            opinions_text = "【既存の意見】\n"
            for op in other_opinions[:3]:  # 最新3件
                opinions_text += f"- {op['role']}: {op['content'][:80]}...\n"
        
        base_prompt = f"""あなたは建設的批判の専門家です。問題や既存の意見に対して、見落とされがちなリスクや課題を指摘し、改善案を提示してください。

【対象の質問】
{user_input}

{opinions_text}

【批判的分析の観点】
1. 見落とされているリスクや問題は何か？
2. 論理的な矛盾や不整合はないか？
3. 実現可能性の課題は何か？
4. より良いアプローチはないか？

【出力形式】
「...という点で課題があります。特に...のリスクを考慮すべきです。代替案として...を検討することで、より堅実な解決が期待できます。」

150文字以内で建設的に批判・改善案を提示してください："""

        return base_prompt
    
    def _build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> str:
        """他Slotの意見を参照した相互議論プロンプト"""
        opinions_text = ""
        if other_opinions:
            opinions_text = "【他のエージェントの意見】\n"
            for op in other_opinions:
                opinions_text += f"- {op['role']}: {op['content'][:100]}...\n"
        
        cross_ref_prompt = f"""他のエージェントの意見を踏まえ、あなたの批判的分析を深化・拡張してください。

【元の質問】
{user_input}

{opinions_text}

【相互参照の指針】
1. 他の意見にはどのような盲点やリスクがあるか？
2. 楽観的すぎる見通しはないか？
3. 実現における具体的な障害は何か？
4. より慎重なアプローチは？

【出力形式】
「{opinions_text.split(':')[0] if opinions_text else '他のエージェント'}の意見について、...という点でリスクがあります。特に...を考慮すると、...のような課題が予想されます。」

他の意見を明示的に参照しながら150文字以内で："""

        return cross_ref_prompt
    
    def execute_cross_reference(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """相互参照モードでの実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_cross_reference_prompt(bb, user_input, other_opinions)
            
            if self.debug:
                print(f"\n🔄 {self.name} (相互参照モード) ---")
                print(f"参照意見数: {len(other_opinions)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # メタデータに相互参照情報を追加
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 2,  # 相互参照フェーズ
                "referenced_opinions": len(other_opinions),
                "cross_reference_mode": True
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

class EnhancedSupporterSlot(BaseSlot):
    """支持的拡張Slot（創造的発展強化版）"""
    
    def get_role_description(self) -> str:
        return "創造的発展・機会創出"
    
    def build_system_prompt(self) -> str:
        return (
            "あなたは創造的発展の専門家です。既存のアイデアを発展させ、"
            "新しい可能性や機会を見出してください。"
        )
    
    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        return self._build_prompt(bb, user_input)
    
    def _get_agent_role(self) -> AgentRole:
        return AgentRole.SUPPORTER
    
    def execute_cross_reference(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """相互参照モードでの実行（SupporterSlot版）"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_cross_reference_prompt(bb, user_input, other_opinions)
            
            if self.debug:
                print(f"\n🔄 {self.name} (相互参照モード) ---")
                print(f"参照意見数: {len(other_opinions)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 2,
                "referenced_opinions": len(other_opinions),
                "cross_reference_mode": True
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> str:
        """他Slotの意見を参照した創造的発展プロンプト"""
        opinions_text = ""
        if other_opinions:
            opinions_text = "【他のエージェントの意見】\n"
            for op in other_opinions:
                opinions_text += f"- {op['role']}: {op['content'][:100]}...\n"
        
        cross_ref_prompt = f"""他のエージェントの意見を統合し、創造的に発展させてください。

【元の質問】
{user_input}

{opinions_text}

【創造的統合の指針】
1. 他の意見をどう組み合わせれば新しい価値が生まれるか？
2. 批判的な指摘をポジティブな機会に変換できないか？
3. 異分野の知見や別の視点を加えるとどうなるか？
4. より大きな可能性やスケールアップの余地は？

【出力形式】
「{opinions_text.split(':')[0] if opinions_text else '他のエージェント'}の指摘を受け、...という新たな可能性が見えてきます。特に...と...を組み合わせることで、...という革新的なアプローチが可能です。」

他の意見を明示的に統合しながら150文字以内で："""

        return cross_ref_prompt
    
    def _build_prompt(self, bb: SlotBlackboard, user_input: str, context: str = "") -> str:
        """創造的発展プロンプト"""
        
        # 既存の分析や批判を取得
        context_info = ""
        if hasattr(bb, 'get_structured_context'):
            ctx = bb.get_structured_context()
            recent_opinions = ctx.get('recent_opinions', [])
            if recent_opinions:
                context_info = "【これまでの議論】\n"
                for op in recent_opinions[:3]:
                    context_info += f"- {op['role']}: {op['content'][:80]}...\n"
        
        base_prompt = f"""あなたは創造的発展の専門家です。既存のアイデアを発展させ、新しい可能性や機会を見出してください。

【元の質問】
{user_input}

{context_info}

【創造的発展の観点】
1. これまでの議論をどう発展させられるか？
2. 新しい機会や可能性は何か？
3. 異なる分野の知見を応用できないか？
4. より大きな価値を生み出すには？

【出力形式】
「...のアイデアを発展させると、...という新しい可能性があります。特に...の観点から考えると、...によってより大きな成果が期待できます。」

150文字以内で創造的に発展・拡張してください："""

        return base_prompt
    
    def _build_cross_reference_prompt(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]]) -> str:
        """他Slotの意見を参照した相互議論プロンプト"""
        opinions_text = ""
        if other_opinions:
            opinions_text = "【他のエージェントの意見】\n"
            for op in other_opinions:
                opinions_text += f"- {op['role']}: {op['content'][:100]}...\n"
        
        cross_ref_prompt = f"""他のエージェントの意見を踏まえ、あなたの創造的発展アイデアを拡張してください。

【元の質問】
{user_input}

{opinions_text}

【相互参照の指針】
1. 他の意見をどう創造的に発展させられるか？
2. 批判的視点も含めて、どんな新しい可能性が見えるか？
3. 異なる視点を統合した革新的アイデアは？
4. より大きな成果を生むシナジーは？

【出力形式】
「{opinions_text.split(':')[0] if opinions_text else '他のエージェント'}の指摘を踏まえ、...という新しい可能性を提案します。特に...と...を組み合わせることで、...のような革新的な成果が期待できます。」

他の意見を明示的に参照しながら150文字以内で："""

        return cross_ref_prompt
    
    def execute_cross_reference(self, bb: SlotBlackboard, user_input: str, other_opinions: List[Dict[str, Any]], embedder=None) -> Optional[SlotEntry]:
        """相互参照モードでの実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_cross_reference_prompt(bb, user_input, other_opinions)
            
            if self.debug:
                print(f"\n🔄 {self.name} (相互参照モード) ---")
                print(f"参照意見数: {len(other_opinions)}")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # メタデータに相互参照情報を追加
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 2,  # 相互参照フェーズ
                "referenced_opinions": len(other_opinions),
                "cross_reference_mode": True
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

class EnhancedSynthesizerSlot(BaseSlot):
    """統合Slot（責任明確な統合強化版）"""
    
    def get_role_description(self) -> str:
        return "多視点統合・責任明確な結論"
    
    def build_system_prompt(self) -> str:
        return (
            "あなたは統合責任者です。各専門家の意見を統合し、"
            "明確で実行可能な結論を導いてください。"
        )
    
    def build_user_prompt(self, bb: SlotBlackboard, user_input: str) -> str:
        return self._build_prompt(bb, user_input)
    
    def _get_agent_role(self) -> AgentRole:
        return AgentRole.SYNTHESIZER
    
    def execute_synthesis_with_citations(self, bb: SlotBlackboard, user_input: str, discussion_history: Dict[str, Any], embedder=None) -> Optional[SlotEntry]:
        """引用付き統合モードでの実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_synthesis_prompt(bb, user_input, discussion_history)
            
            if self.debug:
                print(f"\n📋 {self.name} (引用付き統合モード) ---")
                phases = discussion_history.get('phases', {})
                print(f"統合対象: {len(phases)}フェーズ, {discussion_history.get('total_entries', 0)}意見")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # 統合品質の評価
            quality_score = self._evaluate_synthesis_quality(response, discussion_history)
            
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 3,
                "synthesis_mode": True,
                "discussion_phases": len(discussion_history.get('phases', {})),
                "quality_score": quality_score,
                "total_opinions_synthesized": discussion_history.get('total_entries', 0)
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _build_synthesis_prompt(self, bb: SlotBlackboard, user_input: str, discussion_history: Dict[str, Any]) -> str:
        """引用付き統合プロンプト"""
        timeline = discussion_history.get('timeline', [])
        phases = discussion_history.get('phases', {})
        
        # フェーズ別に議論を整理
        discussion_summary = ""
        citation_map = {}  # 引用用マッピング
        
        for phase_num, phase_entries in phases.items():
            phase_name = phase_entries[0].get('phase_name', f'フェーズ{phase_num}') if phase_entries else f'フェーズ{phase_num}'
            discussion_summary += f"\n【{phase_name}】\n"
            
            for entry in phase_entries:
                role = entry['role']
                content = entry['content'][:100]
                discussion_summary += f"- {role}: {content}...\n"
                
                # 引用用の短縮形を作成
                if role not in citation_map:
                    citation_map[role] = []
                citation_map[role].append(content[:50])
        
        # 合意・対立の分析
        collaboration_metrics = bb.calculate_collaboration_metrics()
        consensus_score = collaboration_metrics.get('consensus_score', 0.5)
        conflict_indicators = collaboration_metrics.get('conflict_indicators', 0)
        
        synthesis_prompt = f"""これまでの議論を統合し、引用付きで明確な結論を提示してください。

【元の質問】
{user_input}

【議論の流れ】{discussion_summary}

【議論の状況】
- コンセンサス度: {consensus_score:.2f}
- 対立指標: {conflict_indicators}件
- 参加者: {len(citation_map)}名

【統合の責任】
1. 各専門家の核心的な洞察を正確に引用する
2. 合意できる部分と対立する部分を明確化する
3. 対立がある場合は、なぜその判断をしたか理由を述べる
4. 具体的で実行可能な最終結論を責任を持って提示する

【引用の要求】
- 必ず「○○は『...』と指摘したように」の形で引用する
- 意見の対立があれば「一方で○○は...だが、△△は...」と明記する
- 最終判断の理由を「...の理由から、...を結論とする」と明示する

【出力形式】
「問題分析者の『...』との指摘、批判的視点からの『...』という懸念、そして創造的発展として『...』という提案を総合すると、最適解は...です。特に...の理由から、...をお勧めします。この判断の責任は私が負います。」

200文字以内で責任と引用を明確にして統合してください："""

        return synthesis_prompt
    
    def _evaluate_synthesis_quality(self, response: str, discussion_history: Dict[str, Any]) -> float:
        """統合品質の評価"""
        quality_score = 0.0
        
        # 引用の存在チェック（0.3点）
        citation_keywords = ['指摘したように', 'という意見', 'は述べた', 'によると']
        if any(keyword in response for keyword in citation_keywords):
            quality_score += 0.3
        
        # 責任明示の存在チェック（0.2点）
        responsibility_keywords = ['責任', '判断', '結論', 'お勧め']
        if any(keyword in response for keyword in responsibility_keywords):
            quality_score += 0.2
        
        # 統合性の評価（0.3点）
        total_entries = discussion_history.get('total_entries', 0)
        if total_entries >= 3:  # 3つ以上の意見を統合
            quality_score += 0.3
        elif total_entries >= 2:
            quality_score += 0.2
        
        # 長さの適切性（0.2点）
        if 100 <= len(response) <= 250:
            quality_score += 0.2
        elif 50 <= len(response) <= 300:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _build_prompt(self, bb: SlotBlackboard, user_input: str, context: str = "") -> str:
        """責任明確な統合プロンプト"""
        
        # 構造化されたコンテキストを取得
        synthesis_context = {}
        if hasattr(bb, 'get_structured_context'):
            synthesis_context = bb.get_structured_context()
        
        # 各役割の意見を整理
        role_summary = ""
        role_opinions = synthesis_context.get('role_opinions', {})
        
        if 'reformulator' in role_opinions:
            role_summary += f"【問題分析】{role_opinions['reformulator'][0][:60]}...\n"
        if 'critic' in role_opinions:
            role_summary += f"【批判的視点】{role_opinions['critic'][0][:60]}...\n"
        if 'supporter' in role_opinions:
            role_summary += f"【創造的発展】{role_opinions['supporter'][0][:60]}...\n"
        
        # 多様性分析
        analysis = synthesis_context.get('analysis', {})
        diversity_info = f"意見の多様性: {analysis.get('diversity_score', 0):.2f}"
        consensus_areas = analysis.get('consensus_areas', [])
        conflict_areas = analysis.get('conflict_areas', [])
        
        base_prompt = f"""あなたは統合責任者です。各専門家の意見を統合し、明確で実行可能な結論を導いてください。

【元の質問】
{user_input}

【専門家の意見】
{role_summary}

【分析結果】
{diversity_info}
合意領域: {', '.join(consensus_areas[:3])}
課題領域: {', '.join(conflict_areas[:3])}

【統合の責任】
1. 各専門家の核心的な洞察を明確に統合する
2. 相反する意見がある場合は、その理由と解決策を示す
3. 具体的で実行可能な結論を提示する
4. 判断の根拠を明確にし、責任を持つ

【出力形式】
「各専門家の分析を総合すると、...が最適解です。問題分析者の...、批判的視点の...、創造的発展の...を統合した結果、...をお勧めします。その理由は...です。」

200文字以内で責任を持って統合結論を提示してください："""

        return base_prompt
    
    def execute_synthesis(self, bb: SlotBlackboard, user_input: str, discussion_history: Dict[str, Any], embedder=None) -> Optional[SlotEntry]:
        """引用付き統合モードでの実行"""
        t0 = time.time()
        
        try:
            sys_prompt = self.build_system_prompt()
            usr_prompt = self._build_synthesis_prompt(user_input, discussion_history)
            
            if self.debug:
                print(f"\n📋 {self.name} (引用付き統合モード) ---")
                phases = discussion_history.get('phases', {})
                print(f"統合対象フェーズ: {len(phases)}フェーズ")
                print("─" * 60)
            
            response = self._generate_response(sys_prompt, usr_prompt)
            
            if not response or response.strip() == "":
                return None
            
            # メタデータに統合情報を追加
            metadata = {
                "role": self.get_role_description(),
                "execution_time": time.time() - t0,
                "user_input": user_input[:100],
                "phase": 3,  # 統合フェーズ
                "synthesis_mode": True,
                "discussion_phases": len(discussion_history.get('phases', {})),
                "total_discussion_entries": discussion_history.get('total_entries', 0)
            }
            
            entry = bb.add_slot_entry(self.name, response, None, metadata)
            
            self.exec_count += 1
            self.last_exec_time = time.time() - t0
            self.total_exec_time += self.last_exec_time
            
            return entry
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _build_synthesis_prompt(self, user_input: str, discussion_history: Dict[str, Any]) -> str:
        """引用付き統合プロンプトを構築"""
        timeline = discussion_history.get('timeline', [])
        phases = discussion_history.get('phases', {})
        
        discussion_summary = ""
        for phase_num, phase_entries in phases.items():
            phase_name = phase_entries[0].get('phase_name', f'フェーズ{phase_num}') if phase_entries else f'フェーズ{phase_num}'
            discussion_summary += f"\n【{phase_name}】\n"
            for entry in phase_entries:
                discussion_summary += f"- {entry['role']}: {entry['content'][:80]}...\n"
        
        return f"""これまでの議論を統合し、引用付きで明確な結論を提示してください。

【元の質問】
{user_input}

【議論の流れ】{discussion_summary}

【統合の指針】
1. 各エージェントの主要な論点を整理
2. 合意できる部分と対立する部分を明確化
3. 最も妥当と思われる結論を論理的に導出
4. 必ず具体的な引用を含める（「Reformulatorが指摘したように〜」等）

【出力形式】
「問題分析者が指摘した...と、批判的視点での...を踏まえ、創造的発展の...を統合すると、...という結論に至ります。特に...の観点から、...をお勧めします。」

引用付きで200文字以内で統合結論を述べてください："""

# RAG機能強化
class RAGEnhancedSlot(BaseSlot):
    """RAG機能付きSlot基底クラス"""
    
    def __init__(self, slot_name: str, config: Dict[str, Any], model_factory):
        super().__init__(slot_name, config, model_factory)
        self.enable_rag = config.get('rag_mode', 'dummy') != 'dummy'
        
        if self.enable_rag:
            try:
                from .rag_retriever import RAGRetriever
                self.rag_retriever = RAGRetriever(config)
                logger.info(f"{slot_name}: RAG機能を有効化")
            except ImportError:
                logger.warning(f"{slot_name}: RAG機能のインポートに失敗、ダミーモードで継続")
                self.enable_rag = False
                self.rag_retriever = None
        else:
            self.rag_retriever = None
    
    def _get_rag_context(self, query: str) -> str:
        """RAGコンテキストを取得"""
        if not self.enable_rag or not self.rag_retriever:
            return ""
        
        try:
            results = self.rag_retriever.retrieve(query, top_k=3)
            if results:
                context = "【関連知識】\n"
                for result in results:
                    context += f"- {result.get('content', '')[:100]}...\n"
                return context
        except Exception as e:
            logger.warning(f"RAG検索エラー: {e}")
        
        return ""

class EnhancedReformulatorSlotWithRAG(RAGEnhancedSlot, EnhancedReformulatorSlot):
    """RAG機能付き問題再定式化Slot"""
    
    def _build_prompt(self, bb: SlotBlackboard, user_input: str, context: str = "") -> str:
        base_prompt = super()._build_prompt(bb, user_input, context)
        rag_context = self._get_rag_context(user_input)
        
        if rag_context:
            return f"{rag_context}\n\n{base_prompt}"
        return base_prompt

# Slot作成ファクトリ
def create_enhanced_slots(config: Dict[str, Any], model_factory) -> Dict[str, BaseSlot]:
    """改良版Slotを作成"""
    slots = {}
    
    # RAG設定の確認
    enable_rag = config.get('rag_mode', 'dummy') != 'dummy'
    
    if enable_rag:
        slots['ReformulatorSlot'] = EnhancedReformulatorSlotWithRAG('ReformulatorSlot', config, model_factory)
    else:
        slots['ReformulatorSlot'] = EnhancedReformulatorSlot('ReformulatorSlot', config, model_factory)
    
    slots['CriticSlot'] = EnhancedCriticSlot('CriticSlot', config, model_factory)
    slots['SupporterSlot'] = EnhancedSupporterSlot('SupporterSlot', config, model_factory)
    slots['SynthesizerSlot'] = EnhancedSynthesizerSlot('SynthesizerSlot', config, model_factory)
    
    return slots
