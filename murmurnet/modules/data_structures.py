#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core Data Structures for MurmurNet (Recreated for current task)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, List

@dataclass
class AgentOutput:
    """
    Represents the output of a single agent.
    """
    agent_id: str
    text: str
    role: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    iteration: Optional[int] = None

    def __repr__(self):
        return (f"AgentOutput(agent_id='{self.agent_id}', role='{self.role}', iter={self.iteration}, "
                f"text='{self.text[:60]}...', ts={self.timestamp:.2f})")

@dataclass
class BlackboardEntry:
    """
    Represents a single entry on the blackboard. (Minimal for dependency)
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str  
    content_type: str 
    data: Any
    iteration: Optional[int] = None

    def __repr__(self):
        return (f"BlackboardEntry(id='{self.entry_id}', source='{self.source}', type='{self.content_type}', "
                f"iter={self.iteration}, data='{str(self.data)[:50]}...')")
