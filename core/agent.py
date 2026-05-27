from __future__ import annotations
 
from copy import deepcopy
from typing import Dict, List
from pydantic import BaseModel, Field
 
from .belief import BeliefState
 
 
class AgentMind(BaseModel):
    name: str
    first_order: BeliefState = Field(default_factory=BeliefState)
    second_order: Dict[str, BeliefState] = Field(default_factory=dict)
 
    def initialize_second_order(
        self,
        agents: List[str],
        initial_world: Dict[str, str]
    ) -> None:
        for agent in agents:
            if agent != self.name:
                self.second_order[agent] = BeliefState(
                    object_locations=deepcopy(initial_world)
                )