from __future__ import annotations

from copy import deepcopy
from typing import Dict, List
from pydantic import BaseModel, Field

from .agent import AgentMind
from .belief import BeliefState
from .engine import BeliefEngine
from .event import Event
from .history import EventHistory


class WorldState(BaseModel):
    world_truth: Dict[str, str] = Field(default_factory=dict)
    minds: Dict[str, AgentMind] = Field(default_factory=dict)
    history: EventHistory = Field(default_factory=EventHistory)

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize(
        self,
        agents: List[str],
        initial_objects: Dict[str, str]
    ) -> None:
        self.world_truth = deepcopy(initial_objects)

        for agent in agents:
            mind = AgentMind(
                name=agent,
                first_order=BeliefState(
                    object_locations=deepcopy(initial_objects)
                )
            )
            self.minds[agent] = mind

        for agent in agents:
            self.minds[agent].initialize_second_order(
                agents=agents,
                initial_world=initial_objects
            )

    # =========================================================
    # PROCESS EVENT
    # =========================================================

    def process_event(self, event: Event) -> None:
        # 1. Ground truth
        self.world_truth[event.target_object] = event.to_location

        # 2. Delegate belief updates to engine
        BeliefEngine.propagate(event=event, minds=self.minds)

        # 3. Build belief snapshot for all agents
        belief_snapshot = {
            agent: {
                "first_order": dict(mind.first_order.object_locations),
                "second_order": {
                    other: dict(bs.object_locations)
                    for other, bs in mind.second_order.items()
                }
            }
            for agent, mind in self.minds.items()
        }

        # 4. Record immutable snapshot
        self.history.append(
            event=event,
            world_truth=self.world_truth,
            belief_snapshot=belief_snapshot
        )

    # =========================================================
    # QUERIES
    # =========================================================

    def query_world(self, obj: str) -> str:
        return self.world_truth.get(obj, "UNKNOWN")

    def query_first_order(self, agent: str, obj: str) -> str:
        return self.minds[agent].first_order.get(obj)

    def query_second_order(self, agent: str, other_agent: str, obj: str) -> str:
        return self.minds[agent].second_order[other_agent].get(obj)

    # =========================================================
    # DEBUG
    # =========================================================

    def print_state(self) -> None:
        print("\n########################################")
        print("WORLD TRUTH")
        print("########################################")
        for obj, loc in self.world_truth.items():
            print(f"  {obj} -> {loc}")

        print("\n########################################")
        print("AGENT BELIEFS")
        print("########################################")
        for agent_name, mind in self.minds.items():
            print(f"\n--- {agent_name} ---")
            print("  FIRST ORDER:")
            for obj, loc in mind.first_order.object_locations.items():
                print(f"    {obj} -> {loc}")
            print("  SECOND ORDER:")
            for other, belief_state in mind.second_order.items():
                print(f"    THINKS {other} BELIEVES:")
                for obj, loc in belief_state.object_locations.items():
                    print(f"      {obj} -> {loc}")