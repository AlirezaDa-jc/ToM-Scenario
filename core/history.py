from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .event import Event


# belief_snapshot structure:
# {
#   "Alice": {
#     "first_order": {"coin": "Box_A"},
#     "second_order": {
#       "Bob": {"coin": "Box_A"},
#       ...
#     }
#   },
#   ...
# }
BeliefSnapshot = Dict[str, Dict]


class EventRecord(BaseModel):
    step: int
    event: Event
    world_snapshot: Dict[str, str]
    belief_snapshot: BeliefSnapshot

    class Config:
        frozen = True  # immutable


class EventHistory(BaseModel):
    records: List[EventRecord] = Field(default_factory=list)

    # =========================================================
    # APPEND
    # =========================================================

    def append(
        self,
        event: Event,
        world_truth: Dict[str, str],
        belief_snapshot: BeliefSnapshot
    ) -> None:
        record = EventRecord(
            step=len(self.records) + 1,
            event=event,
            world_snapshot=deepcopy(world_truth),
            belief_snapshot=deepcopy(belief_snapshot)
        )
        self.records.append(record)

    # =========================================================
    # REPLAY
    # =========================================================

    def get_state_at(self, step: int) -> Optional[EventRecord]:
        for r in self.records:
            if r.step == step:
                return r
        return None

    # =========================================================
    # PRINT
    # =========================================================

    def print_history(self) -> None:
        print("\n########################################")
        print("EVENT HISTORY")
        print("########################################")
        for r in self.records:
            print(f"\n[Step {r.step}]")
            print(f"  {r.event.actor} moves {r.event.target_object} "
                  f"{r.event.from_location} -> {r.event.to_location}")
            print(f"  visible_to: {r.event.visible_to}")
            print(f"  world after: {r.world_snapshot}")

    # =========================================================
    # EXPORT
    # =========================================================

    def export_story(self, all_agents: List[str]) -> str:
        lines = []
        for r in self.records:
            witnesses = r.event.visible_to
            absent = [a for a in all_agents if a not in witnesses]

            line = (
                f"Step {r.step}: {r.event.actor} moved {r.event.target_object} "
                f"from {r.event.from_location} to {r.event.to_location}. "
                f"{', '.join(witnesses)} witnessed the event."
            )
            if absent:
                line += f" {', '.join(absent)} did not witness the event."

            lines.append(line)
        return "\n".join(lines)