from __future__ import annotations

from .agent import AgentMind
from .event import Event
from typing import Dict


class BeliefEngine:
    """
    Pure belief propagation logic.
    Receives an event + agent minds, updates beliefs in-place.
    No world state knowledge. No I/O.
    """

    @staticmethod
    def propagate(event: Event, minds: Dict[str, AgentMind]) -> None:
        for observer in event.visible_to:
            mind = minds[observer]

            # First-order: observer saw it happen
            mind.first_order.update(event.target_object, event.to_location)

            # Second-order: observer knows other visible agents also saw it
            for other in event.visible_to:
                if other == observer:
                    continue
                if other in mind.second_order:
                    mind.second_order[other].update(
                        event.target_object, event.to_location
                    )