from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, Field

from .event import Event


# =========================================================
# TEMPLATE TYPES
# =========================================================

class TemplateType(str, Enum):
    FALSE_BELIEF = "false_belief"
    SECOND_ORDER_FALSE_BELIEF = "second_order_false_belief"
    MULTI_STEP_HIDDEN_TRANSFER = "multi_step_hidden_transfer"


# =========================================================
# SCENARIO CONFIG
# =========================================================

class ScenarioConfig(BaseModel):
    template_type: TemplateType
    seed: int = 42


# =========================================================
# SCENARIO (output of generator)
# No reasoning. No answers. Only structure.
# =========================================================

class Scenario(BaseModel):
    config: ScenarioConfig
    template_type: TemplateType
    agents: List[str]
    objects: List[str]
    locations: List[str]
    initial_objects: Dict[str, str]   # object -> starting location
    events: List[Event]


# =========================================================
# NAME POOLS
# =========================================================

_AGENT_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve",
    "Frank", "Grace", "Henry", "Iris", "Jack"
]

_OBJECT_NAMES = [
    "coin", "key", "marble", "ball", "book",
    "gem", "ring", "token", "chip", "stone"
]

_LOCATION_NAMES = [
    "Box_A", "Box_B", "Basket", "Drawer", "Shelf",
    "Bag", "Chest", "Jar", "Pouch", "Cupboard"
]


# =========================================================
# SCENARIO GENERATOR
# Responsibility: produce symbolic event sequences only.
# No belief reasoning. No answer derivation.
# =========================================================

class ScenarioGenerator:

    @staticmethod
    def generate(config: ScenarioConfig) -> Scenario:
        rng = random.Random(config.seed)

        if config.template_type == TemplateType.FALSE_BELIEF:
            return ScenarioGenerator._false_belief(config, rng)

        if config.template_type == TemplateType.SECOND_ORDER_FALSE_BELIEF:
            return ScenarioGenerator._second_order_false_belief(config, rng)

        if config.template_type == TemplateType.MULTI_STEP_HIDDEN_TRANSFER:
            return ScenarioGenerator._multi_step_hidden_transfer(config, rng)

        raise ValueError(f"Unknown template: {config.template_type}")

    # =====================================================
    # TEMPLATE 1 — False Belief (Sally-Anne)
    # =====================================================
    # Cognitive structure:
    #   Agent A is present at initialization.
    #   Agent B moves object — Agent A does not witness it.
    #   → A holds a false first-order belief.
    #
    # Key question this enables:
    #   "Where does A think the object is?"
    # =====================================================

    @staticmethod
    def _false_belief(config: ScenarioConfig, rng: random.Random) -> Scenario:
        names = rng.sample(_AGENT_NAMES, 2)
        agent_a, agent_b = names[0], names[1]

        obj = rng.choice(_OBJECT_NAMES)
        locs = rng.sample(_LOCATION_NAMES, 2)
        loc_start, loc_end = locs[0], locs[1]

        return Scenario(
            config=config,
            template_type=config.template_type,
            agents=[agent_a, agent_b],
            objects=[obj],
            locations=[loc_start, loc_end],
            initial_objects={obj: loc_start},
            events=[
                Event(
                    actor=agent_b,
                    action="move",
                    target_object=obj,
                    from_location=loc_start,
                    to_location=loc_end,
                    visible_to=[agent_b]         # A is absent
                )
            ]
        )

    # =====================================================
    # TEMPLATE 2 — Second-Order False Belief
    # =====================================================
    # Cognitive structure:
    #   All agents present at initialization.
    #   Agent A leaves.
    #   Agent B moves object — witnessed by B and C only.
    #   → A has false first-order belief (missed the move).
    #   → B and C have true first-order beliefs.
    #   → C has a true second-order belief about B.
    #   → B has a false second-order belief about A
    #      (B knows A missed it, so B knows A's belief is wrong).
    #
    # Key questions this enables:
    #   "Where does C think A believes the object is?"
    #   "Where does B think A believes the object is?"
    # =====================================================

    @staticmethod
    def _second_order_false_belief(config: ScenarioConfig, rng: random.Random) -> Scenario:
        names = rng.sample(_AGENT_NAMES, 3)
        agent_a, agent_b, agent_c = names[0], names[1], names[2]

        obj = rng.choice(_OBJECT_NAMES)
        locs = rng.sample(_LOCATION_NAMES, 2)
        loc_start, loc_end = locs[0], locs[1]

        return Scenario(
            config=config,
            template_type=config.template_type,
            agents=[agent_a, agent_b, agent_c],
            objects=[obj],
            locations=[loc_start, loc_end],
            initial_objects={obj: loc_start},
            events=[
                Event(
                    actor=agent_b,
                    action="move",
                    target_object=obj,
                    from_location=loc_start,
                    to_location=loc_end,
                    visible_to=[agent_b, agent_c]  # A is absent
                )
            ]
        )

    # =====================================================
    # TEMPLATE 3 — Multi-Step Hidden Transfer
    # =====================================================
    # Cognitive structure:
    #   Event 1: B and C witness — A absent.
    #   Event 2: C witnesses only — A and B absent.
    #   → A: missed all events → false belief (loc_start).
    #   → B: missed event 2   → false belief (loc_mid).
    #   → C: witnessed all    → true belief  (loc_end).
    #   Three-level epistemic divergence across agents.
    #
    # Key questions this enables:
    #   "Where does A think the object is?"
    #   "Where does B think the object is?"
    #   "Where does C think A believes the object is?"
    # =====================================================

    @staticmethod
    def _multi_step_hidden_transfer(config: ScenarioConfig, rng: random.Random) -> Scenario:
        names = rng.sample(_AGENT_NAMES, 3)
        agent_a, agent_b, agent_c = names[0], names[1], names[2]

        obj = rng.choice(_OBJECT_NAMES)
        locs = rng.sample(_LOCATION_NAMES, 3)
        loc_start, loc_mid, loc_end = locs[0], locs[1], locs[2]

        return Scenario(
            config=config,
            template_type=config.template_type,
            agents=[agent_a, agent_b, agent_c],
            objects=[obj],
            locations=[loc_start, loc_mid, loc_end],
            initial_objects={obj: loc_start},
            events=[
                Event(
                    actor=agent_b,
                    action="move",
                    target_object=obj,
                    from_location=loc_start,
                    to_location=loc_mid,
                    visible_to=[agent_b, agent_c]  # A absent
                ),
                Event(
                    actor=agent_c,
                    action="move",
                    target_object=obj,
                    from_location=loc_mid,
                    to_location=loc_end,
                    visible_to=[agent_c]            # A and B absent
                )
            ]
        )