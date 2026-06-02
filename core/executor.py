from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict

from core.scenario import Scenario
from core.world import WorldState


# =========================================================
# EXECUTION RESULT
# All answers derived from Simulator — never from Generator.
# =========================================================

class ExecutionResult(BaseModel):
    scenario: Scenario
    world: WorldState

    # Derived from world state after all events
    world_truth: Dict[str, str] = Field(default_factory=dict)

    # first_order_beliefs[agent][object] = location
    first_order_beliefs: Dict[str, Dict[str, str]] = Field(default_factory=dict)

    # second_order_beliefs[agent][other_agent][object] = location
    second_order_beliefs: Dict[str, Dict[str, Dict[str, str]]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# =========================================================
# SCENARIO EXECUTOR
# =========================================================

class ScenarioExecutor:
    """
    Takes a Scenario, runs it through WorldState + BeliefEngine,
    and returns an ExecutionResult with all belief states
    derived from the simulator — not from the generator.
    """

    @staticmethod
    def run(scenario: Scenario) -> ExecutionResult:

        # 1. Initialize world
        world = WorldState()
        world.initialize(
            agents=scenario.agents,
            initial_objects=scenario.initial_objects
        )

        # 2. Process all events
        for event in scenario.events:
            world.process_event(event)

        # 3. Extract ground truth
        world_truth = dict(world.world_truth)

        # 4. Extract first-order beliefs from simulator
        first_order_beliefs: Dict[str, Dict[str, str]] = {}
        for agent in scenario.agents:
            first_order_beliefs[agent] = {
                obj: world.query_first_order(agent, obj)
                for obj in scenario.objects
            }

        # 5. Extract second-order beliefs from simulator
        second_order_beliefs: Dict[str, Dict[str, Dict[str, str]]] = {}
        for agent in scenario.agents:
            second_order_beliefs[agent] = {}
            for other in scenario.agents:
                if other == agent:
                    continue
                second_order_beliefs[agent][other] = {
                    obj: world.query_second_order(agent, other, obj)
                    for obj in scenario.objects
                }

        return ExecutionResult(
            scenario=scenario,
            world=world,
            world_truth=world_truth,
            first_order_beliefs=first_order_beliefs,
            second_order_beliefs=second_order_beliefs
        )