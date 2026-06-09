from __future__ import annotations

from typing import List
from pydantic import BaseModel


# =========================================================
# VALIDATION ERROR
# =========================================================

class ValidationError(BaseModel):
    scenario_id: str
    check: str
    message: str


# =========================================================
# VALIDATION REPORT
# =========================================================

class ValidationReport(BaseModel):
    passed: bool
    total_scenarios: int
    total_questions: int
    errors: List[ValidationError]

    def print_report(self) -> None:
        print("\n########################################")
        print("DATASET VALIDATION REPORT")
        print("########################################")
        print(f"Scenarios : {self.total_scenarios}")
        print(f"Questions : {self.total_questions}")
        print(f"Status    : {'PASSED' if self.passed else 'FAILED'}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  [{e.scenario_id}] {e.check}")
                print(f"    → {e.message}")
        else:
            print("\nAll checks passed.")


# =========================================================
# DATASET VALIDATOR
# =========================================================

class DatasetValidator:
    """
    Runs consistency checks on a loaded dataset.
    Fails loudly — every error is collected and reported.
    Never silently skips.
    """

    @staticmethod
    def validate(scenarios: list) -> ValidationReport:
        errors: List[ValidationError] = []
        total_questions = 0

        for scenario in scenarios:
            sid = scenario.get("scenario_id", "UNKNOWN")
            valid_locations = set(loc.lower() for loc in scenario.get("locations", []))
            agents = scenario.get("agents", [])
            belief_state = scenario.get("belief_state", {})
            questions = scenario.get("questions", [])
            total_questions += len(questions)

            errors += DatasetValidator._check_reality(sid, scenario, valid_locations)
            errors += DatasetValidator._check_first_order(sid, scenario, valid_locations, agents)
            errors += DatasetValidator._check_second_order(sid, scenario, valid_locations, agents)
            errors += DatasetValidator._check_questions(sid, questions, belief_state, valid_locations)
            errors += DatasetValidator._check_no_unknown_agents(sid, belief_state, agents)

        return ValidationReport(
            passed=len(errors) == 0,
            total_scenarios=len(scenarios),
            total_questions=total_questions,
            errors=errors
        )

    # =====================================================
    # CHECK: reality values are valid locations
    # =====================================================

    @staticmethod
    def _check_reality(
        sid: str,
        scenario: dict,
        valid_locations: set
    ) -> List[ValidationError]:
        errors = []
        reality = scenario.get("belief_state", {}).get("reality", {})

        for obj, loc in reality.items():
            if loc.lower() not in valid_locations:
                errors.append(ValidationError(
                    scenario_id=sid,
                    check="reality_in_locations",
                    message=f"Object '{obj}' has reality location '{loc}' not in locations list."
                ))
        return errors

    # =====================================================
    # CHECK: first-order beliefs are valid locations
    # =====================================================

    @staticmethod
    def _check_first_order(
        sid: str,
        scenario: dict,
        valid_locations: set,
        agents: list
    ) -> List[ValidationError]:
        errors = []
        first_order = scenario.get("belief_state", {}).get("first_order", {})

        for agent, beliefs in first_order.items():
            for obj, loc in beliefs.items():
                if loc.lower() not in valid_locations:
                    errors.append(ValidationError(
                        scenario_id=sid,
                        check="first_order_in_locations",
                        message=(
                            f"Agent '{agent}' first-order belief: "
                            f"'{obj}' -> '{loc}' not in locations list."
                        )
                    ))
        return errors

    # =====================================================
    # CHECK: second-order beliefs are valid locations
    # =====================================================

    @staticmethod
    def _check_second_order(
        sid: str,
        scenario: dict,
        valid_locations: set,
        agents: list
    ) -> List[ValidationError]:
        errors = []
        second_order = scenario.get("belief_state", {}).get("second_order", {})

        for agent, others in second_order.items():
            for other, beliefs in others.items():
                for obj, loc in beliefs.items():
                    if loc.lower() not in valid_locations:
                        errors.append(ValidationError(
                            scenario_id=sid,
                            check="second_order_in_locations",
                            message=(
                                f"Agent '{agent}' thinks '{other}' believes "
                                f"'{obj}' -> '{loc}' not in locations list."
                            )
                        ))
        return errors

    # =====================================================
    # CHECK: question expected_location matches belief_state
    # =====================================================

    @staticmethod
    def _check_questions(
        sid: str,
        questions: list,
        belief_state: dict,
        valid_locations: set
    ) -> List[ValidationError]:
        errors = []
        reality = belief_state.get("reality", {})
        first_order = belief_state.get("first_order", {})
        second_order = belief_state.get("second_order", {})

        for q in questions:
            qid = q.get("question_id", "UNKNOWN")
            rt = q.get("reasoning_type")
            obj = q.get("target_object")
            agent = q.get("target_agent")
            other = q.get("target_other_agent")
            expected = q.get("expected_location", "")

            # Check expected_location is a valid location
            if expected.lower() not in valid_locations:
                errors.append(ValidationError(
                    scenario_id=sid,
                    check="answer_in_locations",
                    message=f"Question '{qid}' expected_location '{expected}' not in locations list."
                ))
                continue

            # Check expected_location matches belief_state
            if rt == "reality":
                ground = reality.get(obj)
                if ground and ground.lower() != expected.lower():
                    errors.append(ValidationError(
                        scenario_id=sid,
                        check="answer_matches_belief_state",
                        message=(
                            f"Question '{qid}' (reality): "
                            f"expected '{expected}' but belief_state says '{ground}'."
                        )
                    ))

            elif rt == "first_order":
                ground = first_order.get(agent, {}).get(obj)
                if ground and ground.lower() != expected.lower():
                    errors.append(ValidationError(
                        scenario_id=sid,
                        check="answer_matches_belief_state",
                        message=(
                            f"Question '{qid}' (first_order, agent={agent}): "
                            f"expected '{expected}' but belief_state says '{ground}'."
                        )
                    ))

            elif rt == "second_order":
                ground = second_order.get(agent, {}).get(other, {}).get(obj)
                if ground and ground.lower() != expected.lower():
                    errors.append(ValidationError(
                        scenario_id=sid,
                        check="answer_matches_belief_state",
                        message=(
                            f"Question '{qid}' (second_order, agent={agent}, other={other}): "
                            f"expected '{expected}' but belief_state says '{ground}'."
                        )
                    ))

        return errors

    # =====================================================
    # CHECK: no unknown agents in belief_state
    # =====================================================

    @staticmethod
    def _check_no_unknown_agents(
        sid: str,
        belief_state: dict,
        agents: list
    ) -> List[ValidationError]:
        errors = []
        known = set(agents)

        for agent in belief_state.get("first_order", {}):
            if agent not in known:
                errors.append(ValidationError(
                    scenario_id=sid,
                    check="no_unknown_agents",
                    message=f"Unknown agent '{agent}' found in first_order beliefs."
                ))

        for agent in belief_state.get("second_order", {}):
            if agent not in known:
                errors.append(ValidationError(
                    scenario_id=sid,
                    check="no_unknown_agents",
                    message=f"Unknown agent '{agent}' found in second_order beliefs."
                ))

        return errors