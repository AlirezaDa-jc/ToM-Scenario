from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from core.executor import ExecutionResult


# =========================================================
# REASONING TYPE
# =========================================================

class ReasoningType(str, Enum):
    REALITY = "reality"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


# =========================================================
# QUESTION
# =========================================================

class Question(BaseModel):
    question_id: str
    question_text: str
    answer: str
    reasoning_type: ReasoningType
    target_object: str
    target_agent: Optional[str] = None
    target_other_agent: Optional[str] = None
    expected_location: str


# =========================================================
# QUESTION SET
# =========================================================

class QuestionSet(BaseModel):
    scenario_id: str
    story: str
    questions: List[Question] = Field(default_factory=list)


# =========================================================
# QUESTION GENERATOR
# =========================================================

class QuestionGenerator:
    """
    Converts an ExecutionResult into a QuestionSet.
    Never infers answers — all answers come from ExecutionResult.
    """

    @staticmethod
    def generate(result: ExecutionResult) -> QuestionSet:
        scenario = result.scenario
        scenario_id = f"{scenario.template_type.value}_{scenario.config.seed}"

        questions: List[Question] = []
        index = 0

        # 1. Reality questions
        for q in QuestionGenerator._reality_questions(result, scenario_id, index):
            questions.append(q)
            index += 1

        # 2. First-order belief questions
        for q in QuestionGenerator._first_order_questions(result, scenario_id, index):
            questions.append(q)
            index += 1

        # 3. Second-order belief questions
        for q in QuestionGenerator._second_order_questions(result, scenario_id, index):
            questions.append(q)
            index += 1

        return QuestionSet(
            scenario_id=scenario_id,
            story=result.world.history.export_story(all_agents=scenario.agents),
            questions=questions
        )

    # =====================================================
    # REALITY
    # =====================================================

    @staticmethod
    def _reality_questions(
        result: ExecutionResult,
        scenario_id: str,
        start_index: int
    ) -> List[Question]:
        questions = []
        for i, obj in enumerate(result.scenario.objects):
            answer = result.world_truth[obj]
            questions.append(Question(
                question_id=f"{scenario_id}_reality_{start_index + i}",
                question_text=f"Where is the {obj} really?",
                answer=answer,
                reasoning_type=ReasoningType.REALITY,
                target_object=obj,
                expected_location=answer
            ))
        return questions

    # =====================================================
    # FIRST-ORDER
    # =====================================================

    @staticmethod
    def _first_order_questions(
        result: ExecutionResult,
        scenario_id: str,
        start_index: int
    ) -> List[Question]:
        questions = []
        i = 0
        for agent in result.scenario.agents:
            for obj in result.scenario.objects:
                answer = result.first_order_beliefs[agent][obj]
                questions.append(Question(
                    question_id=f"{scenario_id}_first_order_{start_index + i}",
                    question_text=f"Where does {agent} think the {obj} is?",
                    answer=answer,
                    reasoning_type=ReasoningType.FIRST_ORDER,
                    target_object=obj,
                    target_agent=agent,
                    expected_location=answer
                ))
                i += 1
        return questions

    # =====================================================
    # SECOND-ORDER
    # =====================================================

    @staticmethod
    def _second_order_questions(
        result: ExecutionResult,
        scenario_id: str,
        start_index: int
    ) -> List[Question]:
        questions = []
        i = 0
        for agent in result.scenario.agents:
            for other in result.scenario.agents:
                if other == agent:
                    continue
                for obj in result.scenario.objects:
                    answer = result.second_order_beliefs[agent][other][obj]
                    questions.append(Question(
                        question_id=f"{scenario_id}_second_order_{start_index + i}",
                        question_text=(
                            f"Where does {agent} think "
                            f"{other} believes the {obj} is?"
                        ),
                        answer=answer,
                        reasoning_type=ReasoningType.SECOND_ORDER,
                        target_object=obj,
                        target_agent=agent,
                        target_other_agent=other,
                        expected_location=answer
                    ))
                    i += 1
        return questions