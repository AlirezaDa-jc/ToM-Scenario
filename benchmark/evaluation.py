from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel

from core.questions import ReasoningType


# =========================================================
# PREDICTION NORMALIZER
# Sits between raw model output and Evaluator.
# Strips whitespace, punctuation, casing artifacts.
# =========================================================

class PredictionNormalizer:
    """
    Normalizes raw model output to a clean, comparable string.
    Applied before evaluation — never modifies EvaluationResult directly.
    """

    @staticmethod
    def normalize(raw: str) -> str:
        text = raw.strip()

        # Remove trailing punctuation
        while text and text[-1] in ".,;:!?":
            text = text[:-1]

        # Collapse whitespace
        text = " ".join(text.split())
        return text.lower()


# =========================================================
# PREDICTION EXTRACTOR
# Sits between raw model output and Evaluator.
#
# Rules:
# - Strip parenthetical content first (reasoning/explanation
#   often contains OTHER location names mentioned for context,
#   not the model's actual answer).
# - Case-insensitive match against valid_locations.
# - If multiple valid locations remain after stripping
#   parentheses → return the FIRST one (the model's stated
#   answer typically comes before its justification).
# - If nothing found outside parentheses, fall back to
#   scanning the full text (LAST match), then normalized text.
# =========================================================

class PredictionExtractor:

    @staticmethod
    def _strip_parentheticals(text: str) -> str:
        return re.sub(r"\([^)]*\)", " ", text)

    @staticmethod
    def _find_first(text: str, valid_locations: List[str]) -> Optional[str]:
        normalized = text.lower()
        first_match: Optional[str] = None
        first_pos: int = len(normalized) + 1

        for location in valid_locations:
            pos = normalized.find(location.lower())
            if pos != -1 and pos < first_pos:
                first_pos = pos
                first_match = location.lower()

        return first_match

    @staticmethod
    def _find_last(text: str, valid_locations: List[str]) -> Optional[str]:
        normalized = text.lower()
        last_match: Optional[str] = None
        last_pos: int = -1

        for location in valid_locations:
            pos = normalized.rfind(location.lower())
            if pos != -1 and pos > last_pos:
                last_pos = pos
                last_match = location.lower()

        return last_match

    @staticmethod
    def extract(raw_response: str, valid_locations: List[str]) -> str:
        # 1. Try outside parentheses first, take FIRST match
        stripped = PredictionExtractor._strip_parentheticals(raw_response)
        match = PredictionExtractor._find_first(stripped, valid_locations)
        if match is not None:
            return match

        # 2. Fall back: scan full text (incl. parentheses), take LAST match
        match = PredictionExtractor._find_last(raw_response, valid_locations)
        if match is not None:
            return match

        # 3. Fall back: normalized raw text
        return PredictionNormalizer.normalize(raw_response)


# =========================================================
# EVALUATION RESULT
# =========================================================

class EvaluationResult(BaseModel):
    question_id: str
    scenario_id: str
    model_name: str
    reasoning_type: ReasoningType
    template_type: str
    seed: int
    prompt_template: str
    raw_response: str
    extracted_prediction: str
    expected_answer: str
    correct: bool


# =========================================================
# EVALUATOR
# =========================================================

class Evaluator:

    @staticmethod
    def evaluate(
        question: dict,
        scenario: dict,
        raw_response: str,
        model_name: str,
        prompt_template: str,
    ) -> EvaluationResult:

        extracted = PredictionExtractor.extract(
            raw_response=raw_response,
            valid_locations=scenario["locations"]
        )
        expected = PredictionNormalizer.normalize(question["expected_location"])

        return EvaluationResult(
            question_id=question["question_id"],
            scenario_id=scenario["scenario_id"],
            model_name=model_name,
            reasoning_type=ReasoningType(question["reasoning_type"]),
            template_type=scenario["template_type"],
            seed=scenario["seed"],
            prompt_template=prompt_template,
            raw_response=raw_response,
            extracted_prediction=extracted,
            expected_answer=expected,
            correct=(extracted == expected),
        )