from __future__ import annotations

from typing import Optional, List

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
# =========================================================

class PredictionExtractor:
    """
    Extracts a valid location name from raw model output.
    - Case-insensitive match against valid_locations.
    - If multiple valid locations appear → return the LAST one.
    - If no valid location found → fallback to normalized text.
    """

    @staticmethod
    def extract(raw_response: str, valid_locations: List[str]) -> str:
        normalized_response = raw_response.lower()

        last_match: Optional[str] = None
        last_pos: int = -1

        for location in valid_locations:
            pos = normalized_response.rfind(location.lower())
            if pos != -1 and pos > last_pos:
                last_pos = pos
                last_match = location.lower()

        if last_match is not None:
            return last_match

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
    """
    Produces an EvaluationResult from a question dict,
    raw model response, and scenario metadata.
    Never normalizes or infers — delegates to PredictionNormalizer.
    """

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
