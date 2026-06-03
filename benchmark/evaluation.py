from __future__ import annotations

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

        # Lowercase for comparison
        text = text.lower()

        return text


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
    prediction: str          # normalized
    expected_answer: str     # normalized
    correct: bool


# =========================================================
# EVALUATOR
# Exact-match, case-insensitive.
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

        prediction = PredictionNormalizer.normalize(raw_response)
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
            prediction=prediction,
            expected_answer=expected,
            correct=(prediction == expected),
        )