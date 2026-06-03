from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from pydantic import BaseModel

from benchmark.evaluation import EvaluationResult


# =========================================================
# METRICS REPORT
# =========================================================

class MetricsReport(BaseModel):
    model_name: str
    prompt_template: str
    total_questions: int
    correct_questions: int
    overall_accuracy: float
    by_reasoning_type: Dict[str, float]   # ReasoningType.value -> accuracy
    by_template_type: Dict[str, float]    # template_type -> accuracy


# =========================================================
# METRICS AGGREGATOR
# =========================================================

class MetricsAggregator:
    """
    Computes accuracy metrics from a list of EvaluationResults.
    No evaluation logic — only aggregation.
    """

    @staticmethod
    def compute(results: List[EvaluationResult]) -> MetricsReport:
        if not results:
            raise ValueError("Cannot compute metrics from empty results.")

        model_name = results[0].model_name
        prompt_template = results[0].prompt_template

        total = len(results)
        correct = sum(1 for r in results if r.correct)

        # --- by reasoning type ---
        by_type_total: Dict[str, int] = defaultdict(int)
        by_type_correct: Dict[str, int] = defaultdict(int)
        for r in results:
            by_type_total[r.reasoning_type.value] += 1
            if r.correct:
                by_type_correct[r.reasoning_type.value] += 1

        by_reasoning_type = {
            rt: round(by_type_correct[rt] / by_type_total[rt], 4)
            for rt in by_type_total
        }

        # --- by template type ---
        by_tmpl_total: Dict[str, int] = defaultdict(int)
        by_tmpl_correct: Dict[str, int] = defaultdict(int)
        for r in results:
            by_tmpl_total[r.template_type] += 1
            if r.correct:
                by_tmpl_correct[r.template_type] += 1

        by_template_type = {
            t: round(by_tmpl_correct[t] / by_tmpl_total[t], 4)
            for t in by_tmpl_total
        }

        return MetricsReport(
            model_name=model_name,
            prompt_template=prompt_template,
            total_questions=total,
            correct_questions=correct,
            overall_accuracy=round(correct / total, 4),
            by_reasoning_type=by_reasoning_type,
            by_template_type=by_template_type,
        )