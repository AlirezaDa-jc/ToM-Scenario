from __future__ import annotations

from typing import List

from benchmark.adapter import ModelAdapter
from benchmark.evaluation import EvaluationResult, Evaluator
from benchmark.metrics import MetricsReport, MetricsAggregator
from benchmark.prompt import PromptBuilder


# =========================================================
# BENCHMARK RUNNER
# Orchestrator — connects all benchmark components.
# =========================================================

class BenchmarkRunner:
    """
    Iterates through a loaded dataset, builds prompts,
    queries the model adapter, and evaluates each response.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        prompt_template: str = "default"
    ):
        self.adapter = adapter
        self.prompt_template = prompt_template

    def run(self, dataset: list) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for scenario in dataset:
            story = scenario["story"]

            for question in scenario["questions"]:
                prompt = PromptBuilder.build(
                    story=story,
                    question_text=question["question_text"],
                    template_name=self.prompt_template
                )

                raw_response = self.adapter.generate(prompt)

                result = Evaluator.evaluate(
                    question=question,
                    scenario=scenario,
                    raw_response=raw_response,
                    model_name=self.adapter.name,
                    prompt_template=self.prompt_template,
                )

                results.append(result)

        return results

    def run_and_report(self, dataset: list) -> MetricsReport:
        results = self.run(dataset)
        return MetricsAggregator.compute(results)