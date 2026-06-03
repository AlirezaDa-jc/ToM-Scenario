from __future__ import annotations

import json
from pathlib import Path

from benchmark.adapters.gemini_adapter import GeminiAdapter
from benchmark.loader import DatasetLoader
from benchmark.metrics import MetricsAggregator
from benchmark.runner import BenchmarkRunner

DATASET_PATH = Path("output/scenarios.json")
OUTPUT_PATH  = Path("output/evaluation.json")


if __name__ == "__main__":

    # 1. Load dataset
    dataset = DatasetLoader.load(DATASET_PATH)
    print(f"Loaded {len(dataset)} scenarios.")

    # 2. Run benchmark with Gemini Adapter
    adapter = GeminiAdapter(model="gemini-3-flash-preview")
    runner  = BenchmarkRunner(adapter=adapter, prompt_template="default")
    results = runner.run(dataset)

    # 3. Compute metrics
    report = MetricsAggregator.compute(results)

    print(f"\n=== METRICS: {report.model_name} ===")
    print(f"Overall accuracy     : {report.overall_accuracy:.2%}")
    print(f"\nBy reasoning type:")
    for rt, acc in report.by_reasoning_type.items():
        print(f"  {rt:20s}: {acc:.2%}")
    print(f"\nBy template type:")
    for tt, acc in report.by_template_type.items():
        print(f"  {tt:35s}: {acc:.2%}")

    # 4. Export results
    output = {
        "metrics": report.model_dump(),
        "results": [r.model_dump() for r in results]
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nEvaluation saved → {OUTPUT_PATH}")