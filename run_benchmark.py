from __future__ import annotations

import json
from pathlib import Path

from benchmark.adapters.gemini_adapter import GeminiAdapter
from benchmark.adapters.hf_adapter import HuggingFaceAdapter
from benchmark.loader import DatasetLoader
from benchmark.metrics import MetricsAggregator
from benchmark.runner import BenchmarkRunner
import os

os.environ["HF_HOME"] = r".\hf_cache"
DATASET_PATH = Path("output/scenarios.json")
OUTPUT_DIR   = Path("output/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

PROMPT_TEMPLATE = "default"


def run_model(dataset: list, model_id: str) -> dict:
    adapter = HuggingFaceAdapter(model_id=model_id, max_new_tokens=64)
    runner  = BenchmarkRunner(adapter=adapter, prompt_template=PROMPT_TEMPLATE)
    results = runner.run(dataset)
    report  = MetricsAggregator.compute(results)

    print(f"\n=== {report.model_name} ===")
    print(f"Overall accuracy : {report.overall_accuracy:.2%}")
    print("By reasoning type:")
    for rt, acc in report.by_reasoning_type.items():
        print(f"  {rt:20s}: {acc:.2%}")
    print("By template type:")
    for tt, acc in report.by_template_type.items():
        print(f"  {tt:35s}: {acc:.2%}")

    return {
        "metrics": report.model_dump(),
        "results": [r.model_dump() for r in results]
    }


if __name__ == "__main__":
    dataset = DatasetLoader.load(DATASET_PATH)
    print(f"Loaded {len(dataset)} scenarios.")

    all_reports = {}

    for model_id in MODELS:
        try:
            output = run_model(dataset, model_id)
            model_name = model_id.split("/")[-1]
            all_reports[model_name] = output

            # Save per-model result
            out_file = OUTPUT_DIR / f"{model_name}.json"
            out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"Saved → {out_file}")

        except Exception as e:
            print(f"[FAILED] {model_id}: {e}")

    # Save combined leaderboard
    leaderboard = {
        name: data["metrics"]
        for name, data in all_reports.items()
    }
    lb_file = OUTPUT_DIR / "leaderboard.json"
    lb_file.write_text(json.dumps(leaderboard, indent=2, ensure_ascii=False))
    print(f"\nLeaderboard saved → {lb_file}")