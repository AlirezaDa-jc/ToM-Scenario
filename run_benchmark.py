from __future__ import annotations

import json
from pathlib import Path
import gc
import torch

from benchmark.adapter import ModelAdapter, RandomBaselineAdapter
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
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]
PROMPT_TEMPLATE = "cot"
OUTPUT_SUFFIX = "_cot"


def run_adapter(dataset: list, adapter: ModelAdapter, prompt_template: str = PROMPT_TEMPLATE) -> dict:
    runner = BenchmarkRunner(adapter=adapter, prompt_template=prompt_template)
    results = runner.run(dataset)
    report = MetricsAggregator.compute(results)

    print(f"\n=== {report.model_name} [{prompt_template}] ===")
    print(f"Overall accuracy : {report.overall_accuracy:.2%}")
    print("By reasoning type:")
    for rt, acc in report.by_reasoning_type.items():
        print(f"  {rt:20s}: {acc:.2%}")
    print("By template type:")
    for tt, acc in report.by_template_type.items():
        print(f"  {tt:35s}: {acc:.2%}")

    return {
        "prompt_template": prompt_template,
        "metrics": report.model_dump(),
        "results": [r.model_dump() for r in results]
    }


def run_model(dataset: list, model_id: str) -> dict:
    adapter = HuggingFaceAdapter(model_id=model_id, max_new_tokens=64)
    return run_adapter(dataset, adapter, PROMPT_TEMPLATE)


if __name__ == "__main__":
    dataset = DatasetLoader.load(DATASET_PATH)
    print(f"Loaded {len(dataset)} scenarios.")

    all_reports = {}

    try:
        output = run_adapter(dataset, RandomBaselineAdapter())
        all_reports["random_baseline"] = output

        out_file = OUTPUT_DIR / "random_baseline.json"
        out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"Saved → {out_file}")

    except Exception as e:
        print(f"[FAILED] random_baseline: {e}")

    finally:
        if 'output' in locals():
            del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for model_id in MODELS:
        try:
            output = run_model(dataset, model_id)
            model_name = model_id.split("/")[-1]
            output_name = f"{model_name}{OUTPUT_SUFFIX}"
            all_reports[output_name] = output

            # Save per-model result without overwriting default-template results.
            out_file = OUTPUT_DIR / f"{output_name}.json"
            out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"Saved → {out_file}")

        except Exception as e:
            print(f"[FAILED] {model_id}: {e}")

        finally:
            # Clean up memory aggressively after every model
            if 'output' in locals():
                del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save COT-only leaderboard separately so the existing/default leaderboard is not overwritten.
    leaderboard = {
        name: data["metrics"]
        for name, data in all_reports.items()
    }
    lb_file = OUTPUT_DIR / f"leaderboard_{PROMPT_TEMPLATE}.json"
    lb_file.write_text(json.dumps(leaderboard, indent=2, ensure_ascii=False))
    print(f"\nCOT leaderboard saved → {lb_file}")