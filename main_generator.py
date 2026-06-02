from __future__ import annotations

import json
from pathlib import Path

from core.executor import ScenarioExecutor
from core.questions import QuestionGenerator
from core.scenario import ScenarioConfig, ScenarioGenerator, TemplateType

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_output(result, qset) -> dict:
    scenario = result.scenario
    return {
        "scenario_id": qset.scenario_id,
        "template_type": scenario.template_type.value,
        "seed": scenario.config.seed,
        "agents": scenario.agents,
        "objects": scenario.objects,
        "locations": scenario.locations,
        "initial_objects": scenario.initial_objects,
        "events": [
            {
                "step": i + 1,
                "actor": e.actor,
                "target_object": e.target_object,
                "from_location": e.from_location,
                "to_location": e.to_location,
                "visible_to": e.visible_to,
            }
            for i, e in enumerate(scenario.events)
        ],
        "story": qset.story,
        "ground_truth": result.world_truth,
        "first_order_beliefs": result.first_order_beliefs,
        "second_order_beliefs": result.second_order_beliefs,
        "questions": [
            {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "answer": q.answer,
                "reasoning_type": q.reasoning_type.value,
                "target_object": q.target_object,
                "target_agent": q.target_agent,
                "target_other_agent": q.target_other_agent,
                "expected_location": q.expected_location,
            }
            for q in qset.questions
        ]
    }


if __name__ == "__main__":

    all_scenarios = []

    for template in TemplateType:
        for seed in [42, 99, 7]:
            config = ScenarioConfig(template_type=template, seed=seed)
            scenario = ScenarioGenerator.generate(config)
            result = ScenarioExecutor.run(scenario)
            qset = QuestionGenerator.generate(result)
            all_scenarios.append(build_output(result, qset))

    output_file = OUTPUT_DIR / "scenarios.json"
    output_file.write_text(json.dumps(all_scenarios, indent=2, ensure_ascii=False))

    total_questions = sum(len(s["questions"]) for s in all_scenarios)
    print(f"Generated {len(all_scenarios)} scenarios, {total_questions} questions → {output_file}")