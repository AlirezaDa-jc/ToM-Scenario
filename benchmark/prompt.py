from __future__ import annotations

from typing import Dict


# =========================================================
# PROMPT TEMPLATES
# =========================================================

_TEMPLATES: Dict[str, str] = {
    "default": (
        "Story: {story}\n"
        "Question: {question}\n"
        "Answer with only the location name."
    ),
    "minimal": (
        "{story}\n"
        "{question}"
    ),
    "cot": (
        "Story: {story}\n"
        "Question: {question}\n"
        "Think step by step, then answer with only the location name."
    ),
}


# =========================================================
# PROMPT BUILDER
# =========================================================

class PromptBuilder:
    """
    Converts a story + question text into a model prompt.
    Supports multiple named templates.
    """

    @staticmethod
    def build(
        story: str,
        question_text: str,
        template_name: str = "default"
    ) -> str:
        if template_name not in _TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(_TEMPLATES.keys())}"
            )
        return _TEMPLATES[template_name].format(
            story=story,
            question=question_text
        )

    @staticmethod
    def available_templates() -> list[str]:
        return list(_TEMPLATES.keys())