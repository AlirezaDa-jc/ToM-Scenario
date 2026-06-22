from __future__ import annotations

from typing import Dict


# =========================================================
# PROMPT TEMPLATES
# =========================================================

_TEMPLATES: Dict[str, str] = {
    "default": (
        "Story:\n{story}\n\n"
        "Question:\n{question}\n\n"
        "**STRICT OUTPUT RULES:**\n"
        "- Answer with exactly one location name.\n"
        "- Do not explain.\n"
        "- Do not add punctuation.\n"
        "- Do not mention any other location.\n\n"
        "Location:"
    ),
    "minimal": (
        "{story}\n"
        "{question}"
    ),
    "cot": (
        "Story:\n{story}\n\n"
        "Question:\n{question}\n\n"

        "MAX 64 TOKENS.\n"
        "OUTPUT FORMAT:\n"
        "ANSWER: <location>\n"
        "REASON: <10 words max>\n\n"

        "The ANSWER line is mandatory and must appear first.\n"
        "One location name only.\n"
    )
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