from __future__ import annotations

from abc import ABC, abstractmethod


# =========================================================
# MODEL ADAPTER — abstract interface
# All model integrations must implement this.
# =========================================================

class ModelAdapter(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to model, return raw string response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier for EvaluationResult and reporting."""
        ...


# =========================================================
# DUMMY ADAPTER — for pipeline testing without any API
# =========================================================

class DummyAdapter(ModelAdapter):
    """
    Always returns a fixed answer.
    Used to verify the benchmark pipeline end-to-end
    without any model API.
    """

    def __init__(self, fixed_answer: str = "Box_A"):
        self._fixed_answer = fixed_answer

    def generate(self, prompt: str) -> str:
        return self._fixed_answer

    @property
    def name(self) -> str:
        return f"dummy({self._fixed_answer})"