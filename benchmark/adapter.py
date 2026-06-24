from __future__ import annotations

import random
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


class RandomBaselineAdapter(ModelAdapter):
    """
    Randomly selects a valid location from the scenario's locations list.
    Used as a statistical baseline — expected accuracy ≈ 1/num_locations.
    With 10 locations in dataset, expected ≈ 10%.

    Since generate() only receives the prompt string, we extract locations
    from the prompt by matching against the known location pool.
    """

    _KNOWN_LOCATIONS = [
        "Box_A", "Box_B", "Basket", "Drawer", "Shelf",
        "Bag", "Chest", "Jar", "Pouch", "Cupboard"
    ]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        valid = [loc for loc in self._KNOWN_LOCATIONS if loc.lower() in prompt_lower]
        if not valid:
            valid = self._KNOWN_LOCATIONS
        return self._rng.choice(valid)

    @property
    def name(self) -> str:
        return "random_baseline"