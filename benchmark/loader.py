from __future__ import annotations

import json
from pathlib import Path
from typing import List


class DatasetLoader:
    """
    Loads generated scenarios/questions JSON from disk.
    No logic — only I/O.
    """

    @staticmethod
    def load(path: str | Path) -> List[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)