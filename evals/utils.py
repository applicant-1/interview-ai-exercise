"""Utility functions for evaluation data processing."""

import json

from .models import EvalData


def load_evals(filepath: str) -> list[EvalData]:
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(EvalData(**obj))
    return data
