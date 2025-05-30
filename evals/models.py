"""Evaluation data model."""

from pydantic import BaseModel


class EvalData(BaseModel):
    """Data model for evaluation input and output."""

    id: int
    input: str
    expected: str
    actual: str | None = None

    # Evaluation results
    is_consistent: bool | None = None
    explanation: str | None = None


class EvaluatorOutput(BaseModel):
    """Output from an evaluator run."""

    is_consistent: bool | None = None
    explanation: str | None = None
