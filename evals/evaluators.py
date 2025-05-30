"""Simple evaluator implementations."""

from pydantic_ai import Agent

from ai_exercise.constants import SETTINGS
from evals.models import EvalData, EvaluatorOutput

consistency = Agent(
    SETTINGS.completions_model,
    system_prompt="""\
The user will provide a query, the groundtruth expected response to that query. \
And the actual response provided by an LLM model. \
Evaluate whether the actual response is consistent with the expected response. \
""",
    result_type=EvaluatorOutput,
    model_settings={
        "temperature": 0.0,
    },
)


def create_consistency_prompt(eval: EvalData) -> str:
    """Create a prompt combining query, expected and actual responses"""
    return f"""Query:
{eval.input}

Expected response: {eval.expected}

Actual response: {eval.actual}"""
