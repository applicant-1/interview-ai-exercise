"""Agent implementations."""

from pydantic_ai import Agent

from ai_exercise.constants import SETTINGS
from ai_exercise.models import ContextRailOutput

rewriter = Agent(
    SETTINGS.completions_model,
    system_prompt="""\
The user will provide a query to you relating to OpenAPI specifications. \
Rewrite the query in a way that is optimized for searching through embedded \
chunks of OpenAPI specification (chunks of JSON) within a vector store.\
""",
    result_type=str,
    model_settings={
        "temperature": 0.0,
    },
)

responder = Agent(
    SETTINGS.completions_model,
    system_prompt="Please answer the question based on the user-provided context",
    result_type=str,
    model_settings={
        "temperature": 0.0,
    },
)

context_rail = Agent(
    SETTINGS.completions_model,
    system_prompt="Please answer the question based on the user-provided context",
    result_type=ContextRailOutput,
    model_settings={
        "temperature": 0.0,
    },
)


def create_responder_prompt(query: str, context: list[str]) -> str:
    """Create a prompt combining query and context"""
    context_str = "\n\n".join(context)
    return f"""Context:
{context_str}

Question: {query}

Answer:"""
