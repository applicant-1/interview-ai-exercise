"""Generate a response using an LLM."""

from openai import AsyncOpenAI


def create_prompt(query: str, context: list[str]) -> str:
    """Create a prompt combining query and context"""
    context_str = "\n\n".join(context)
    return f"""Please answer the question based on the following context:

Context:
{context_str}

Question: {query}

Answer:"""


async def get_completion(client: AsyncOpenAI, prompt: str, model: str) -> str:
    """Get completion from OpenAI"""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
