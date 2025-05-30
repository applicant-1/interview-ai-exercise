"""FastAPI app creation, main API routes."""

import asyncio

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client
from ai_exercise.llm.agents import (
    context_rail,
    create_responder_prompt,
    responder,
    rewriter,
)
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    add_documents,
    build_docs,
    get_json_data_list,
    split_docs,
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    LoadDocumentsOutput,
)
from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.retrieval.vector_store import create_collection

app = FastAPI()

collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to load documents into vector store."""
    json_data_list = get_json_data_list()

    documents = []
    for json_data in json_data_list:
        documents.extend(build_docs(json_data))

    # split docs
    documents = split_docs(documents)

    # load documents into vector store
    add_documents(collection, documents)

    # check the number of documents in the collection
    print(f"Number of documents in collection: {collection.count()}")

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
async def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Rewrite the chat query into a search query
    print(f"Chat query: {chat_query.query}")
    rewritten_query = await rewriter.run(chat_query.query)
    print(f"Rewritten query: {rewritten_query}")

    # Get relevant chunks from the collection
    relevant_chunks = get_relevant_chunks(
        collection=collection, query=rewritten_query.data, k=SETTINGS.k_neighbors
    )

    # TODO: Rerank the chunks based on relevance

    # Create prompt with context
    prompt = create_responder_prompt(query=chat_query.query, context=relevant_chunks)

    response_task = responder.run(prompt)
    context_rail_task = context_rail.run(prompt)

    print(f"Prompt: {prompt}")

    # Get completion and context check from LLM concurrently
    responder_res, context_res = await asyncio.gather(response_task, context_rail_task)

    # Simple conditional logic to determine if the response is answerable
    if not context_res.data.is_answerable:
        return ChatOutput(message=context_res.data.response_to_user)

    message = responder_res.data
    return ChatOutput(message=message)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
