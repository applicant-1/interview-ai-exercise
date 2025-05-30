"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.loading.chunk_json import chunk_data
from ai_exercise.models import Document


def get_json_data(url: str) -> dict[str, Any]:
    """Gets JSON data from the specified URL."""
    response = requests.get(url)
    response.raise_for_status()
    json_data = response.json()

    return json_data


def get_json_data_list() -> list[dict[str, Any]]:
    """Loads multiple JSON data from a list of URLs."""
    with open(SETTINGS.docs_config) as file:
        config = yaml.safe_load(file)
    urls = config["urls"]

    json_data = []
    for url in urls:
        data = get_json_data(url)
        json_data.append(data)

    return json_data


def document_json_array(data: list[dict[str, Any]], source: str) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects."""
    return [
        Document(page_content=json.dumps(item), metadata={"source": source})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects."""
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)
        docs.extend(document_json_array(chunks, attribute))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def add_documents(
    collection: chromadb.Collection, docs: list[Document], batch_size: int = 100
) -> None:
    """Add documents to the collection in batches to avoid exceeding token limits."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        collection.add(
            documents=[doc.page_content for doc in batch_docs],
            metadatas=[doc.metadata or {} for doc in batch_docs],
            ids=[f"doc_{i + j}" for j in range(len(batch_docs))],
        )
