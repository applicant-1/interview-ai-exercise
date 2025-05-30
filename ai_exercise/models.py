"""Types for the API."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class Document:
    """A document to be added to the vector store."""

    page_content: str
    metadata: dict = None


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class LoadDocumentsOutput(BaseModel):
    """Model for the load documents route output."""

    status: str


class ChatQuery(BaseModel):
    """Model for the chat input."""

    query: str


class ChatOutput(BaseModel):
    """Model for the chat route output."""

    message: str


class ContextRailOutput(BaseModel):
    """Model for the context rail output."""

    is_answerable: bool = Field(
        description="Indicates if the question can be answered based on the context provided."
    )
    response_to_user: str = Field(
        description="""\
A polite response to the user if unable to answer the question from the context. \
Otherwise, empty."""
    )
