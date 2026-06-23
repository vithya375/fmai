"""
Pydantic request/response models for the RAG API.
Defines all data structures used across API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Chat Models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request model for the /api/chat endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's question to answer using RAG.",
    )


class SourceDocument(BaseModel):
    """Represents a source chunk used to generate an answer."""
    document_id: str = Field(..., description="ID of the parent document.")
    filename: str = Field(..., description="Original filename of the document.")
    chunk_text: str = Field(..., description="The text content of the retrieved chunk.")
    relevance_score: float = Field(
        ..., description="Similarity score (lower is more relevant for L2 distance)."
    )


class ChatResponse(BaseModel):
    """Response model for the /api/chat endpoint."""
    answer: str = Field(..., description="The generated answer from the LLM.")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents referenced in the answer.",
    )


# ─── Document Models ─────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata for an uploaded and processed document."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    filename: str = Field(..., description="Original filename.")
    file_type: str = Field(..., description="MIME type or extension of the file.")
    file_size: int = Field(..., description="File size in bytes.")
    num_chunks: int = Field(..., description="Number of chunks created from this document.")
    upload_date: str = Field(..., description="ISO format upload timestamp.")


class UploadResponse(BaseModel):
    """Response model for the /api/upload endpoint."""
    message: str = Field(..., description="Status message about the upload.")
    document: DocumentMetadata = Field(
        ..., description="Metadata of the uploaded document."
    )


class DocumentListResponse(BaseModel):
    """Response model for the GET /api/documents endpoint."""
    documents: list[DocumentMetadata] = Field(
        default_factory=list, description="List of all uploaded documents."
    )


class DeleteResponse(BaseModel):
    """Response model for the DELETE /api/documents/{doc_id} endpoint."""
    message: str = Field(..., description="Deletion status message.")
    document_id: str = Field(..., description="ID of the deleted document.")


# ─── Health Check Models ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response model for the /api/health endpoint."""
    status: str = Field(default="healthy", description="Service health status.")
    version: str = Field(default="1.0.0", description="API version.")
    chroma_status: str = Field(..., description="ChromaDB connection status.")
    mistral_configured: bool = Field(
        ..., description="Whether the Mistral API key is set."
    )
    document_count: int = Field(
        default=0, description="Total number of documents in the store."
    )
