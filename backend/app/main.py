"""
FastAPI application — main entry point for the RAG backend.

Endpoints:
  POST   /api/upload              Upload and process a document
  POST   /api/chat                Ask a question (RAG pipeline)
  GET    /api/documents           List all uploaded documents
  DELETE /api/documents/{doc_id}  Delete a document by ID
  GET    /api/health              Health check

Static files:
  The frontend directory (../frontend/) is served at "/" so the
  single-page application works out of the box.
"""

import logging
import mimetypes
import os
import uuid
from contextlib import asynccontextmanager

# Fix MIME types for Windows registry issues
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.chunker import TextChunker
from app.config import get_settings
from app.document_processor import DocumentProcessor
from app.embeddings import EmbeddingsManager
from app.models import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    HealthResponse,
    UploadResponse,
)
from app.rag_engine import RAGEngine
from app.vector_store import VectorStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Settings & directory setup ────────────────────────────────────────────────
settings = get_settings()

UPLOAD_DIR = Path(settings.upload_directory)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path(settings.frontend_directory)

# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavyweight services on application startup."""
    global vector_store, embeddings_manager, rag_engine

    logger.info("Starting RAG backend …")

    # Vector store (ChromaDB) — always available
    try:
        vector_store = VectorStore()
        logger.info("ChromaDB vector store ready")
    except Exception as exc:
        logger.error("Failed to initialise ChromaDB: %s", exc)

    # Embeddings & RAG engine — require API key
    if settings.mistral_api_key:
        try:
            embeddings_manager = EmbeddingsManager()
            rag_engine = RAGEngine(embeddings_manager, vector_store)
            logger.info("Mistral embeddings & RAG engine ready")
        except Exception as exc:
            logger.error("Failed to initialise Mistral services: %s", exc)
    else:
        logger.warning(
            "MISTRAL_API_KEY not set — upload and chat endpoints will not work"
        )

    yield  # Application runs here

    # Shutdown logic (if needed in the future) goes after yield
    logger.info("RAG backend shutting down…")


# ── Application factory ──────────────────────────────────────────────────────
app = FastAPI(
    title="Mistral RAG API",
    description="Retrieval-Augmented Generation backend powered by Mistral AI and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared service instances (created once at startup) ────────────────────────
document_processor = DocumentProcessor()
chunker = TextChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

# These will be initialised in the startup event so we can handle missing keys
# gracefully at import time.
vector_store: VectorStore | None = None
embeddings_manager: EmbeddingsManager | None = None
rag_engine: RAGEngine | None = None


# ═════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract its text, chunk it, embed the chunks,
    and store everything in ChromaDB.
    """
    # ── Guards ────────────────────────────────────────────────────────────
    if embeddings_manager is None or vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Check that MISTRAL_API_KEY is configured.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not document_processor.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: PDF, DOCX, TXT, MD, audio files.",
        )

    # ── Save uploaded file ────────────────────────────────────────────────
    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()
        file_path.write_bytes(content)
        file_size = len(content)
        logger.info("Saved upload: %s (%d bytes)", safe_filename, file_size)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save file.")

    # ── Extract text ──────────────────────────────────────────────────────
    try:
        text = await document_processor.extract_text(str(file_path))
    except Exception as exc:
        # Clean up the saved file on failure
        file_path.unlink(missing_ok=True)
        logger.error("Text extraction failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    if not text.strip():
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="No text content could be extracted from the file."
        )

    # ── Chunk text ────────────────────────────────────────────────────────
    chunks = chunker.split_text(text)
    if not chunks:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Document produced no chunks.")

    logger.info("Chunked into %d pieces", len(chunks))

    # ── Generate embeddings ───────────────────────────────────────────────
    try:
        embeddings = await embeddings_manager.embed_texts(chunks)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=502, detail=f"Embedding generation failed: {exc}"
        )

    # ── Store in ChromaDB ─────────────────────────────────────────────────
    file_type = document_processor.get_file_type(file.filename)
    try:
        num_chunks = vector_store.add_document(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
        )
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        logger.error("ChromaDB insert failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Vector store insert failed: {exc}")

    # ── Build response ────────────────────────────────────────────────────
    doc_meta = DocumentMetadata(
        document_id=document_id,
        filename=file.filename,
        file_type=file_type,
        file_size=file_size,
        num_chunks=num_chunks,
        upload_date=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Upload complete: %s → %d chunks stored", file.filename, num_chunks
    )
    return UploadResponse(
        message=f"Successfully processed '{file.filename}' into {num_chunks} chunks.",
        document=doc_meta,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question and receive a RAG-powered answer with source references.
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not available. Check that MISTRAL_API_KEY is configured.",
        )

    try:
        response = await rag_engine.query(request.question)
        return response
    except Exception as exc:
        logger.error("Chat query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """Return metadata for all uploaded documents."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available.")

    try:
        docs = vector_store.get_all_documents()
        documents = [DocumentMetadata(**doc) for doc in docs]
        return DocumentListResponse(documents=documents)
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}")


@app.delete("/api/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the vector store."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available.")

    # Also remove the uploaded file from disk
    try:
        for f in UPLOAD_DIR.iterdir():
            if f.name.startswith(doc_id):
                f.unlink(missing_ok=True)
                logger.info("Deleted file from disk: %s", f.name)
    except Exception as exc:
        logger.warning("Could not clean up file on disk: %s", exc)

    # Remove from ChromaDB
    deleted = vector_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Document '{doc_id}' not found."
        )

    return DeleteResponse(
        message=f"Document '{doc_id}' and all its chunks have been deleted.",
        document_id=doc_id,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Return service health status."""
    chroma_ok = vector_store.is_healthy() if vector_store else False
    doc_count = vector_store.get_document_count() if vector_store else 0

    return HealthResponse(
        status="healthy" if chroma_ok else "degraded",
        version="1.0.0",
        chroma_status="connected" if chroma_ok else "disconnected",
        mistral_configured=bool(settings.mistral_api_key),
        document_count=doc_count,
    )


# ── Serve frontend static files ──────────────────────────────────────────────
# Mount AFTER API routes so /api/* takes priority.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Serving frontend from %s", FRONTEND_DIR.resolve())
else:
    logger.warning(
        "Frontend directory %s does not exist — static file serving disabled",
        FRONTEND_DIR.resolve(),
    )
