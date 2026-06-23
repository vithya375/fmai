"""
ChromaDB vector store wrapper for persistent document storage and retrieval.
Manages document chunks, embeddings, and metadata in a ChromaDB collection.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent ChromaDB wrapper for the RAG system.

    Each document is stored as multiple chunks, each with:
      - id: unique chunk ID  (format: "{doc_id}_chunk_{idx}")
      - embedding: 1024-dim vector from Mistral
      - document: the chunk text
      - metadata: document_id, filename, file_type, upload_date,
                  file_size, num_chunks, chunk_index
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

        logger.info(
            "ChromaDB initialized — collection '%s' has %d items",
            settings.chroma_collection_name,
            self.collection.count(),
        )

    # ── Add documents ─────────────────────────────────────────────────────

    def add_document(
        self,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        filename: str,
        file_type: str,
        file_size: int,
    ) -> int:
        """
        Add a document's chunks and embeddings to the vector store.

        Returns the number of chunks added.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        upload_date = datetime.now(timezone.utc).isoformat()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embedding_list: list[list[float]] = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            embedding_list.append(embedding)
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size,
                "num_chunks": len(chunks),
                "chunk_index": idx,
                "upload_date": upload_date,
            })

        # Upsert into ChromaDB (handles both insert and update)
        self.collection.upsert(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Added %d chunks for document '%s' (id=%s)",
            len(chunks),
            filename,
            document_id,
        )
        return len(chunks)

    # ── Query / similarity search ─────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """
        Perform a similarity search against stored embeddings.

        Returns the raw ChromaDB query result dict with keys:
          ids, documents, metadatas, distances
        """
        if self.collection.count() == 0:
            logger.warning("Query against empty collection – returning no results")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        logger.debug("Query returned %d results", len(results["ids"][0]))
        return results

    # ── Document management ───────────────────────────────────────────────

    def get_all_documents(self) -> list[dict]:
        """
        Return metadata for every unique document in the store.

        De-duplicates by document_id (since each document has many chunks).
        """
        if self.collection.count() == 0:
            return []

        # Fetch all items (only metadata, no embeddings)
        all_items = self.collection.get(include=["metadatas"])

        # De-duplicate by document_id
        seen: dict[str, dict] = {}
        for meta in all_items["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "file_type": meta["file_type"],
                    "file_size": meta.get("file_size", 0),
                    "num_chunks": meta.get("num_chunks", 0),
                    "upload_date": meta.get("upload_date", ""),
                }

        return list(seen.values())

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks belonging to a given document_id.

        Returns True if chunks were found and deleted, False otherwise.
        """
        # Find all chunk IDs for this document
        all_items = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        if not all_items["ids"]:
            logger.warning("No chunks found for document_id=%s", document_id)
            return False

        self.collection.delete(ids=all_items["ids"])
        logger.info(
            "Deleted %d chunks for document_id=%s",
            len(all_items["ids"]),
            document_id,
        )
        return True

    def document_exists(self, document_id: str) -> bool:
        """Check whether any chunks exist for the given document_id."""
        results = self.collection.get(
            where={"document_id": document_id},
            include=[],
            limit=1,
        )
        return len(results["ids"]) > 0

    def get_document_count(self) -> int:
        """Return the total number of unique documents (not chunks)."""
        return len(self.get_all_documents())

    def get_collection_count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self.collection.count()

    def is_healthy(self) -> bool:
        """Quick health check — tries to read the collection count."""
        try:
            _ = self.collection.count()
            return True
        except Exception as exc:
            logger.error("ChromaDB health check failed: %s", exc)
            return False
