"""
RAG (Retrieval-Augmented Generation) engine.
Orchestrates the full pipeline: query embedding → retrieval → context
assembly → LLM generation with source attribution.
"""

import logging
from typing import Optional

from mistralai.client import Mistral

from app.config import get_settings
from app.embeddings import EmbeddingsManager
from app.vector_store import VectorStore
from app.models import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)

# ── Relevance threshold ──────────────────────────────────────────────────────
# Cosine distance: 0 = identical, 2 = opposite.  Chunks above this threshold
# are considered irrelevant and will NOT be included as context.
RELEVANCE_THRESHOLD = 0.75

# System prompt that instructs the LLM how to behave as a RAG assistant
RAG_SYSTEM_PROMPT = """You are RAG a friendly, knowledgeable AI assistant that helps users explore their uploaded documents.

PERSONALITY:
- Be warm, conversational, and approachable.  Greet users naturally when they say hello.
- If the user makes casual conversation (e.g. "hi", "thanks", "how are you"), respond like a helpful friend — don't refuse or say you can't answer.
- Use a friendly but professional tone.

WHEN ANSWERING DOCUMENT QUESTIONS:
- Use the provided context documents to give accurate, well-structured answers.
- Cite which source documents you are referencing (e.g. "According to filename.pdf…").
- If multiple sources are relevant, synthesize them into a coherent answer.
- If the context doesn't fully cover the question, say what you *can* answer from the context and note what's missing.
- Do NOT fabricate facts that aren't in the documents.

WHEN NO CONTEXT IS RELEVANT:
- If the retrieved context doesn't seem related to the user's question, let them know you couldn't find relevant information in their documents and suggest they upload more material or rephrase the question.
- You can still answer general greetings, thank-yous, and simple conversational messages without needing document context.

FORMATTING:
- Use markdown for clarity: **bold** for emphasis, bullet lists for multiple points, and headings for long answers.
- Keep answers concise but complete.
"""


class RAGEngine:
    """
    Full RAG pipeline: embed query → retrieve top-K chunks → generate answer.
    """

    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        vector_store: VectorStore,
    ) -> None:
        self.settings = get_settings()
        self.embeddings = embeddings_manager
        self.vector_store = vector_store

        # Mistral chat client
        if not self.settings.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY is required for the RAG engine")
        self.client = Mistral(api_key=self.settings.mistral_api_key)

    async def query(self, question: str) -> ChatResponse:
        """
        Run the full RAG pipeline for a user question.

        Steps:
          1. Embed the question using Mistral embeddings.
          2. Retrieve the top-K most relevant chunks from ChromaDB.
          3. Filter by relevance threshold.
          4. Assemble a context prompt from the retrieved chunks.
          5. Send context + question to the Mistral chat model.
          6. Return the answer with source document references.
        """
        logger.info("RAG query: %s", question[:100])

        # ── Step 1: Embed the question ────────────────────────────────────
        query_embedding = await self.embeddings.embed_query(question)

        # ── Step 2: Retrieve relevant chunks ──────────────────────────────
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.settings.top_k,
        )

        # Unpack ChromaDB results (they come wrapped in an outer list)
        retrieved_docs = results["documents"][0] if results["documents"][0] else []
        retrieved_metas = results["metadatas"][0] if results["metadatas"][0] else []
        retrieved_dists = results["distances"][0] if results["distances"][0] else []

        # ── Step 3: Filter by relevance threshold ─────────────────────────
        context_parts: list[str] = []
        source_documents: list[SourceDocument] = []

        for idx, (doc_text, meta, distance) in enumerate(
            zip(retrieved_docs, retrieved_metas, retrieved_dists), start=1
        ):
            if distance > RELEVANCE_THRESHOLD:
                logger.debug(
                    "Skipping chunk %d (distance=%.4f > threshold=%.2f)",
                    idx, distance, RELEVANCE_THRESHOLD,
                )
                continue

            context_parts.append(
                f"[Source {len(context_parts) + 1} — {meta.get('filename', 'unknown')}]\n{doc_text}"
            )
            source_documents.append(
                SourceDocument(
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", "unknown"),
                    chunk_text=doc_text[:500],  # truncate for response
                    relevance_score=round(float(distance), 4),
                )
            )

        # ── Step 4: Build the user message ────────────────────────────────
        if context_parts:
            context_block = "\n\n---\n\n".join(context_parts)
            user_message = (
                f"Here are relevant excerpts from the user's documents:\n\n"
                f"{context_block}\n\n"
                f"---\n\n"
                f"User's message: {question}"
            )
        else:
            # No relevant context — let the LLM handle conversationally
            user_message = f"User's message: {question}"
            if retrieved_docs:
                user_message += (
                    "\n\n(Note: Some documents were retrieved but none were "
                    "relevant enough to the user's message. Respond naturally.)"
                )
            else:
                user_message += (
                    "\n\n(Note: The user has no documents uploaded yet, or the "
                    "knowledge base is empty. Respond naturally and let them know "
                    "they can upload documents to ask questions about them.)"
                )

        logger.debug("Sending %d relevant chunks to LLM", len(context_parts))

        # ── Step 5: Generate answer via Mistral chat ──────────────────────
        try:
            response = await self.client.chat.complete_async(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            answer = response.choices[0].message.content
            logger.info("Generated answer (%d chars) with %d sources", len(answer), len(source_documents))

        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError(f"Failed to generate answer: {exc}") from exc

        # ── Step 6: Return structured response ────────────────────────────
        return ChatResponse(
            answer=answer,
            sources=source_documents,
        )

