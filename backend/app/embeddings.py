"""
Mistral AI embeddings wrapper with automatic batching.
Handles splitting large input lists into batches to respect API limits.
"""

import logging
from typing import Optional

from mistralai.client import Mistral

from app.config import get_settings

logger = logging.getLogger(__name__)

# Mistral embedding API accepts up to 16 texts per request
MAX_BATCH_SIZE = 16


class EmbeddingsManager:
    """
    Wraps the Mistral embeddings API with batching and error handling.

    The mistral-embed model produces 1024-dimensional float vectors.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY must be set in the environment or passed explicitly."
            )
        self.client = Mistral(api_key=self.api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Automatically splits *texts* into batches of MAX_BATCH_SIZE to stay
        within API limits. Returns a list of embedding vectors (each a list
        of 1024 floats) in the same order as the input.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[batch_start : batch_start + MAX_BATCH_SIZE]
            batch_num = (batch_start // MAX_BATCH_SIZE) + 1
            total_batches = (len(texts) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

            logger.debug(
                "Embedding batch %d/%d (%d texts)", batch_num, total_batches, len(batch)
            )

            try:
                response = await self.client.embeddings.create_async(
                    model=self.model,
                    inputs=batch,  # SDK uses `inputs` (plural)
                )

                # Extract embedding vectors from the response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as exc:
                logger.error(
                    "Embedding batch %d failed: %s", batch_num, exc
                )
                raise RuntimeError(
                    f"Failed to generate embeddings for batch {batch_num}: {exc}"
                ) from exc

        logger.info(
            "Generated %d embeddings (dimension=%d)", len(all_embeddings), self.dimension
        )
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single query string.
        Convenience wrapper around embed_texts for single inputs.
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]
