"""
Text chunking module for the RAG pipeline.
Splits text into overlapping chunks using paragraph and sentence boundaries
to preserve semantic coherence within each chunk.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into chunks of a target size with configurable overlap.

    Strategy:
      1. Split on double-newlines (paragraph boundaries) first.
      2. If a paragraph exceeds chunk_size, split it on sentence boundaries.
      3. If a single sentence exceeds chunk_size, hard-split on character count.
      4. Merge small segments into chunks ≤ chunk_size, carrying over `overlap`
         characters from the previous chunk to maintain context continuity.
    """

    # Sentence-ending punctuation followed by whitespace or end-of-string
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────

    def split_text(self, text: str) -> list[str]:
        """
        Split *text* into a list of overlapping chunks.

        Returns an empty list when the input is blank or whitespace-only.
        """
        if not text or not text.strip():
            return []

        # Normalise whitespace while preserving paragraph breaks
        text = text.strip()

        # Step 1 – break into paragraphs
        paragraphs = self._split_paragraphs(text)

        # Step 2 – break oversized paragraphs into sentences
        segments = self._split_into_segments(paragraphs)

        # Step 3 – merge segments into properly sized chunks with overlap
        chunks = self._merge_segments(segments)

        logger.info("Split text (%d chars) into %d chunks", len(text), len(chunks))
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text on double-newline paragraph boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_segments(self, paragraphs: list[str]) -> list[str]:
        """
        Ensure every segment is ≤ chunk_size.
        Paragraphs that fit are kept whole; oversized ones are broken into
        sentences, and oversized sentences are hard-split.
        """
        segments: list[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                segments.append(para)
            else:
                # Paragraph too big → split into sentences
                sentences = self.SENTENCE_SPLIT_RE.split(para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(sentence) <= self.chunk_size:
                        segments.append(sentence)
                    else:
                        # Sentence still too big → hard-split
                        segments.extend(self._hard_split(sentence))
        return segments

    def _hard_split(self, text: str) -> list[str]:
        """
        Split *text* into pieces of at most *chunk_size* characters,
        trying to break on the last whitespace before the limit.
        """
        pieces: list[str] = []
        while len(text) > self.chunk_size:
            # Try to find a whitespace boundary near the limit
            split_idx = text.rfind(" ", 0, self.chunk_size)
            if split_idx == -1:
                # No whitespace found – force split at chunk_size
                split_idx = self.chunk_size
            pieces.append(text[:split_idx].strip())
            text = text[split_idx:].strip()
        if text:
            pieces.append(text)
        return pieces

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """
        Greedily merge segments into chunks of up to *chunk_size* characters.
        Each new chunk starts with an overlap window from the end of the
        previous chunk to preserve cross-boundary context.
        """
        if not segments:
            return []

        chunks: list[str] = []
        current_chunk = segments[0]

        for segment in segments[1:]:
            # Check if appending this segment (with a space) still fits
            candidate = current_chunk + "\n\n" + segment
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Flush the current chunk
                chunks.append(current_chunk)

                # Create overlap from the tail of the finished chunk
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""

                # Find a clean word boundary for the overlap start
                if overlap_text and not overlap_text[0].isspace():
                    space_idx = overlap_text.find(" ")
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx:].strip()

                # Start new chunk with overlap + new segment
                if overlap_text:
                    current_chunk = overlap_text + "\n\n" + segment
                else:
                    current_chunk = segment

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
