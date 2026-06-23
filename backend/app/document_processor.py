"""
Multi-format document processor for the RAG pipeline.
Supports: PDF (via PyMuPDF/fitz), DOCX (via python-docx), plain text,
and audio files (via Mistral Voxtral transcription).
"""

import logging
import os
from pathlib import Path

from mistralai.client import Mistral

from app.config import get_settings

logger = logging.getLogger(__name__)

# File extensions recognized by each handler
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".html", ".htm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm", ".mp4"}

ALL_SUPPORTED = PDF_EXTENSIONS | DOCX_EXTENSIONS | TEXT_EXTENSIONS | AUDIO_EXTENSIONS


class DocumentProcessor:
    """Extracts text content from various document formats."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def is_supported(self, filename: str) -> bool:
        """Check whether the file extension is supported."""
        ext = Path(filename).suffix.lower()
        return ext in ALL_SUPPORTED

    def get_file_type(self, filename: str) -> str:
        """Return a human-readable file type string."""
        ext = Path(filename).suffix.lower()
        if ext in PDF_EXTENSIONS:
            return "pdf"
        elif ext in DOCX_EXTENSIONS:
            return "docx"
        elif ext in AUDIO_EXTENSIONS:
            return "audio"
        else:
            return "text"

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from *file_path*.
        Dispatches to the appropriate handler based on file extension.
        Returns the extracted text as a single string.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in PDF_EXTENSIONS:
            return self._extract_pdf(path)
        elif ext in DOCX_EXTENSIONS:
            return self._extract_docx(path)
        elif ext in AUDIO_EXTENSIONS:
            return await self._transcribe_audio(path)
        elif ext in TEXT_EXTENSIONS:
            return self._extract_text(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── PDF extraction via PyMuPDF (fitz) ────────────────────────────────

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from all pages of a PDF document."""
        try:
            import fitz  # PyMuPDF

            text_parts: list[str] = []
            with fitz.open(str(path)) as doc:
                for page_num, page in enumerate(doc, start=1):
                    page_text = page.get_text("text")
                    if page_text.strip():
                        text_parts.append(page_text.strip())
                    logger.debug("Extracted page %d/%d from %s", page_num, len(doc), path.name)

            full_text = "\n\n".join(text_parts)
            logger.info("Extracted %d characters from PDF: %s", len(full_text), path.name)
            return full_text

        except ImportError:
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF processing. "
                "Install it with: pip install PyMuPDF"
            )
        except Exception as exc:
            logger.error("Failed to extract PDF %s: %s", path.name, exc)
            raise

    # ── DOCX extraction via python-docx ──────────────────────────────────

    def _extract_docx(self, path: Path) -> str:
        """Extract text from a DOCX document, preserving paragraph structure."""
        try:
            from docx import Document

            doc = Document(str(path))
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            full_text = "\n\n".join(paragraphs)
            logger.info("Extracted %d characters from DOCX: %s", len(full_text), path.name)
            return full_text

        except ImportError:
            raise RuntimeError(
                "python-docx is required for DOCX processing. "
                "Install it with: pip install python-docx"
            )
        except Exception as exc:
            logger.error("Failed to extract DOCX %s: %s", path.name, exc)
            raise

    # ── Plain text extraction ────────────────────────────────────────────

    def _extract_text(self, path: Path) -> str:
        """Read a plain-text file with UTF-8 encoding (falls back to latin-1)."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s, falling back to latin-1", path.name)
            text = path.read_text(encoding="latin-1")

        logger.info("Read %d characters from text file: %s", len(text), path.name)
        return text

    # ── Audio transcription via Mistral Voxtral ──────────────────────────

    async def _transcribe_audio(self, path: Path) -> str:
        """
        Transcribe an audio file using Mistral's Voxtral model.
        Saves the transcription to a .txt file alongside the audio file.
        """
        if not self.settings.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY is required for audio transcription")

        client = Mistral(api_key=self.settings.mistral_api_key)

        try:
            # Read the audio file bytes
            file_bytes = path.read_bytes()
            file_name = path.name

            logger.info("Transcribing audio file: %s (%d bytes)", file_name, len(file_bytes))

            # Call Mistral Voxtral for transcription (SDK v2 uses complete_async)
            response = await client.audio.transcriptions.complete_async(
                model=self.settings.audio_model,
                file={
                    "file_name": file_name,
                    "content": file_bytes,
                },
            )

            transcribed_text = response.text if hasattr(response, "text") else str(response)

            # Save transcription alongside the audio file
            transcript_path = path.with_suffix(".transcript.txt")
            transcript_path.write_text(transcribed_text, encoding="utf-8")
            logger.info(
                "Transcription saved to %s (%d chars)",
                transcript_path.name,
                len(transcribed_text),
            )

            return transcribed_text

        except Exception as exc:
            logger.error("Audio transcription failed for %s: %s", path.name, exc)
            raise
