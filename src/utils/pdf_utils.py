"""Small helpers for extracting text from PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_text_from_pdf(file: str | Path | BinaryIO) -> str:
    """Extract and combine text from every page in a PDF.

    ``file`` may be a filesystem path or an already-open binary file object.
    Pages without extractable text are skipped. Empty PDFs return an empty
    string, while unreadable PDFs raise a clear ``ValueError``.
    """
    try:
        reader = PdfReader(file)
        page_text: list[str] = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                page_text.append(text.strip())

        return "\n\n".join(text for text in page_text if text)
    except (OSError, AttributeError, PdfReadError, TypeError, ValueError) as exc:
        raise ValueError("Unable to read or extract text from the PDF.") from exc