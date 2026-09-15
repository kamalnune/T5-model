"""File-reading helpers for supported document formats."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from docx import Document

from src.utils.pdf_utils import extract_text_from_pdf


FileSource = str | Path | BinaryIO


def read_txt_file(file: FileSource, encoding: str = "utf-8") -> str:
    """Read and return the contents of a plain-text file."""
    try:
        if isinstance(file, (str, Path)):
            return Path(file).read_text(encoding=encoding)

        data = file.read()
        if isinstance(data, str):
            return data
        return data.decode(encoding)
    except (OSError, AttributeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("Unable to read the TXT file.") from exc


def read_pdf_file(file: FileSource) -> str:
    """Extract and return text from every page of a PDF file."""
    try:
        return extract_text_from_pdf(file)
    except ValueError as exc:
        raise ValueError("Unable to read the PDF file.") from exc


def read_docx_file(file: FileSource) -> str:
    """Read and combine paragraph text from a DOCX file."""
    try:
        document = Document(file)
        return "\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Unable to read the DOCX file.") from exc


def extract_text_from_uploaded_file(file: FileSource) -> str:
    """Detect a TXT, PDF, or DOCX extension and extract its text.

    File-like objects should expose a ``name`` attribute, as uploaded-file
    objects typically do. Unsupported or missing extensions raise ``ValueError``.
    """
    if isinstance(file, (str, Path)):
        filename = Path(file).name
    else:
        filename = getattr(file, "name", "")

    extension = Path(filename).suffix.lower()
    if extension == ".txt":
        return read_txt_file(file)
    if extension == ".pdf":
        return read_pdf_file(file)
    if extension == ".docx":
        return read_docx_file(file)

    raise ValueError(
        "Unsupported file type. Please provide a TXT, PDF, or DOCX file."
    )