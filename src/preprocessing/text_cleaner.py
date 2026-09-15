"""Utilities for preparing text before T5 summarization."""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Normalize tabs and repeated horizontal whitespace without removing punctuation.

    Newlines are retained so that paragraph boundaries remain meaningful.
    Empty or whitespace-only input returns an empty string.
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    return normalized.strip()


def remove_extra_spaces(text: str) -> str:
    """Remove spaces directly inside lines and around paragraph boundaries."""
    if not text:
        return ""

    lines = (line.strip() for line in text.split("\n"))
    return "\n".join(lines)


def remove_empty_lines(text: str) -> str:
    """Remove blank lines while preserving the order of non-empty lines."""
    if not text:
        return ""

    return "\n".join(line for line in text.splitlines() if line.strip())


def clean_text(text: str) -> str:
    """Clean document formatting while preserving words and meaningful punctuation."""
    if not text:
        return ""

    cleaned = normalize_whitespace(text)
    cleaned = remove_extra_spaces(cleaned)
    cleaned = remove_empty_lines(cleaned)
    return cleaned


def prepare_text_for_t5(text: str) -> str:
    """Clean text and add the task prefix expected by the T5 summarization model."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    prefix = "summarize: "
    return cleaned if cleaned.lower().startswith(prefix) else prefix + cleaned


def split_long_text_into_chunks(
    text: str,
    max_chunk_length: int = 2000,
) -> list[str]:
    """Split text into word-boundary chunks suitable for T5 input.

    The chunk limit is measured in characters, which provides a simple,
    predictable safeguard before tokenization. Words are never split in half.
    """
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be greater than zero.")

    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        separator_length = 1 if current_words else 0
        proposed_length = current_length + separator_length + len(word)

        if current_words and proposed_length > max_chunk_length:
            chunks.append(" ".join(current_words))
            current_words = []
            current_length = 0

        # Keep an unusually long word intact rather than silently truncating it.
        current_words.append(word)
        current_length += (1 if len(current_words) > 1 else 0) + len(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks