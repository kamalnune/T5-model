"""General text statistics used by the summarization application."""

from __future__ import annotations

import re
from typing import TypedDict


class TextStatistics(TypedDict):
    """Structured text measurements returned by :func:`get_text_statistics`."""

    word_count: int
    character_count: int
    compression_ratio: float
    reduction_percentage: float
    estimated_reading_time_minutes: float


def count_words(text: str) -> int:
    """Return the number of word-like tokens in ``text``."""
    if not text:
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def count_characters(text: str) -> int:
    """Return the number of characters in ``text``, including spaces."""
    return len(text) if text else 0


def calculate_compression_ratio(original: str, summary: str) -> float:
    """Return original word count divided by summary word count.

    A value greater than one indicates that the summary is shorter. If the
    summary has no words, ``0.0`` is returned because a finite ratio is
    undefined.
    """
    summary_words = count_words(summary)
    if summary_words == 0:
        return 0.0
    return count_words(original) / summary_words


def calculate_reduction_percentage(original: str, summary: str) -> float:
    """Return the percentage of original words removed by the summary."""
    original_words = count_words(original)
    if original_words == 0:
        return 0.0
    reduction = (original_words - count_words(summary)) / original_words
    return round(reduction * 100, 2)


def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """Estimate reading time in minutes using the supplied reading speed."""
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be greater than zero.")
    return round(count_words(text) / words_per_minute, 2)


def get_text_statistics(text: str) -> TextStatistics:
    """Return common word, character, compression, and reading-time metrics."""
    return {
        "word_count": count_words(text),
        "character_count": count_characters(text),
        "compression_ratio": calculate_compression_ratio(text, ""),
        "reduction_percentage": calculate_reduction_percentage(text, ""),
        "estimated_reading_time_minutes": estimate_reading_time(text),
    }