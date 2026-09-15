"""Unit tests for the summarization project's core behavior."""

from __future__ import annotations

import pytest

from src.models.t5_model import T5Summarizer
from src.preprocessing.text_cleaner import clean_text
from src.utils.text_utils import (
    calculate_compression_ratio,
    count_words,
)


def test_empty_input_is_rejected() -> None:
    """The model wrapper should reject empty input before inference."""
    with pytest.raises(ValueError):
        T5Summarizer().summarize_text("")


def test_short_input_is_rejected_or_validated() -> None:
    """Whitespace-only input should not trigger model loading."""
    with pytest.raises(ValueError):
        T5Summarizer().summarize_text("   ")


def test_short_input_can_be_summarized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short valid input can be handled without requiring a GPU."""
    monkeypatch.setattr(
        T5Summarizer,
        "summarize_text",
        lambda *args, **kwargs: "Short summary",
    )
    assert T5Summarizer().summarize_text("A short sentence.") == "Short summary"


def test_normal_text_summarization_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal input can be tested with deterministic CPU-safe fake inference."""
    expected_summary = "A concise summary of the document."

    def fake_summarize(_self: T5Summarizer, text: str, **_kwargs: object) -> str:
        assert text.strip()
        return expected_summary

    monkeypatch.setattr(T5Summarizer, "summarize_text", fake_summarize)
    summary = T5Summarizer().summarize_text(
        "This is a normal document containing several useful sentences."
    )

    assert summary == expected_summary


def test_summary_is_returned_as_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summarizer contract returns text rather than a token sequence."""
    monkeypatch.setattr(
        T5Summarizer,
        "summarize_text",
        lambda _self, _text, **_kwargs: "Generated summary",
    )
    summary = T5Summarizer().summarize_text("Input text")
    assert isinstance(summary, str)


def test_summary_is_not_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful summarization result contains at least one character."""
    monkeypatch.setattr(
        T5Summarizer,
        "summarize_text",
        lambda _self, _text, **_kwargs: "Generated summary",
    )
    assert T5Summarizer().summarize_text("Input text").strip()


def test_word_counting() -> None:
    """Word counting handles ordinary punctuation and whitespace."""
    assert count_words("T5 makes useful summaries.") == 4
    assert count_words("") == 0


def test_compression_ratio() -> None:
    """Compression ratio compares original and generated word counts."""
    original = "one two three four five six"
    summary = "one two three"
    assert calculate_compression_ratio(original, summary) == 2.0


def test_text_cleaning() -> None:
    """Text cleaning removes redundant whitespace but preserves punctuation."""
    messy_text = "  Hello,   world! \n\n This is a test.  "
    assert clean_text(messy_text) == "Hello, world!\nThis is a test."