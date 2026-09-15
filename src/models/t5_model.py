"""T5-based text summarization inference utilities.

This module contains the model-loading and inference layer only. Streamlit pages
can import :class:`T5Summarizer` without needing to manage model lifecycle or
device selection themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 512


@st.cache_resource
def load_t5_model() -> tuple[T5ForConditionalGeneration, T5Tokenizer, torch.device]:
    """Load and cache the pretrained T5 model, tokenizer, and inference device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Download the model artifacts once, then reuse them across Streamlit
        # reruns and sessions through the resource cache.
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to load the pretrained summarization model '{MODEL_NAME}'."
        ) from exc

    return model, tokenizer, device


@dataclass
class T5Summarizer:
    """Beginner-friendly wrapper around T5 text summarization inference."""

    max_input_length: int = MAX_INPUT_LENGTH

    def summarize_text(
        self,
        text: str,
        *,
        max_length: int = 150,
        min_length: int = 40,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        temperature: float = 1.0,
    ) -> str:
        """Generate a summary for ``text`` using configurable decoding options."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        if min_length < 0 or max_length <= 0 or min_length > max_length:
            raise ValueError("Use valid generation lengths with min_length <= max_length.")
        if num_beams < 1:
            raise ValueError("num_beams must be at least 1.")
        if length_penalty <= 0:
            raise ValueError("length_penalty must be greater than zero.")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        model, tokenizer, device = load_t5_model()

        try:
            # T5 is trained with this task prefix; truncation bounds memory use
            # for documents longer than the model's supported input window.
            model_input = tokenizer(
                f"summarize: {text.strip()}",
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
            )
            input_ids = model_input["input_ids"].to(device)
            attention_mask = model_input["attention_mask"].to(device)

            # Beam search improves summary quality; sampling is enabled only
            # when a non-default temperature requests stochastic decoding.
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "temperature": temperature,
                "early_stopping": True,
            }
            if temperature != 1.0:
                generation_kwargs["do_sample"] = True

            with torch.inference_mode():
                generated_ids = model.generate(**generation_kwargs)

            return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError("Text summarization failed during model inference.") from exc


def summarize_text(
    text: str,
    *,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    temperature: float = 1.0,
) -> str:
    """Summarize text using a cached default :class:`T5Summarizer` instance."""
    summarizer = T5Summarizer()
    return summarizer.summarize_text(
        text,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
    )