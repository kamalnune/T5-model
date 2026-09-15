"""Interactive page for summarizing pasted text with T5."""

from __future__ import annotations

import streamlit as st

from src.models.t5_model import T5Summarizer
from src.utils.text_utils import (
    calculate_compression_ratio,
    calculate_reduction_percentage,
    count_words,
    estimate_reading_time,
)


SUMMARY_PRESETS = {
    "Short": {"min_length": 20, "max_length": 60},
    "Medium": {"min_length": 40, "max_length": 120},
    "Long": {"min_length": 80, "max_length": 200},
}


def _clear_page() -> None:
    """Clear the current text and generated result."""
    st.session_state["summary_text"] = ""
    st.session_state.pop("generated_summary", None)


def main() -> None:
    """Render the interactive text summarization page."""
    st.set_page_config(
        page_title="Summarize Text | T5",
        page_icon="✍️",
        layout="wide",
    )
    st.title("Summarize Text")
    st.caption("Generate a concise summary from long-form text using T5-base.")

    text = st.text_area(
        "Text to summarize",
        key="summary_text",
        height=300,
        placeholder="Paste an article, report, or other long-form text here...",
    )

    control_columns = st.columns([1.2, 1, 1, 1])
    with control_columns[0]:
        mode = st.selectbox("Summary mode", ["Short", "Medium", "Long", "Custom"])

    preset = SUMMARY_PRESETS.get(mode, {"min_length": 40, "max_length": 120})
    with control_columns[1]:
        min_length = st.number_input(
            "Minimum length",
            min_value=1,
            max_value=500,
            value=preset["min_length"],
            disabled=mode != "Custom",
        )
    with control_columns[2]:
        max_length = st.number_input(
            "Maximum length",
            min_value=1,
            max_value=500,
            value=preset["max_length"],
            disabled=mode != "Custom",
        )
    with control_columns[3]:
        num_beams = st.slider("Number of beams", min_value=1, max_value=8, value=4)

    advanced_columns = st.columns(2)
    with advanced_columns[0]:
        length_penalty = st.slider(
            "Length penalty",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
        )
    with advanced_columns[1]:
        no_repeat_ngram_size = st.slider(
            "No-repeat n-gram size",
            min_value=0,
            max_value=6,
            value=3,
        )

    if min_length > max_length:
        st.error("Minimum length must be less than or equal to maximum length.")

    action_columns = st.columns([1, 1, 4])
    generate = action_columns[0].button(
        "Generate Summary",
        type="primary",
        use_container_width=True,
    )
    clear = action_columns[1].button("Clear", use_container_width=True)

    if clear:
        _clear_page()
        st.rerun()

    if generate:
        if not text.strip():
            st.warning("Please enter text before generating a summary.")
        elif min_length > max_length:
            st.error("Please correct the length settings before continuing.")
        else:
            try:
                summarizer = T5Summarizer()
                with st.spinner("Generating your summary..."):
                    summary = summarizer.summarize_text(
                        text,
                        max_length=int(max_length),
                        min_length=int(min_length),
                        num_beams=int(num_beams),
                        length_penalty=float(length_penalty),
                        no_repeat_ngram_size=int(no_repeat_ngram_size),
                    )
                st.session_state["generated_summary"] = summary
            except (RuntimeError, ValueError) as exc:
                st.error(f"Unable to generate the summary: {exc}")

    summary = st.session_state.get("generated_summary", "")
    if summary:
        st.divider()
        st.subheader("Summary Results")
        result_columns = st.columns(2)
        with result_columns[0]:
            st.markdown("**Original text**")
            st.text_area("Original text", value=text, height=260, disabled=True, label_visibility="collapsed")
        with result_columns[1]:
            st.markdown("**Generated summary**")
            st.text_area("Generated summary", value=summary, height=260, disabled=True, label_visibility="collapsed")

        original_words = count_words(text)
        summary_words = count_words(summary)
        metrics = st.columns(5)
        metrics[0].metric("Original words", original_words)
        metrics[1].metric("Summary words", summary_words)
        metrics[2].metric("Compression ratio", f"{calculate_compression_ratio(text, summary):.2f}x")
        metrics[3].metric("Reduction", f"{calculate_reduction_percentage(text, summary):.2f}%")
        metrics[4].metric("Reading time", f"{estimate_reading_time(summary):.2f} min")

        st.download_button(
            "Download Summary as TXT",
            data=summary,
            file_name="t5_summary.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()