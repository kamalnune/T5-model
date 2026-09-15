"""Streamlit page for summarizing uploaded TXT, PDF, and DOCX files."""

from __future__ import annotations

import streamlit as st

from src.models.t5_model import T5Summarizer
from src.utils.file_utils import extract_text_from_uploaded_file
from src.utils.text_utils import (
    calculate_compression_ratio,
    calculate_reduction_percentage,
    count_characters,
    count_words,
    estimate_reading_time,
)


SUMMARY_LENGTHS = {
    "Short": (20, 60),
    "Medium": (40, 120),
    "Long": (80, 200),
}


def _document_statistics(text: str) -> None:
    """Display basic statistics for extracted document text."""
    columns = st.columns(3)
    columns[0].metric("Words", count_words(text))
    columns[1].metric("Characters", count_characters(text))
    columns[2].metric("Reading time", f"{estimate_reading_time(text):.2f} min")


def main() -> None:
    """Render the uploaded-file summarization workflow."""
    st.set_page_config(
        page_title="File Summarization | T5",
        page_icon="📄",
        layout="wide",
    )
    st.title("File Summarization")
    st.caption("Upload a document and generate a concise T5-base summary.")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        help="Supported formats: PDF, TXT, and DOCX.",
    )

    if uploaded_file is None:
        st.info("Upload a PDF, TXT, or DOCX file to get started.")
        return

    st.success(f"Uploaded document: **{uploaded_file.name}**")
    try:
        with st.spinner("Extracting document text..."):
            extracted_text = extract_text_from_uploaded_file(uploaded_file)
    except ValueError as exc:
        st.error(f"Unable to read this file: {exc}")
        return

    if not extracted_text.strip():
        st.warning("The uploaded document does not contain extractable text.")
        return

    st.subheader("Document Statistics")
    _document_statistics(extracted_text)

    with st.expander("Preview extracted text"):
        st.text_area(
            "Extracted text",
            value=extracted_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    summary_length = st.selectbox(
        "Summary length",
        options=list(SUMMARY_LENGTHS),
        index=1,
    )
    min_length, max_length = SUMMARY_LENGTHS[summary_length]

    if st.button("Generate Summary", type="primary", use_container_width=True):
        try:
            summarizer = T5Summarizer()
            with st.spinner("Generating summary..."):
                summary = summarizer.summarize_text(
                    extracted_text,
                    min_length=min_length,
                    max_length=max_length,
                )
            st.session_state["file_summary"] = summary
        except (RuntimeError, ValueError) as exc:
            st.error(f"Unable to generate the summary: {exc}")
            return

    summary = st.session_state.get("file_summary", "")
    if summary:
        st.divider()
        st.subheader("Summary Results")
        columns = st.columns(2)
        with columns[0]:
            st.markdown("**Original text**")
            st.text_area(
                "Original document text",
                value=extracted_text,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
        with columns[1]:
            st.markdown("**Generated summary**")
            st.text_area(
                "Generated document summary",
                value=summary,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

        metric_columns = st.columns(4)
        metric_columns[0].metric("Original words", count_words(extracted_text))
        metric_columns[1].metric("Summary words", count_words(summary))
        metric_columns[2].metric(
            "Compression ratio",
            f"{calculate_compression_ratio(extracted_text, summary):.2f}x",
        )
        metric_columns[3].metric(
            "Reduction",
            f"{calculate_reduction_percentage(extracted_text, summary):.2f}%",
        )
        st.download_button(
            "Download Summary as TXT",
            data=summary,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()