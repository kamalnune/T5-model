"""Streamlit page for summarizing multiple TXT files in one batch."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.t5_model import T5Summarizer
from src.utils.file_utils import read_txt_file
from src.utils.text_utils import calculate_compression_ratio, count_words


def _process_files(uploaded_files: list[object]) -> pd.DataFrame:
    """Extract and summarize each file, retaining errors in the results."""
    summarizer = T5Summarizer()
    results: list[dict[str, object]] = []
    progress = st.progress(0, text="Preparing batch...")

    for index, uploaded_file in enumerate(uploaded_files):
        filename = getattr(uploaded_file, "name", f"file_{index + 1}.txt")
        result: dict[str, object] = {
            "Filename": filename,
            "Original Word Count": 0,
            "Summary": "",
            "Summary Word Count": 0,
            "Compression Ratio": 0.0,
        }
        try:
            text = read_txt_file(uploaded_file)
            if not text.strip():
                raise ValueError("The file is empty.")

            summary = summarizer.summarize_text(text)
            result.update(
                {
                    "Original Word Count": count_words(text),
                    "Summary": summary,
                    "Summary Word Count": count_words(summary),
                    "Compression Ratio": round(
                        calculate_compression_ratio(text, summary),
                        2,
                    ),
                }
            )
        except (RuntimeError, ValueError, OSError, UnicodeDecodeError) as exc:
            result["Summary"] = f"Error: {exc}"
        results.append(result)
        progress.progress(
            (index + 1) / len(uploaded_files),
            text=f"Processed {index + 1} of {len(uploaded_files)} files",
        )

    progress.empty()
    return pd.DataFrame(results)


def main() -> None:
    """Render the batch summarization workflow."""
    st.set_page_config(
        page_title="Batch Summarization | T5",
        page_icon="📚",
        layout="wide",
    )
    st.title("Batch Summarization")
    st.caption("Summarize multiple TXT documents with one streamlined workflow.")

    uploaded_files = st.file_uploader(
        "Upload TXT files",
        type=["txt"],
        accept_multiple_files=True,
        help="Only plain-text TXT files are supported for batch processing.",
    )

    if not uploaded_files:
        st.info("Upload one or more TXT files to begin.")
        return

    st.write(f"**{len(uploaded_files)} file(s) selected**")
    if st.button("Process Batch", type="primary", use_container_width=True):
        with st.spinner("Summarizing batch..."):
            st.session_state["batch_results"] = _process_files(uploaded_files)

    results = st.session_state.get("batch_results")
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.subheader("Batch Results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="batch_summarization_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()