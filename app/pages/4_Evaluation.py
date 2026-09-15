"""Streamlit page for evaluating generated summaries with ROUGE."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluation.rouge_score import calculate_rouge
from src.models.t5_model import T5Summarizer


def main() -> None:
    """Render the summary evaluation workflow."""
    st.set_page_config(
        page_title="Evaluation | T5",
        page_icon="📊",
        layout="wide",
    )
    st.title("Summary Evaluation")
    st.caption("Compare a T5-generated summary with a human-written reference.")

    with st.expander("What is ROUGE?"):
        st.write(
            "ROUGE measures overlap between a generated summary and a reference "
            "summary. ROUGE-1 compares individual words, ROUGE-2 compares "
            "two-word sequences, and ROUGE-L measures the longest common "
            "subsequence. Each score includes precision, recall, and F1."
        )

    input_columns = st.columns(2)
    with input_columns[0]:
        original_text = st.text_area(
            "Original/reference text",
            height=280,
            placeholder="Paste the original document text here...",
        )
    with input_columns[1]:
        reference_summary = st.text_area(
            "Human/reference summary",
            height=280,
            placeholder="Paste the human-written reference summary here...",
        )

    if st.button("Generate and Evaluate", type="primary", use_container_width=True):
        if not original_text.strip():
            st.warning("Please enter the original text.")
            return
        if not reference_summary.strip():
            st.warning("Please enter a human/reference summary.")
            return

        try:
            summarizer = T5Summarizer()
            with st.spinner("Generating summary and calculating ROUGE scores..."):
                generated_summary = summarizer.summarize_text(original_text)
                scores = calculate_rouge(reference_summary, generated_summary)
            st.session_state["evaluation_summary"] = generated_summary
            st.session_state["evaluation_scores"] = scores
        except (RuntimeError, ValueError) as exc:
            st.error(f"Evaluation could not be completed: {exc}")
            return

    generated_summary = st.session_state.get("evaluation_summary", "")
    scores = st.session_state.get("evaluation_scores")
    if generated_summary and isinstance(scores, dict):
        st.divider()
        st.subheader("Generated Summary")
        st.text_area(
            "T5-generated summary",
            value=generated_summary,
            height=180,
            disabled=True,
            label_visibility="collapsed",
        )

        st.subheader("ROUGE Results")
        rows = []
        for metric_name, metric_scores in scores.items():
            rows.append(
                {
                    "Metric": metric_name.upper().replace("ROUGEL", "ROUGE-L"),
                    "Precision": metric_scores["precision"],
                    "Recall": metric_scores["recall"],
                    "F1": metric_scores["f1"],
                }
            )
        results_table = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(
            results_table.style.format("{:.4f}"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()