"""Main Streamlit landing page for the T5 summarization application."""

from __future__ import annotations

import streamlit as st
import torch


st.set_page_config(
    page_title="AI Text Summarization using T5",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_sidebar() -> None:
    """Render application navigation and model details."""
    with st.sidebar:
        st.title("T5 Summarizer")
        st.caption("Advanced NLP toolkit")
        st.divider()

        st.subheader("Navigation")
        st.page_link("main.py", label="Home", icon="🏠")
        st.page_link("pages/1_Summarize.py", label="Summarize Text", icon="✍️")
        st.page_link(
            "pages/2_File_Summarization.py",
            label="File Summarization",
            icon="📄",
        )
        st.page_link(
            "pages/3_Batch_Summarization.py",
            label="Batch Summarization",
            icon="📚",
        )
        st.page_link("pages/4_Evaluation.py", label="Evaluation", icon="📊")
        st.page_link("pages/5_About_Model.py", label="About the Model", icon="🤖")

        st.divider()
        st.subheader("Model Information")
        st.write("**Model:** T5-base")
        st.write("**Framework:** Hugging Face Transformers")
        st.write("**Backend:** PyTorch")
        backend = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        st.write(f"**Compute:** {backend}")


def _render_feature_card(title: str, description: str, icon: str) -> None:
    """Render one concise feature section."""
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the application home page."""
    _render_sidebar()

    st.markdown(
        """
        <style>
        .hero {
            padding: 2.25rem 2.5rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #12355b, #1d7874);
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin-bottom: 0.5rem; }
        .hero p { font-size: 1.1rem; margin-bottom: 0; }
        .feature-card {
            min-height: 175px;
            padding: 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.75rem;
            background: rgba(128, 128, 128, 0.06);
        }
        .feature-icon { font-size: 1.75rem; }
        .feature-card h3 { margin: 0.5rem 0; }
        .feature-card p { color: #6b7280; }
        </style>
        <section class="hero">
            <h1>AI Text Summarization using T5</h1>
            <p>Transform lengthy documents into clear, meaningful summaries with
            transformer-based natural language processing.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project Overview")
    st.write(
        "This application uses Google's T5-base sequence-to-sequence model to "
        "generate concise summaries from text and supported documents. It is "
        "designed as an accessible demonstration of an end-to-end AIML project."
    )

    st.subheader("What you can do")
    feature_columns = st.columns(3)
    features = [
        ("Summarize text", "Generate summaries with configurable decoding controls.", "✍️"),
        ("Process documents", "Extract and summarize TXT, PDF, and DOCX files.", "📄"),
        ("Evaluate quality", "Compare generated output with reference summaries using ROUGE.", "📊"),
    ]
    for column, feature in zip(feature_columns, features):
        with column:
            _render_feature_card(*feature)

    st.info(
        "Choose a tool from the sidebar to begin. The model is loaded only when "
        "a summarization workflow requires it."
    )


if __name__ == "__main__":
    main()