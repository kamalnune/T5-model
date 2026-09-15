"""Educational Streamlit page explaining the T5 summarization model."""

from __future__ import annotations

import streamlit as st


def _section(title: str, content: str) -> None:
    """Render a consistent educational content section."""
    with st.expander(title, expanded=True):
        st.markdown(content)


def main() -> None:
    """Render the educational T5 model presentation."""
    st.set_page_config(
        page_title="About T5 Model",
        page_icon="🤖",
        layout="wide",
    )
    st.title("About the T5 Model")
    st.caption("A simple guide to the transformer behind this summarization system.")

    st.markdown(
        """
        <div style="padding: 1.5rem; border-radius: 1rem;
                    background: linear-gradient(135deg, #12355b, #1d7874);
                    color: white;">
            <h2>Text-to-Text Transfer Transformer</h2>
            <p>T5 treats every language task as text input transformed into text output.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("How a summary is produced")
    st.code(
        "Input Text\n"
        "    ↓\n"
        "Tokenization\n"
        "    ↓\n"
        "T5 Encoder\n"
        "    ↓\n"
        "Attention\n"
        "    ↓\n"
        "T5 Decoder\n"
        "    ↓\n"
        "Generated Summary",
        language="text",
    )

    _section(
        "1. What is T5?",
        "T5 is a transformer-based language model developed by Google. It can "
        "translate, answer questions, classify text, and summarize documents by "
        "representing each task as a text-to-text problem.",
    )
    _section(
        "2. What does the name mean?",
        "**Text-to-Text Transfer Transformer** describes three ideas: text is "
        "both the input and output format, knowledge transfers from pretraining "
        "to many tasks, and the model uses the Transformer neural architecture.",
    )
    _section(
        "3. T5 architecture",
        "T5 uses an encoder-decoder Transformer. The encoder reads the input "
        "context, while the decoder generates the output one token at a time. "
        "The two parts work together through learned representations and attention.",
    )
    _section(
        "4. Encoder-decoder architecture",
        "The encoder converts the source document into contextual representations. "
        "The decoder consults those representations while predicting each next "
        "summary token. This makes the design well suited to sequence-to-sequence tasks.",
    )
    _section(
        "5. Tokenization",
        "Tokenization converts text into smaller pieces called tokens. T5 uses "
        "a subword tokenizer, so uncommon words can be represented by meaningful "
        "parts instead of requiring every complete word in its vocabulary.",
    )
    _section(
        "6. Attention mechanism",
        "Attention lets the model focus on the most relevant parts of the input "
        "when interpreting or generating text. It helps connect important words "
        "even when they are far apart in a document.",
    )
    _section(
        "7. Pretraining",
        "During pretraining, T5 learns general language patterns from a large "
        "collection of text. It reconstructs masked or corrupted spans, learning "
        "grammar, context, and relationships between words.",
    )
    _section(
        "8. Fine-tuning",
        "Fine-tuning adapts the pretrained model to a specific task using labeled "
        "examples. For summarization, the examples pair documents with shorter "
        "human-written summaries.",
    )
    _section(
        "9. How T5 performs summarization",
        "The document is supplied as input and the decoder generates a shorter "
        "sequence that captures important information. Beam search can explore "
        "several likely continuations before selecting a fluent result.",
    )
    _section(
        '10. What does "summarize:" mean?',
        'The prefix `"summarize: "` is a task instruction. T5 was trained to '
        "recognize task prefixes, so this tells the model to produce a summary "
        "rather than translate, classify, or answer a question.",
    )

    model_columns = st.columns(2)
    with model_columns[0]:
        _section(
            "11. T5-base model",
            "This project uses `t5-base`, a balanced version of T5 with about "
            "220 million parameters. It offers strong language performance while "
            "being more practical to run than larger T5 variants.",
        )
    with model_columns[1]:
        _section(
            "12. Advantages",
            "- One model supports many text-to-text tasks.\n"
            "- Pretraining provides strong language understanding.\n"
            "- Encoder-decoder generation suits summarization naturally.\n"
            "- It can run on CPU and accelerate on a CUDA GPU.",
        )

    _section(
        "13. Limitations",
        "- Long inputs must be truncated or processed in chunks.\n"
        "- Generated text can occasionally omit details or contain inaccuracies.\n"
        "- Results depend on decoding settings and input quality.\n"
        "- Inference can require substantial memory and processing time.",
    )
    _section(
        "14. Applications",
        "T5-style systems can support news and research summarization, meeting "
        "notes, document analysis, customer-support workflows, question answering, "
        "translation, and other natural language processing applications.",
    )

    st.info(
        "Tip: Start with the Summarize Text page to experiment with different "
        "summary lengths and generation settings."
    )


if __name__ == "__main__":
    main()