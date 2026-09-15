# AI Text Summarization using T5 Transformers

An advanced natural language processing application that converts long-form
text and documents into concise, meaningful summaries using Google's T5-base
sequence-to-sequence Transformer model.

This project is designed as a professional final-year CSE (AIML) project,
combining transformer-based inference, document processing, evaluation, and a
multi-page Streamlit interface.

## Project Overview

The application provides several summarization workflows:

- Paste text directly and generate a configurable summary.
- Upload TXT, PDF, or DOCX documents for summarization.
- Process multiple TXT files in one batch and download CSV results.
- Compare generated summaries with human-written references using ROUGE.
- Learn how T5 works through an educational model page.

## Problem Statement

Long reports, articles, research papers, and business documents are difficult
to read and review efficiently. Manual summarization is time-consuming and can
be inconsistent. This project addresses that problem by automatically
extracting the most important information while preserving the central meaning
of the source text.

## Motivation

The project demonstrates how modern Transformer models can solve a practical
NLP problem. It provides an accessible implementation for understanding
preprocessing, sequence-to-sequence generation, document extraction, model
evaluation, and responsible interpretation of generated text.

## Objectives

1. Build a reliable T5-based text summarization pipeline.
2. Support plain text and common document formats.
3. Provide configurable generation controls.
4. Enable single-document and batch summarization.
5. Evaluate generated output using ROUGE metrics.
6. Present the complete workflow through a clean Streamlit application.
7. Keep the code modular, testable, and beginner-friendly.

## Features

- T5-base text summarization
- Automatic CPU/CUDA device selection
- Cached model loading for efficient Streamlit usage
- Short, medium, long, and custom summary modes
- Beam search and decoding controls
- TXT, PDF, and DOCX extraction
- Multiple-file TXT batch processing
- Word count, reading time, compression ratio, and reduction metrics
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation
- Download summaries as TXT and batch results as CSV
- Educational explanation of T5 architecture
- Unit tests for core utilities and summarizer behavior

## Technology Stack

- **Language:** Python 3.10+
- **UI:** Streamlit
- **Model:** Hugging Face `t5-base`
- **NLP framework:** Hugging Face Transformers
- **Deep learning backend:** PyTorch
- **PDF extraction:** pypdf
- **DOCX extraction:** python-docx
- **Evaluation:** rouge-score
- **Data processing:** pandas
- **Testing:** pytest

## System Architecture

```text
User
  |
  v
Streamlit Pages
  |
  +--> Text input / File upload / Batch upload
  |
  v
Preprocessing and File Utilities
  |
  v
T5Summarizer --> Cached T5-base Model
  |
  v
Generated Summary
  |
  +--> Text statistics
  +--> TXT/CSV download
  +--> ROUGE evaluation
```

## Project Folder Structure

```text
T5-Advanced-Summarizer/
├── app/
│   ├── main.py
│   └── pages/
│       ├── 1_Summarize.py
│       ├── 2_File_Summarization.py
│       ├── 3_Batch_Summarization.py
│       ├── 4_Evaluation.py
│       └── 5_About_Model.py
├── src/
│   ├── models/t5_model.py
│   ├── preprocessing/text_cleaner.py
│   ├── utils/
│   │   ├── file_utils.py
│   │   ├── pdf_utils.py
│   │   └── text_utils.py
│   └── evaluation/rouge_score.py
├── tests/test_summarizer.py
├── data/
│   ├── input/
│   └── output/
├── assets/
├── logs/
├── requirements.txt
├── run.py
└── README.md
```

## T5 Architecture

T5 stands for **Text-to-Text Transfer Transformer**. It frames many NLP tasks
as text input transformed into text output.

T5 uses an encoder-decoder Transformer architecture:

- The **encoder** reads the complete input sequence and builds contextual
  representations.
- The **decoder** generates the output sequence one token at a time.
- **Self-attention** helps the model understand relationships within the text.
- **Cross-attention** helps the decoder focus on relevant encoder information.
- A subword tokenizer converts text into model-readable token IDs.

The `t5-base` checkpoint contains approximately 220 million parameters and
provides a practical balance between quality and local inference requirements.

## Text Summarization Workflow

```text
Input Text
    ↓
Clean and normalize text
    ↓
Add the "summarize: " task prefix
    ↓
Tokenize and truncate long input
    ↓
T5 encoder
    ↓
Attention and decoder generation
    ↓
Decode output tokens
    ↓
Generated summary and statistics
```

## Preprocessing

The preprocessing layer:

- Normalizes line endings and repeated whitespace.
- Removes unnecessary empty lines and surrounding spaces.
- Preserves meaningful punctuation.
- Handles empty input safely.
- Splits very long documents into word-boundary chunks when required.
- Adds the `summarize: ` prefix expected by T5.

## Model Inference

The model module loads `T5ForConditionalGeneration` and `T5Tokenizer` from
Hugging Face. `load_t5_model()` is decorated with Streamlit
`@st.cache_resource`, so the model is loaded only once per application
process. CUDA is selected when available; otherwise inference runs on CPU.

Generation supports configurable:

- Minimum and maximum output length
- Number of beams
- Length penalty
- No-repeat n-gram size
- Temperature
- Early stopping

## Beam Search

Beam search keeps several likely output sequences while decoding instead of
choosing only the locally highest-probability token. The `num_beams` setting
controls how many candidates are explored. Higher values can improve
fluency and coverage but increase inference time and memory usage.

## ROUGE Evaluation

ROUGE compares a generated summary with a human-written reference summary:

- **ROUGE-1:** overlap of individual words
- **ROUGE-2:** overlap of two-word sequences
- **ROUGE-L:** longest common subsequence overlap

For each metric, the application reports:

- **Precision:** how much generated content overlaps the reference
- **Recall:** how much reference content was captured
- **F1:** harmonic mean of precision and recall

## File Summarization

The file workflow accepts TXT, PDF, and DOCX files. It detects the extension,
extracts text through the utility layer, displays document statistics, and
passes the extracted content to the same T5 inference module used for direct
text summarization.

## Batch Summarization

The batch page accepts multiple TXT files, processes each document
independently, and stores the results in a pandas DataFrame. A failure in one
file is recorded as an error without stopping the remaining files. Results can
be downloaded as CSV.

## Installation

Clone or download the project, then open a terminal in the project directory:

```bash
git clone <repository-url>
cd T5-Advanced-Summarizer
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The first summarization run downloads the `t5-base` model from Hugging Face.
Internet access and sufficient disk space are required for this initial
download.

## Virtual Environment Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Use the project launcher:

```bash
python run.py
```

Or start Streamlit directly:

```bash
streamlit run app/main.py
```

Streamlit will display a local URL, normally
`http://localhost:8501`.

## Example

Input:

```text
Artificial intelligence is changing how organizations analyze information.
Transformer models can process language at scale and support applications such
as question answering, translation, and summarization. Summarization helps
users review long documents more efficiently.
```

Possible output:

```text
Transformer-based AI supports large-scale language analysis, including
summarization, helping users review documents efficiently.
```

Generated text may vary because output depends on model decoding settings and
the supplied input.

## Screenshots

Add project screenshots here after running the application:

```text
![Home page](assets/screenshots/home.png)
![Text summarization](assets/screenshots/summarize.png)
![File summarization](assets/screenshots/file-summarization.png)
![Evaluation page](assets/screenshots/evaluation.png)
```

## Advantages

- Uses a strong pretrained Transformer model.
- Provides a modular and reusable inference layer.
- Supports common document formats.
- Offers both interactive and batch workflows.
- Includes quantitative ROUGE evaluation.
- Runs on CPU and can use CUDA acceleration.
- Demonstrates an end-to-end AIML application suitable for academic review.

## Limitations

- `t5-base` has a finite input context, so very long documents need truncation
  or chunking.
- Summaries may omit details or occasionally produce inaccurate statements.
- CPU inference can be slow for large documents and batches.
- The quality of ROUGE scores depends on the chosen reference summary.
- Scanned PDFs without an embedded text layer require OCR, which is not
  included in the current version.

## Future Enhancements

- Add OCR support for scanned PDFs.
- Implement hierarchical summarization for very long documents.
- Add multilingual T5 model support.
- Add user-selectable model checkpoints.
- Persist history and export richer reports.
- Add authentication and deployment configuration.
- Add human feedback collection and quality dashboards.
- Add automated model benchmarking on a labeled dataset.

## Testing

Run the test suite with pytest:

```bash
pytest -q
```

The tests cover empty and short inputs, normal summarization contracts,
word-counting, compression ratio calculation, and text cleaning. Inference
tests are designed to avoid requiring a GPU.

## Author

**Final-Year CSE (AIML) Student**

This project was developed as an academic demonstration of practical
Transformer-based natural language processing. Update this section with the
author's name, institution, department, and contact links before publishing.