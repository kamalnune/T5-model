"""ROUGE evaluation helpers for generated text summaries."""

from __future__ import annotations

from typing import TypedDict

from rouge_score import rouge_scorer


class RougeMetric(TypedDict):
    """Precision, recall, and F1 values for one ROUGE metric."""

    precision: float
    recall: float
    f1: float


class RougeScores(TypedDict):
    """ROUGE-1, ROUGE-2, and ROUGE-L evaluation results."""

    rouge1: RougeMetric
    rouge2: RougeMetric
    rougeL: RougeMetric


def calculate_rouge(
    reference_summary: str,
    generated_summary: str,
) -> RougeScores:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Empty inputs return zero for every score because no meaningful overlap can
    be measured.
    """
    if not reference_summary or not generated_summary:
        zero_score: RougeMetric = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"rouge1": zero_score, "rouge2": zero_score, "rougeL": zero_score}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference_summary, generated_summary)

    return {
        "rouge1": _metric_to_dict(scores["rouge1"]),
        "rouge2": _metric_to_dict(scores["rouge2"]),
        "rougeL": _metric_to_dict(scores["rougeL"]),
    }


def _metric_to_dict(metric: rouge_scorer.Score) -> RougeMetric:
    """Convert a rouge-score metric object into a plain dictionary."""
    return {
        "precision": float(metric.precision),
        "recall": float(metric.recall),
        "f1": float(metric.fmeasure),
    }