"""
Evaluation metrics, benchmarking suite, and failure analysis.
"""

from src.evaluation.metrics import (
    evaluate_benchmark,
    format_table5_markdown,
    classify_question_constraints,
    compute_hits_at_k,
)

__all__ = [
    "evaluate_benchmark",
    "format_table5_markdown",
    "classify_question_constraints",
    "compute_hits_at_k",
]
