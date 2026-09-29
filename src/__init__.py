"""
Spatial-Temporal Knowledge Graph Question Answering (STKGQA) Core Package.
"""

from src.data import (
    load_stqad,
    get_vocabularies,
    STQADataset,
    collate_stqad_fn,
    clean_entity,
    classify_question_clues,
)
from src.evaluation import (
    evaluate_benchmark,
    format_table5_markdown,
    classify_question_constraints,
    compute_hits_at_k,
)
from src.utils import (
    haversine_distance,
    satisfies_distance_constraint,
    satisfies_direction_constraint,
    satisfies_temporal_constraint,
    ensure_checkpoint,
    load_model_checkpoint,
    DEFAULT_CHECKPOINT_URLS,
)

__all__ = [
    # Data
    "load_stqad",
    "get_vocabularies",
    "STQADataset",
    "collate_stqad_fn",
    "clean_entity",
    "classify_question_clues",
    # Evaluation
    "evaluate_benchmark",
    "format_table5_markdown",
    "classify_question_constraints",
    "compute_hits_at_k",
    # Utils
    "haversine_distance",
    "satisfies_distance_constraint",
    "satisfies_direction_constraint",
    "satisfies_temporal_constraint",
    "ensure_checkpoint",
    "load_model_checkpoint",
    "DEFAULT_CHECKPOINT_URLS",
]
