"""
Spatio-Temporal utilities and model checkpoint helpers.
"""

from src.utils.geo import (
    haversine_distance,
    satisfies_distance_constraint,
    satisfies_direction_constraint,
)
from src.utils.time import satisfies_temporal_constraint
from src.utils.checkpoint import (
    ensure_checkpoint,
    load_model_checkpoint,
    DEFAULT_CHECKPOINT_URLS,
)

__all__ = [
    "haversine_distance",
    "satisfies_distance_constraint",
    "satisfies_direction_constraint",
    "satisfies_temporal_constraint",
    "ensure_checkpoint",
    "load_model_checkpoint",
    "DEFAULT_CHECKPOINT_URLS",
]
