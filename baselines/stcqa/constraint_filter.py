"""
Dynamic Spatio-Temporal Constraint Filter for STCQA.
Implements the Answer Filtering Module from Section 5.4 of:
    Dai et al., "Question answering over spatio-temporal knowledge graph",
    Knowledge-Based Systems 329 (2025) 114314.

Theoretical Motivation:
    Purely embedding-based KGQA methods (ComplEx, TComplEx, DistMult) map symbolic
    facts into latent geometric vector spaces. While effective for semantic association,
    they struggle with:
      1. Metric continuous distance calculations (e.g. spherical Haversine distance)
      2. Strict temporal ordering intervals (e.g. before/after/during)
      3. Precise cardinal coordinate inequalities (e.g. lat_cand > lat_clue)

    The Answer Filtering Module acts as a neuro-symbolic post-processor:
    It takes the top-K candidate answers retrieved by the neural scoring function
    and verifies them against explicit spatio-temporal rules, boosting compliant
    entities to the top of the ranked list.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from src.utils_geo import haversine_distance, satisfies_distance_constraint, satisfies_direction_constraint
from src.utils_time import satisfies_temporal_constraint


class ConstraintFilter:
    """
    Evaluates spatio-temporal constraints and re-ranks neural candidate entities.
    """
    def __init__(self, entity_meta: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Args:
            entity_meta: Optional dictionary mapping entity names to metadata:
                - 'coords': Tuple[float, float] -> (latitude, longitude)
                - 'interval': Tuple[int, int]   -> (start_year, end_year)
        """
        self.entity_meta = entity_meta or {}

    def extract_constraints_from_question(self, question: str) -> Dict[str, Any]:
        """
        Extracts explicit spatial and temporal constraints from the natural language question.

        Returns:
            Dict containing:
              - 'temp_constraint': str or None (e.g., 'before', 'after', 'during')
              - 'geo_direction':   str or None (e.g., 'northeast', 'north')
              - 'geo_max_miles':   float or None (e.g., 300.0)
        """
        q = question.lower()
        extracted = {
            "temp_constraint": None,
            "geo_direction": None,
            "geo_max_miles": None
        }

        # ---------------------------------------------------------------------
        # [Step A] Extract Distance Constraint (DC)
        # e.g., "within 300 miles of London" -> max_miles = 300.0
        # ---------------------------------------------------------------------
        dist_match = re.search(r"within\s+(\d+(?:\.\d+)?)\s+miles", q)
        if dist_match:
            extracted["geo_max_miles"] = float(dist_match.group(1))

        # ---------------------------------------------------------------------
        # [Step B] Extract Directional Constraint (DDC / SDC)
        # Ordered so compound directions (northeast) are checked before single (north)
        # ---------------------------------------------------------------------
        directional_keywords = [
            "northeast", "northwest", "southeast", "southwest",
            "north", "south", "east", "west"
        ]
        for d in directional_keywords:
            if re.search(r"\b" + d + r"\b", q):
                extracted["geo_direction"] = d
                break

        # ---------------------------------------------------------------------
        # [Step C] Extract Temporal Ordering / Interval Constraint (DTC / STC)
        # Covers both DTC (during, while) and STC (before, after, etc.)
        # ---------------------------------------------------------------------
        temporal_keywords = [
            "during", "while",
            "before", "prior to",
            "after", "posterior to", "later than"
        ]
        for t in temporal_keywords:
            if re.search(r"\b" + t + r"\b", q):
                extracted["temp_constraint"] = t
                break

        return extracted

    def filter_and_rerank(self,
                          candidate_entities: List[str],
                          scores: List[float],
                          question: str,
                          spatial_clue: Optional[str] = None,
                          temporal_clue: Optional[str] = None) -> List[str]:
        """
        Executes the full 4-step Answer Filtering pipeline:
          Step 1: Parse and classify constraints from question text
          Step 2: Retrieve spatial/temporal reference metadata for clues
          Step 3: Test each candidate against active constraints
          Step 4: Re-rank candidates, prioritizing compliant answers
        """

        # =====================================================================
        # STEP 1: Parse and classify constraints from the question
        # =====================================================================
        constraints = self.extract_constraints_from_question(question)
        temp_c = constraints["temp_constraint"]
        geo_dir = constraints["geo_direction"]
        geo_dist = constraints["geo_max_miles"]

        # =====================================================================
        # STEP 2: Retrieve reference metadata (coordinates & intervals) for clues
        # =====================================================================
        # Coordinates of the spatial clue (e.g. Munich -> <48.14, 11.58>)
        clue_coords = self.entity_meta.get(spatial_clue, {}).get("coords") if spatial_clue else None
        # Interval of the temporal clue (e.g. World War II -> [1939, 1945])
        clue_interval = self.entity_meta.get(temporal_clue, {}).get("interval") if temporal_clue else None

        satisfying = []
        violating = []

        # =====================================================================
        # STEP 3: Iterate over candidates and verify constraint satisfaction
        # =====================================================================
        for cand, score in zip(candidate_entities, scores):
            cand_meta = self.entity_meta.get(cand, {})
            cand_coords = cand_meta.get("coords")
            cand_interval = cand_meta.get("interval")

            passes = True

            # -----------------------------------------------------------------
            # Step 3.1: Check Distance Constraint (DC)
            # Evaluated via spherical Haversine formula (rounded to 0.1 mile)
            # -----------------------------------------------------------------
            if geo_dist is not None and clue_coords:
                if cand_coords:
                    if not satisfies_distance_constraint(cand_coords, clue_coords, geo_dist):
                        passes = False
                elif self.entity_meta:
                    # Entity lacks coordinate data required for distance validation
                    passes = False

            # -----------------------------------------------------------------
            # Step 3.2: Check Directional Constraint (DDC / SDC)
            # Strictly checks latitude and/or longitude inequalities
            # (Per paper: equality on coordinate dimension fails requirement)
            # -----------------------------------------------------------------
            if passes and geo_dir and clue_coords:
                if cand_coords:
                    if not satisfies_direction_constraint(cand_coords, clue_coords, geo_dir):
                        passes = False
                elif self.entity_meta:
                    passes = False

            # -----------------------------------------------------------------
            # Step 3.3: Check Temporal Constraint (DTC / STC)
            # - DTC ('during'): [start, end]_cand must be fully contained within clue
            # - STC ('before'): end_cand < start_clue (strictly earlier)
            # - STC ('after'):  start_cand > end_clue (strictly later)
            # -----------------------------------------------------------------
            if passes and temp_c and clue_interval:
                if cand_interval:
                    if not satisfies_temporal_constraint(cand_interval, clue_interval, temp_c):
                        passes = False
                elif self.entity_meta:
                    passes = False

            # -----------------------------------------------------------------
            # Step 3.4: Place into corresponding bucket
            # -----------------------------------------------------------------
            if passes:
                satisfying.append((cand, score))
            else:
                violating.append((cand, score))

        # =====================================================================
        # STEP 4: Re-rank and recombine candidate list
        # Satisfying entities preserve their relative neural scores and appear first.
        # Violating entities are demoted to the back of the queue.
        # =====================================================================
        satisfying.sort(key=lambda x: x[1], reverse=True)
        violating.sort(key=lambda x: x[1], reverse=True)

        reranked_cands = [c for c, _ in satisfying] + [c for c, _ in violating]
        return reranked_cands

