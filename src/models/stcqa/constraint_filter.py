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
from src.utils.geo import haversine_distance, satisfies_distance_constraint, satisfies_direction_constraint
from src.utils.time import satisfies_temporal_constraint


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

    def get_coords(self, entity: Optional[str]) -> Optional[Tuple[float, float]]:
        """Retrieves coordinates for an entity from metadata."""
        if not entity:
            return None
        clean = entity.strip().rstrip(",").strip()
        if clean in self.entity_meta and "coords" in self.entity_meta[clean]:
            return tuple(self.entity_meta[clean]["coords"])
        return None

    def get_interval(self, entity: Optional[str]) -> Optional[Tuple[int, int]]:
        """Retrieves temporal interval for an entity from metadata."""
        if not entity:
            return None
        clean = entity.strip().rstrip(",").strip()
        if clean in self.entity_meta and "interval" in self.entity_meta[clean]:
            return tuple(self.entity_meta[clean]["interval"])
        return None

    def filter_and_rerank(self,
                          candidate_entities: List[str],
                          scores: List[float],
                          question: str,
                          spatial_clue: Optional[str] = None,
                          temporal_clue: Optional[str] = None) -> List[str]:
        """
        Executes the neuro-symbolic Answer Filtering pipeline:
          Step 1: Parse and classify constraints from question text
          Step 2: Retrieve spatial/temporal reference metadata for clues
          Step 3: Test each candidate against active constraints (3-tier: satisfying, neutral, violating)
          Step 4: Re-rank candidates:
                  - Verified compliant entities boosted to the front
                  - Unverified/neutral entities retain relative neural ranking
                  - Proven violating entities demoted to the back
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
        clue_coords = self.get_coords(spatial_clue)
        clue_interval = self.get_interval(temporal_clue)

        satisfying = []
        neutral = []
        violating = []

        # =====================================================================
        # STEP 3: Iterate over candidates and verify constraint satisfaction
        # =====================================================================
        for cand, score in zip(candidate_entities, scores):
            cand_coords = self.get_coords(cand)
            cand_interval = self.get_interval(cand)

            is_violating = False
            has_positive_verification = False

            # -----------------------------------------------------------------
            # Step 3.1: Check Distance Constraint (DC)
            # Evaluated via spherical Haversine formula (rounded to 0.1 mile)
            # -----------------------------------------------------------------
            if geo_dist is not None and clue_coords:
                if cand_coords:
                    if not satisfies_distance_constraint(cand_coords, clue_coords, geo_dist):
                        is_violating = True
                    else:
                        has_positive_verification = True

            # -----------------------------------------------------------------
            # Step 3.2: Check Directional Constraint (DDC / SDC)
            # Strictly checks latitude and/or longitude inequalities
            # -----------------------------------------------------------------
            if not is_violating and geo_dir and clue_coords:
                if cand_coords:
                    if not satisfies_direction_constraint(cand_coords, clue_coords, geo_dir):
                        is_violating = True
                    else:
                        has_positive_verification = True

            # -----------------------------------------------------------------
            # Step 3.3: Check Temporal Constraint (DTC / STC)
            # - DTC ('during'): [start, end]_cand must be fully contained within clue
            # - STC ('before'): end_cand < start_clue (strictly earlier)
            # - STC ('after'):  start_cand > end_clue (strictly later)
            # -----------------------------------------------------------------
            if not is_violating and temp_c and clue_interval:
                if cand_interval:
                    if not satisfies_temporal_constraint(cand_interval, clue_interval, temp_c):
                        is_violating = True
                    else:
                        has_positive_verification = True

            # -----------------------------------------------------------------
            # Step 3.4: Place into 3-tier bucket
            # -----------------------------------------------------------------
            if is_violating:
                violating.append((cand, score))
            elif has_positive_verification:
                satisfying.append((cand, score))
            else:
                neutral.append((cand, score))

        # =====================================================================
        # STEP 4: Re-rank and recombine candidate list
        # 1. Verified satisfying entities appear first (ordered by neural score)
        # 2. Unverified neutral entities appear next (preserving neural score)
        # 3. Explicitly violating entities demoted to the back of the queue
        # =====================================================================
        satisfying.sort(key=lambda x: x[1], reverse=True)
        neutral.sort(key=lambda x: x[1], reverse=True)
        violating.sort(key=lambda x: x[1], reverse=True)

        reranked_cands = [c for c, _ in satisfying] + [c for c, _ in neutral] + [c for c, _ in violating]
        return reranked_cands

