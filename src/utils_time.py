"""
Temporal interval and order utility functions for STKGQA.
Supports Single Timestamp Constraint (STC) and Double Timestamp Constraint (DTC) checking.
"""

from typing import Tuple


def satisfies_temporal_constraint(cand_interval: Tuple[int, int],
                                  clue_interval: Tuple[int, int],
                                  constraint_type: str) -> bool:
    """
    Checks if cand_interval satisfies the temporal constraint relative to clue_interval.
    cand_interval: (start_year, end_year)
    clue_interval: (start_year, end_year)
    constraint_type: 'before'/'prior to', 'after'/'later than'/'posterior to', 'during'/'while'
    
    Per paper Section 5.4:
    - DTC ('during', 'while'): cand start and end times are entirely contained within clue span.
    - STC ('before'): cand end time strictly earlier than clue start time (equality fails).
    - STC ('after'): cand start time strictly later than clue end time (equality fails).
    """
    c_start, c_end = cand_interval
    ref_start, ref_end = clue_interval
    ct = constraint_type.lower().strip()

    if "during" in ct or "while" in ct:
        return c_start >= ref_start and c_end <= ref_end
    elif "before" in ct or "prior" in ct:
        return c_end < ref_start
    elif "after" in ct or "posterior" in ct or "later" in ct:
        return c_start > ref_end
    return True
