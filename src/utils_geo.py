"""
Geographical distance and orientation utility functions for STKGQA.
Supports Haversine distance calculation and cardinal/intercardinal direction validation.
"""

import math
from typing import Tuple, Optional

# Earth's mean radius in miles
EARTH_RADIUS_MILES = 3958.8


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Computes great-circle distance between two coordinates in miles.
    coord: (latitude, longitude) in decimal degrees.
    Returns distance rounded to nearest 0.1 mile per paper specification.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    distance = EARTH_RADIUS_MILES * c
    return round(distance, 1)


def satisfies_distance_constraint(cand_coord: Tuple[float, float],
                                  clue_coord: Tuple[float, float],
                                  max_miles: float) -> bool:
    """
    Checks if cand_coord is within max_miles of clue_coord.
    """
    dist = haversine_distance(cand_coord, clue_coord)
    return dist <= max_miles


def satisfies_direction_constraint(cand_coord: Tuple[float, float],
                                   clue_coord: Tuple[float, float],
                                   direction: str) -> bool:
    """
    Checks if cand_coord satisfies the directional constraint relative to clue_coord.
    Strict inequality required per paper: equality on coordinate dimension fails requirement.
    """
    c_lat, c_lon = cand_coord
    ref_lat, ref_lon = clue_coord
    d = direction.lower().strip()

    if d == "north":
        return c_lat > ref_lat
    elif d == "south":
        return c_lat < ref_lat
    elif d == "east":
        return c_lon > ref_lon
    elif d == "west":
        return c_lon < ref_lon
    elif d == "northeast":
        return c_lat > ref_lat and c_lon > ref_lon
    elif d == "northwest":
        return c_lat > ref_lat and c_lon < ref_lon
    elif d == "southeast":
        return c_lat < ref_lat and c_lon > ref_lon
    elif d == "southwest":
        return c_lat < ref_lat and c_lon < ref_lon
    return True
