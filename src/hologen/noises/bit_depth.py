"""
This module provides a function for generating bit depth noise.
"""

import numpy

ALLOWED_BIT_DEPTHS: list[int] = [8, 10, 12]


def add_bit_depth_noise(
    field: numpy.ndarray[float], bit_depth: int
) -> numpy.ndarray[float]:
    """Add bit depth noise to a given field."""
    max_val: int = 2**bit_depth - 1
    # Normalize current max to sensor max (saturation)
    # Or just clip if we assume field was already calibrated
    field = numpy.clip(field, 0, max_val)
    field = numpy.round(field)
    return field
