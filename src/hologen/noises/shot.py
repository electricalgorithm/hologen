"""
This module provides a function for generating shot noise.
"""

import numpy
import hologen.utils as utils


def add_shot_noise(field: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """Add shot noise to a given field."""
    return utils.RandomNumberGenerator().poisson(field).astype(float)
