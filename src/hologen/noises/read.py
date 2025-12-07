"""
This module provides a function for generating read noise.
"""
import numpy
import hologen.utils as utils


def add_read_noise(field: numpy.ndarray[float], sigma: float) -> numpy.ndarray[float]:
    """Add read noise to a given field."""
    read_noise = utils.RandomNumberGenerator().normal(0, sigma, field.shape)
    field += read_noise
    return field