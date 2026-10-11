"""
This module provides a function for generating shot noise.

Light is absorbed as discrete photons arriving at random times, so the number of
photo-electrons a pixel collects during one exposure follows a Poisson
distribution: for a mean of N electrons the standard deviation is sqrt(N)
(EMVA Standard 1288, Release 4.0 Linear, 2021, Sec. 2.4, Eq. 13).

The input must be the expected number of photo-electrons per pixel, not a
normalised intensity: the noise depends on the absolute count.
"""

import numpy
import hologen.utils as utils


def add_shot_noise(field: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Sample the photo-electron count of every pixel.

    Args:
        field (numpy.ndarray[float]): Expected photo-electrons per pixel.

    Returns:
        numpy.ndarray[float]: Poisson-distributed photo-electron counts.
    """
    return utils.RandomNumberGenerator().poisson(field).astype(float)
