"""
This module provides the analog-to-digital conversion of a camera.

The ADC turns electrons into digital numbers (DN) with the overall system gain K
(DN per electron), clips at the largest k-bit value 2^k - 1 and rounds to whole
DN, which adds uniform quantisation noise of variance 1/12 DN^2 (EMVA Standard
1288, Release 4.0 Linear, 2021, Sec. 2.4, Eq. 14). The saturation capacity is the
charge at which the digital signal clips (Sec. 2.7), so the gain that maps it to
full scale is K = (2^k - 1) / saturation capacity.
"""

import numpy


def adc_gain(bit_depth: int, saturation_electrons: float) -> float:
    """
    Return the system gain K (DN per electron) that maps the saturation capacity
    to the largest digital value.

    Args:
        bit_depth (int): ADC resolution in bits.
        saturation_electrons (float): Saturation capacity (in electrons).

    Returns:
        float: The system gain K (DN per electron).
    """
    if bit_depth < 1:
        raise ValueError(f"bit_depth must be a positive integer, got {bit_depth}.")
    if saturation_electrons <= 0:
        raise ValueError(
            f"saturation_electrons must be positive, got {saturation_electrons}."
        )
    return (2**bit_depth - 1) / saturation_electrons


def add_bit_depth_noise(
    field: numpy.ndarray[float], bit_depth: int, saturation_electrons: float
) -> numpy.ndarray[float]:
    """
    Convert electrons to clipped, quantised digital numbers.

    Args:
        field (numpy.ndarray[float]): Electrons per pixel.
        bit_depth (int): ADC resolution in bits.
        saturation_electrons (float): Saturation capacity (in electrons).

    Returns:
        numpy.ndarray[float]: Digital numbers in [0, 2^bit_depth - 1].
    """
    gain: float = adc_gain(bit_depth, saturation_electrons)
    digital: numpy.ndarray[float] = numpy.clip(field * gain, 0, 2**bit_depth - 1)
    return numpy.round(digital)
