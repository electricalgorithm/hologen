"""
This module provides speckle noise implementation for object domain.
"""
import numpy
from scipy.fft import fft2, ifft2
import hologen.utils as utils


def add_speckle_noise(
    field: numpy.ndarray[numpy.complex128],
    resolution: int,
    pixel_size: float,
    strength: float,
    roughness_scale: float,
) -> numpy.ndarray[numpy.complex128]:
    """
    Adds speckle noise by simulating a rough phase background.
    This simulates passing light through dirty optics or a rough glass slide.

    Args:
        field (numpy.ndarray[numpy.complex128]): The field to apply speckle noise.
        resolution (int): The resolution of the object domain.
        pixel_size (float): The pixel size of the object domain.
        strength (float): Magnitude of the random phase shifts (0.0 to 2*pi).
        roughness_scale (float): Scaling factor for the grain size of the speckle.
                                    (Smaller = finer grain).

    Returns:
        numpy.ndarray[numpy.complex128]: The given field with speckle noise.
    """
    # Create a random phase screen
    random_phase = utils.RandomNumberGenerator().uniform(
        -numpy.pi, numpy.pi, (resolution, resolution)
    )

    # Create frequency grid to work on.
    freq_grid: utils.FrequencyGrid = utils.create_frequency_grid(resolution, pixel_size)

    # Smooth it slightly to create "grains" instead of white noise
    # Using a Gaussian filter in frequency domain for efficiency
    sigma = roughness_scale * (resolution / 50.0)
    kernel = numpy.exp(-(freq_grid.FX**2 + freq_grid.FY**2) * sigma**2)
    smooth_phase = ifft2(fft2(random_phase) * kernel).real

    # Normalize and apply strength
    smooth_phase = (smooth_phase - smooth_phase.mean()) / (
        smooth_phase.std() + 1e-10
    )
    final_phase_screen = smooth_phase * strength

    # Multiply the field by this phase screen
    field *= numpy.exp(1j * final_phase_screen)
    return field
