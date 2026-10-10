"""
This module provides a speckle noise implementation for the sensor plane.

Speckle is modelled following Goodman, "Speckle Phenomena in Optics" (2007):
coherent light scattered by dust and rough optical surfaces reaches every sensor
pixel as the sum of many random phasors. By the central limit theorem this sum is
a circular complex Gaussian field A, whose intensity |A|^2 is exponentially
distributed (fully developed speckle, contrast C = std/mean = 1).

The speckle pattern multiplies the recorded intensity:

    I_recorded(x, y) = I_ideal(x, y) * S(x, y),    E[S] = 1

so it is multiplicative in intensity, not in phase.

Physical parameters:
    grain_size (microns): mean speckle grain diameter d_s = wavelength * z / D,
        where D is the width of the scattering region seen from the sensor.
        The intensity correlation of a circular pupil falls to its first zero at
        1.22 * d_s (Goodman, Sec. 4.4).
    contrast (unitless ratio): C = std(S) / mean(S), in (0, 1]. C = 1 is fully
        developed speckle; C < 1 is partially developed speckle.
"""

import numpy
import hologen.utils as utils


def generate_circular_gaussian_field(
    resolution: int, pixel_size: float, grain_size: float
) -> numpy.ndarray[numpy.complex128]:
    """
    Generate a band-limited circular complex Gaussian field with E[|A|^2] = 1.

    White complex Gaussian noise is filtered by a circular pupil of cutoff
    frequency f_c = 1 / (2 * grain_size), so that the mean grain diameter of
    |A|^2 is grain_size.

    Args:
        resolution (int): The resolution of the sensor plane.
        pixel_size (float): The pixel size of the sensor plane (in microns).
        grain_size (float): Mean speckle grain diameter (in microns).

    Returns:
        numpy.ndarray[numpy.complex128]: The complex speckle field.
    """
    if grain_size < 2 * pixel_size:
        raise ValueError(
            f"grain_size ({grain_size} um) must be at least two pixels "
            f"({2 * pixel_size} um) to be sampled without aliasing."
        )

    rng: numpy.random.Generator = utils.RandomNumberGenerator()

    # Circular white noise: independent real and imaginary parts, E[|w|^2] = 1.
    white: numpy.ndarray[numpy.complex128] = (
        rng.normal(0.0, 1.0, (resolution, resolution))
        + 1j * rng.normal(0.0, 1.0, (resolution, resolution))
    ) / numpy.sqrt(2.0)

    # Pupil on the unshifted fft2 layout (DC at index [0, 0]).
    freq: numpy.ndarray[float] = numpy.fft.fftfreq(resolution, d=pixel_size)
    fx: numpy.ndarray[float]
    fy: numpy.ndarray[float]
    fx, fy = numpy.meshgrid(freq, freq)
    cutoff: float = 1.0 / (2.0 * grain_size)
    pupil: numpy.ndarray[bool] = (fx**2 + fy**2) <= cutoff**2

    # Filtering keeps only the pupil's share of the power; rescale to E[|A|^2] = 1.
    field: numpy.ndarray[numpy.complex128] = numpy.fft.ifft2(
        numpy.fft.fft2(white) * pupil
    )
    return field / numpy.sqrt(pupil.mean())


def generate_speckle_pattern(
    resolution: int, pixel_size: float, grain_size: float, contrast: float
) -> numpy.ndarray[float]:
    """
    Generate a speckle intensity pattern S with E[S] = 1 and std(S) = contrast.

    Args:
        resolution (int): The resolution of the sensor plane.
        pixel_size (float): The pixel size of the sensor plane (in microns).
        grain_size (float): Mean speckle grain diameter (in microns).
        contrast (float): Speckle contrast std(S) / mean(S), in (0, 1].

    Returns:
        numpy.ndarray[float]: The speckle intensity pattern.
    """
    if not 0.0 < contrast <= 1.0:
        raise ValueError(f"contrast must be in (0, 1], got {contrast}.")

    field: numpy.ndarray[numpy.complex128] = generate_circular_gaussian_field(
        resolution, pixel_size, grain_size
    )

    # Fully developed speckle: all light is scattered, S = |A|^2 (Goodman, Sec. 3.2).
    if contrast == 1.0:
        return numpy.abs(field) ** 2

    # Partially developed speckle: the unscattered beam (a constant phasor of
    # power 1) interferes with scattered light of power beta (Goodman, Sec. 3.3).
    # Then E[S] = 1 and C = sqrt(2 * beta + beta^2) / (1 + beta), which inverts to
    # beta = 1 / sqrt(1 - C^2) - 1.
    beta: float = 1.0 / numpy.sqrt(1.0 - contrast**2) - 1.0
    return numpy.abs(1.0 + numpy.sqrt(beta) * field) ** 2 / (1.0 + beta)


def add_speckle_noise(
    intensity: numpy.ndarray[float],
    pixel_size: float,
    grain_size: float,
    contrast: float,
) -> numpy.ndarray[float]:
    """
    Multiply a sensor-plane intensity by a speckle pattern.

    Args:
        intensity (numpy.ndarray[float]): The ideal intensity |U|^2 at the sensor.
        pixel_size (float): The pixel size of the sensor plane (in microns).
        grain_size (float): Mean speckle grain diameter (in microns).
        contrast (float): Speckle contrast std(S) / mean(S), in (0, 1].

    Returns:
        numpy.ndarray[float]: The intensity with speckle noise.
    """
    if intensity.ndim != 2 or intensity.shape[0] != intensity.shape[1]:
        raise ValueError(f"intensity must be a square 2D array, got {intensity.shape}.")

    speckle: numpy.ndarray[float] = generate_speckle_pattern(
        intensity.shape[0], pixel_size, grain_size, contrast
    )
    return intensity * speckle
