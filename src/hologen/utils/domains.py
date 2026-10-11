"""
This module exposes the domain operations for easy-to-use.
"""

import numpy
import dataclasses


@dataclasses.dataclass
class CoordinateGrid:
    X: numpy.ndarray
    Y: numpy.ndarray


@dataclasses.dataclass
class PolarGrid:
    R: numpy.ndarray
    Theta: numpy.ndarray
    Rho_norm: numpy.ndarray


@dataclasses.dataclass
class FrequencyGrid:
    FX: numpy.ndarray
    FY: numpy.ndarray


def create_coordinate_grid(resolution: int, pixel_size: float) -> CoordinateGrid:
    """Create a coordinate grid and return it."""
    # Pixel centres spaced exactly pixel_size apart, with 0 at index N // 2 as in
    # the centred frequency grid. numpy.linspace over the full length would space
    # them length / (N - 1) apart, 0.45% too far at 224 pixels.
    axis = (numpy.arange(resolution) - resolution // 2) * pixel_size
    x, y = numpy.meshgrid(axis, axis)
    return CoordinateGrid(X=x, Y=y)


def create_polar_grid(resolution: int, pixel_size: float) -> PolarGrid:
    """Create a polar grid and return it."""
    coordinates = create_coordinate_grid(resolution, pixel_size)
    length = resolution * pixel_size
    r = numpy.sqrt(coordinates.X**2 + coordinates.Y**2)
    theta = numpy.arctan2(coordinates.Y, coordinates.X)
    rho_norm = r / (length / 2)
    return PolarGrid(R=r, Theta=theta, Rho_norm=rho_norm)


def create_frequency_grid(resolution: int, pixel_size: float) -> FrequencyGrid:
    """Create a frequency grid and return it."""
    # Centred frequency axis with exactly `resolution` samples. numpy.arange with a
    # float step can return one sample too many through rounding (e.g. 224 pixels
    # of 0.24 um), so build it from fftfreq instead.
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(resolution, d=pixel_size))
    freq_x_dim, freq_y_dim = numpy.meshgrid(freq, freq)
    return FrequencyGrid(FX=freq_x_dim, FY=freq_y_dim)
