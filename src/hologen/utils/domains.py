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
    length = resolution * pixel_size
    x_dim = numpy.linspace(-length / 2, length / 2, resolution)
    y_dim = numpy.linspace(-length / 2, length / 2, resolution)
    x, y = numpy.meshgrid(x_dim, y_dim)
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
    sampling_frequency = 1.0 / pixel_size
    df = sampling_frequency / resolution
    fx = numpy.arange(-sampling_frequency / 2, sampling_frequency / 2, df)
    fy = numpy.arange(-sampling_frequency / 2, sampling_frequency / 2, df)
    freq_x_dim, freq_y_dim = numpy.meshgrid(fx, fy)
    return FrequencyGrid(FX=freq_x_dim, FY=freq_y_dim)
