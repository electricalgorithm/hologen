"""Coordinate, polar and frequency grids of the simulated planes."""

import numpy
import dataclasses


@dataclasses.dataclass
class CoordinateGrid:
    """Positions of the pixel centres, in um, each of shape (N, N)."""

    X: numpy.ndarray
    Y: numpy.ndarray


@dataclasses.dataclass
class PolarGrid:
    """
    Polar coordinates of the pixel centres, each of shape (N, N).

    R is the radius in um, Theta the angle in radians and Rho_norm the radius
    divided by half the field width, so it is 1 at the edge.
    """

    R: numpy.ndarray
    Theta: numpy.ndarray
    Rho_norm: numpy.ndarray


@dataclasses.dataclass
class FrequencyGrid:
    """Centred spatial frequencies, in cycles per um, each of shape (N, N)."""

    FX: numpy.ndarray
    FY: numpy.ndarray


def create_coordinate_grid(resolution: int, pixel_size: float) -> CoordinateGrid:
    """
    Create the grid of pixel centres.

    Args:
        resolution (int): Number of pixels N along each side.
        pixel_size (float): Pixel size, in um.

    Returns:
        CoordinateGrid: x and y positions, with 0 at index N // 2.
    """
    # Pixel centres spaced exactly pixel_size apart, with 0 at index N // 2 as in
    # the centred frequency grid. numpy.linspace over the full length would space
    # them length / (N - 1) apart, 0.45% too far at 224 pixels.
    axis: numpy.ndarray = (numpy.arange(resolution) - resolution // 2) * pixel_size
    x, y = numpy.meshgrid(axis, axis)
    return CoordinateGrid(X=x, Y=y)


def create_polar_grid(resolution: int, pixel_size: float) -> PolarGrid:
    """
    Create the polar grid of pixel centres.

    Args:
        resolution (int): Number of pixels N along each side.
        pixel_size (float): Pixel size, in um.

    Returns:
        PolarGrid: Radius, angle and radius normalised to 1 at the field edge.
    """
    coordinates: CoordinateGrid = create_coordinate_grid(resolution, pixel_size)
    length: float = resolution * pixel_size
    r: numpy.ndarray = numpy.sqrt(coordinates.X**2 + coordinates.Y**2)
    theta: numpy.ndarray = numpy.arctan2(coordinates.Y, coordinates.X)
    rho_norm: numpy.ndarray = r / (length / 2)
    return PolarGrid(R=r, Theta=theta, Rho_norm=rho_norm)


def create_frequency_grid(resolution: int, pixel_size: float) -> FrequencyGrid:
    """
    Create the centred grid of spatial frequencies.

    Args:
        resolution (int): Number of pixels N along each side.
        pixel_size (float): Pixel size, in um.

    Returns:
        FrequencyGrid: fx and fy, in cycles per um, with 0 at index N // 2.
    """
    # Centred frequency axis with exactly `resolution` samples. numpy.arange with a
    # float step can return one sample too many through rounding (e.g. 224 pixels
    # of 0.24 um), so build it from fftfreq instead.
    freq: numpy.ndarray = numpy.fft.fftshift(
        numpy.fft.fftfreq(resolution, d=pixel_size)
    )
    freq_x_dim, freq_y_dim = numpy.meshgrid(freq, freq)
    return FrequencyGrid(FX=freq_x_dim, FY=freq_y_dim)
