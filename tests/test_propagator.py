"""Tests for the frequency grid and the objective pupil of the magnified setup."""

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.objects import add_random_cells
from hologen.propagator import forwards_propagate, objective_pupil
from hologen.utils.domains import create_frequency_grid

WAVELENGTH: float = 0.532  # um
NA: float = 0.45
PIXEL: float = 0.24  # um, IMX183 behind a 10x objective


@pytest.mark.parametrize("resolution", [64, 224, 225, 512, 1024])
@pytest.mark.parametrize("pixel_size", [0.1, 0.24, 0.3, 0.465, 1.0, 4.65])
def test_frequency_grid_has_one_sample_per_pixel(
    resolution: int, pixel_size: float
) -> None:
    """
    The frequency grid has exactly one sample per pixel.

    This is a regression test. numpy.arange with a float step gave 225 samples
    for 224 pixels of 0.24 um, which broke every propagation on that grid.
    """
    grid = create_frequency_grid(resolution, pixel_size)
    assert grid.FX.shape == (resolution, resolution)
    step = 1 / (resolution * pixel_size)
    assert numpy.allclose(numpy.diff(grid.FX[0]), step)
    assert grid.FX[0, resolution // 2] == 0.0  # DC sits in the centre


def test_pupil_passes_frequencies_up_to_na_over_wavelength() -> None:
    pupil = objective_pupil(224, PIXEL, WAVELENGTH, NA)
    grid = create_frequency_grid(224, PIXEL)
    radius = numpy.hypot(grid.FX, grid.FY)
    assert pupil[radius <= 0.99 * NA / WAVELENGTH].all()
    assert not pupil[radius > 1.01 * NA / WAVELENGTH].any()


@pytest.mark.parametrize("numerical_aperture", [0.0, -0.1, 2.0])
def test_invalid_numerical_aperture_raises(numerical_aperture: float) -> None:
    with pytest.raises(ValueError, match="numerical_aperture"):
        objective_pupil(64, PIXEL, WAVELENGTH, numerical_aperture)


def random_object(seed: int) -> numpy.ndarray:
    """
    Create an object with fine detail on the magnified grid.

    Args:
        seed (int): Random seed of the object.

    Returns:
        numpy.ndarray: The complex object field, shape (224, 224).
    """
    RandomNumberGenerator.set_seed(seed)
    sim = InlineHologramSimulator(resolution=224, pixel_size=PIXEL)
    add_random_cells(sim, num_cells=5, radius_range=(2, 6))
    return sim.object_field


def test_pupil_removes_frequencies_beyond_the_cutoff() -> None:
    field = forwards_propagate(
        random_object(1), 20.0, WAVELENGTH, PIXEL, 224, numerical_aperture=NA
    )
    spectrum = numpy.fft.fftshift(numpy.fft.fft2(field))
    grid = create_frequency_grid(224, PIXEL)
    outside = numpy.hypot(grid.FX, grid.FY) > 1.01 * NA / WAVELENGTH
    energy_outside = numpy.sum(numpy.abs(spectrum[outside]) ** 2)
    assert energy_outside < 1e-20 * numpy.sum(numpy.abs(spectrum) ** 2)


def test_plane_wave_passes_the_pupil_unchanged() -> None:
    """The unscattered background (DC) is inside every pupil."""
    field = forwards_propagate(
        numpy.ones((224, 224), complex), 20.0, WAVELENGTH, PIXEL, 224, NA
    )
    numpy.testing.assert_allclose(numpy.abs(field) ** 2, 1.0, atol=1e-12)


def test_no_aperture_keeps_the_lensless_propagation() -> None:
    obj = random_object(2)
    without = forwards_propagate(obj, 20.0, WAVELENGTH, PIXEL, 224)
    explicit = forwards_propagate(obj, 20.0, WAVELENGTH, PIXEL, 224, None)
    numpy.testing.assert_array_equal(without, explicit)
    with_pupil = forwards_propagate(obj, 20.0, WAVELENGTH, PIXEL, 224, NA)
    assert not numpy.allclose(without, with_pupil)


def test_simulator_uses_its_numerical_aperture() -> None:
    RandomNumberGenerator.set_seed(3)
    sim = InlineHologramSimulator(
        resolution=224, pixel_size=PIXEL, numerical_aperture=NA
    )
    add_random_cells(sim, num_cells=5, radius_range=(2, 6))
    hologram = sim.generate_hologram(z_distance=20.0, background_electrons=1.0)
    expected = (
        numpy.abs(
            forwards_propagate(sim.object_field, 20.0, WAVELENGTH, PIXEL, 224, NA)
        )
        ** 2
    )
    numpy.testing.assert_allclose(hologram, expected)
