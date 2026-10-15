"""Tests for the frequency grid, the transfer function and the objective pupil."""

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.objects import add_random_cells
from hologen.propagator import (
    backwards_propagate,
    forwards_propagate,
    objective_pupil,
)
from hologen.utils.domains import create_frequency_grid
from hologen.utils.maths import get_angular_spectrum_transfer_function

WAVELENGTH: float = 0.532  # um
NA: float = 0.45
PIXEL: float = 0.24  # um, IMX183 behind a 10x objective
Z: float = 20.0  # um


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


def evanescent_mask(resolution: int, pixel_size: float) -> numpy.ndarray:
    """
    Mark the spatial frequencies beyond 1 / wavelength.

    Args:
        resolution (int): Number of pixels along each side.
        pixel_size (float): Pixel size, in um.

    Returns:
        numpy.ndarray: True where the plane wave is evanescent.
    """
    grid = create_frequency_grid(resolution, pixel_size)
    return numpy.hypot(grid.FX, grid.FY) > 1 / WAVELENGTH


@pytest.mark.parametrize("is_forward", [True, False])
def test_evanescent_waves_are_removed(is_forward: bool) -> None:
    """The magnified grid has evanescent frequencies, and they do not pass."""
    transfer: numpy.ndarray = get_angular_spectrum_transfer_function(
        224, PIXEL, WAVELENGTH, Z, is_forward
    )
    evanescent: numpy.ndarray = evanescent_mask(224, PIXEL)
    assert 0.35 < evanescent.mean() < 0.37  # 36% of the 0.24 um grid
    assert numpy.all(transfer[evanescent] == 0)
    numpy.testing.assert_allclose(numpy.abs(transfer[~evanescent]), 1.0)


def test_zero_matches_the_physical_decay_away_from_the_cutoff() -> None:
    """
    Forwards, zero equals the physical decay except in a thin ring past 1 / wavelength.

    The decay is exp(-k z sqrt((wavelength f)^2 - 1)). At 20 um it is below 1e-12
    once f is 0.02 cycles per um past 1 / wavelength. Closer to the circle a few
    grid points keep more (at most 0.0076 on the 224 px grid), but those lie far
    beyond the objective pupil (0.85 cycles per um), so no hologram contains them.
    """
    grid = create_frequency_grid(224, PIXEL)
    radius: numpy.ndarray = numpy.hypot(grid.FX, grid.FY)
    arg: numpy.ndarray = 1 - (WAVELENGTH * radius) ** 2
    k: float = 2 * numpy.pi / WAVELENGTH
    exact: numpy.ndarray = numpy.where(
        arg >= 0,
        numpy.exp(1j * k * Z * numpy.sqrt(numpy.maximum(arg, 0))),
        numpy.exp(-k * Z * numpy.sqrt(numpy.maximum(-arg, 0))),
    )
    transfer: numpy.ndarray = get_angular_spectrum_transfer_function(
        224, PIXEL, WAVELENGTH, Z, True
    )
    error: numpy.ndarray = numpy.abs(transfer - exact)
    away: numpy.ndarray = radius > 1 / WAVELENGTH + 0.02
    assert error[away].max() < 1e-12
    assert error.max() < 0.01
    assert radius[error > 1e-12].min() > NA / WAVELENGTH  # never inside the pupil


def test_lensless_grid_has_no_evanescent_waves() -> None:
    """With 4.65 um pixels every frequency propagates, so nothing changed there."""
    transfer: numpy.ndarray = get_angular_spectrum_transfer_function(
        224, 4.65, WAVELENGTH, 20000.0, True
    )
    assert not evanescent_mask(224, 4.65).any()
    numpy.testing.assert_allclose(numpy.abs(transfer), 1.0)


def test_back_propagation_removes_evanescent_noise() -> None:
    """
    White noise back-propagated on the magnified grid loses its evanescent part.

    This is what the network input and reconstructed_XXXX.parquet see.
    """
    noise: numpy.ndarray = numpy.random.default_rng(3).normal(size=(224, 224))
    field: numpy.ndarray = backwards_propagate(noise, Z, WAVELENGTH, PIXEL, 224)
    spectrum: numpy.ndarray = numpy.abs(
        numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(field)))
    )
    evanescent: numpy.ndarray = evanescent_mask(224, PIXEL)
    assert spectrum[evanescent].max() < 1e-9 * spectrum.max()
    kept: float = float(
        numpy.sum(spectrum**2) / numpy.sum(numpy.abs(numpy.fft.fft2(noise)) ** 2)
    )
    assert kept == pytest.approx(1 - evanescent.mean(), abs=0.02)
