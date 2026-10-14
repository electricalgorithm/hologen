"""
Unit tests for shot noise and its interaction with dark current.

An empty object field gives |U|^2 = 1 everywhere, so every pixel of the recorded
hologram is an independent sample with known mean and variance. That makes the
Poisson statistics directly testable.
"""

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.noises import add_shot_noise

RESOLUTION: int = 256
Z_DISTANCE: float = 20000.0  # microns


@pytest.fixture(autouse=True)
def seed() -> None:
    """Seed the shared generator so every test is reproducible."""
    RandomNumberGenerator.set_seed(4321)


def empty_hologram(**kwargs: float | bool) -> numpy.ndarray:
    """
    Record a hologram of an empty field, where |U|^2 = 1 at every pixel.

    Args:
        **kwargs (float | bool): Noise and camera options of generate_hologram.

    Returns:
        numpy.ndarray: The recorded hologram.
    """
    sim = InlineHologramSimulator(resolution=RESOLUTION, pixel_size=4.65)
    return sim.generate_hologram(z_distance=Z_DISTANCE, **kwargs)


@pytest.mark.parametrize("mean", [5.0, 100.0, 1000.0])
def test_shot_noise_is_poisson(mean: float) -> None:
    """Mean is preserved and variance equals the mean."""
    counts = add_shot_noise(numpy.full((RESOLUTION, RESOLUTION), mean))
    assert counts.mean() == pytest.approx(mean, rel=0.01)
    assert counts.var() == pytest.approx(mean, rel=0.03)


def test_shot_noise_counts_are_whole_electrons() -> None:
    counts = add_shot_noise(numpy.full((64, 64), 12.3))
    numpy.testing.assert_array_equal(counts, numpy.round(counts))
    assert counts.min() >= 0


def test_shot_noise_keeps_dark_pixels_dark() -> None:
    """No light, no photo-electrons."""
    counts = add_shot_noise(numpy.zeros((64, 64)))
    numpy.testing.assert_array_equal(counts, 0.0)


def test_relative_shot_noise_falls_as_one_over_sqrt_n() -> None:
    """Ten times more light gives sqrt(10) times less relative noise."""
    dim = add_shot_noise(numpy.full((RESOLUTION, RESOLUTION), 100.0))
    bright = add_shot_noise(numpy.full((RESOLUTION, RESOLUTION), 1000.0))
    ratio = (dim.std() / dim.mean()) / (bright.std() / bright.mean())
    assert ratio == pytest.approx(numpy.sqrt(10), rel=0.05)


def test_background_electrons_sets_signal_level() -> None:
    hologram = empty_hologram(background_electrons=2500.0)
    numpy.testing.assert_allclose(hologram, 2500.0, rtol=1e-9)


def test_simulator_shot_noise_statistics() -> None:
    hologram = empty_hologram(shot_noise=True, background_electrons=1000.0)
    assert hologram.mean() == pytest.approx(1000.0, rel=0.005)
    assert hologram.var() == pytest.approx(1000.0, rel=0.03)


def test_dark_current_is_not_counted_twice() -> None:
    """
    Photo-electrons and dark electrons add their variances once, N + d.

    This is a regression test. They are independent Poisson processes, so the
    total variance is N + d. Sampling shot noise after adding the dark signal
    gave N + 2d.
    """
    signal, dark = 1000.0, 500.0
    hologram = empty_hologram(
        shot_noise=True, dark_noise_mean=dark, background_electrons=signal
    )
    assert hologram.mean() == pytest.approx(signal + dark, rel=0.005)
    assert hologram.var() == pytest.approx(signal + dark, rel=0.03)
