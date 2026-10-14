"""
Unit tests for the speckle noise model (Goodman, "Speckle Phenomena in Optics", 2007).

The statistical tests use fixed seeds and tolerances of several standard errors,
so they are deterministic and not flaky.
"""

import numpy
import pytest
from scipy.special import j1

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.noises import add_speckle_noise
from hologen.noises.speckle import (
    generate_circular_gaussian_field,
    generate_speckle_pattern,
)

RESOLUTION: int = 512
PIXEL_SIZE: float = 1.0  # microns
GRAIN_SIZE: float = 8.0  # microns


@pytest.fixture(autouse=True)
def seed() -> None:
    """Seed the shared generator so every test is reproducible."""
    RandomNumberGenerator.set_seed(1234)


def test_field_has_unit_power() -> None:
    field = generate_circular_gaussian_field(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE)
    assert field.shape == (RESOLUTION, RESOLUTION)
    assert numpy.iscomplexobj(field)
    assert numpy.mean(numpy.abs(field) ** 2) == pytest.approx(1.0, abs=0.05)


def test_field_is_circular() -> None:
    """Real and imaginary parts carry equal power and are uncorrelated."""
    field = generate_circular_gaussian_field(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE)
    assert numpy.var(field.real) == pytest.approx(0.5, abs=0.03)
    assert numpy.var(field.imag) == pytest.approx(0.5, abs=0.03)
    assert numpy.mean(field.real * field.imag) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("contrast", [0.05, 0.1, 0.3, 0.6, 1.0])
def test_pattern_mean_and_contrast(contrast: float) -> None:
    pattern = generate_speckle_pattern(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE, contrast)
    assert pattern.mean() == pytest.approx(1.0, abs=0.05 * contrast + 0.002)
    assert pattern.std() / pattern.mean() == pytest.approx(contrast, rel=0.08)


def test_pattern_is_non_negative() -> None:
    """S is a brightness, so it can never be negative."""
    for contrast in [0.1, 0.6, 1.0]:
        pattern = generate_speckle_pattern(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE, contrast)
        assert pattern.min() >= 0.0


def test_fully_developed_speckle_is_exponential() -> None:
    """For C = 1, P(S > s) = exp(-s) (Goodman, Sec. 3.2)."""
    pattern = generate_speckle_pattern(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE, 1.0)
    for s in [0.5, 1.0, 2.0]:
        assert numpy.mean(pattern > s) == pytest.approx(numpy.exp(-s), abs=0.03)


@pytest.mark.parametrize("grain_size", [6.0, 10.0])
def test_grain_size_matches_theory(grain_size: float) -> None:
    """
    Speckle has the intensity autocovariance of a circular pupil.

    For fully developed speckle it is |2 J1(x) / x|^2 with
    x = pi * r / grain_size (Goodman, Sec. 4.4).
    """
    autocov = numpy.zeros(RESOLUTION)
    for _ in range(8):
        pattern = generate_speckle_pattern(RESOLUTION, PIXEL_SIZE, grain_size, 1.0)
        pattern = pattern - pattern.mean()
        corr = numpy.real(numpy.fft.ifft2(numpy.abs(numpy.fft.fft2(pattern)) ** 2))
        autocov += (corr[0, :] + corr[:, 0]) / 2
    autocov /= autocov[0]

    r = numpy.arange(1, int(2 * grain_size))
    x = numpy.pi * r * PIXEL_SIZE / grain_size
    theory = (2 * j1(x) / x) ** 2
    numpy.testing.assert_allclose(autocov[r], theory, atol=0.02)


def test_field_correlation_matches_pupil() -> None:
    """
    The speckle field has the autocorrelation 2 J1(x) / x of a circular pupil.

    See Goodman, Sec. 4.4. Unlike |A|^2, it changes sign if the pupil is placed
    off DC, so this catches a pupil built on a centred (fftshift-ed) frequency
    grid.
    """
    autocorr = numpy.zeros(RESOLUTION)
    for _ in range(8):
        field = generate_circular_gaussian_field(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE)
        corr = numpy.real(numpy.fft.ifft2(numpy.abs(numpy.fft.fft2(field)) ** 2))
        autocorr += (corr[0, :] + corr[:, 0]) / 2
    autocorr /= autocorr[0]

    r = numpy.arange(1, int(2 * GRAIN_SIZE))
    x = numpy.pi * r * PIXEL_SIZE / GRAIN_SIZE
    numpy.testing.assert_allclose(autocorr[r], 2 * j1(x) / x, atol=0.03)


def test_partial_speckle_is_smooth_at_pixel_scale() -> None:
    """
    Neighbouring pixels inside one grain are strongly correlated for C < 1.

    For C < 1 the linear term 2 sqrt(beta) Re(A) is visible in S, so the
    correlation must be strongly positive.
    """
    pattern = generate_speckle_pattern(RESOLUTION, PIXEL_SIZE, GRAIN_SIZE, 0.1)
    pattern = pattern - pattern.mean()
    neighbour = numpy.mean(pattern[:, :-1] * pattern[:, 1:]) / numpy.var(pattern)
    assert neighbour > 0.8


def test_same_seed_gives_same_pattern() -> None:
    RandomNumberGenerator.set_seed(7)
    first = generate_speckle_pattern(64, PIXEL_SIZE, GRAIN_SIZE, 0.2)
    RandomNumberGenerator.set_seed(7)
    second = generate_speckle_pattern(64, PIXEL_SIZE, GRAIN_SIZE, 0.2)
    numpy.testing.assert_array_equal(first, second)


def test_add_speckle_noise_is_multiplicative() -> None:
    """Doubling the input doubles the output for the same speckle realisation."""
    intensity = numpy.full((64, 64), 3.0)
    RandomNumberGenerator.set_seed(7)
    single = add_speckle_noise(intensity, PIXEL_SIZE, GRAIN_SIZE, 0.2)
    RandomNumberGenerator.set_seed(7)
    double = add_speckle_noise(2 * intensity, PIXEL_SIZE, GRAIN_SIZE, 0.2)
    numpy.testing.assert_allclose(double, 2 * single)


def test_add_speckle_noise_keeps_zero_intensity_dark() -> None:
    """Speckle redistributes existing light; it cannot light up a dark pixel."""
    intensity = numpy.zeros((64, 64))
    noisy = add_speckle_noise(intensity, PIXEL_SIZE, GRAIN_SIZE, 0.5)
    numpy.testing.assert_array_equal(noisy, intensity)


@pytest.mark.parametrize("contrast", [0.0, -0.1, 1.01])
def test_invalid_contrast_raises(contrast: float) -> None:
    with pytest.raises(ValueError, match="contrast"):
        generate_speckle_pattern(64, PIXEL_SIZE, GRAIN_SIZE, contrast)


def test_unresolved_grain_size_raises() -> None:
    """Grains smaller than two pixels would alias, so they are rejected."""
    with pytest.raises(ValueError, match="grain_size"):
        generate_circular_gaussian_field(64, pixel_size=4.65, grain_size=9.0)


def test_non_square_intensity_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        add_speckle_noise(numpy.ones((32, 64)), PIXEL_SIZE, GRAIN_SIZE, 0.1)


def test_simulator_applies_speckle_at_sensor() -> None:
    """Speckle multiplies the hologram and leaves the object field untouched."""

    def make_simulator() -> InlineHologramSimulator:
        """
        Create an empty simulator with a fixed seed.

        Returns:
            InlineHologramSimulator: The simulator.
        """
        RandomNumberGenerator.set_seed(99)
        return InlineHologramSimulator(resolution=128, pixel_size=4.65)

    clean_sim = make_simulator()
    clean = clean_sim.generate_hologram(z_distance=20000)

    noisy_sim = make_simulator()
    object_before = noisy_sim.object_field.copy()
    noisy = noisy_sim.generate_hologram(
        z_distance=20000, speckle_grain_size=20.0, speckle_contrast=0.1
    )

    numpy.testing.assert_array_equal(noisy_sim.object_field, object_before)
    ratio = noisy / clean
    assert ratio.mean() == pytest.approx(1.0, abs=0.02)
    assert ratio.std() == pytest.approx(0.1, rel=0.2)


def test_simulator_requires_grain_size_with_speckle() -> None:
    sim = InlineHologramSimulator(resolution=64, pixel_size=4.65)
    with pytest.raises(ValueError, match="speckle_grain_size"):
        sim.generate_hologram(z_distance=20000, speckle_contrast=0.1)
