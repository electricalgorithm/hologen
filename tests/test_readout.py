"""
Unit tests for the camera readout.

The readout adds read noise and performs the analog-to-digital conversion
(EMVA Standard 1288, Release 4.0 Linear, Sec. 2.4, Eq. 14).
"""

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.dataset.recipe import default_recipe
from hologen.inline_simulator import (
    DEFAULT_SATURATION_ELECTRONS,
)
from hologen.noises import add_bit_depth_noise, add_read_noise
from hologen.noises.bit_depth import adc_gain
from hologen.objects import add_random_cells

SHAPE: tuple[int, int] = (256, 256)


@pytest.fixture(autouse=True)
def seed() -> None:
    """Seed the shared generator so every test is reproducible."""
    RandomNumberGenerator.set_seed(1357)


# Read noise ---------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1.0, 10.0, 11.48])
def test_read_noise_is_zero_mean_gaussian(sigma: float) -> None:
    noise = add_read_noise(numpy.zeros(SHAPE), sigma)
    assert noise.mean() == pytest.approx(0.0, abs=0.02 * sigma)
    assert noise.std() == pytest.approx(sigma, rel=0.02)


def test_read_noise_does_not_depend_on_the_signal() -> None:
    """Unlike shot noise, read noise is the same for dim and bright pixels."""
    RandomNumberGenerator.set_seed(1)
    dim = add_read_noise(numpy.full(SHAPE, 10.0), 10.0) - 10.0
    RandomNumberGenerator.set_seed(1)
    bright = add_read_noise(numpy.full(SHAPE, 9000.0), 10.0) - 9000.0
    numpy.testing.assert_allclose(dim, bright, atol=1e-9)


def test_read_noise_does_not_modify_its_input() -> None:
    """Regression test: the function used to add to its input in place."""
    signal = numpy.full(SHAPE, 4000.0)
    before = signal.copy()
    add_read_noise(signal, 10.0)
    numpy.testing.assert_array_equal(signal, before)


# Analog-to-digital conversion ---------------------------------------------


def test_gain_maps_saturation_to_full_scale() -> None:
    assert adc_gain(12, 11000.0) == pytest.approx(4095 / 11000)
    assert adc_gain(8, 11000.0) == pytest.approx(255 / 11000)


@pytest.mark.parametrize(
    ("bit_depth", "saturation"), [(0, 11000.0), (12, 0.0), (12, -5.0)]
)
def test_invalid_adc_settings_raise(bit_depth: int, saturation: float) -> None:
    with pytest.raises(ValueError):
        adc_gain(bit_depth, saturation)


def test_adc_output_is_whole_numbers_in_range() -> None:
    electrons = numpy.linspace(-500.0, 15000.0, 10000)
    digital = add_bit_depth_noise(electrons, 12, 11000.0)
    numpy.testing.assert_array_equal(digital, numpy.round(digital))
    assert digital.min() == 0.0
    assert digital.max() == 4095.0


def test_adc_clips_only_above_saturation() -> None:
    gain = adc_gain(12, 11000.0)
    below = add_bit_depth_noise(numpy.array([10900.0]), 12, 11000.0)
    above = add_bit_depth_noise(numpy.array([11100.0, 50000.0]), 12, 11000.0)
    assert below[0] == pytest.approx(10900.0 * gain, abs=0.5)
    numpy.testing.assert_array_equal(above, 4095.0)


def test_quantisation_noise_is_one_twelfth() -> None:
    """Rounding a smooth signal adds a uniform error of variance 1/12 DN^2."""
    rng = numpy.random.default_rng(0)
    electrons = rng.uniform(1000.0, 9000.0, 200_000)
    digital = add_bit_depth_noise(electrons, 12, 11000.0)
    error = digital - electrons * adc_gain(12, 11000.0)
    assert error.var() == pytest.approx(1 / 12, rel=0.02)


# Whole recording chain ----------------------------------------------------


def test_recording_follows_the_emva_noise_law() -> None:
    """
    The variance of the digital output follows EMVA 1288, Eq. 14.

    On an empty field every pixel has the same mean, so the variance must equal
    K^2 (sigma_d^2 + mu_e + mu_dark) + 1/12, with the dark electrons as an extra
    Poisson term.
    """
    background, dark, read, bits = 4000.0, 7.1, 10.0, 12
    sim = InlineHologramSimulator(resolution=256, pixel_size=4.65)
    digital = sim.generate_hologram(
        z_distance=20000,
        shot_noise=True,
        dark_noise_mean=dark,
        read_noise_sigma=read,
        bit_depth=bits,
        background_electrons=background,
    )
    gain = adc_gain(bits, DEFAULT_SATURATION_ELECTRONS)
    assert digital.mean() == pytest.approx(gain * (background + dark), rel=0.002)
    expected = gain**2 * (read**2 + background + dark) + 1 / 12
    assert digital.var() == pytest.approx(expected, rel=0.03)


def test_dataset_settings_do_not_saturate() -> None:
    """
    No hologram of the dataset recipe reaches full scale.

    This is a regression test. With the 4000 e- exposure, the old ADC step
    (gain 1) clipped 27% of the pixels at 4095. With the gain set by the
    saturation capacity, no hologram of the recipe's geometry, camera and laser
    speckle should clip.
    """
    recipe = default_recipe()
    config = recipe.hologram_config()
    noise = recipe.camera_noise(recipe.speckle.laser_contrast)
    clipped = 0.0
    for sample in range(20):
        RandomNumberGenerator.set_seed(10000 + sample)
        sim = InlineHologramSimulator(
            config.resolution,
            config.pixel_size,
            config.wavelength,
            config.numerical_aperture,
        )
        add_random_cells(sim, num_cells=config.num_cells)
        digital = sim.generate_hologram(
            z_distance=config.z_distance,
            speckle_grain_size=noise.speckle_grain_size,
            speckle_contrast=noise.speckle_contrast,
            shot_noise=True,
            dark_noise_mean=noise.dark_noise_mean,
            read_noise_sigma=noise.read_noise_sigma,
            bit_depth=config.bit_depth,
            background_electrons=config.background_electrons,
            saturation_electrons=config.saturation_electrons,
        )
        clipped = max(clipped, float(numpy.mean(digital >= 4095)))
    assert clipped == 0.0
