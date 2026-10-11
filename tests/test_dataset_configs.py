"""
Tests for the noise configurations of the dataset recipe (hologen.dataset.recipe).

Every recording made by a real camera contains shot noise, dark current, read
noise and quantisation, so only the ideal reference may switch them off, and each
configuration must add a noise level that is clearly different from the others.
"""

import numpy
import pytest

from hologen import InlineHologramSimulator, NoiseConfig, RandomNumberGenerator
from hologen.dataset.recipe import default_recipe
from hologen.objects import add_random_cells

RECIPE = default_recipe()
CONFIGURATIONS: dict[str, NoiseConfig] = RECIPE.noise_configurations()

# rms deviation from the ideal hologram, in % of the background, as documented in
# Recipe.noise_configurations().
DOCUMENTED_NOISE: dict[str, float] = {
    "ideal": 0.02,
    "camera": 1.6,
    "camera_led_speckle": 2.1,
    "camera_laser_speckle": 12.1,
}


def record(noise: NoiseConfig, seed: int) -> numpy.ndarray:
    """Record a dataset hologram with a noise configuration, normalised by its mean."""
    RandomNumberGenerator.set_seed(seed)
    sim = InlineHologramSimulator(resolution=224, pixel_size=4.65)
    add_random_cells(sim, num_cells=5)
    hologram = sim.generate_hologram(
        z_distance=20000,
        speckle_grain_size=noise.speckle_grain_size if noise.speckle else None,
        speckle_contrast=noise.speckle_contrast if noise.speckle else 0.0,
        shot_noise=noise.shot,
        dark_noise_mean=noise.dark_noise_mean if noise.dark else 0.0,
        read_noise_sigma=noise.read_noise_sigma if noise.read else 0.0,
        bit_depth=12,
    )
    return hologram / hologram.mean()


def ideal_hologram(seed: int) -> numpy.ndarray:
    """The noise-free intensity |ASM(O)|^2 of the same sample, normalised."""
    RandomNumberGenerator.set_seed(seed)
    sim = InlineHologramSimulator(resolution=224, pixel_size=4.65)
    add_random_cells(sim, num_cells=5)
    hologram = sim.generate_hologram(z_distance=20000, background_electrons=1.0)
    return hologram / hologram.mean()


@pytest.fixture(scope="module")
def measured_noise() -> dict[str, float]:
    """rms deviation from the ideal hologram per configuration, in %."""
    seeds = range(40000, 40010)
    ideals = {seed: ideal_hologram(seed) for seed in seeds}
    return {
        name: 100
        * float(numpy.mean([numpy.std(record(noise, s) - ideals[s]) for s in seeds]))
        for name, noise in CONFIGURATIONS.items()
    }


def test_documented_configurations_are_the_ones_used() -> None:
    assert list(CONFIGURATIONS) == list(DOCUMENTED_NOISE)


def test_samples_are_split_equally() -> None:
    """Each configuration gets the same share and no sample is lost to rounding."""
    counts = RECIPE.samples_per_configuration()
    assert set(counts.values()) == {RECIPE.dataset.total_samples // len(CONFIGURATIONS)}
    assert sum(counts.values()) == RECIPE.dataset.total_samples


def test_only_the_ideal_reference_is_noise_free() -> None:
    ideal = CONFIGURATIONS["ideal"]
    assert not (ideal.speckle or ideal.shot or ideal.read or ideal.dark)


@pytest.mark.parametrize("name", [n for n in CONFIGURATIONS if n != "ideal"])
def test_every_recording_has_the_full_camera(name: str) -> None:
    """A real camera always adds shot, dark and read noise with the cited values."""
    noise = CONFIGURATIONS[name]
    assert noise.shot and noise.dark and noise.read
    assert noise.read_noise_sigma == RECIPE.camera.read_noise
    assert noise.dark_noise_mean == RECIPE.camera.dark_electrons


@pytest.mark.parametrize(
    ("name", "contrast"),
    [
        ("camera", None),
        ("camera_led_speckle", RECIPE.speckle.led_contrast),
        ("camera_laser_speckle", RECIPE.speckle.laser_contrast),
    ],
)
def test_light_source_sets_the_speckle(name: str, contrast: float | None) -> None:
    noise = CONFIGURATIONS[name]
    assert noise.speckle == (contrast is not None)
    assert noise.speckle_contrast == contrast


def test_measured_noise_matches_the_documentation(
    measured_noise: dict[str, float],
) -> None:
    for name, documented in DOCUMENTED_NOISE.items():
        assert measured_noise[name] == pytest.approx(documented, rel=0.15), name


def test_each_configuration_adds_a_distinct_noise_level(
    measured_noise: dict[str, float],
) -> None:
    """
    Regression test for the former 8 configurations, where dark-only and read-only
    sets were practically noise-free and four speckle sets were indistinguishable:
    each configuration must be at least 25% noisier than the previous one.
    """
    levels = [measured_noise[name] for name in DOCUMENTED_NOISE]
    for quieter, noisier in zip(levels, levels[1:]):
        assert noisier > 1.25 * quieter
