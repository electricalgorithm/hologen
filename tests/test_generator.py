"""
Tests for the dataset generator: that every noise configuration reaches the
simulator unchanged, and that the stored reconstruction has the physical scale.
"""

import pathlib

import numpy
import pandas
import pytest

from hologen import DatasetConfig, DatasetGenerator, InlineHologramSimulator
from hologen.dataset.recipe import Recipe, default_recipe, recipe_from_dict
from hologen.noises.bit_depth import adc_gain
from hologen.utils import load_samples_from_parquet

SAMPLES_PER_CONFIGURATION: int = 2


def tiny_recipe() -> Recipe:
    """The default recipe on a 64 x 64 grid, small enough for a test."""
    return recipe_from_dict(
        {"geometry": {"resolution": 64}, "dataset": {"total_samples": 8}}
    )


def generate(recipe: Recipe, output: pathlib.Path) -> None:
    generator = DatasetGenerator(
        DatasetConfig(
            output_dir=output,
            base_seed=recipe.dataset.seed,
            hologram_config=recipe.hologram_config(),
            generate_visualization=False,
        ),
        batch_size=recipe.dataset.batch_size,
    )
    counts = recipe.samples_per_configuration()
    for name, noise in recipe.noise_configurations().items():
        generator.add_config(name, num_samples=counts[name], noise=noise)
    generator.generate(verbose=False)


def test_each_configuration_reaches_the_simulator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generator passes every noise and camera setting of the recipe on."""
    calls: list[dict[str, object]] = []
    original = InlineHologramSimulator.generate_hologram

    def spy(
        self: InlineHologramSimulator, *args: object, **kwargs: object
    ) -> numpy.ndarray:
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(InlineHologramSimulator, "generate_hologram", spy)
    recipe = tiny_recipe()
    generate(recipe, tmp_path)

    camera, speckle = recipe.camera, recipe.speckle
    expected = {
        "ideal": (0.0, False, 0.0, 0.0),
        "camera": (0.0, True, camera.dark_electrons, camera.read_noise),
        "camera_led_speckle": (
            speckle.led_contrast,
            True,
            camera.dark_electrons,
            camera.read_noise,
        ),
        "camera_laser_speckle": (
            speckle.laser_contrast,
            True,
            camera.dark_electrons,
            camera.read_noise,
        ),
    }
    assert len(calls) == 8
    for index, name in enumerate(recipe.dataset.configurations):
        for call in calls[2 * index : 2 * index + 2]:
            got = (
                call["speckle_contrast"],
                call["shot_noise"],
                call["dark_noise_mean"],
                call["read_noise_sigma"],
            )
            assert got == expected[name], name
            assert call["bit_depth"] == camera.bit_depth
            assert call["background_electrons"] == camera.background_electrons
            assert call["saturation_electrons"] == camera.saturation_electrons
            if call["speckle_contrast"]:
                assert call["speckle_grain_size"] == speckle.grain_size


def test_stored_data_has_the_physical_scale(tmp_path: pathlib.Path) -> None:
    """
    Holograms are stored in digital numbers with the illumination at K * electrons,
    and the stored back-propagation is normalised by that, so its amplitude has
    the scale of the ground-truth object.
    """
    recipe = tiny_recipe()
    generate(recipe, tmp_path)

    config = recipe.hologram_config()
    background_dn = config.background_electrons * adc_gain(
        config.bit_depth, config.saturation_electrons
    )
    names = pandas.read_parquet(
        tmp_path / "hologram_0000.parquet", columns=["sample_id", "config_name"]
    ).drop_duplicates()
    assert names["config_name"].value_counts().to_dict() == dict.fromkeys(
        recipe.dataset.configurations, SAMPLES_PER_CONFIGURATION
    )

    holograms = load_samples_from_parquet(tmp_path, "hologram", resolution=64)
    reconstructed = load_samples_from_parquet(tmp_path, "reconstructed", resolution=64)
    truths = load_samples_from_parquet(tmp_path, "ground_truth", resolution=64)
    for sample_id, hologram in holograms.items():
        # Propagation conserves power, so the mean recorded intensity is the
        # background times the mean transmission |O|^2 of this sample's object.
        transmission = numpy.mean(numpy.abs(truths[sample_id]) ** 2)
        assert hologram.mean() == pytest.approx(background_dn * transmission, rel=0.01)
        assert hologram.max() < 2**config.bit_depth - 1
        # Back-propagation conserves power too, so the reconstruction of sqrt(H)
        # has the same mean |.|^2 as the object when H is normalised correctly.
        power = numpy.mean(numpy.abs(reconstructed[sample_id]) ** 2)
        assert power == pytest.approx(transmission, rel=0.01)


def test_default_recipe_is_unchanged_by_the_tiny_one() -> None:
    """The helper must not leak its overrides into the defaults."""
    assert default_recipe().geometry.resolution == 224
