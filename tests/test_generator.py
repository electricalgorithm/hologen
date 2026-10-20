"""
Tests for the dataset generator.

Every noise configuration must reach the simulator unchanged, and the stored
reconstruction must have the physical scale.
"""

from pandas.core.frame import DataFrame
import pathlib

import numpy
import pandas
import pytest

from hologen import DatasetConfig, DatasetGenerator, InlineHologramSimulator
from hologen.dataset.recipe import Recipe, default_recipe, recipe_from_dict
from hologen.noises.bit_depth import adc_gain
from hologen.dataset.storage import load_samples

SAMPLES_PER_CONFIGURATION: int = 2


def tiny_recipe() -> Recipe:
    """
    Build a recipe small enough for a test.

    It has red blood cells on a 128 x 128 grid, which is 31 um wide and holds
    about 4 cells.

    Returns:
        Recipe: The recipe with 8 holograms.
    """
    return recipe_from_dict(
        {
            "geometry": {"resolution": 128},
            "dataset": {"total_samples": 8, "objects": ["red_blood_cells"]},
        }
    )


def generate(
    recipe: Recipe, output: pathlib.Path, write_reconstructed: bool = False
) -> None:
    """
    Generate a recipe's dataset the way hologen generate does.

    Args:
        recipe (Recipe): The recipe to generate.
        output (pathlib.Path): Folder for the dataset files.
        write_reconstructed (bool): Whether to store the back-propagated field.
    """
    generator = DatasetGenerator(
        DatasetConfig(
            output_dir=output,
            base_seed=recipe.dataset.seed,
            hologram_config=recipe.hologram_config(),
            generate_visualization=False,
            write_reconstructed=write_reconstructed,
        ),
        batch_size=recipe.dataset.batch_size,
    )
    counts: dict[str, int] = recipe.samples_per_group()
    for name, noise, family in recipe.groups():
        generator.add_config(
            name, num_samples=counts[name], noise=noise, objects=family
        )
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
        """
        Record the keyword arguments and call the real generate_hologram.

        Args:
            self (InlineHologramSimulator): The simulator.
            *args (object): Positional arguments of generate_hologram.
            **kwargs (object): Keyword arguments of generate_hologram.

        Returns:
            numpy.ndarray: The hologram from the real method.
        """
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
            assert (
                call["background_electrons"]
                == recipe.objects.red_blood_cells_background_electrons
            )
            assert call["saturation_electrons"] == camera.saturation_electrons
            if call["speckle_contrast"]:
                assert call["speckle_grain_size"] == speckle.grain_size


def test_stored_data_has_the_physical_scale(tmp_path: pathlib.Path) -> None:
    """
    The stored reconstruction has the amplitude scale of the ground truth.

    Holograms are stored in digital numbers with the illumination at
    K * electrons, and the stored back-propagation is normalised by that.
    """
    recipe = tiny_recipe()
    generate(recipe, tmp_path, write_reconstructed=True)

    config = recipe.hologram_config()
    # Each object family is recorded with its own exposure.
    background_dn: float = (
        recipe.objects.red_blood_cells_background_electrons
        * adc_gain(config.bit_depth, config.saturation_electrons)
    )
    names = pandas.read_parquet(
        tmp_path / "metadata_0000.parquet", columns=["sample_id", "config_name"]
    )
    assert names["config_name"].value_counts().to_dict() == dict.fromkeys(
        recipe.samples_per_group(), SAMPLES_PER_CONFIGURATION
    )

    holograms: dict[int, numpy.ndarray] = load_samples(tmp_path, "hologram")
    reconstructed: dict[int, numpy.ndarray] = load_samples(tmp_path, "reconstructed")
    truths: dict[int, numpy.ndarray] = load_samples(tmp_path, "ground_truth")
    for sample_id, hologram in holograms.items():
        # Propagation conserves power, so the mean recorded intensity is the
        # background times the mean transmission |O|^2 of this sample's object.
        transmission = numpy.mean(numpy.abs(truths[sample_id]) ** 2)
        assert hologram.mean() == pytest.approx(background_dn * transmission, rel=0.01)
        # Back-propagation conserves power too, so the reconstruction of sqrt(H)
        # has the same mean |.|^2 as the object when H is normalised correctly.
        power = numpy.mean(numpy.abs(reconstructed[sample_id]) ** 2)
        assert power == pytest.approx(transmission, rel=0.01)


def test_metadata_records_family_exposure_and_clipping(tmp_path: pathlib.Path) -> None:
    """One metadata row per hologram, consistent with the stored images."""
    recipe: Recipe = tiny_recipe()
    generate(recipe, tmp_path)
    metadata: DataFrame = pandas.read_parquet(tmp_path / "metadata_0000.parquet")
    holograms: dict[int, numpy.ndarray] = load_samples(tmp_path, "hologram")

    assert len(metadata) == recipe.dataset.total_samples
    assert set(metadata["object_family"]) == {"red_blood_cells"}
    assert (metadata["objects_placed"] > 0).all()
    assert (
        metadata["background_electrons"]
        == recipe.objects.red_blood_cells_background_electrons
    ).all()
    full_scale = 2**recipe.camera.bit_depth - 1
    for row in metadata.itertuples():
        expected = numpy.mean(holograms[row.sample_id] >= full_scale)
        assert row.clipped_fraction == pytest.approx(expected)


def test_default_recipe_is_unchanged_by_the_tiny_one() -> None:
    """The helper must not leak its overrides into the defaults."""
    assert default_recipe().geometry.resolution == 224
