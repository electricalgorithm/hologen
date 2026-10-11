"""
Tests for the dataset recipe: cited defaults, TOML round trip and validation.
"""

import dataclasses
import pathlib
import tomllib

import pytest

from hologen.dataset.recipe import (
    CONFIGURATION_NAMES,
    Recipe,
    default_recipe,
    load_recipe,
    recipe_from_dict,
    recipe_to_toml,
)


def test_defaults_are_the_cited_values() -> None:
    """The recipe reproduces the values recorded in the HOLO-015 register."""
    recipe = default_recipe()
    assert (recipe.dataset.total_samples, recipe.dataset.seed) == (25000, 10000)
    assert recipe.dataset.configurations == list(CONFIGURATION_NAMES)
    assert dataclasses.astuple(recipe.geometry) == (
        224,
        2.4,
        10.0,
        0.45,
        0.532,
        20.0,
        5,
    )
    assert recipe.geometry.pixel_size == pytest.approx(0.24)
    assert recipe.camera.background_electrons == 5400.0
    assert recipe.camera.saturation_electrons == 14837.0
    assert recipe.camera.bit_depth == 12
    assert recipe.camera.read_noise == 3.30
    assert recipe.camera.dark_electrons == pytest.approx(0.0245, abs=0.0005)
    assert dataclasses.astuple(recipe.speckle) == (0.59, 0.014, 0.12)


def test_every_value_has_a_unit_or_source() -> None:
    recipe = default_recipe()
    for section in dataclasses.fields(recipe):
        for field in dataclasses.fields(getattr(recipe, section.name)):
            assert field.metadata.get("source"), f"{section.name}.{field.name}"


def test_toml_round_trip_keeps_every_value() -> None:
    recipe = default_recipe()
    text = recipe_to_toml(recipe)
    assert recipe_from_dict(tomllib.loads(text)) == recipe


def test_toml_lists_the_sources() -> None:
    text = recipe_to_toml(default_recipe())
    assert "EMVA 1288" in text
    assert "arXiv:2006.09296" in text
    assert "background_electrons = 5400.0  # e- per pixel" in text


def test_partial_file_keeps_the_other_defaults(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text("[camera]\nread_noise = 20.0\n\n[dataset]\nseed = 7\n")
    recipe = load_recipe(path)
    assert recipe.camera.read_noise == 20.0
    assert recipe.dataset.seed == 7
    assert recipe.camera.background_electrons == 5400.0
    assert recipe.geometry == default_recipe().geometry


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"cameras": {"read_noise": 1.0}}, "Unknown recipe section"),
        ({"camera": {"read_nosie": 1.0}}, "Unknown key"),
        ({"dataset": {"configurations": ["camera", "laser"]}}, "Unknown configuration"),
        ({"dataset": {"configurations": []}}, "At least one"),
        ({"dataset": {"configurations": ["camera", "camera"]}}, "must not repeat"),
        ({"dataset": {"total_samples": 2}}, "at least the number"),
    ],
)
def test_invalid_recipes_are_rejected(data: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        recipe_from_dict(data)


def test_samples_are_split_with_the_remainder_first() -> None:
    recipe = recipe_from_dict(
        {
            "dataset": {
                "total_samples": 10,
                "configurations": ["ideal", "camera", "camera_led_speckle"],
            }
        }
    )
    assert recipe.samples_per_configuration() == {
        "ideal": 4,
        "camera": 3,
        "camera_led_speckle": 3,
    }


def test_configuration_subset_keeps_the_order_given() -> None:
    recipe = recipe_from_dict(
        {"dataset": {"configurations": ["camera_laser_speckle", "ideal"]}}
    )
    assert list(recipe.noise_configurations()) == ["camera_laser_speckle", "ideal"]


def test_hologram_config_uses_the_recipe() -> None:
    recipe = recipe_from_dict(
        {"geometry": {"resolution": 128}, "camera": {"bit_depth": 10}}
    )
    config = recipe.hologram_config()
    assert (config.resolution, config.bit_depth) == (128, 10)
    assert config.saturation_electrons == 14837.0
    assert config.numerical_aperture == 0.45


def test_recipe_is_a_plain_value() -> None:
    """Two recipes with the same values are equal, which the round trip relies on."""
    assert Recipe() == default_recipe()
