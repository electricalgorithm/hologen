"""Tests for the dataset recipe: cited defaults, TOML round trip and validation."""

import dataclasses
import pathlib
import tomllib

import pytest

from hologen.dataset.measure import exposures_are_current
from hologen.dataset.recipe import (
    CONFIGURATION_NAMES,
    Recipe,
    default_recipe,
    load_recipe,
    recipe_from_dict,
    recipe_to_toml,
)


def test_defaults_are_the_cited_values() -> None:
    """The recipe reproduces the cited default values."""
    recipe = default_recipe()
    assert (recipe.dataset.total_samples, recipe.dataset.seed) == (25000, 10000)
    assert recipe.dataset.configurations == list(CONFIGURATION_NAMES)
    assert dataclasses.astuple(recipe.geometry) == (
        1024,
        2.4,
        10.0,
        0.45,
        0.532,
        20.0,
        0.0,
        0.0,
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
        ({"dataset": {"objects": ["neurons"]}}, "Unknown object"),
        ({"dataset": {"objects": []}}, "At least one object"),
        ({"dataset": {"objects": ["ellipses", "ellipses"]}}, "must not repeat"),
        ({"dataset": 3}, "must be a table"),
        ({"dataset": {"total_samples": "x"}}, "must be a whole number"),
        ({"dataset": {"total_samples": 100.0}}, "must be a whole number"),
        ({"dataset": {"configurations": "camera"}}, "must be a list of strings"),
        ({"camera": {"read_noise": "3"}}, "must be a number"),
        ({"camera": {"read_noise": True}}, "must be a number"),
        ({"objects": {"exposure_calibration": 1}}, "must be a string"),
        ({"dataset": {"seed": -1}}, "seed must be zero or more"),
        ({"dataset": {"batch_size": 0}}, "batch_size must be at least 1"),
        ({"geometry": {"resolution": 4}}, "resolution must be at least 8"),
        ({"geometry": {"wavelength": 0.0}}, "wavelength must be positive"),
        ({"geometry": {"numerical_aperture": 1.4}}, "numerical_aperture must be"),
        ({"geometry": {"num_cells": -1}}, "num_cells must be zero or more"),
        ({"camera": {"bit_depth": 40}}, "bit_depth must be between 1 and 16"),
        ({"camera": {"saturation_electrons": 0.0}}, "must be positive"),
        ({"camera": {"read_noise": -1.0}}, "read_noise must be zero or more"),
        ({"speckle": {"laser_contrast": 1.5}}, "laser_contrast must be in"),
        ({"speckle": {"grain_size": 0.0}}, "grain_size must be positive"),
        ({"objects": {"adherent_coverage": 1.0}}, "adherent_coverage must be in"),
        ({"objects": {"red_blood_cell_density": -0.1}}, "must be zero or more"),
        ({"objects": {"exposure_percentile": 0.0}}, "exposure_percentile must be"),
        ({"objects": {"adherent_cells_background_electrons": 0.0}}, "positive"),
    ],
)
def test_invalid_recipes_are_rejected(data: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        recipe_from_dict(data)


def test_whole_numbers_are_accepted_for_decimal_values() -> None:
    """z_distance = 20 in a file means 20.0, so the fingerprint stays the same."""
    recipe: Recipe = recipe_from_dict({"geometry": {"z_distance": 20}})
    assert isinstance(recipe.geometry.z_distance, float)
    assert recipe == default_recipe()
    assert exposures_are_current(recipe)


def test_toml_keeps_quotes_and_backslashes_in_strings() -> None:
    text: str = 'peak "p95" at C:\\data\\run'
    recipe: Recipe = recipe_from_dict({"objects": {"exposure_calibration": text}})
    parsed: dict = tomllib.loads(recipe_to_toml(recipe))
    assert parsed["objects"]["exposure_calibration"] == text


def test_samples_are_split_with_the_remainder_first() -> None:
    recipe = recipe_from_dict(
        {
            "dataset": {
                "total_samples": 10,
                "configurations": ["ideal", "camera", "camera_led_speckle"],
            }
        }
    )
    assert recipe.samples_per_group() == {
        "ideal/red_blood_cells": 2,
        "ideal/adherent_cells": 2,
        "camera/red_blood_cells": 2,
        "camera/adherent_cells": 2,
        "camera_led_speckle/red_blood_cells": 1,
        "camera_led_speckle/adherent_cells": 1,
    }


def test_groups_pair_every_configuration_with_every_family() -> None:
    recipe = recipe_from_dict(
        {
            "dataset": {
                "configurations": ["camera"],
                "objects": ["ellipses", "red_blood_cells"],
            }
        }
    )
    names = [(name, family.name) for name, _, family in recipe.groups()]
    assert names == [
        ("camera/ellipses", "ellipses"),
        ("camera/red_blood_cells", "red_blood_cells"),
    ]


def test_each_family_has_its_own_exposure() -> None:
    recipe = recipe_from_dict(
        {"objects": {"red_blood_cells_background_electrons": 1234.0}}
    )
    families = recipe.object_families()
    assert families["red_blood_cells"].background_electrons == 1234.0
    assert (
        families["adherent_cells"].background_electrons
        == recipe.objects.adherent_cells_background_electrons
    )
    assert (
        families["ellipses"].background_electrons == recipe.camera.background_electrons
    )


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
