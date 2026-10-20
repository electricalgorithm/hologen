"""Tests for the recipe values that scale the phase and absorption of the cells."""

import hashlib

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.dataset.recipe import Recipe, default_recipe, recipe_from_dict

SEED: int = 4321
SIZE: int = 256  # pixels; holds several cells of each family and stays fast


def object_field(recipe: Recipe, family: str, seed: int = SEED) -> numpy.ndarray:
    """
    Build the object field of one family the way the generator does.

    Args:
        recipe (Recipe): Recipe whose geometry and object settings are used.
        family (str): red_blood_cells or adherent_cells.
        seed (int): Seed of the hologram.

    Returns:
        numpy.ndarray: The complex object field, shape (SIZE, SIZE).
    """
    config = recipe.hologram_config()
    RandomNumberGenerator.set_seed(seed)
    sim = InlineHologramSimulator(
        SIZE, config.pixel_size, config.wavelength, config.numerical_aperture
    )
    recipe.object_families()[family].place(sim)
    return sim.object_field


def field_digest(recipe: Recipe, family: str, seed: int = SEED) -> str:
    """
    Hash an object field exactly.

    Args:
        recipe (Recipe): Recipe to build the field from.
        family (str): red_blood_cells or adherent_cells.
        seed (int): Seed of the hologram.

    Returns:
        str: The first 16 hexadecimal digits of the SHA-256 of the field's bytes.
    """
    return hashlib.sha256(object_field(recipe, family, seed).tobytes()).hexdigest()[:16]


def test_training_values_leave_the_fields_unchanged() -> None:
    """The digests were recorded with the code before the scaling values existed."""
    assert field_digest(default_recipe(), "red_blood_cells") == "126f310fc3d0dfbf"
    assert field_digest(default_recipe(), "adherent_cells") == "dd3c87d739151dc0"


def scaled(**objects: float) -> Recipe:
    """
    Build the default recipe with some object values changed.

    Args:
        **objects (float): Values of the [objects] section to change.

    Returns:
        Recipe: The changed recipe.
    """
    return recipe_from_dict({"objects": objects})


def test_doubling_the_opd_doubles_the_phase_of_the_same_cells() -> None:
    """Transparent cells: twice the phase is the square of the field."""
    plain: numpy.ndarray = object_field(default_recipe(), "adherent_cells")
    doubled: numpy.ndarray = object_field(scaled(opd_scale=2.0), "adherent_cells")
    numpy.testing.assert_allclose(doubled, plain**2, atol=1e-12)


def test_doubling_the_opd_keeps_the_red_blood_cell_absorption() -> None:
    """A field A exp(i phi) becomes A exp(2 i phi) = field^2 / A."""
    plain: numpy.ndarray = object_field(default_recipe(), "red_blood_cells")
    doubled: numpy.ndarray = object_field(scaled(opd_scale=2.0), "red_blood_cells")
    numpy.testing.assert_allclose(doubled, plain**2 / numpy.abs(plain), atol=1e-12)


def test_adherent_absorption_is_proportional_to_the_opd() -> None:
    """The phase stays, and the log transmission scales with the coefficient."""
    plain: numpy.ndarray = object_field(default_recipe(), "adherent_cells")
    weak: numpy.ndarray = object_field(
        scaled(adherent_absorption=0.2), "adherent_cells"
    )
    strong: numpy.ndarray = object_field(
        scaled(adherent_absorption=0.6), "adherent_cells"
    )
    numpy.testing.assert_allclose(weak / numpy.abs(weak), plain, atol=1e-12)
    assert numpy.abs(weak).min() < 1.0
    numpy.testing.assert_allclose(
        numpy.log(numpy.abs(strong)), 3 * numpy.log(numpy.abs(weak)), atol=1e-12
    )


def test_haemoglobin_scale_multiplies_the_log_transmission() -> None:
    """Three times the absorption coefficient gives three times log |field|."""
    plain: numpy.ndarray = object_field(default_recipe(), "red_blood_cells")
    dark: numpy.ndarray = object_field(
        scaled(haemoglobin_absorption_scale=3.0), "red_blood_cells"
    )
    numpy.testing.assert_allclose(
        numpy.log(numpy.abs(dark)), 3 * numpy.log(numpy.abs(plain)), atol=1e-12
    )
    numpy.testing.assert_allclose(
        dark / numpy.abs(dark), plain / numpy.abs(plain), atol=1e-12
    )


@pytest.mark.parametrize(
    "objects",
    [
        {"opd_scale": 0.0},
        {"opd_scale": -1.0},
        {"adherent_absorption": -0.1},
        {"haemoglobin_absorption_scale": -1.0},
    ],
)
def test_impossible_values_are_rejected(objects: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must"):
        scaled(**objects)
