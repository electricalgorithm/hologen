"""
Tests for holograms recorded at a distance other than the nominal one.

A hologram is recorded at z_distance + z_offset + U(-z_error, z_error) and
back-propagated at z_distance, the distance a user would assume. The random
error comes from its own stream, so a hologram keeps its objects and speckle
whatever the error. Its camera noise changes realisation with the distance,
because Poisson sampling consumes a number of random values that depends on
the intensity.
"""

import numpy
import pytest

from hologen.dataset.configs import HologramConfig
from hologen.dataset.exposure import peak_ratios
from hologen.dataset.measure import (
    exposure_fingerprint,
    exposures_are_current,
)
from hologen.dataset.recipe import Recipe, default_recipe, recipe_from_dict

NOMINAL: float = 20.0  # um, the default z_distance


def recipe_with(geometry: dict[str, float], samples: int = 4) -> Recipe:
    """
    Build a small laser-speckle red blood cell recipe with geometry overrides.

    Laser speckle and camera noise draw from the shared random stream, so a
    recipe with noise shows whether the distance draw disturbs that stream.

    Args:
        geometry (dict[str, float]): Values for the geometry section.
        samples (int): Number of holograms.

    Returns:
        Recipe: A 96 x 96 pixel recipe with the given geometry.
    """
    return recipe_from_dict(
        {
            "geometry": {"resolution": 96, **geometry},
            "dataset": {
                "total_samples": samples,
                "objects": ["red_blood_cells"],
                "configurations": ["camera_laser_speckle"],
            },
        }
    )


def test_without_error_the_true_distance_is_nominal_plus_offset() -> None:
    config: HologramConfig = recipe_with({"z_offset": -4.0}).hologram_config()
    assert config.true_distance(seed=123) == NOMINAL - 4.0


def test_true_distance_stays_in_range_and_repeats_for_a_seed() -> None:
    config: HologramConfig = recipe_with({"z_error": 6.0}).hologram_config()
    distances: numpy.ndarray = numpy.array(
        [config.true_distance(seed) for seed in range(2000)]
    )
    assert distances.min() >= NOMINAL - 6.0
    assert distances.max() <= NOMINAL + 6.0
    # A uniform draw over 12 um has its mean at 20 um and a std of 12 / sqrt(12).
    assert distances.mean() == pytest.approx(NOMINAL, abs=0.3)
    assert distances.std() == pytest.approx(12.0 / numpy.sqrt(12.0), rel=0.05)
    assert config.true_distance(seed=7) == config.true_distance(seed=7)


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"z_error": -1.0}, "z_error must not be negative"),
        ({"z_offset": -15.0, "z_error": 6.0}, "must be positive"),
    ],
)
def test_impossible_distances_are_rejected(
    geometry: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        recipe_from_dict({"geometry": geometry})


def test_distance_errors_keep_the_exposure_of_the_setup() -> None:
    """
    A wrong distance does not change the camera's exposure.

    The stored default fingerprint stays valid, and recipes with a distance
    offset or error share the fingerprint of their setup.
    """
    assert exposures_are_current(default_recipe())
    for geometry in ({"z_offset": -4.0}, {"z_error": 6.0}):
        varied: Recipe = recipe_from_dict({"geometry": geometry})
        assert exposure_fingerprint(varied) == exposure_fingerprint(default_recipe())


@pytest.mark.parametrize("geometry", [{"z_offset": 4.0}, {"z_error": 6.0}])
def test_exposure_is_calibrated_at_the_nominal_distance(
    geometry: dict[str, float],
) -> None:
    """
    Calibration ignores the distance offset and error.

    Args:
        geometry (dict[str, float]): Distance values that must not matter.
    """
    numpy.testing.assert_array_equal(
        peak_ratios(recipe_with(geometry), "red_blood_cells", 3, seed=11),
        peak_ratios(recipe_with({}), "red_blood_cells", 3, seed=11),
    )
