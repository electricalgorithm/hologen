"""
Tests for holograms recorded at a distance other than the nominal one.

A hologram is recorded at z_distance + z_offset + U(-z_error, z_error) and
back-propagated at z_distance, the distance a user would assume. The random
error comes from its own stream, so a hologram keeps its objects and speckle
whatever the error. Its camera noise changes realisation with the distance,
because Poisson sampling consumes a number of random values that depends on
the intensity.
"""

import pathlib

import numpy
import pandas
import pytest
from pandas import DataFrame

from hologen.dataset.configs import HologramConfig
from hologen.dataset.exposure import peak_ratios
from hologen.dataset.measure import (
    exposure_fingerprint,
    exposures_are_current,
)
from hologen.dataset.recipe import Recipe, default_recipe, recipe_from_dict
from hologen.propagator import backwards_propagate
from hologen.utils import load_samples_from_parquet
from test_generator import generate

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


def test_the_distance_draw_leaves_the_shared_stream_alone(
    tmp_path: pathlib.Path,
) -> None:
    """
    A random-distance hologram equals one generated directly at its distance.

    Both record at the same distance, so any difference could only come from the
    distance draw consuming values of the shared random stream.

    Args:
        tmp_path (pathlib.Path): Folder for the two datasets.
    """
    varied: Recipe = recipe_with({"z_error": 6.0}, samples=1)
    generate(varied, tmp_path / "varied")
    z_true: float = float(
        pandas.read_parquet(tmp_path / "varied" / "metadata_0000.parquet")["z_true"][0]
    )
    assert z_true != NOMINAL

    direct: Recipe = recipe_with({"z_distance": z_true}, samples=1)
    generate(direct, tmp_path / "direct")
    for kind in ("ground_truth", "hologram"):
        a: numpy.ndarray = load_samples_from_parquet(tmp_path / "varied", kind, 96)[0]
        b: numpy.ndarray = load_samples_from_parquet(tmp_path / "direct", kind, 96)[0]
        numpy.testing.assert_array_equal(a, b)


def test_the_same_seed_keeps_its_objects_whatever_the_error(
    tmp_path: pathlib.Path,
) -> None:
    """
    Fixed-distance and z-augmented datasets from one seed hold the same objects.

    Args:
        tmp_path (pathlib.Path): Folder for the two datasets.
    """
    generate(recipe_with({}, samples=2), tmp_path / "fixed")
    generate(recipe_with({"z_error": 6.0}, samples=2), tmp_path / "varied")
    for index in range(2):
        fixed: numpy.ndarray = load_samples_from_parquet(
            tmp_path / "fixed", "ground_truth", 96
        )[index]
        varied: numpy.ndarray = load_samples_from_parquet(
            tmp_path / "varied", "ground_truth", 96
        )[index]
        numpy.testing.assert_array_equal(fixed, varied)
    metadata: DataFrame = pandas.read_parquet(
        tmp_path / "varied" / "metadata_0000.parquet"
    )
    assert (metadata["z_true"] != NOMINAL).all()


def test_offset_records_at_the_shifted_distance_and_reconstructs_at_nominal(
    tmp_path: pathlib.Path,
) -> None:
    shifted: Recipe = recipe_with({"z_offset": 4.0}, samples=1)
    generate(shifted, tmp_path / "shifted")
    direct: Recipe = recipe_with({"z_distance": NOMINAL + 4.0}, samples=1)
    generate(direct, tmp_path / "direct")

    hologram: numpy.ndarray = load_samples_from_parquet(
        tmp_path / "shifted", "hologram", 96
    )[0]
    numpy.testing.assert_array_equal(
        hologram, load_samples_from_parquet(tmp_path / "direct", "hologram", 96)[0]
    )

    # The stored reconstruction is the nominal-distance back-propagation.
    config: HologramConfig = shifted.hologram_config()
    background_dn: float = (
        float(
            pandas.read_parquet(tmp_path / "shifted" / "metadata_0000.parquet")[
                "background_electrons"
            ][0]
        )
        * (2**config.bit_depth - 1)
        / config.saturation_electrons
    )
    expected: numpy.ndarray = backwards_propagate(
        numpy.sqrt(hologram / background_dn).astype(complex),
        NOMINAL,
        config.wavelength,
        config.pixel_size,
        config.resolution,
    )
    stored: numpy.ndarray = load_samples_from_parquet(
        tmp_path / "shifted", "reconstructed", 96
    )[0]
    numpy.testing.assert_allclose(stored, expected, atol=1e-6)


def test_metadata_records_true_and_nominal_distance(tmp_path: pathlib.Path) -> None:
    recipe: Recipe = recipe_with({"z_error": 6.0, "z_offset": 1.0})
    generate(recipe, tmp_path)
    metadata: DataFrame = pandas.read_parquet(tmp_path / "metadata_0000.parquet")
    assert (metadata["z_nominal"] == NOMINAL).all()
    assert metadata["z_true"].between(NOMINAL + 1.0 - 6.0, NOMINAL + 1.0 + 6.0).all()
    assert metadata["z_true"].nunique() == len(metadata)


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
