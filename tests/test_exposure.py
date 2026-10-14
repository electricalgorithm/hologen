"""
Tests for the exposure rule: the largest background at which q percent of
holograms keep their brightest point plus 3 sigma of shot noise below saturation.
"""

from typing import Any
import math

import numpy
import pytest

from hologen.dataset.exposure import calibrate_exposure, exposure_for_peak, peak_ratios
from hologen.dataset.recipe import recipe_from_dict

SATURATION: float = 14837.0


@pytest.mark.parametrize("peak", [1.0, 2.5, 4.85, 10.0])
def test_exposure_puts_the_peak_exactly_at_saturation(peak: float) -> None:
    background: float = exposure_for_peak(peak, SATURATION)
    brightest: float = peak * background
    assert brightest + 3 * math.sqrt(brightest) == pytest.approx(SATURATION)


def test_brighter_peaks_need_less_exposure() -> None:
    assert exposure_for_peak(5.0, SATURATION) < exposure_for_peak(2.0, SATURATION)


@pytest.mark.parametrize(("peak", "saturation"), [(0.0, SATURATION), (2.0, 0.0)])
def test_invalid_input_raises(peak: float, saturation: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        exposure_for_peak(peak, saturation)


def small_recipe():
    """Red blood cells on a 128 px grid: about 4 cells, fast to simulate."""
    return recipe_from_dict(
        {"geometry": {"resolution": 128}, "dataset": {"objects": ["red_blood_cells"]}}
    )


def test_peaks_are_reproducible_and_brighter_than_the_background() -> None:
    recipe = small_recipe()
    first: numpy.ndarray = peak_ratios(recipe, "red_blood_cells", 10, seed=123)
    second: numpy.ndarray = peak_ratios(recipe, "red_blood_cells", 10, seed=123)
    numpy.testing.assert_array_equal(first, second)
    assert (first > 1.0).all()  # cells focus light above the background


def test_calibrated_exposure_lets_about_the_rest_clip() -> None:
    """
    Calibrated on one set of seeds at q = 90, the exposure keeps about 90% of an
    independent set of holograms below saturation.
    """
    recipe = small_recipe()
    background, peak = calibrate_exposure(
        recipe, "red_blood_cells", percentile=90, count=200, seed=1000
    )
    assert background == pytest.approx(exposure_for_peak(peak, SATURATION))

    fresh: numpy.ndarray = peak_ratios(recipe, "red_blood_cells", 200, seed=5000)
    brightest: numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy.float64]] = (
        fresh * background
    )
    clipping = numpy.mean(brightest + 3 * numpy.sqrt(brightest) > SATURATION)
    assert clipping == pytest.approx(0.10, abs=0.06)
