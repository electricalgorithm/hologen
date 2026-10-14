"""
Tests for the red blood cell objects.

They cover the Evans and Fung 1972 shape, the Gienger et al. 2019 refractive
index and Prahl's haemoglobin absorption.
"""

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.objects import (
    RedBloodCellOptics,
    add_red_blood_cells,
    evans_fung_thickness,
)
from hologen.utils.domains import create_coordinate_grid

OPTICS = RedBloodCellOptics()
PIXEL: float = 0.24  # um
WAVELENGTH: float = 0.532  # um


@pytest.fixture(autouse=True)
def seed() -> None:
    """Seed the shared generator before every test."""
    RandomNumberGenerator.set_seed(97)


def test_profile_reproduces_the_published_dimensions() -> None:
    """Evans and Fung report 7.82 um diameter, 0.81 um centre, 2.58 um rim."""
    r = numpy.linspace(0.0, 4.0, 40001)
    h = evans_fung_thickness(r, OPTICS)
    diameter = 2 * r[h > 0].max()
    assert diameter == pytest.approx(7.82, abs=0.01)
    assert h[0] == pytest.approx(0.81, abs=0.005)
    assert h.max() == pytest.approx(2.57, abs=0.02)
    assert r[h.argmax()] == pytest.approx(2.74, abs=0.05)  # rim radius


def test_profile_scales_uniformly() -> None:
    r = numpy.linspace(0.0, 5.0, 5001)
    large = evans_fung_thickness(r * 1.1, OPTICS, scale=1.1)
    numpy.testing.assert_allclose(
        large, 1.1 * evans_fung_thickness(r, OPTICS), atol=1e-12
    )


def single_cell() -> InlineHologramSimulator:
    """
    Place one red blood cell of the Evans-Fung size on a 128 px grid.

    Returns:
        InlineHologramSimulator: The simulator with the cell.
    """
    sim = InlineHologramSimulator(
        resolution=128, pixel_size=PIXEL, wavelength=WAVELENGTH
    )
    assert add_red_blood_cells(sim, 1, RedBloodCellOptics(size_cv=0.0)) == 1
    return sim


def test_phase_follows_refractive_index_times_thickness() -> None:
    sim = single_cell()
    peak = numpy.angle(sim.object_field).max()
    expected = 2 * numpy.pi / WAVELENGTH * OPTICS.delta_n * 2.57
    assert peak == pytest.approx(expected, rel=0.02)
    assert 1.9 < peak < 2.2  # 0.6-0.7 rad centre, 1.9-2.2 rad rim at 532 nm


def test_haemoglobin_absorbs_at_532_nm() -> None:
    sim = single_cell()
    amplitude = numpy.abs(sim.object_field)
    assert amplitude.min() == pytest.approx(
        numpy.exp(-OPTICS.absorption * 2.57 / 2), rel=0.005
    )
    assert amplitude.max() == pytest.approx(1.0)  # the background is not absorbed


def test_cells_are_whole_and_do_not_overlap() -> None:
    sim = InlineHologramSimulator(
        resolution=224, pixel_size=PIXEL, wavelength=WAVELENGTH
    )
    placed = add_red_blood_cells(sim, 25)
    phase = numpy.angle(sim.object_field)
    assert placed == 25
    # Every edge pixel is background: no cell is cut by the border.
    edges = numpy.concatenate([phase[0], phase[-1], phase[:, 0], phase[:, -1]])
    assert numpy.all(edges == 0.0)
    # Non-overlapping cells never exceed the single-cell peak phase.
    single_peak = 2 * numpy.pi / WAVELENGTH * OPTICS.delta_n * 2.57
    assert phase.max() < single_peak * (1 + 4 * OPTICS.size_cv)


def test_crowded_field_places_fewer_cells() -> None:
    sim = InlineHologramSimulator(
        resolution=64, pixel_size=PIXEL, wavelength=WAVELENGTH
    )
    assert add_red_blood_cells(sim, 50, max_attempts=200) < 50


def test_cell_size_varies_with_the_red_cell_distribution_width() -> None:
    sizes = []
    for _ in range(200):
        sim = InlineHologramSimulator(
            resolution=64, pixel_size=PIXEL, wavelength=WAVELENGTH
        )
        add_red_blood_cells(sim, 1)
        area = numpy.count_nonzero(numpy.angle(sim.object_field)) * PIXEL**2
        sizes.append(2 * numpy.sqrt(area / numpy.pi))
    assert numpy.mean(sizes) == pytest.approx(7.82, rel=0.02)
    assert numpy.std(sizes) / numpy.mean(sizes) == pytest.approx(
        OPTICS.size_cv, abs=0.012
    )


def test_coordinate_grid_spacing_is_the_pixel_size() -> None:
    """Regression test: linspace spaced pixels length / (N - 1) apart (0.45% too far at 224)."""
    grid = create_coordinate_grid(224, PIXEL)
    numpy.testing.assert_allclose(numpy.diff(grid.X[0]), PIXEL)
    assert grid.X[0, 112] == 0.0
