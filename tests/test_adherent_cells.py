"""
Tests for the procedural adherent cells.

Their statistics must match the values measured on 7476 real cells by
scripts/calibrate_adherent_cells.py.
"""

import numpy
import pytest

from hologen import InlineHologramSimulator, RandomNumberGenerator
from hologen.objects import (
    AdherentCellStatistics,
    add_adherent_cells,
    cell_opd,
    sample_cell_shape,
)

STATS = AdherentCellStatistics()
PIXEL: float = 0.24  # um
WAVELENGTH: float = 0.532  # um


@pytest.fixture(scope="module")
def sampled_cells() -> dict[str, numpy.ndarray]:
    """
    Draw 400 cells and measure their shape parameters and fill ratios.

    Returns:
        dict[str, numpy.ndarray]: Diameters, aspect ratios, peak OPDs and fill
            ratios of the cells, each of length 400.
    """
    rng = numpy.random.default_rng(11)
    sim = InlineHologramSimulator(512, PIXEL, WAVELENGTH)
    diameters, aspects, peaks, fills = [], [], [], []
    for _ in range(400):
        diameter, aspect, peak = sample_cell_shape(rng, STATS)
        diameters.append(diameter)
        aspects.append(aspect)
        peaks.append(peak)
        cell = cell_opd(sim, (0.0, 0.0), min(diameter, 40.0), aspect, peak, rng, STATS)
        values = cell[cell > 0]
        fills.append(values.mean() / numpy.percentile(values, 95))
    return {
        "diameter": numpy.array(diameters),
        "aspect": numpy.array(aspects),
        "peak": numpy.array(peaks),
        "fill": numpy.array(fills),
    }


def test_size_matches_the_measured_cells(sampled_cells: dict) -> None:
    assert numpy.median(sampled_cells["diameter"]) == pytest.approx(27.87, rel=0.05)
    low, high = STATS.diameter_range
    assert sampled_cells["diameter"].min() >= low
    assert sampled_cells["diameter"].max() <= high


def test_peak_opd_matches_the_measured_cells(sampled_cells: dict) -> None:
    assert numpy.median(sampled_cells["peak"]) == pytest.approx(0.114, rel=0.08)
    low, high = STATS.peak_opd_range
    assert sampled_cells["peak"].min() >= low
    assert sampled_cells["peak"].max() <= high


def test_cells_are_elongated_like_the_measured_cells(sampled_cells: dict) -> None:
    assert numpy.median(sampled_cells["aspect"]) == pytest.approx(2.346, rel=0.08)
    assert sampled_cells["aspect"].min() >= 1.0
    assert sampled_cells["aspect"].max() <= STATS.max_aspect


def test_dome_profile_matches_the_measured_fill_ratio(sampled_cells: dict) -> None:
    """Measured mean / peak OPD: median 0.433, 10-90 percent range 0.312-0.588."""
    fill = sampled_cells["fill"]
    assert numpy.median(fill) == pytest.approx(0.433, abs=0.02)
    assert numpy.percentile(fill, 10) == pytest.approx(0.312, abs=0.04)
    assert numpy.percentile(fill, 90) == pytest.approx(0.588, abs=0.04)


def test_cell_peak_is_scaled_to_the_drawn_value() -> None:
    rng = numpy.random.default_rng(3)
    sim = InlineHologramSimulator(256, PIXEL, WAVELENGTH)
    cell = cell_opd(sim, (0.0, 0.0), 25.0, 2.0, 0.2, rng, STATS)
    assert numpy.percentile(cell[cell > 0], 95) == pytest.approx(0.2, rel=1e-9)


@pytest.fixture(scope="module")
def culture() -> tuple[InlineHologramSimulator, int]:
    """
    Fill a 1024 px field with adherent cells at the measured coverage.

    Returns:
        tuple[InlineHologramSimulator, int]: The simulator with the cells and
            the number of cells placed.
    """
    RandomNumberGenerator.set_seed(5)
    sim = InlineHologramSimulator(1024, PIXEL, WAVELENGTH)
    return sim, add_adherent_cells(sim)


def test_cells_cover_the_measured_fraction(culture) -> None:
    sim, placed = culture
    covered = numpy.mean(numpy.angle(sim.object_field) != 0)
    assert placed > 3
    assert covered == pytest.approx(STATS.coverage_median, abs=0.05)


def test_cells_are_transparent_and_inside_the_field(culture) -> None:
    sim, _ = culture
    numpy.testing.assert_allclose(numpy.abs(sim.object_field), 1.0, atol=1e-12)
    phase = numpy.angle(sim.object_field)
    edges = numpy.concatenate([phase[0], phase[-1], phase[:, 0], phase[:, -1]])
    assert numpy.all(edges == 0.0)


def test_empty_coverage_adds_nothing() -> None:
    sim = InlineHologramSimulator(128, PIXEL, WAVELENGTH)
    assert add_adherent_cells(sim, coverage=0.0) == 0
    numpy.testing.assert_array_equal(sim.object_field, 1.0)
