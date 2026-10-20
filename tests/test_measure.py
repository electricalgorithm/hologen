"""Tests for the exposure calibration, its fingerprint and the group measurement."""

import pathlib

import numpy
import pytest

from hologen.dataset.exposure import exposure_for_peak, peak_ratios
from hologen.dataset.measure import (
    ExposureCalibration,
    GroupStatistics,
    apply_calibration,
    calibrate_families,
    calibration_table,
    exposure_fingerprint,
    exposures_are_current,
    groups_table,
    measure_groups,
    write_calibration,
)
from hologen.dataset.recipe import (
    Recipe,
    default_recipe,
    load_recipe,
    recipe_from_dict,
    recipe_to_toml,
)


def small_recipe() -> Recipe:
    """
    Red blood cells on a 64 px grid, which is about one cell and fast to simulate.

    Returns:
        Recipe: The default recipe with a small image and one object family.
    """
    return recipe_from_dict(
        {
            "geometry": {"resolution": 64},
            "dataset": {
                "objects": ["red_blood_cells"],
                "configurations": ["ideal", "camera"],
            },
        }
    )


def test_default_exposures_match_the_default_settings() -> None:
    """
    The fingerprint stored in the code must follow the default settings.

    If this fails, a default changed and the exposures need `hologen calibrate`
    again.
    """
    assert exposures_are_current(default_recipe())


def test_fingerprint_is_deterministic() -> None:
    assert exposure_fingerprint(default_recipe()) == exposure_fingerprint(
        default_recipe()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"geometry": {"resolution": 512}},
        {"geometry": {"numerical_aperture": 0.3}},
        {"speckle": {"laser_contrast": 0.5}},
        {"camera": {"saturation_electrons": 10000.0}},
        {"objects": {"red_blood_cell_density": 0.002}},
        {"objects": {"exposure_percentile": 99.0}},
        {"objects": {"opd_scale": 2.0}},
        {"objects": {"adherent_absorption": 0.55}},
        {"objects": {"haemoglobin_absorption_scale": 3.0}},
    ],
)
def test_settings_that_move_the_peak_change_the_fingerprint(
    change: dict[str, dict[str, object]],
) -> None:
    changed: Recipe = recipe_from_dict(change)
    assert exposure_fingerprint(changed) != exposure_fingerprint(default_recipe())
    assert not exposures_are_current(changed)


@pytest.mark.parametrize(
    "change",
    [
        {"camera": {"read_noise": 10.0}},
        {"camera": {"dark_current": 5.0}},
        {"dataset": {"total_samples": 100, "seed": 1}},
        {"dataset": {"configurations": ["ideal"]}},
        {"objects": {"red_blood_cells_background_electrons": 1000.0}},
        {"objects": {"exposure_calibration": "edited by hand"}},
    ],
)
def test_settings_that_leave_the_peak_alone_keep_the_fingerprint(
    change: dict[str, dict[str, object]],
) -> None:
    assert exposure_fingerprint(recipe_from_dict(change)) == exposure_fingerprint(
        default_recipe()
    )


def test_calibration_follows_the_exposure_rule() -> None:
    recipe: Recipe = small_recipe()
    calibration: ExposureCalibration = calibrate_families(recipe, 20, seed=7)[0]
    peaks: numpy.ndarray = peak_ratios(recipe, "red_blood_cells", 20, seed=7)

    assert calibration.family == "red_blood_cells"
    assert calibration.peak_percentile == pytest.approx(numpy.percentile(peaks, 95))
    assert calibration.peak_max == pytest.approx(peaks.max())
    assert calibration.exposure == pytest.approx(
        exposure_for_peak(calibration.peak_percentile, 14837.0)
    )


def fake_calibration(family: str, exposure: float) -> ExposureCalibration:
    """
    A calibration with made-up numbers, for tests that do not need a simulation.

    Args:
        family (str): Object family name.
        exposure (float): Background exposure in electrons.

    Returns:
        ExposureCalibration: The calibration.
    """
    return ExposureCalibration(
        family=family,
        exposure=exposure,
        peak_median=2.0,
        peak_percentile=3.0,
        peak_max=4.0,
        percentile=95.0,
        count=10,
        seed=100,
    )


def test_applied_calibration_is_current_and_keeps_other_families() -> None:
    stale: Recipe = recipe_from_dict({"geometry": {"resolution": 512}})
    updated: Recipe = apply_calibration(
        stale, [fake_calibration("red_blood_cells", 1234.4)]
    )

    assert updated.objects.red_blood_cells_background_electrons == 1234
    assert (
        updated.objects.adherent_cells_background_electrons
        == stale.objects.adherent_cells_background_electrons
    )
    assert updated.objects.exposure_calibration == (
        "red_blood_cells p95 peak 3.00x (median 2.00x, max 4.00x); "
        "10 holograms per family, seeds 100-109, 512 px"
    )
    assert exposures_are_current(updated)


def test_ellipses_calibration_does_not_touch_the_camera() -> None:
    recipe: Recipe = default_recipe()
    updated: Recipe = apply_calibration(recipe, [fake_calibration("ellipses", 999.0)])
    assert updated.camera == recipe.camera


def test_written_calibration_round_trips(tmp_path: pathlib.Path) -> None:
    path: pathlib.Path = tmp_path / "recipe.toml"
    path.write_text(recipe_to_toml(recipe_from_dict({"geometry": {"resolution": 512}})))

    written: Recipe = write_calibration(
        path, [fake_calibration("adherent_cells", 2000.0)]
    )

    assert load_recipe(path) == written
    assert written.objects.adherent_cells_background_electrons == 2000
    assert exposures_are_current(load_recipe(path))


def test_ideal_group_has_no_noise_and_no_clipping() -> None:
    rows: list[GroupStatistics] = measure_groups(small_recipe(), 3, seed=11)
    by_group: dict[str, GroupStatistics] = {row.group: row for row in rows}

    assert list(by_group) == ["ideal/red_blood_cells", "camera/red_blood_cells"]
    ideal: GroupStatistics = by_group["ideal/red_blood_cells"]
    camera: GroupStatistics = by_group["camera/red_blood_cells"]
    assert ideal.noise_rms < 0.002  # only quantisation is left
    assert camera.noise_rms > 5 * ideal.noise_rms
    assert ideal.clipping_fraction == 0.0
    assert ideal.clipped_pixels == 0.0


def test_tables_have_one_row_per_entry() -> None:
    calibrations: list[ExposureCalibration] = [
        fake_calibration("red_blood_cells", 2982.0),
        fake_calibration("adherent_cells", 2593.0),
    ]
    row: GroupStatistics = GroupStatistics(
        group="camera/red_blood_cells",
        exposure=2982.0,
        noise_rms=0.0191,
        mean_offset=0.0227,
        mean_offset_sd=0.004,
        median_offset=0.0023,
        median_offset_sd=0.001,
        clipping_fraction=0.0,
        clipped_pixels=0.0,
    )

    calibration_lines: list[str] = calibration_table(calibrations).splitlines()
    group_lines: list[str] = groups_table([row]).splitlines()

    assert len(calibration_lines) == 4  # header, separator, two families
    assert calibration_lines[2].startswith(
        "| red_blood_cells | 2.00x | 3.00x | 4.00x | 2982 |"
    )
    assert group_lines[2] == (
        "| camera/red_blood_cells | 2982 | 1.91 | +2.27 (s.d. 0.40) | "
        "+0.23 (s.d. 0.10) | 0.0 | 0.000 |"
    )
