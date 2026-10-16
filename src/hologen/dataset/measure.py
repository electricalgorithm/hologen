"""
Measure a dataset recipe and calibrate its exposures.

Two measurements back the numbers a paper quotes about a dataset. The exposure
calibration finds, for every object family, the background exposure at which
the recipe's exposure percentile of holograms stays unsaturated
(hologen.dataset.exposure). The group measurement then reports, for every noise
configuration and object family, how noisy the recorded holograms are, how far
flat-field normalisation moves the background and how often holograms clip.

Both use fixed seeds, so every number can be reproduced. The calibration seeds
start at CALIBRATION_SEED and the measurement seeds at MEASURE_SEED, far from
the seeds a dataset uses by default.

The exposure depends on the geometry, the speckle, the saturation capacity and
the objects. A fingerprint of those settings is stored next to the calibrated
exposures, so a later change to any of them can be detected. The fingerprint
covers the recipe only. A change to the object models in the code itself, such
as new cell statistics, needs a recalibration that the fingerprint cannot see.
"""

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import numpy

from hologen.dataset.configs import HologramConfig, NoiseConfig
from hologen.dataset.exposure import exposure_for_peak, peak_ratios
from hologen.dataset.recipe import Recipe, load_recipe, recipe_to_toml
from hologen.inline_simulator import InlineHologramSimulator
from hologen.noises.bit_depth import adc_gain
from hologen.objects.families import ObjectFamily
from hologen.utils import RandomNumberGenerator

CALIBRATION_SEED: int = 90000
MEASURE_SEED: int = 50000


@dataclasses.dataclass(frozen=True)
class ExposureCalibration:
    """The calibrated exposure of one object family and what it was derived from."""

    family: str
    exposure: float  # background electrons per pixel
    peak_median: float  # brightest point relative to the background
    peak_percentile: float
    peak_max: float
    percentile: float
    count: int
    seed: int


@dataclasses.dataclass(frozen=True)
class GroupStatistics:
    """Measured statistics of one group (noise configuration x object family)."""

    group: str
    exposure: float  # background electrons per pixel
    noise_rms: float  # rms deviation from the noise-free hologram, fraction
    mean_offset: float  # normalised background minus 1 when dividing by the mean
    mean_offset_sd: float
    median_offset: float  # the same when dividing by the median
    median_offset_sd: float
    clipping_fraction: float  # fraction of holograms with a pixel at full scale
    clipped_pixels: float  # mean clipped pixel fraction of those holograms


def exposure_fingerprint(recipe: Recipe) -> str:
    """
    Fingerprint the recipe settings that the calibrated exposures depend on.

    These are the geometry (including the image size), the speckle of the
    light source, the saturation capacity and the object settings. The distance
    offset and error are left out. A real camera keeps its exposure when the
    sample sits at the wrong distance, so the exposure is calibrated at the
    nominal distance and holograms recorded elsewhere may clip more often,
    which their metadata records. Camera noise,
    the noise configurations and the dataset size do not change the brightest
    point of a noise-free hologram, so they are left out.

    Args:
        recipe (Recipe): The recipe to fingerprint.

    Returns:
        str: The first 12 hexadecimal digits of a SHA-256 hash of the settings.
    """
    objects: dict[str, Any] = dataclasses.asdict(recipe.objects)
    for field in (
        "red_blood_cells_background_electrons",
        "adherent_cells_background_electrons",
        "exposure_calibration",
        "calibration_fingerprint",
    ):
        objects.pop(field)
    geometry: dict[str, Any] = dataclasses.asdict(recipe.geometry)
    # Exposures belong to the setup and are calibrated at the nominal distance.
    geometry.pop("z_offset")
    geometry.pop("z_error")
    settings: dict[str, Any] = {
        "geometry": geometry,
        "speckle": dataclasses.asdict(recipe.speckle),
        "saturation_electrons": recipe.camera.saturation_electrons,
        "objects": objects,
    }
    encoded: bytes = json.dumps(settings, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def exposures_are_current(recipe: Recipe) -> bool:
    """
    Check that the recipe's exposures were calibrated for its current settings.

    Args:
        recipe (Recipe): The recipe to check.

    Returns:
        bool: True if the stored fingerprint matches the recipe's settings.
    """
    return recipe.objects.calibration_fingerprint == exposure_fingerprint(recipe)


def calibrate_families(
    recipe: Recipe, count: int, seed: int = CALIBRATION_SEED
) -> list[ExposureCalibration]:
    """
    Calibrate the exposure of every object family selected in the recipe.

    Args:
        recipe (Recipe): Recipe with the geometry, camera, speckle, exposure
            percentile and object families.
        count (int): Number of noise-free holograms per family.
        seed (int): Seed of the first hologram of each family.

    Returns:
        list[ExposureCalibration]: One calibration per object family, in the
            order of the recipe's object list.
    """
    percentile: float = recipe.objects.exposure_percentile
    saturation: float = recipe.camera.saturation_electrons
    calibrations: list[ExposureCalibration] = []
    for family in recipe.dataset.objects:
        peaks: numpy.ndarray = peak_ratios(recipe, family, count, seed)
        peak: float = float(numpy.percentile(peaks, percentile))
        calibrations.append(
            ExposureCalibration(
                family=family,
                exposure=exposure_for_peak(peak, saturation),
                peak_median=float(numpy.median(peaks)),
                peak_percentile=peak,
                peak_max=float(peaks.max()),
                percentile=percentile,
                count=count,
                seed=seed,
            )
        )
    return calibrations


def _record(
    sim: InlineHologramSimulator,
    config: HologramConfig,
    noise: NoiseConfig,
    family: ObjectFamily,
    z_distance: float,
) -> numpy.ndarray:
    """
    Record a hologram of the simulator's objects with a group's noise and exposure.

    Args:
        sim (InlineHologramSimulator): Simulator with the objects placed.
        config (HologramConfig): Geometry and camera of the recipe.
        noise (NoiseConfig): Noise configuration of the group.
        family (ObjectFamily): Object family of the group, for its exposure.
        z_distance (float): Recording distance of this hologram, in microns.

    Returns:
        numpy.ndarray: The recorded hologram in digital numbers.
    """
    return sim.generate_hologram(
        z_distance,
        speckle_grain_size=noise.speckle_grain_size if noise.speckle else None,
        speckle_contrast=noise.speckle_contrast if noise.speckle else 0.0,
        shot_noise=noise.shot,
        dark_noise_mean=noise.dark_noise_mean if noise.dark else 0.0,
        read_noise_sigma=noise.read_noise_sigma if noise.read else 0.0,
        bit_depth=config.bit_depth,
        background_electrons=family.background_electrons,
        saturation_electrons=config.saturation_electrons,
    )


def measure_groups(
    recipe: Recipe, count: int, seed: int = MEASURE_SEED
) -> list[GroupStatistics]:
    """
    Measure noise, normalisation and clipping of every group of the recipe.

    Each group (noise configuration x object family) is simulated on the same
    seeds and recorded with the exposure of its object family.

    Args:
        recipe (Recipe): The recipe to measure.
        count (int): Number of holograms per group.
        seed (int): Seed of the first hologram of each group.

    Returns:
        list[GroupStatistics]: One row per group, in the order of the recipe.
    """
    config: HologramConfig = recipe.hologram_config()
    gain: float = adc_gain(config.bit_depth, config.saturation_electrons)
    full_scale: int = 2**config.bit_depth - 1
    rows: list[GroupStatistics] = []
    for name, noise, family in recipe.groups():
        background_dn: float = gain * family.background_electrons
        rms: list[float] = []
        by_mean: list[float] = []
        by_median: list[float] = []
        clipped: list[float] = []
        for i in range(count):
            RandomNumberGenerator.set_seed(seed + i)
            sim: InlineHologramSimulator = InlineHologramSimulator(
                config.resolution,
                config.pixel_size,
                config.wavelength,
                config.numerical_aperture,
            )
            family.place(sim)
            distance: float = config.true_distance(seed + i)
            ideal: numpy.ndarray = sim.generate_hologram(
                distance, background_electrons=1.0
            )
            recorded: numpy.ndarray = _record(sim, config, noise, family, distance)
            rms.append(float(numpy.std(recorded / background_dn - ideal)))
            by_mean.append(background_dn / float(recorded.mean()) - 1)
            by_median.append(background_dn / float(numpy.median(recorded)) - 1)
            clipped.append(float(numpy.mean(recorded >= full_scale)))
        clipped_fractions: numpy.ndarray = numpy.asarray(clipped)
        # Clipped fractions of only the holograms that clip at all.
        hit: numpy.ndarray = clipped_fractions[clipped_fractions > 0]
        rows.append(
            GroupStatistics(
                group=name,
                exposure=family.background_electrons,
                noise_rms=float(numpy.mean(rms)),
                mean_offset=float(numpy.mean(by_mean)),
                mean_offset_sd=float(numpy.std(by_mean)),
                median_offset=float(numpy.mean(by_median)),
                median_offset_sd=float(numpy.std(by_median)),
                clipping_fraction=float(numpy.mean(clipped_fractions > 0)),
                clipped_pixels=float(hit.mean()) if hit.size else 0.0,
            )
        )
    return rows


def calibration_table(calibrations: list[ExposureCalibration]) -> str:
    """
    Format exposure calibrations as a Markdown table.

    Args:
        calibrations (list[ExposureCalibration]): Results of calibrate_families.

    Returns:
        str: A Markdown table with one row per object family.
    """
    lines: list[str] = [
        "| family | peak median | peak at percentile | peak max | exposure (e-) | "
        "shot noise at background (%) |",
        "|---|---|---|---|---|---|",
    ]
    for c in calibrations:
        lines.append(
            f"| {c.family} | {c.peak_median:.2f}x | {c.peak_percentile:.2f}x | "
            f"{c.peak_max:.2f}x | {c.exposure:.0f} | {100 / numpy.sqrt(c.exposure):.2f} |"
        )
    return "\n".join(lines) + "\n"


def groups_table(rows: list[GroupStatistics]) -> str:
    """
    Format group statistics as a Markdown table, with fractions in percent.

    Args:
        rows (list[GroupStatistics]): Results of measure_groups.

    Returns:
        str: A Markdown table with one row per group.
    """
    lines: list[str] = [
        "| group | exposure (e-) | noise (rms, % of background) | normalised background, "
        "mean (%) | normalised background, median (%) | holograms clipping (%) | "
        "clipped pixels when clipping (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.group} | {r.exposure:.0f} | {100 * r.noise_rms:.2f} | "
            f"{100 * r.mean_offset:+.2f} (s.d. {100 * r.mean_offset_sd:.2f}) | "
            f"{100 * r.median_offset:+.2f} (s.d. {100 * r.median_offset_sd:.2f}) | "
            f"{100 * r.clipping_fraction:.1f} | {100 * r.clipped_pixels:.3f} |"
        )
    return "\n".join(lines) + "\n"


def apply_calibration(
    recipe: Recipe, calibrations: list[ExposureCalibration]
) -> Recipe:
    """
    Return a copy of the recipe with calibrated exposures, provenance and fingerprint.

    Families that were not calibrated keep their exposure.

    Args:
        recipe (Recipe): The recipe the calibrations were measured on.
        calibrations (list[ExposureCalibration]): Results of calibrate_families.

    Returns:
        Recipe: The updated recipe.
    """
    changes: dict[str, Any] = {
        f"{c.family}_background_electrons": round(c.exposure)
        for c in calibrations
        if c.family in ("red_blood_cells", "adherent_cells")
    }
    notes: list[str] = [
        f"{c.family} p{c.percentile:g} peak {c.peak_percentile:.2f}x "
        f"(median {c.peak_median:.2f}x, max {c.peak_max:.2f}x)"
        for c in calibrations
    ]
    first: ExposureCalibration = calibrations[0]
    changes["exposure_calibration"] = (
        "; ".join(notes)
        + f"; {first.count} holograms per family, seeds {first.seed}-"
        + f"{first.seed + first.count - 1}, {recipe.geometry.resolution} px"
    )
    updated: Recipe = dataclasses.replace(
        recipe, objects=dataclasses.replace(recipe.objects, **changes)
    )
    return dataclasses.replace(
        updated,
        objects=dataclasses.replace(
            updated.objects, calibration_fingerprint=exposure_fingerprint(updated)
        ),
    )


def write_calibration(
    path: pathlib.Path, calibrations: list[ExposureCalibration]
) -> Recipe:
    """
    Write calibrated exposures into a recipe file.

    The file is read, updated with apply_calibration and written back in the
    standard recipe layout, with the unit and source of every value.

    Args:
        path (pathlib.Path): The recipe file to update.
        calibrations (list[ExposureCalibration]): Results of calibrate_families
            on the recipe in that file.

    Returns:
        Recipe: The recipe as written.
    """
    recipe: Recipe = apply_calibration(load_recipe(path), calibrations)
    path.write_text(recipe_to_toml(recipe))
    return recipe
