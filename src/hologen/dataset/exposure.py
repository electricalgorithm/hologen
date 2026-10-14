"""
Calibrate how many photo-electrons the background of a hologram may collect.

Cells act as small lenses and focus light into spots brighter than the
background, so the brightest point of a hologram limits the exposure. How much
brighter that point is (the peak ratio p) varies from hologram to hologram. For
adherent cells the distribution has a long tail, and its maximum keeps growing
as more holograms are generated. An exposure chosen for the worst case would
therefore depend on the size of the dataset.

Instead, the exposure is set the way an operator sets it on a few representative
fields at the microscope. It is the largest background B for which q percent of
holograms keep their brightest point, plus 3 sigma of shot noise, below the
saturation capacity. With p_q the q-th percentile of the peak ratio, B solves

    p_q * B + 3 sqrt(p_q * B) = saturation

The other (100 - q) percent of holograms clip at the brightest spots of the most
strongly refracting cells, as happens in real recordings.
"""

import math
from collections.abc import Callable

import numpy

from hologen.dataset.configs import HologramConfig
from hologen.dataset.recipe import Recipe
from hologen.inline_simulator import InlineHologramSimulator
from hologen.utils import RandomNumberGenerator


def exposure_for_peak(peak_ratio: float, saturation_electrons: float) -> float:
    """
    Find the largest background exposure that keeps a bright point unsaturated.

    The point is peak_ratio times as bright as the background. Together with
    3 sigma of its shot noise it must stay below the saturation capacity.

    Args:
        peak_ratio (float): Brightness of the point relative to the background.
        saturation_electrons (float): Saturation capacity of a pixel (in e-).

    Returns:
        float: The background exposure (in e- per pixel).

    Raises:
        ValueError: If peak_ratio or saturation_electrons is not positive.
    """
    if peak_ratio <= 0 or saturation_electrons <= 0:
        raise ValueError("peak_ratio and saturation_electrons must be positive.")
    # With x = peak_ratio * B, solve x + 3 sqrt(x) = saturation for sqrt(x).
    root: float = (-3 + math.sqrt(9 + 4 * saturation_electrons)) / 2
    return root**2 / peak_ratio


def peak_ratios(recipe: Recipe, family: str, count: int, seed: int) -> numpy.ndarray:
    """
    Measure the brightest point of noise-free holograms of an object family.

    Each hologram is simulated with the laser speckle of the recipe, the
    configuration with the brightest spots, and without camera noise. Hologram i
    uses seed + i, so the seeds should not overlap those of a dataset.

    Args:
        recipe (Recipe): Recipe with the geometry, speckle and object families.
        family (str): Name of the object family, e.g. "red_blood_cells".
        count (int): Number of holograms to simulate.
        seed (int): Seed of the first hologram.

    Returns:
        numpy.ndarray: The brightest point of each hologram, relative to the
            background, with shape (count,).
    """
    config: HologramConfig = recipe.hologram_config()
    place: Callable[[InlineHologramSimulator], int] = recipe.object_families()[
        family
    ].place
    ratios: list[float] = []
    for i in range(count):
        RandomNumberGenerator.set_seed(seed + i)
        sim: InlineHologramSimulator = InlineHologramSimulator(
            config.resolution,
            config.pixel_size,
            config.wavelength,
            config.numerical_aperture,
        )
        place(sim)
        intensity: numpy.ndarray = sim.generate_hologram(
            config.z_distance,
            speckle_grain_size=recipe.speckle.grain_size,
            speckle_contrast=recipe.speckle.laser_contrast,
            background_electrons=1.0,
        )
        ratios.append(float(intensity.max()))
    return numpy.asarray(ratios)


def calibrate_exposure(
    recipe: Recipe, family: str, percentile: float, count: int, seed: int
) -> tuple[float, float]:
    """
    Calibrate the background exposure of an object family.

    The exposure keeps `percentile` percent of the family's holograms below
    saturation (see exposure_for_peak and peak_ratios).

    Args:
        recipe (Recipe): Recipe with the geometry, camera and object families.
        family (str): Name of the object family, e.g. "adherent_cells".
        percentile (float): Percent of holograms that must not saturate.
        count (int): Number of holograms to measure.
        seed (int): Seed of the first hologram.

    Returns:
        tuple[float, float]: The background exposure (in e- per pixel) and the
            peak ratio at the percentile it was derived from.
    """
    peaks: numpy.ndarray = peak_ratios(recipe, family, count, seed)
    peak: float = float(numpy.percentile(peaks, percentile))
    background: float = exposure_for_peak(peak, recipe.camera.saturation_electrons)
    return background, peak
