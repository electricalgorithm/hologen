"""
Procedural adherent cells as phase objects, calibrated to measured cells.

The calibration (scripts/calibrate_adherent_cells.py) measures 7476 segmented
cells of five lines (PC-3, PNT1A, HOB, G361, A2058) in the quantitative phase
dataset of Vicar, Balvan, Gumulec, Zenodo, doi:10.5281/zenodo.4771831 (CC BY
4.0), with OPD = 0.18 um^3/pg x dry-mass density (Vicar et al., Sci. Rep.
10:1566, 2020) and an inferred 0.63 um pixel. Each cell draws its size and peak
optical path difference (OPD) jointly from the measured lognormal distributions
and its elongation from the measured aspect ratios; its dome profile matches the
measured ratio of mean to peak OPD.

Features finer than about 1 um (irregular outline, nucleus, granular texture) are
not resolved in that data (NA 0.3 at 650 nm), so their parameters are design
choices, not measurements.

Unstained cells are transparent: the amplitude stays 1 and only the phase
2 pi OPD / wavelength changes.
"""

import dataclasses

import numpy
from scipy.ndimage import gaussian_filter

import hologen.utils as utils
from hologen.inline_simulator import InlineHologramSimulator


@dataclasses.dataclass(frozen=True)
class AdherentCellStatistics:
    """Cell statistics; lengths and OPD in microns."""

    # Measured (scripts/calibrate_adherent_cells.py, 7476 cells).
    diameter_median: float = 27.87  # equivalent diameter
    diameter_log_sd: float = 0.432
    aspect_median: float = 2.346  # long / short axis
    aspect_log_sd: float = 0.545
    peak_opd_median: float = 0.114  # 95th percentile of OPD within a cell
    peak_opd_log_sd: float = 0.750
    size_opd_correlation: float = 0.266  # of log diameter and log peak OPD
    coverage_median: float = 0.263  # fraction of the field covered by cells
    # Profile exponent: OPD ~ (1 - rho^2)^p with p lognormal per cell, chosen so
    # the generated ratio of mean to peak OPD matches the measured median 0.433
    # and its 10-90 percent range 0.312-0.588.
    profile_exponent: float = 1.2
    profile_exponent_log_sd: float = 0.5
    # Design choices (below the resolution of the calibration data).
    outline_roughness: float = 0.15  # relative amplitude of outline harmonics 2-5
    nucleus_fraction: float = 0.35  # nucleus size relative to the cell
    nucleus_contrast: float = 0.3  # extra OPD of the nucleus relative to the dome
    texture: float = 0.1  # relative rms of the granular texture
    texture_scale: float = 1.0  # correlation length of the texture, um
    max_aspect: float = 6.4  # measured 95th percentile
    # Size and peak OPD are drawn within their measured 5-95 percent ranges, so the
    # joint lognormal tails cannot combine into giant bright cells, which the data
    # do not contain (large cells are flat, bright cells are small).
    diameter_range: tuple[float, float] = (13.71, 52.23)
    peak_opd_range: tuple[float, float] = (0.042, 0.407)


def sample_cell_shape(
    rng: numpy.random.Generator, stats: AdherentCellStatistics
) -> tuple[float, float, float]:
    """
    Draw the size, elongation and peak optical path difference of one cell.

    Size and peak OPD come jointly from the measured lognormal distributions,
    redrawn until both fall in their measured 5-95 percent ranges.

    Args:
        rng (numpy.random.Generator): Random number generator.
        stats (AdherentCellStatistics): Measured cell statistics.

    Returns:
        tuple[float, float, float]: Equivalent diameter in um, aspect ratio
            (at least 1) and peak OPD in um.
    """
    covariance = [
        [
            stats.diameter_log_sd**2,
            stats.size_opd_correlation * stats.diameter_log_sd * stats.peak_opd_log_sd,
        ],
        [
            stats.size_opd_correlation * stats.diameter_log_sd * stats.peak_opd_log_sd,
            stats.peak_opd_log_sd**2,
        ],
    ]
    mean = [numpy.log(stats.diameter_median), numpy.log(stats.peak_opd_median)]
    for _ in range(1000):
        log_diameter, log_peak = rng.multivariate_normal(mean, covariance)
        diameter, peak = numpy.exp(log_diameter), numpy.exp(log_peak)
        if (
            stats.diameter_range[0] <= diameter <= stats.diameter_range[1]
            and stats.peak_opd_range[0] <= peak <= stats.peak_opd_range[1]
        ):
            break
    aspect = float(
        numpy.clip(
            rng.lognormal(numpy.log(stats.aspect_median), stats.aspect_log_sd),
            1.0,
            stats.max_aspect,
        )
    )
    return float(diameter), aspect, float(peak)


def cell_opd(
    simulator: InlineHologramSimulator,
    centre: tuple[float, float],
    diameter: float,
    aspect: float,
    peak_opd: float,
    rng: numpy.random.Generator,
    stats: AdherentCellStatistics,
) -> numpy.ndarray:
    """
    Compute the optical path difference map of one cell on the full grid.

    The map is scaled so that its 95th percentile inside the cell is peak_opd.
    The cell is computed only inside its bounding window, which keeps dense
    cultures on large grids fast.

    Args:
        simulator (InlineHologramSimulator): Simulator whose grid is used.
        centre (tuple[float, float]): Cell centre (x, y), in um.
        diameter (float): Equivalent diameter, in um.
        aspect (float): Ratio of the long to the short axis.
        peak_opd (float): 95th percentile of the OPD inside the cell, in um.
        rng (numpy.random.Generator): Random number generator for the
            orientation, outline, dome exponent, nucleus position and texture.
        stats (AdherentCellStatistics): Measured statistics and design choices.

    Returns:
        numpy.ndarray: OPD in um, zero outside the cell, shape (resolution,
            resolution).
    """
    semi_major = diameter / 2 * numpy.sqrt(aspect)
    semi_minor = diameter / 2 / numpy.sqrt(aspect)
    angle = rng.uniform(0, numpy.pi)

    # Bounding window: the outline harmonics can stretch the cell beyond its
    # ellipse by up to the sum of their amplitudes (a few times the roughness).
    pixel = simulator.pixel_size
    reach = semi_major * (1 + 6 * stats.outline_roughness) + 2 * pixel
    size = simulator.resolution
    col = int(round(centre[0] / pixel)) + size // 2
    row = int(round(centre[1] / pixel)) + size // 2
    half = int(numpy.ceil(reach / pixel))
    rows = slice(max(row - half, 0), min(row + half + 1, size))
    cols = slice(max(col - half, 0), min(col + half + 1, size))
    full = numpy.zeros((size, size))

    dx, dy = simulator.X[rows, cols] - centre[0], simulator.Y[rows, cols] - centre[1]
    u = dx * numpy.cos(angle) + dy * numpy.sin(angle)
    v = -dx * numpy.sin(angle) + dy * numpy.cos(angle)
    theta = numpy.arctan2(v / semi_minor, u / semi_major)

    # Irregular outline: low harmonics modulate the boundary radius.
    boundary = numpy.ones_like(theta)
    for k in range(2, 6):
        boundary += rng.normal(0, stats.outline_roughness / k) * numpy.cos(
            k * theta + rng.uniform(0, 2 * numpy.pi)
        )
    rho2 = ((u / semi_major) ** 2 + (v / semi_minor) ** 2) / boundary**2
    inside = rho2 < 1
    exponent = stats.profile_exponent * numpy.exp(
        rng.normal(0, stats.profile_exponent_log_sd)
    )
    dome = numpy.where(inside, numpy.clip(1 - rho2, 0, 1) ** exponent, 0.0)

    # Nucleus: a smoother bump near, but not exactly at, the centre.
    offset = rng.uniform(-0.25, 0.25, size=2)
    nu = (u / semi_major - offset[0]) / stats.nucleus_fraction
    nv = (v / semi_minor - offset[1]) / stats.nucleus_fraction
    nucleus = numpy.clip(1 - nu**2 - nv**2, 0, 1) ** 1.5 * inside

    # Granular texture: band-limited noise, only inside the cell.
    noise = gaussian_filter(rng.normal(0, 1, dx.shape), stats.texture_scale / pixel)
    noise /= noise.std() + 1e-12
    profile = (dome + stats.nucleus_contrast * nucleus) * (1 + stats.texture * noise)
    profile = numpy.clip(profile, 0, None) * inside
    values = profile[inside]
    if values.size == 0:
        return full
    full[rows, cols] = profile * (peak_opd / numpy.percentile(values, 95))
    return full


def add_adherent_cells(
    simulator: InlineHologramSimulator,
    coverage: float | None = None,
    stats: AdherentCellStatistics = AdherentCellStatistics(),
    max_overlap: float = 0.1,
    max_attempts: int = 200,
) -> int:
    """
    Fill the field with adherent cells and apply their phase.

    Cells are added until they cover the requested fraction of the field. Each
    lies fully inside the field and overlaps the others by at most max_overlap
    of its area.

    Args:
        simulator (InlineHologramSimulator): Simulator to add the cells to.
        coverage (float | None): Fraction of the field to cover, or None for
            the measured median.
        stats (AdherentCellStatistics): Measured statistics and design choices.
        max_overlap (float): Largest fraction of a cell's area that may overlap
            other cells.
        max_attempts (int): Placement attempts before giving up.

    Returns:
        int: The number of cells placed.
    """
    rng: numpy.random.Generator = utils.RandomNumberGenerator()
    target = stats.coverage_median if coverage is None else coverage
    margin = simulator.pixel_size
    low, high = float(simulator.X.min()) + margin, float(simulator.X.max()) - margin
    opd = numpy.zeros_like(simulator.X)
    occupied = numpy.zeros(simulator.X.shape, dtype=bool)
    placed = 0
    for _ in range(max_attempts):
        if occupied.mean() >= target:
            break
        diameter, aspect, peak = sample_cell_shape(rng, stats)
        reach = diameter / 2 * numpy.sqrt(aspect) * (1 + 2 * stats.outline_roughness)
        if 2 * reach >= high - low:
            continue
        centre = tuple(rng.uniform(low + reach, high - reach, size=2))
        cell = cell_opd(simulator, centre, diameter, aspect, peak, rng, stats)
        mask = cell > 0
        if not mask.any() or (mask & occupied).sum() > max_overlap * mask.sum():
            continue
        opd += cell
        occupied |= mask
        placed += 1
    simulator.apply_transmission(
        numpy.ones_like(opd), 2 * numpy.pi / simulator.wavelength * opd
    )
    return placed
