"""
Red blood cells as phase and amplitude objects.

Shape: the biconcave thickness profile of Evans and Fung, Improved measurements of
the erythrocyte geometry, Microvasc. Res. 4:335-347 (1972),

    h(r) = sqrt(1 - (r / R0)^2) * (C0 + C2 (r / R0)^2 + C4 (r / R0)^4),

with R0 = 3.91 um, C0 = 0.81 um, C2 = 7.83 um, C4 = -4.39 um at 300 mOsm, which
gives a 7.82 um diameter, a 0.81 um centre and a 2.57 um rim.

Phase: phi = 2 pi / lambda * delta_n * h, where delta_n = alpha * MCHC is the
refractive index difference to the medium (Gienger et al., Sci. Rep. 9:4623, 2019).

Amplitude: haemoglobin absorbs at 532 nm (Q band), so the amplitude transmission
is exp(-mu_a * h / 2) with mu_a = 2.303 * eps * MCHC / 64,500 (Prahl, Tabulated
molar extinction coefficient for hemoglobin in water, omlc.org, 1998).
"""

import dataclasses

import numpy

import hologen.utils as utils
from hologen.inline_simulator import InlineHologramSimulator


@dataclasses.dataclass(frozen=True)
class RedBloodCellOptics:
    """Shape and optical constants of a red blood cell (lengths in microns)."""

    radius: float = 3.91  # R0, Evans and Fung 1972
    c0: float = 0.81
    c2: float = 7.83
    c4: float = -4.39
    delta_n: float = 0.066  # alpha 0.20 mL/g x MCHC 0.33 g/mL
    absorption: float = 0.0517  # mu_a in 1/um at 532 nm, MCHC 330 g/L
    size_cv: float = 0.043  # diameter spread: RDW (volume CV) about 13% / 3


def evans_fung_thickness(
    radial_distance: numpy.ndarray, optics: RedBloodCellOptics, scale: float = 1.0
) -> numpy.ndarray:
    """
    Thickness (um) of a red blood cell at a distance from its centre; zero
    outside the cell. A scale other than 1 enlarges the whole cell uniformly.
    """
    u = (radial_distance / (optics.radius * scale)) ** 2
    inside = u < 1.0
    thickness = numpy.zeros_like(radial_distance, dtype=float)
    ui = u[inside]
    thickness[inside] = (
        scale * numpy.sqrt(1.0 - ui) * (optics.c0 + optics.c2 * ui + optics.c4 * ui**2)
    )
    return thickness


def add_red_blood_cells(
    simulator: InlineHologramSimulator,
    num_cells: int,
    optics: RedBloodCellOptics = RedBloodCellOptics(),
    max_attempts: int = 1000,
) -> int:
    """
    Place red blood cells flat in the object plane, fully inside the field of
    view and without overlap, and apply their transmission.

    Returns:
        int: The number of cells placed (fewer than requested if the field is
            too crowded after max_attempts tries).
    """
    rng: numpy.random.Generator = utils.RandomNumberGenerator()
    # Cell centres stay one pixel inside the outermost pixel centres, so no cell
    # reaches the border (the FFT would wrap it to the opposite side).
    low: float = float(simulator.X.min()) + simulator.pixel_size
    high: float = float(simulator.X.max()) - simulator.pixel_size
    placed: list[tuple[float, float, float]] = []  # (x, y, radius)
    thickness = numpy.zeros_like(simulator.X)

    attempts = 0
    while len(placed) < num_cells and attempts < max_attempts:
        attempts += 1
        scale = max(rng.normal(1.0, optics.size_cv), 0.5)
        radius = optics.radius * scale
        if 2 * radius >= high - low:
            continue
        x, y = rng.uniform(low + radius, high - radius, size=2)
        if any(numpy.hypot(x - px, y - py) < radius + pr for px, py, pr in placed):
            continue
        placed.append((x, y, radius))
        distance = numpy.hypot(simulator.X - x, simulator.Y - y)
        thickness += evans_fung_thickness(distance, optics, scale)

    phase = 2 * numpy.pi / simulator.wavelength * optics.delta_n * thickness
    amplitude = numpy.exp(-optics.absorption * thickness / 2)
    simulator.apply_transmission(amplitude, phase)
    return len(placed)
