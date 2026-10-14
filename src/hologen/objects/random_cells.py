"""Random elliptical phase objects, the first-version "cells" kept as a baseline."""

from numpy import cos as npCos
from numpy import sin as npSin
from numpy import pi as npPi
from numpy.random import Generator as npGenerator
from hologen.utils import RandomNumberGenerator
from hologen.inline_simulator import InlineHologramSimulator


def add_random_cells(
    hologram: InlineHologramSimulator,
    num_cells: int = 10,
    radius_range: tuple[int, int] = (5, 15),
    phase_max: float = 0.5,
) -> None:
    """
    Add random ellipses with a uniform phase and amplitude to the field.

    Centres are drawn anywhere in the field, so ellipses may overlap or reach the
    border.

    Args:
        hologram (InlineHologramSimulator): Simulator to add the ellipses to.
        num_cells (int): Number of ellipses.
        radius_range (tuple[int, int]): Range of the semi-axes, in pixels.
        phase_max (float): Largest phase shift, in radians; the smallest is 0.1.
    """
    rng: npGenerator = RandomNumberGenerator()

    for _ in range(num_cells):
        cx = rng.uniform(hologram.X.min(), hologram.X.max())
        cy = rng.uniform(hologram.Y.min(), hologram.Y.max())
        rx = rng.uniform(*radius_range) * hologram.pixel_size
        ry = rng.uniform(*radius_range) * hologram.pixel_size
        rotation = rng.uniform(0, npPi)

        xx = (hologram.X - cx) * npCos(rotation) + (hologram.Y - cy) * npSin(rotation)
        yy = (hologram.X - cx) * npSin(rotation) - (hologram.Y - cy) * npCos(rotation)

        mask = (xx**2 / rx**2) + (yy**2 / ry**2) <= 1.0
        phi = rng.uniform(0.1, phase_max)
        amp = rng.uniform(0.90, 1.0)

        hologram.add_object(mask, amplitude=amp, phase_shift=phi)
