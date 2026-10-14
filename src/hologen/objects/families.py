"""
Object families describe what a hologram shows and how it is exposed.

Each family pairs a placement function with its own background exposure. An
operator sets the exposure separately for each type of sample, because red blood
cells and adherent cells focus light differently (see hologen.dataset.exposure).
"""

import dataclasses
from collections.abc import Callable

from hologen.inline_simulator import InlineHologramSimulator


@dataclasses.dataclass(frozen=True)
class ObjectFamily:
    """A kind of sample and how it is recorded."""

    name: str
    place: Callable[[InlineHologramSimulator], int]  # returns objects placed
    background_electrons: float  # e- per pixel for the unit-intensity background
