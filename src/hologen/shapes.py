from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.random import Generator

from .types import ArrayFloat, GridSpec, ObjectShapeGenerator


@dataclass(slots=True)
class BaseShapeGenerator(ObjectShapeGenerator):
    """Abstract base for object-domain shape generators.

    Args:
        name: Canonical name used when recording generated samples.
    """

    name: str

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        """Create a binary object-domain image.

        Args:
            grid: Grid specification describing the desired output resolution.
            rng: Random number generator providing stochastic parameters.

        Returns:
            Binary amplitude image with values in ``{0.0, 1.0}``.

        Raises:
            NotImplementedError: If the subclass does not override the method.
        """

        raise NotImplementedError

    def _empty_canvas(self, grid: GridSpec) -> ArrayFloat:
        """Allocate a zero-initialized canvas matching the grid.

        Args:
            grid: Grid specification describing the desired output resolution.

        Returns:
            Two-dimensional floating-point array filled with zeros.
        """

        return np.zeros((grid.height, grid.width), dtype=np.float64)

    def _clamp(self, canvas: ArrayFloat) -> ArrayFloat:
        """Clamp canvas values to the range ``[0.0, 1.0]``.

        Args:
            canvas: Canvas to clamp in-place.

        Returns:
            The provided canvas with values constrained to ``[0.0, 1.0]``.
        """

        np.clip(canvas, 0.0, 1.0, out=canvas)
        return canvas


class CircleGenerator(BaseShapeGenerator):
    """Generator producing filled discs."""

    __slots__ = ("min_radius", "max_radius")

    def __init__(self, name: str, min_radius: float, max_radius: float) -> None:
        super().__init__(name=name)
        self.min_radius = min_radius
        self.max_radius = max_radius

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        canvas = self._empty_canvas(grid)
        radius = rng.uniform(self.min_radius, self.max_radius) * min(grid.height, grid.width)
        center_y = rng.uniform(0.3, 0.7) * grid.height
        center_x = rng.uniform(0.3, 0.7) * grid.width
        yy, xx = np.ogrid[: grid.height, : grid.width]
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        canvas[mask] = 1.0
        return self._clamp(canvas)


class RectangleGenerator(BaseShapeGenerator):
    """Generator producing filled rectangles."""

    __slots__ = ("min_scale", "max_scale")

    def __init__(self, name: str, min_scale: float, max_scale: float) -> None:
        super().__init__(name=name)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        canvas = self._empty_canvas(grid)
        half_height = rng.uniform(self.min_scale, self.max_scale) * grid.height * 0.5
        half_width = rng.uniform(self.min_scale, self.max_scale) * grid.width * 0.5
        center_y = rng.uniform(0.4, 0.6) * grid.height
        center_x = rng.uniform(0.4, 0.6) * grid.width
        min_y = int(max(center_y - half_height, 0))
        max_y = int(min(center_y + half_height, grid.height))
        min_x = int(max(center_x - half_width, 0))
        max_x = int(min(center_x + half_width, grid.width))
        canvas[min_y:max_y, min_x:max_x] = 1.0
        return self._clamp(canvas)


class RingGenerator(BaseShapeGenerator):
    """Generator producing annular rings."""

    __slots__ = ("min_radius", "max_radius", "min_thickness", "max_thickness")

    def __init__(
        self,
        name: str,
        min_radius: float,
        max_radius: float,
        min_thickness: float,
        max_thickness: float,
    ) -> None:
        super().__init__(name=name)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        canvas = self._empty_canvas(grid)
        outer_radius = rng.uniform(self.min_radius, self.max_radius) * min(grid.height, grid.width)
        thickness = rng.uniform(self.min_thickness, self.max_thickness) * outer_radius
        inner_radius = max(outer_radius - thickness, 2.0)
        center_y = rng.uniform(0.4, 0.6) * grid.height
        center_x = rng.uniform(0.4, 0.6) * grid.width
        yy, xx = np.ogrid[: grid.height, : grid.width]
        radial_distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        mask = (radial_distance <= outer_radius) & (radial_distance >= inner_radius)
        canvas[mask] = 1.0
        return self._clamp(canvas)


def available_generators() -> Iterable[ObjectShapeGenerator]:
    """Return the default suite of shape generators."""

    return (
        CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18),
        RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.35),
        RingGenerator(
            name="ring",
            min_radius=0.12,
            max_radius=0.25,
            min_thickness=0.1,
            max_thickness=0.3,
        ),
    )
