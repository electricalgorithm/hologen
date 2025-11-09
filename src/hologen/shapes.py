from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from hologen.types import ArrayFloat, GridSpec, ObjectShapeGenerator


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
        radius = rng.uniform(self.min_radius, self.max_radius) * min(
            grid.height, grid.width
        )
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
        outer_radius = rng.uniform(self.min_radius, self.max_radius) * min(
            grid.height, grid.width
        )
        thickness = rng.uniform(self.min_thickness, self.max_thickness) * outer_radius
        inner_radius = max(outer_radius - thickness, 2.0)
        center_y = rng.uniform(0.4, 0.6) * grid.height
        center_x = rng.uniform(0.4, 0.6) * grid.width
        yy, xx = np.ogrid[: grid.height, : grid.width]
        radial_distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        mask = (radial_distance <= outer_radius) & (radial_distance >= inner_radius)
        canvas[mask] = 1.0
        return self._clamp(canvas)


class CircleCheckerGenerator(BaseShapeGenerator):
    """Generator producing filled discs with a checkerboard pattern inside."""

    __slots__ = ("min_radius", "max_radius", "checker_size")

    def __init__(
        self, name: str, min_radius: float, max_radius: float, checker_size: int = 8
    ) -> None:
        super().__init__(name=name)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.checker_size = max(int(checker_size), 1)

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        canvas = self._empty_canvas(grid)
        radius = rng.uniform(self.min_radius, self.max_radius) * min(
            grid.height, grid.width
        )
        center_y = rng.uniform(0.3, 0.7) * grid.height
        center_x = rng.uniform(0.3, 0.7) * grid.width
        yy, xx = np.ogrid[: grid.height, : grid.width]
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2

        # bounding box for checker tiling to keep pattern stable
        ys, xs = np.where(mask)
        if ys.size == 0:
            return self._clamp(canvas)
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        yy_bb, xx_bb = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]
        checker = ((yy_bb // self.checker_size) + (xx_bb // self.checker_size)) % 2
        # embed checker only where mask is True
        submask = mask[min_y : max_y + 1, min_x : max_x + 1]
        canvas[min_y : max_y + 1, min_x : max_x + 1][submask] = checker[submask].astype(
            np.float64
        )
        # ensure binary values: set everything else 0.0
        canvas[~mask] = 0.0
        return self._clamp(canvas)


class EllipseCheckerGenerator(BaseShapeGenerator):
    """Generator producing filled ellipses with a checkerboard pattern inside."""

    __slots__ = (
        "min_radius_y",
        "max_radius_y",
        "min_radius_x",
        "max_radius_x",
        "checker_size",
    )

    def __init__(
        self,
        name: str,
        min_radius_y: float,
        max_radius_y: float,
        min_radius_x: float,
        max_radius_x: float,
        checker_size: int = 8,
    ) -> None:
        super().__init__(name=name)
        self.min_radius_y = min_radius_y
        self.max_radius_y = max_radius_y
        self.min_radius_x = min_radius_x
        self.max_radius_x = max_radius_x
        self.checker_size = max(int(checker_size), 1)

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        canvas = self._empty_canvas(grid)
        ry = rng.uniform(self.min_radius_y, self.max_radius_y) * grid.height
        rx = rng.uniform(self.min_radius_x, self.max_radius_x) * grid.width
        center_y = rng.uniform(0.3, 0.7) * grid.height
        center_x = rng.uniform(0.3, 0.7) * grid.width
        yy, xx = np.ogrid[: grid.height, : grid.width]
        norm = ((yy - center_y) / max(ry, 1e-8)) ** 2 + (
            (xx - center_x) / max(rx, 1e-8)
        ) ** 2
        mask = norm <= 1.0

        ys, xs = np.where(mask)
        if ys.size == 0:
            return self._clamp(canvas)
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        yy_bb, xx_bb = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]
        checker = ((yy_bb // self.checker_size) + (xx_bb // self.checker_size)) % 2
        submask = mask[min_y : max_y + 1, min_x : max_x + 1]
        # apply checkerboard inside the ellipse
        area = np.zeros_like(checker, dtype=np.float64)
        area[submask] = checker[submask].astype(np.float64)
        canvas[min_y : max_y + 1, min_x : max_x + 1] = area
        canvas[~mask] = 0.0
        return self._clamp(canvas)


class TriangleCheckerGenerator(BaseShapeGenerator):
    """Generator producing filled triangles with a checkerboard pattern inside."""

    __slots__ = ("min_scale", "max_scale", "checker_size")

    def __init__(
        self, name: str, min_scale: float, max_scale: float, checker_size: int = 8
    ) -> None:
        super().__init__(name=name)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.checker_size = max(int(checker_size), 1)

    def _polygon_mask(self, grid_h: int, grid_w: int, verts: np.ndarray) -> np.ndarray:
        """Vectorized winding test for a triangle (convex polygon)."""
        yy, xx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        # Using barycentric / edge-sign test for triangle:
        v0 = verts[1] - verts[0]
        v1 = verts[2] - verts[1]
        v2 = verts[0] - verts[2]
        p0 = np.stack((xx - verts[0, 1], yy - verts[0, 0]), axis=-1)
        p1 = np.stack((xx - verts[1, 1], yy - verts[1, 0]), axis=-1)
        p2 = np.stack((xx - verts[2, 1], yy - verts[2, 0]), axis=-1)

        # 2D cross product z-component for each edge
        cross0 = v0[0] * p0[..., 1] - v0[1] * p0[..., 0]
        cross1 = v1[0] * p1[..., 1] - v1[1] * p1[..., 0]
        cross2 = v2[0] * p2[..., 1] - v2[1] * p2[..., 0]

        # For consistent winding, all crosses should have the same sign (>=0 or <=0)
        mask_pos = (cross0 >= 0) & (cross1 >= 0) & (cross2 >= 0)
        mask_neg = (cross0 <= 0) & (cross1 <= 0) & (cross2 <= 0)
        return mask_pos | mask_neg

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        canvas = self._empty_canvas(grid)
        scale = (
            rng.uniform(self.min_scale, self.max_scale)
            * min(grid.height, grid.width)
            * 0.5
        )
        center_y = rng.uniform(0.35, 0.65) * grid.height
        center_x = rng.uniform(0.35, 0.65) * grid.width
        base_angle = rng.uniform(0.0, 2.0 * np.pi)
        jitter = rng.uniform(-0.2, 0.2, size=3)
        angles = (
            base_angle + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]) + jitter
        )
        radii = rng.uniform(0.8, 1.0, size=3) * scale

        verts = np.zeros((3, 2), dtype=np.float64)  # (y, x) pairs
        verts[:, 0] = center_y + radii * np.sin(angles)
        verts[:, 1] = center_x + radii * np.cos(angles)

        # Clip vertices into image bounds
        verts[:, 0] = np.clip(verts[:, 0], 0, grid.height - 1)
        verts[:, 1] = np.clip(verts[:, 1], 0, grid.width - 1)

        mask = self._polygon_mask(grid.height, grid.width, verts)

        ys, xs = np.where(mask)
        if ys.size == 0:
            return self._clamp(canvas)
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        yy_bb, xx_bb = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]
        checker = ((yy_bb // self.checker_size) + (xx_bb // self.checker_size)) % 2
        submask = mask[min_y : max_y + 1, min_x : max_x + 1]
        board = np.zeros_like(checker, dtype=np.float64)
        board[submask] = checker[submask].astype(np.float64)
        canvas[min_y : max_y + 1, min_x : max_x + 1] = board
        canvas[~mask] = 0.0
        return self._clamp(canvas)


class RectangleCheckerGenerator(BaseShapeGenerator):
    """Generator producing filled rectangles with a checkerboard pattern inside."""

    __slots__ = ("min_scale", "max_scale", "checker_size")

    def __init__(
        self, name: str, min_scale: float, max_scale: float, checker_size: int = 8
    ) -> None:
        super().__init__(name=name)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.checker_size = max(int(checker_size), 1)

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

        if min_y >= max_y or min_x >= max_x:
            return self._clamp(canvas)

        yy_bb, xx_bb = np.ogrid[min_y:max_y, min_x:max_x]
        checker = ((yy_bb // self.checker_size) + (xx_bb // self.checker_size)) % 2
        canvas[min_y:max_y, min_x:max_x] = checker.astype(np.float64)
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
        CircleCheckerGenerator(
            name="circle_checker", min_radius=0.1, max_radius=0.2, checker_size=16
        ),
        RectangleCheckerGenerator(
            name="rectangle_checker", min_scale=0.1, max_scale=0.35, checker_size=16
        ),
        EllipseCheckerGenerator(
            name="ellipse_checker",
            min_radius_y=0.1,
            max_radius_y=0.35,
            min_radius_x=0.1,
            max_radius_x=0.35,
            checker_size=16,
        ),
    )
