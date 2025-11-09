"""Test configuration and fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from hologen.types import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    OffAxisCarrier,
    OpticalConfig,
)


@pytest.fixture
def rng() -> Generator:
    """Provide a seeded random number generator."""
    return default_rng(seed=42)


@pytest.fixture
def grid_spec() -> GridSpec:
    """Provide a standard grid specification."""
    return GridSpec(height=64, width=64, pixel_pitch=5e-6)


@pytest.fixture
def optical_config() -> OpticalConfig:
    """Provide standard optical parameters."""
    return OpticalConfig(wavelength=532e-9, propagation_distance=0.01)


@pytest.fixture
def off_axis_carrier() -> OffAxisCarrier:
    """Provide off-axis carrier configuration."""
    return OffAxisCarrier(frequency_x=1000.0, frequency_y=1000.0, gaussian_width=300.0)


@pytest.fixture
def inline_config(grid_spec: GridSpec, optical_config: OpticalConfig) -> HolographyConfig:
    """Provide inline holography configuration."""
    return HolographyConfig(
        grid=grid_spec,
        optics=optical_config,
        method=HolographyMethod.INLINE,
        carrier=None,
    )


@pytest.fixture
def off_axis_config(
    grid_spec: GridSpec, optical_config: OpticalConfig, off_axis_carrier: OffAxisCarrier
) -> HolographyConfig:
    """Provide off-axis holography configuration."""
    return HolographyConfig(
        grid=grid_spec,
        optics=optical_config,
        method=HolographyMethod.OFF_AXIS,
        carrier=off_axis_carrier,
    )


@pytest.fixture
def sample_object_field(grid_spec: GridSpec) -> np.ndarray:
    """Provide a sample object field for testing."""
    field = np.zeros((grid_spec.height, grid_spec.width), dtype=np.float64)
    center_y, center_x = grid_spec.height // 2, grid_spec.width // 2
    radius = min(grid_spec.height, grid_spec.width) // 8
    y, x = np.ogrid[:grid_spec.height, :grid_spec.width]
    mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
    field[mask] = 1.0
    return field