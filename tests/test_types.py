"""Tests for hologen.types module."""

from __future__ import annotations

import numpy as np
import pytest

from hologen.types import (
    GridSpec,
    HologramSample,
    HolographyConfig,
    HolographyMethod,
    ObjectSample,
    OffAxisCarrier,
    OpticalConfig,
)


class TestGridSpec:
    """Test GridSpec dataclass."""

    def test_creation(self) -> None:
        """Test GridSpec creation with valid parameters."""
        grid = GridSpec(height=256, width=512, pixel_pitch=4.65e-6)
        assert grid.height == 256
        assert grid.width == 512
        assert grid.pixel_pitch == pytest.approx(4.65e-6)

    def test_slots(self) -> None:
        """Test that GridSpec uses slots."""
        grid = GridSpec(height=64, width=64, pixel_pitch=1e-6)
        assert hasattr(grid, "__slots__")


class TestOpticalConfig:
    """Test OpticalConfig dataclass."""

    def test_creation(self) -> None:
        """Test OpticalConfig creation with valid parameters."""
        optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.02)
        assert optics.wavelength == pytest.approx(532e-9)
        assert optics.propagation_distance == pytest.approx(0.02)

    def test_slots(self) -> None:
        """Test that OpticalConfig uses slots."""
        optics = OpticalConfig(wavelength=632e-9, propagation_distance=0.01)
        assert hasattr(optics, "__slots__")


class TestOffAxisCarrier:
    """Test OffAxisCarrier dataclass."""

    def test_creation(self) -> None:
        """Test OffAxisCarrier creation with valid parameters."""
        carrier = OffAxisCarrier(
            frequency_x=1500.0, frequency_y=1200.0, gaussian_width=400.0
        )
        assert carrier.frequency_x == pytest.approx(1500.0)
        assert carrier.frequency_y == pytest.approx(1200.0)
        assert carrier.gaussian_width == pytest.approx(400.0)

    def test_slots(self) -> None:
        """Test that OffAxisCarrier uses slots."""
        carrier = OffAxisCarrier(
            frequency_x=1000.0, frequency_y=1000.0, gaussian_width=300.0
        )
        assert hasattr(carrier, "__slots__")


class TestHolographyConfig:
    """Test HolographyConfig dataclass."""

    def test_inline_config(
        self, grid_spec: GridSpec, optical_config: OpticalConfig
    ) -> None:
        """Test inline holography configuration."""
        config = HolographyConfig(
            grid=grid_spec,
            optics=optical_config,
            method=HolographyMethod.INLINE,
            carrier=None,
        )
        assert config.grid is grid_spec
        assert config.optics is optical_config
        assert config.method == HolographyMethod.INLINE
        assert config.carrier is None

    def test_off_axis_config(
        self,
        grid_spec: GridSpec,
        optical_config: OpticalConfig,
        off_axis_carrier: OffAxisCarrier,
    ) -> None:
        """Test off-axis holography configuration."""
        config = HolographyConfig(
            grid=grid_spec,
            optics=optical_config,
            method=HolographyMethod.OFF_AXIS,
            carrier=off_axis_carrier,
        )
        assert config.grid is grid_spec
        assert config.optics is optical_config
        assert config.method == HolographyMethod.OFF_AXIS
        assert config.carrier is off_axis_carrier

    def test_slots(self, grid_spec: GridSpec, optical_config: OpticalConfig) -> None:
        """Test that HolographyConfig uses slots."""
        config = HolographyConfig(
            grid=grid_spec,
            optics=optical_config,
            method=HolographyMethod.INLINE,
        )
        assert hasattr(config, "__slots__")


class TestObjectSample:
    """Test ObjectSample dataclass."""

    def test_creation(self) -> None:
        """Test ObjectSample creation."""
        pixels = np.ones((32, 32), dtype=np.float64)
        sample = ObjectSample(name="test_shape", pixels=pixels)
        assert sample.name == "test_shape"
        assert sample.pixels.shape == (32, 32)
        np.testing.assert_array_equal(sample.pixels, pixels)

    def test_slots(self) -> None:
        """Test that ObjectSample uses slots."""
        pixels = np.zeros((16, 16), dtype=np.float64)
        sample = ObjectSample(name="test", pixels=pixels)
        assert hasattr(sample, "__slots__")


class TestHologramSample:
    """Test HologramSample dataclass."""

    def test_creation(self) -> None:
        """Test HologramSample creation."""
        pixels = np.ones((32, 32), dtype=np.float64)
        object_sample = ObjectSample(name="test_shape", pixels=pixels)
        hologram = np.random.rand(32, 32).astype(np.float64)
        reconstruction = np.random.rand(32, 32).astype(np.float64)

        sample = HologramSample(
            object_sample=object_sample,
            hologram=hologram,
            reconstruction=reconstruction,
        )
        assert hasattr(sample, "__slots__")


class TestHolographyMethod:
    """Test HolographyMethod enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert HolographyMethod.INLINE == "inline"
        assert HolographyMethod.OFF_AXIS == "off_axis"

    def test_string_conversion(self) -> None:
        """Test string conversion."""
        assert str(HolographyMethod.INLINE) == "HolographyMethod.INLINE"
        assert str(HolographyMethod.OFF_AXIS) == "HolographyMethod.OFF_AXIS"
        # Test value access
        assert HolographyMethod.INLINE.value == "inline"
        assert HolographyMethod.OFF_AXIS.value == "off_axis"
