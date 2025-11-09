"""Tests for hologen package initialization."""

from __future__ import annotations

import pytest

import hologen
from hologen.converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
)
from hologen.shapes import CircleGenerator, RectangleGenerator, RingGenerator
from hologen.types import (
    GridSpec,
    HologramSample,
    HolographyConfig,
    HolographyMethod,
    ObjectSample,
    OffAxisCarrier,
    OpticalConfig,
)
from hologen.utils.io import NumpyDatasetWriter


class TestPackageImports:
    """Test that package imports work correctly."""

    def test_types_available(self) -> None:
        """Test that type definitions are available."""
        assert hasattr(hologen, "GridSpec")
        assert hasattr(hologen, "OpticalConfig")
        assert hasattr(hologen, "OffAxisCarrier")
        assert hasattr(hologen, "HolographyConfig")
        assert hasattr(hologen, "HolographyMethod")
        assert hasattr(hologen, "ObjectSample")
        assert hasattr(hologen, "HologramSample")

    def test_converters_available(self) -> None:
        """Test that converter classes are available."""
        assert hasattr(hologen, "ObjectDomainProducer")
        assert hasattr(hologen, "ObjectToHologramConverter")
        assert hasattr(hologen, "HologramDatasetGenerator")

    def test_shapes_available(self) -> None:
        """Test that shape generators are available."""
        assert hasattr(hologen, "CircleGenerator")
        assert hasattr(hologen, "RectangleGenerator")
        assert hasattr(hologen, "RingGenerator")

    def test_io_available(self) -> None:
        """Test that IO utilities are available."""
        assert hasattr(hologen, "NumpyDatasetWriter")

    def test_imported_classes_are_correct(self) -> None:
        """Test that imported classes are the correct types."""
        assert hologen.GridSpec is GridSpec
        assert hologen.OpticalConfig is OpticalConfig
        assert hologen.OffAxisCarrier is OffAxisCarrier
        assert hologen.HolographyConfig is HolographyConfig
        assert hologen.HolographyMethod is HolographyMethod
        assert hologen.ObjectSample is ObjectSample
        assert hologen.HologramSample is HologramSample
        assert hologen.ObjectDomainProducer is ObjectDomainProducer
        assert hologen.ObjectToHologramConverter is ObjectToHologramConverter
        assert hologen.HologramDatasetGenerator is HologramDatasetGenerator
        assert hologen.CircleGenerator is CircleGenerator
        assert hologen.RectangleGenerator is RectangleGenerator
        assert hologen.RingGenerator is RingGenerator
        assert hologen.NumpyDatasetWriter is NumpyDatasetWriter

    def test_package_has_version(self) -> None:
        """Test that package has version attribute."""
        # This would typically be set by setuptools, but we can check it exists
        # or skip if not available in development
        try:
            version = hologen.__version__
            assert isinstance(version, str)
            assert len(version) > 0
        except AttributeError:
            pytest.skip("Version not available in development mode")

    def test_package_docstring(self) -> None:
        """Test that package has docstring."""
        assert hologen.__doc__ is not None
        assert len(hologen.__doc__.strip()) > 0

    def test_all_attribute(self) -> None:
        """Test that __all__ is properly defined."""
        assert hasattr(hologen, "__all__")
        assert isinstance(hologen.__all__, list | tuple)
        assert len(hologen.__all__) > 0

        # Check that all items in __all__ are actually available
        for item in hologen.__all__:
            assert hasattr(hologen, item), f"Item '{item}' in __all__ but not available"
