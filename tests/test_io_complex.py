"""Tests for complex field I/O in hologen.utils.io module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hologen.types import (
    ComplexHologramSample,
    ComplexObjectSample,
    FieldRepresentation,
    ObjectSample,
)
from hologen.utils.io import ComplexFieldWriter, load_complex_sample


class TestComplexFieldWriterNPZ:
    """Test .npz export for complex fields."""

    def test_save_complex_representation(self, tmp_path: Path) -> None:
        """Test saving complex representation to .npz."""
        writer = ComplexFieldWriter(save_preview=False)
        
        # Create complex sample
        field = np.array([[1 + 1j, 2 + 0j], [0 + 2j, 3 + 3j]], dtype=np.complex128)
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.COMPLEX
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.COMPLEX,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.COMPLEX,
        )
        
        # Save
        writer.save([hologram_sample], tmp_path)
        
        # Load and verify
        npz_path = tmp_path / "sample_00000_test_object.npz"
        assert npz_path.exists()
        
        data = np.load(npz_path)
        assert "real" in data
        assert "imag" in data
        assert "representation" in data
        
        reconstructed = data["real"] + 1j * data["imag"]
        assert np.allclose(reconstructed, field)

    def test_save_amplitude_representation(self, tmp_path: Path) -> None:
        """Test saving amplitude representation to .npz."""
        writer = ComplexFieldWriter(save_preview=False)
        
        field = np.array([[1 + 1j, 2 + 0j]], dtype=np.complex128)
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.AMPLITUDE
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.AMPLITUDE,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.AMPLITUDE,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        npz_path = tmp_path / "sample_00000_test_object.npz"
        data = np.load(npz_path)
        
        assert "amplitude" in data
        assert np.allclose(data["amplitude"], np.abs(field))

    def test_save_phase_representation(self, tmp_path: Path) -> None:
        """Test saving phase representation to .npz."""
        writer = ComplexFieldWriter(save_preview=False)
        
        field = np.array([[1 + 1j, 2 + 0j]], dtype=np.complex128)
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.PHASE
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.PHASE,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.PHASE,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        npz_path = tmp_path / "sample_00000_test_object.npz"
        data = np.load(npz_path)
        
        assert "phase" in data
        assert np.allclose(data["phase"], np.angle(field))

    def test_save_intensity_representation(self, tmp_path: Path) -> None:
        """Test saving intensity representation to .npz."""
        writer = ComplexFieldWriter(save_preview=False)
        
        field = np.array([[1 + 1j, 2 + 0j]], dtype=np.complex128)
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.INTENSITY
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.INTENSITY,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.INTENSITY,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        npz_path = tmp_path / "sample_00000_test_object.npz"
        data = np.load(npz_path)
        
        assert "intensity" in data
        assert np.allclose(data["intensity"], np.abs(field) ** 2)


class TestComplexFieldWriterPNG:
    """Test PNG export for complex fields."""

    def test_png_complex_creates_two_files(self, tmp_path: Path) -> None:
        """Test that complex representation creates amplitude and phase PNGs."""
        writer = ComplexFieldWriter(save_preview=True, phase_colormap=None)
        
        field = np.ones((32, 32), dtype=np.complex128) * (1 + 1j)
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.COMPLEX
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.COMPLEX,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.COMPLEX,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        # Check that both amplitude and phase PNGs exist
        amplitude_path = tmp_path / "sample_00000_test_object_amplitude.png"
        phase_path = tmp_path / "sample_00000_test_object_phase.png"
        
        assert amplitude_path.exists()
        assert phase_path.exists()

    def test_png_amplitude_single_file(self, tmp_path: Path) -> None:
        """Test that amplitude representation creates single PNG."""
        writer = ComplexFieldWriter(save_preview=True)
        
        field = np.ones((32, 32), dtype=np.complex128) * 2.0
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.AMPLITUDE
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.AMPLITUDE,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.AMPLITUDE,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        png_path = tmp_path / "sample_00000_test_object.png"
        assert png_path.exists()

    def test_png_phase_single_file(self, tmp_path: Path) -> None:
        """Test that phase representation creates single PNG."""
        writer = ComplexFieldWriter(save_preview=True, phase_colormap=None)
        
        field = np.exp(1j * np.pi / 4 * np.ones((32, 32)))
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.PHASE
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.PHASE,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.PHASE,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        png_path = tmp_path / "sample_00000_test_object.png"
        assert png_path.exists()

    def test_no_preview_no_png(self, tmp_path: Path) -> None:
        """Test that save_preview=False doesn't create PNGs."""
        writer = ComplexFieldWriter(save_preview=False)
        
        field = np.ones((32, 32), dtype=np.complex128)
        object_sample = ComplexObjectSample(
            name="test", field=field, representation=FieldRepresentation.COMPLEX
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=field,
            hologram_representation=FieldRepresentation.COMPLEX,
            reconstruction_field=field,
            reconstruction_representation=FieldRepresentation.COMPLEX,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        # Check that no PNG files exist
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) == 0


class TestLoadComplexSample:
    """Test load_complex_sample function."""

    def test_load_complex_format(self, tmp_path: Path) -> None:
        """Test loading complex format with real/imag arrays."""
        # Create complex .npz file
        field = np.array([[1 + 1j, 2 + 0j]], dtype=np.complex128)
        npz_path = tmp_path / "complex_sample.npz"
        np.savez(
            npz_path,
            real=field.real,
            imag=field.imag,
            representation="complex",
        )
        
        # Load
        sample = load_complex_sample(npz_path)
        
        assert isinstance(sample, ComplexObjectSample)
        assert np.allclose(sample.field, field)
        assert sample.representation == FieldRepresentation.COMPLEX

    def test_load_amplitude_format(self, tmp_path: Path) -> None:
        """Test loading amplitude format."""
        amplitude = np.array([[1.0, 2.0]], dtype=np.float64)
        npz_path = tmp_path / "amplitude_sample.npz"
        np.savez(npz_path, amplitude=amplitude, representation="amplitude")
        
        sample = load_complex_sample(npz_path)
        
        assert isinstance(sample, ComplexObjectSample)
        assert np.allclose(np.abs(sample.field), amplitude)
        assert sample.representation == FieldRepresentation.AMPLITUDE

    def test_load_phase_format(self, tmp_path: Path) -> None:
        """Test loading phase format."""
        phase = np.array([[0.0, np.pi / 2]], dtype=np.float64)
        npz_path = tmp_path / "phase_sample.npz"
        np.savez(npz_path, phase=phase, representation="phase")
        
        sample = load_complex_sample(npz_path)
        
        assert isinstance(sample, ComplexObjectSample)
        assert np.allclose(np.angle(sample.field), phase)
        assert sample.representation == FieldRepresentation.PHASE

    def test_load_intensity_format(self, tmp_path: Path) -> None:
        """Test loading intensity format."""
        intensity = np.array([[1.0, 4.0]], dtype=np.float64)
        npz_path = tmp_path / "intensity_sample.npz"
        np.savez(npz_path, intensity=intensity, representation="intensity")
        
        sample = load_complex_sample(npz_path)
        
        assert isinstance(sample, ComplexObjectSample)
        assert np.allclose(np.abs(sample.field) ** 2, intensity)
        assert sample.representation == FieldRepresentation.INTENSITY

    def test_load_legacy_format(self, tmp_path: Path) -> None:
        """Test loading legacy format with 'object' key."""
        pixels = np.array([[1.0, 0.5]], dtype=np.float64)
        npz_path = tmp_path / "legacy_sample.npz"
        np.savez(npz_path, object=pixels)
        
        sample = load_complex_sample(npz_path)
        
        assert isinstance(sample, ObjectSample)
        assert np.allclose(sample.pixels, pixels)

    def test_load_unknown_format_error(self, tmp_path: Path) -> None:
        """Test error for unknown format."""
        npz_path = tmp_path / "unknown_sample.npz"
        np.savez(npz_path, unknown_key=np.array([[1.0]]))
        
        with pytest.raises(ValueError, match="Unknown .npz format"):
            load_complex_sample(npz_path)


class TestRoundTripExportImport:
    """Test round-trip export and import."""

    def test_complex_round_trip(self, tmp_path: Path) -> None:
        """Test export → import → verify for complex representation."""
        writer = ComplexFieldWriter(save_preview=False)
        
        # Create original field
        original_field = np.array(
            [[1 + 1j, 2 + 0j], [0 + 2j, 3 + 3j]], dtype=np.complex128
        )
        object_sample = ComplexObjectSample(
            name="test", field=original_field, representation=FieldRepresentation.COMPLEX
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=original_field,
            hologram_representation=FieldRepresentation.COMPLEX,
            reconstruction_field=original_field,
            reconstruction_representation=FieldRepresentation.COMPLEX,
        )
        
        # Export
        writer.save([hologram_sample], tmp_path)
        
        # Import
        npz_path = tmp_path / "sample_00000_test_object.npz"
        loaded_sample = load_complex_sample(npz_path)
        
        # Verify
        assert isinstance(loaded_sample, ComplexObjectSample)
        assert np.allclose(loaded_sample.field, original_field)
        assert loaded_sample.representation == FieldRepresentation.COMPLEX

    def test_amplitude_round_trip(self, tmp_path: Path) -> None:
        """Test export → import → verify for amplitude representation."""
        writer = ComplexFieldWriter(save_preview=False)
        
        original_field = np.array([[1.0, 2.0]], dtype=np.complex128)
        object_sample = ComplexObjectSample(
            name="test", field=original_field, representation=FieldRepresentation.AMPLITUDE
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=original_field,
            hologram_representation=FieldRepresentation.AMPLITUDE,
            reconstruction_field=original_field,
            reconstruction_representation=FieldRepresentation.AMPLITUDE,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        npz_path = tmp_path / "sample_00000_test_object.npz"
        loaded_sample = load_complex_sample(npz_path)
        
        # Amplitude should match
        assert np.allclose(np.abs(loaded_sample.field), np.abs(original_field))

    def test_phase_round_trip(self, tmp_path: Path) -> None:
        """Test export → import → verify for phase representation."""
        writer = ComplexFieldWriter(save_preview=False)
        
        # Create field with specific phase
        phase = np.array([[0.0, np.pi / 2, -np.pi / 2]], dtype=np.float64)
        original_field = np.exp(1j * phase)
        
        object_sample = ComplexObjectSample(
            name="test", field=original_field, representation=FieldRepresentation.PHASE
        )
        hologram_sample = ComplexHologramSample(
            object_sample=object_sample,
            hologram_field=original_field,
            hologram_representation=FieldRepresentation.PHASE,
            reconstruction_field=original_field,
            reconstruction_representation=FieldRepresentation.PHASE,
        )
        
        writer.save([hologram_sample], tmp_path)
        
        npz_path = tmp_path / "sample_00000_test_object.npz"
        loaded_sample = load_complex_sample(npz_path)
        
        # Phase should match
        assert np.allclose(np.angle(loaded_sample.field), phase)

    def test_multiple_samples_round_trip(self, tmp_path: Path) -> None:
        """Test round-trip with multiple samples."""
        writer = ComplexFieldWriter(save_preview=False)
        
        # Create multiple samples
        samples = []
        for i in range(3):
            field = np.ones((16, 16), dtype=np.complex128) * (i + 1)
            object_sample = ComplexObjectSample(
                name=f"test_{i}",
                field=field,
                representation=FieldRepresentation.COMPLEX,
            )
            hologram_sample = ComplexHologramSample(
                object_sample=object_sample,
                hologram_field=field,
                hologram_representation=FieldRepresentation.COMPLEX,
                reconstruction_field=field,
                reconstruction_representation=FieldRepresentation.COMPLEX,
            )
            samples.append(hologram_sample)
        
        # Export
        writer.save(samples, tmp_path)
        
        # Import and verify each
        for i in range(3):
            npz_path = tmp_path / f"sample_{i:05d}_test_{i}_object.npz"
            loaded_sample = load_complex_sample(npz_path)
            
            expected_field = np.ones((16, 16), dtype=np.complex128) * (i + 1)
            assert np.allclose(loaded_sample.field, expected_field)
