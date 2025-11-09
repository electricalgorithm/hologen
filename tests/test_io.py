"""Tests for hologen.utils.io module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from hologen.types import HologramSample, ObjectSample
from hologen.utils.io import NumpyDatasetWriter


class TestNumpyDatasetWriter:
    """Test NumpyDatasetWriter class."""

    def test_creation_default(self) -> None:
        """Test NumpyDatasetWriter creation with defaults."""
        writer = NumpyDatasetWriter()
        assert writer.save_preview is True

    def test_creation_no_preview(self) -> None:
        """Test NumpyDatasetWriter creation without preview."""
        writer = NumpyDatasetWriter(save_preview=False)
        assert writer.save_preview is False

    def test_slots(self) -> None:
        """Test that NumpyDatasetWriter uses slots."""
        writer = NumpyDatasetWriter()
        assert hasattr(writer, "__slots__")

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save creates output directory."""
        writer = NumpyDatasetWriter(save_preview=False)
        output_dir = tmp_path / "test_output"

        # Create sample data
        sample = HologramSample(
            object_sample=ObjectSample(
                name="test_shape",
                pixels=np.ones((16, 16), dtype=np.float64),
            ),
            hologram=np.random.rand(16, 16).astype(np.float64),
            reconstruction=np.random.rand(16, 16).astype(np.float64),
        )

        writer.save([sample], output_dir)
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_save_npz_files(self, tmp_path: Path) -> None:
        """Test that NPZ files are created correctly."""
        writer = NumpyDatasetWriter(save_preview=False)

        # Create sample data
        object_pixels = np.ones((8, 8), dtype=np.float64)
        hologram_data = np.random.rand(8, 8).astype(np.float64)
        reconstruction_data = np.random.rand(8, 8).astype(np.float64)

        sample = HologramSample(
            object_sample=ObjectSample(name="circle", pixels=object_pixels),
            hologram=hologram_data,
            reconstruction=reconstruction_data,
        )

        writer.save([sample], tmp_path)

        # Check NPZ file exists
        npz_file = tmp_path / "sample_00000_circle.npz"
        assert npz_file.exists()

        # Load and verify contents
        data = np.load(npz_file)
        np.testing.assert_array_equal(data["object"], object_pixels)
        np.testing.assert_array_equal(data["hologram"], hologram_data)
        np.testing.assert_array_equal(data["reconstruction"], reconstruction_data)

    def test_save_multiple_samples(self, tmp_path: Path) -> None:
        """Test saving multiple samples."""
        writer = NumpyDatasetWriter(save_preview=False)

        samples = []
        for i in range(3):
            sample = HologramSample(
                object_sample=ObjectSample(
                    name=f"shape_{i}",
                    pixels=np.full((4, 4), i, dtype=np.float64),
                ),
                hologram=np.full((4, 4), i * 2, dtype=np.float64),
                reconstruction=np.full((4, 4), i * 3, dtype=np.float64),
            )
            samples.append(sample)

        writer.save(samples, tmp_path)

        # Check all files exist
        for i in range(3):
            npz_file = tmp_path / f"sample_0000{i}_shape_{i}.npz"
            assert npz_file.exists()

    @patch("hologen.utils.io.Image")
    def test_save_with_preview(self, mock_image_class, tmp_path: Path) -> None:
        """Test saving with PNG previews."""
        mock_image = Mock()
        mock_image_class.fromarray.return_value = mock_image

        writer = NumpyDatasetWriter(save_preview=True)

        sample = HologramSample(
            object_sample=ObjectSample(
                name="test_shape",
                pixels=np.random.rand(8, 8).astype(np.float64),
            ),
            hologram=np.random.rand(8, 8).astype(np.float64),
            reconstruction=np.random.rand(8, 8).astype(np.float64),
        )

        writer.save([sample], tmp_path)

        # Verify PIL Image was called for each domain
        assert mock_image_class.fromarray.call_count == 3
        assert mock_image.save.call_count == 3

    def test_save_preview_filenames(self, tmp_path: Path) -> None:
        """Test that preview filenames are correct."""
        with patch("hologen.utils.io.Image") as mock_image_class:
            mock_image = Mock()
            mock_image_class.fromarray.return_value = mock_image

            writer = NumpyDatasetWriter(save_preview=True)

            sample = HologramSample(
                object_sample=ObjectSample(
                    name="circle",
                    pixels=np.random.rand(4, 4).astype(np.float64),
                ),
                hologram=np.random.rand(4, 4).astype(np.float64),
                reconstruction=np.random.rand(4, 4).astype(np.float64),
            )

            writer.save([sample], tmp_path)

            # Check that save was called with correct paths
            save_calls = mock_image.save.call_args_list
            assert len(save_calls) == 3

            saved_paths = [call[0][0] for call in save_calls]
            expected_suffixes = ["_object.png", "_hologram.png", "_reconstruction.png"]

            for suffix in expected_suffixes:
                assert any(str(path).endswith(suffix) for path in saved_paths)

    def test_write_png_conversion(self, tmp_path: Path) -> None:
        """Test PNG conversion from float to uint8."""
        with patch("hologen.utils.io.Image") as mock_image_class:
            mock_image = Mock()
            mock_image_class.fromarray.return_value = mock_image

            writer = NumpyDatasetWriter()

            # Test image with known values
            test_image = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float64)
            test_path = tmp_path / "test.png"

            writer._write_png(test_path, test_image)

            # Verify conversion to uint8
            mock_image_class.fromarray.assert_called_once()
            call_args = mock_image_class.fromarray.call_args
            converted_array = call_args[0][0]
            expected_array = np.array([[0, 127], [255, 63]], dtype=np.uint8)

            np.testing.assert_array_equal(converted_array, expected_array)
            assert call_args[1]["mode"] == "L"

    def test_normalization_in_preview(self, tmp_path: Path) -> None:
        """Test that images are normalized before PNG conversion."""
        with patch("hologen.utils.io.Image") as mock_image_class:
            mock_image = Mock()
            mock_image_class.fromarray.return_value = mock_image

            writer = NumpyDatasetWriter(save_preview=True)

            # Create sample with non-normalized data
            sample = HologramSample(
                object_sample=ObjectSample(
                    name="test",
                    pixels=np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64),
                ),
                hologram=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
                reconstruction=np.array(
                    [[100.0, 200.0], [300.0, 400.0]], dtype=np.float64
                ),
            )

            writer.save([sample], tmp_path)

            # All fromarray calls should receive normalized data (0-255 range)
            for call in mock_image_class.fromarray.call_args_list:
                array = call[0][0]
                assert array.min() >= 0
                assert array.max() <= 255
                assert array.dtype == np.uint8

    def test_empty_samples_list(self, tmp_path: Path) -> None:
        """Test handling of empty samples list."""
        writer = NumpyDatasetWriter()
        writer.save([], tmp_path)

        # Should create directory but no files
        assert tmp_path.exists()
        assert len(list(tmp_path.iterdir())) == 0

    def test_sample_indexing(self, tmp_path: Path) -> None:
        """Test that samples are indexed correctly."""
        writer = NumpyDatasetWriter(save_preview=False)

        samples = []
        for i in range(12):  # Test zero-padding
            sample = HologramSample(
                object_sample=ObjectSample(
                    name="test",
                    pixels=np.zeros((2, 2), dtype=np.float64),
                ),
                hologram=np.zeros((2, 2), dtype=np.float64),
                reconstruction=np.zeros((2, 2), dtype=np.float64),
            )
            samples.append(sample)

        writer.save(samples, tmp_path)

        # Check file naming with zero-padding
        expected_files = [f"sample_{i:05d}_test.npz" for i in range(12)]
        actual_files = [f.name for f in tmp_path.glob("*.npz")]

        assert len(actual_files) == 12
        for expected in expected_files:
            assert expected in actual_files
