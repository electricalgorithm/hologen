"""Tests for the dataset file format 2: one row and one row group per hologram."""

import pathlib

import numpy
import pyarrow.parquet as pq
import pytest
from pandas import DataFrame

from hologen.dataset.storage import FORMAT_VERSION, ImageFile


def test_holograms_round_trip_as_exact_counts(tmp_path: pathlib.Path) -> None:
    """Counts are stored as uint16 and read back unchanged, one row group each."""
    counts: numpy.ndarray = (
        numpy.random.default_rng(0).integers(0, 4096, (3, 8, 8)).astype(float)
    )
    path: pathlib.Path = tmp_path / "hologram_0000.parquet"
    ImageFile.write_holograms(path, [10, 11, 12], counts)
    file: pq.ParquetFile = pq.ParquetFile(path)
    assert file.num_row_groups == 3
    assert file.schema_arrow.field("intensity").type.value_type.value_type == "uint16"
    for row in range(3):
        numpy.testing.assert_array_equal(ImageFile(path).hologram(row), counts[row])


def test_fields_round_trip_to_float32_precision(tmp_path: pathlib.Path) -> None:
    """Complex fields are stored as float32 real and imaginary parts."""
    rng: numpy.random.Generator = numpy.random.default_rng(1)
    fields: numpy.ndarray = rng.standard_normal((2, 6, 6)) + 1j * rng.standard_normal(
        (2, 6, 6)
    )
    path: pathlib.Path = tmp_path / "ground_truth_0000.parquet"
    ImageFile.write_fields(path, [0, 1], fields, kind="ground_truth")
    file: ImageFile = ImageFile(path)
    assert (len(file), file.kind, file.shape) == (2, "ground_truth", (6, 6))
    numpy.testing.assert_allclose(file.field(1), fields[1], atol=1e-6)
    assert file.field(1).dtype == numpy.complex64


def test_every_file_records_its_format(tmp_path: pathlib.Path) -> None:
    """The key-value metadata names the format, kind, units and shape."""
    path: pathlib.Path = tmp_path / "hologram_0000.parquet"
    ImageFile.write_holograms(path, [0], numpy.zeros((1, 4, 5)))
    metadata: dict[bytes, bytes] = pq.ParquetFile(path).schema_arrow.metadata
    assert metadata[b"hologen.format"] == FORMAT_VERSION.encode()
    assert metadata[b"hologen.kind"] == b"hologram"
    assert metadata[b"hologen.shape"] == b"4,5"
    assert b"digital numbers" in metadata[b"hologen.units"]


def test_counts_outside_16_bits_are_rejected(tmp_path: pathlib.Path) -> None:
    """Values that do not fit uint16 exactly would be silently changed, so they are refused."""
    with pytest.raises(ValueError, match="whole camera counts"):
        ImageFile.write_holograms(
            tmp_path / "h.parquet", [0], numpy.full((1, 2, 2), 0.5)
        )


def test_the_old_per_pixel_layout_is_rejected(tmp_path: pathlib.Path) -> None:
    """A file without the format marker gives a message to regenerate the dataset."""
    path: pathlib.Path = tmp_path / "hologram_0000.parquet"
    DataFrame({"sample_id": [0] * 4, "intensity": [1.0] * 4}).to_parquet(path)
    with pytest.raises(ValueError, match="regenerate"):
        ImageFile(path)
