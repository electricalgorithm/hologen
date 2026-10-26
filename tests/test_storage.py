"""Tests for the dataset file format 2: one row and one row group per hologram."""

import pathlib

import numpy
import pandas
import pyarrow.parquet as pq
import pytest
from pandas import DataFrame
from test_generator import generate

from hologen.dataset.recipe import Recipe, recipe_from_dict
from hologen.dataset.storage import FORMAT_VERSION, ImageFile, load_samples


def small_recipe(samples: int = 4) -> Recipe:
    """
    Build a small camera-noise recipe.

    Args:
        samples (int): Number of holograms.

    Returns:
        Recipe: A 48 x 48 pixel red blood cell recipe.
    """
    return recipe_from_dict(
        {
            "geometry": {"resolution": 48},
            "dataset": {
                "total_samples": samples,
                "objects": ["red_blood_cells"],
                "configurations": ["camera"],
            },
        }
    )


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
    """Counts that do not fit uint16 exactly are refused, as they would change."""
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


def test_generated_datasets_are_compact_and_complete(tmp_path: pathlib.Path) -> None:
    """
    A generated dataset is compact and complete.

    It has one row per hologram, the seeds in the metadata and no reconstruction.
    """
    recipe: Recipe = small_recipe()
    generate(recipe, tmp_path)
    assert not (tmp_path / "reconstructed_0000.parquet").exists()
    holograms: dict[int, numpy.ndarray] = load_samples(tmp_path, "hologram")
    assert sorted(holograms) == [0, 1, 2, 3]
    assert holograms[0].shape == (48, 48)
    assert pq.ParquetFile(tmp_path / "ground_truth_0000.parquet").metadata.num_rows == 4
    metadata: DataFrame = pandas.read_parquet(tmp_path / "metadata_0000.parquet")
    assert metadata["seed"].tolist() == [
        recipe.dataset.seed + index for index in range(4)
    ]


def test_the_reconstruction_is_written_on_request(tmp_path: pathlib.Path) -> None:
    """With write_reconstructed the back-propagated field is stored too."""
    generate(small_recipe(2), tmp_path, write_reconstructed=True)
    reconstructed: dict[int, numpy.ndarray] = load_samples(tmp_path, "reconstructed")
    assert reconstructed[1].shape == (48, 48)
    assert numpy.iscomplexobj(reconstructed[1])
