"""
Reading and writing dataset files in hologen format 2.

Every image file holds one row per hologram, with the image as a 2-D list in
the layout Hugging Face datasets read as Array2D, and one row group per
hologram, so that a reader can load a single hologram without the rest of the
file. Holograms are stored as uint16 camera counts and complex fields as
float32 real and imaginary parts, compressed with zstd. The Parquet key-value
metadata of each file records the format version, the kind of data, its units
and the image shape. Rows join to metadata_XXXX.parquet by sample_id.
"""

import pathlib

import numpy
import pyarrow
import pyarrow.parquet as pq

FORMAT_VERSION: str = "2"
UNITS: dict[str, str] = {
    "hologram": "digital numbers of the camera's ADC",
    "ground_truth": "complex object field relative to the illumination",
    "reconstructed": "field back-propagated at the nominal distance, relative to the illumination",
}


class ImageFile:
    """
    One image file of a dataset: holograms, ground truths or reconstructions.

    Opening a file checks that it is in format 2 and reads its kind and image
    shape from the Parquet metadata. Each row is then read on its own, from its
    own row group.

    Attributes:
        path (pathlib.Path): The Parquet file.
        kind (str): hologram, ground_truth or reconstructed.
        shape (tuple[int, int]): Height and width of the images.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """
        Open a file and check that it is in format 2.

        Args:
            path (pathlib.Path): Parquet file.

        Raises:
            ValueError: If the file was written in another format.
        """
        self.path: pathlib.Path = path
        self._file: pq.ParquetFile = pq.ParquetFile(path)
        metadata: dict[bytes, bytes] = self._file.schema_arrow.metadata or {}
        if metadata.get(b"hologen.format") != FORMAT_VERSION.encode():
            raise ValueError(
                f"{path} was written before hologen format {FORMAT_VERSION} (one pixel per "
                "row); regenerate the dataset with the current hologen."
            )
        self.kind: str = metadata[b"hologen.kind"].decode()
        height: str
        width: str
        height, width = metadata[b"hologen.shape"].decode().split(",")
        self.shape: tuple[int, int] = (int(height), int(width))

    def __len__(self) -> int:
        """
        Count the holograms in the file.

        Returns:
            int: The number of rows, one per hologram.
        """
        return self._file.metadata.num_rows

    def sample_ids(self) -> list[int]:
        """
        Read the sample id of every row.

        Returns:
            list[int]: The sample ids in row order.
        """
        return self._file.read(columns=["sample_id"]).column("sample_id").to_pylist()

    def _column(self, row: int, column: str) -> numpy.ndarray:
        """
        Read one image column of one row group.

        Args:
            row (int): Row (and row group) of the hologram.
            column (str): Column to read.

        Returns:
            numpy.ndarray: The image, shape (H, W), in its stored type.
        """
        images: pyarrow.ChunkedArray = self._file.read_row_group(
            row, columns=[column]
        ).column(column)
        values: pyarrow.Array = images.combine_chunks().flatten().flatten()
        return values.to_numpy(zero_copy_only=False).reshape(self.shape)

    def hologram(self, row: int) -> numpy.ndarray:
        """
        Read one recorded hologram.

        Args:
            row (int): Row of the hologram in the file.

        Returns:
            numpy.ndarray: The hologram in digital numbers, uint16, shape (H, W).
        """
        return self._column(row, "intensity")

    def field(self, row: int) -> numpy.ndarray:
        """
        Read one complex field.

        Args:
            row (int): Row of the field in the file.

        Returns:
            numpy.ndarray: The field, complex64, shape (H, W).
        """
        field: numpy.ndarray = self._column(row, "real") + 1j * self._column(
            row, "imag"
        )
        return field.astype(numpy.complex64)

    def image(self, row: int) -> numpy.ndarray:
        """
        Read one row as the kind of data the file holds.

        Args:
            row (int): Row in the file.

        Returns:
            numpy.ndarray: uint16 counts for a hologram file, complex64 otherwise.
        """
        return self.hologram(row) if self.kind == "hologram" else self.field(row)

    @staticmethod
    def _image_column(
        images: numpy.ndarray, value_type: pyarrow.DataType
    ) -> pyarrow.Array:
        """
        Build a column of 2-D lists from a stack of images.

        Args:
            images (numpy.ndarray): Images of shape (N, H, W).
            value_type (pyarrow.DataType): Type of the stored values.

        Returns:
            pyarrow.Array: One list of H lists of W values per image.
        """
        count: int
        height: int
        width: int
        count, height, width = images.shape
        values: pyarrow.Array = pyarrow.array(images.reshape(-1), type=value_type)
        rows: pyarrow.Array = pyarrow.ListArray.from_arrays(
            pyarrow.array(
                numpy.arange(0, count * height * width + 1, width),
                type=pyarrow.int32(),
            ),
            values,
        )
        return pyarrow.ListArray.from_arrays(
            pyarrow.array(
                numpy.arange(0, count * height + 1, height), type=pyarrow.int32()
            ),
            rows,
        )

    @staticmethod
    def _write(
        path: pathlib.Path, table: pyarrow.Table, kind: str, shape: tuple[int, int]
    ) -> None:
        """
        Write a table with one row group per hologram and the format metadata.

        Args:
            path (pathlib.Path): Parquet file to write.
            table (pyarrow.Table): One row per hologram.
            kind (str): hologram, ground_truth or reconstructed.
            shape (tuple[int, int]): Height and width of the images.
        """
        metadata: dict[str, str] = {
            "hologen.format": FORMAT_VERSION,
            "hologen.kind": kind,
            "hologen.units": UNITS[kind],
            "hologen.shape": f"{shape[0]},{shape[1]}",
        }
        pq.write_table(
            table.replace_schema_metadata(metadata),
            path,
            row_group_size=1,
            compression="zstd",
        )

    @staticmethod
    def write_holograms(
        path: pathlib.Path, sample_ids: list[int], counts: numpy.ndarray
    ) -> None:
        """
        Write recorded holograms as uint16 camera counts.

        Args:
            path (pathlib.Path): Parquet file to write.
            sample_ids (list[int]): Sample id of each hologram.
            counts (numpy.ndarray): Holograms in digital numbers, shape (N, H, W).

        Raises:
            ValueError: If a value is not a whole count between 0 and 65535, which
                uint16 could not store exactly.
        """
        if (
            counts.min() < 0
            or counts.max() > numpy.iinfo(numpy.uint16).max
            or not numpy.all(counts == numpy.round(counts))
        ):
            raise ValueError(
                "Holograms must hold whole camera counts between 0 and 65535."
            )
        table: pyarrow.Table = pyarrow.table(
            {
                "sample_id": pyarrow.array(sample_ids, type=pyarrow.int64()),
                "intensity": ImageFile._image_column(
                    counts.astype(numpy.uint16), pyarrow.uint16()
                ),
            }
        )
        ImageFile._write(path, table, "hologram", (counts.shape[1], counts.shape[2]))

    @staticmethod
    def write_fields(
        path: pathlib.Path, sample_ids: list[int], fields: numpy.ndarray, kind: str
    ) -> None:
        """
        Write complex fields as float32 real and imaginary parts.

        Args:
            path (pathlib.Path): Parquet file to write.
            sample_ids (list[int]): Sample id of each field.
            fields (numpy.ndarray): Complex fields, shape (N, H, W).
            kind (str): ground_truth or reconstructed.
        """
        table: pyarrow.Table = pyarrow.table(
            {
                "sample_id": pyarrow.array(sample_ids, type=pyarrow.int64()),
                "real": ImageFile._image_column(
                    fields.real.astype(numpy.float32), pyarrow.float32()
                ),
                "imag": ImageFile._image_column(
                    fields.imag.astype(numpy.float32), pyarrow.float32()
                ),
            }
        )
        ImageFile._write(path, table, kind, (fields.shape[1], fields.shape[2]))


def load_samples(dataset_dir: pathlib.Path, kind: str) -> dict[int, numpy.ndarray]:
    """
    Load every image of one kind from a dataset folder.

    Args:
        dataset_dir (pathlib.Path): The dataset folder.
        kind (str): hologram, ground_truth or reconstructed.

    Returns:
        dict[int, numpy.ndarray]: The images by sample id, each of shape (H, W):
            uint16 counts for holograms, complex64 for fields.

    Raises:
        FileNotFoundError: If the folder has no file of that kind.
    """
    paths: list[pathlib.Path] = sorted(dataset_dir.glob(f"{kind}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {kind}_*.parquet files in {dataset_dir}.")
    samples: dict[int, numpy.ndarray] = {}
    path: pathlib.Path
    for path in paths:
        file: ImageFile = ImageFile(path)
        row: int
        sample_id: int
        for row, sample_id in enumerate(file.sample_ids()):
            samples[sample_id] = file.image(row)
    return samples
