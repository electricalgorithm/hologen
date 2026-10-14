"""Saving and loading hologram datasets as parquet files."""

import pandas as pd
import numpy as np

from pathlib import Path
from glob import glob


def save_complex_field_to_parquet(data: np.ndarray, filepath: str | Path) -> None:
    """
    Save a complex field as parquet with separate real and imaginary parts.

    The data is flattened. To rebuild it, use
    field = (real + 1j * imag).reshape(resolution, resolution).

    Args:
        data (np.ndarray): Complex field to save.
        filepath (str | Path): Parquet file to write.
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)

    df: pd.DataFrame = pd.DataFrame(
        {
            "real": data.real.flatten(),
            "imag": data.imag.flatten(),
        }
    )
    df.to_parquet(filepath, compression="snappy")


def save_intensity_to_parquet(data: np.ndarray, filepath: str | Path) -> None:
    """
    Save a real-valued intensity as parquet.

    The data is flattened. To rebuild it, use
    intensity = data.reshape(resolution, resolution).

    Args:
        data (np.ndarray): Real-valued intensity to save.
        filepath (str | Path): Parquet file to write.
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)

    df: pd.DataFrame = pd.DataFrame(
        {
            "intensity": data.flatten(),
        }
    )
    df.to_parquet(filepath, compression="snappy")


def load_samples_from_parquet(
    dataset_dir: str | Path, data_type: str = "hologram", resolution: int = 256
) -> dict[int, np.ndarray]:
    """
    Load samples from parquet files (handles both single and multi-file datasets).

    Args:
        dataset_dir (str | Path): The dataset folder.
        data_type (str): Kind of data to load, one of "hologram",
            "ground_truth" or "reconstructed".
        resolution (int): Image size in pixels; images are square.

    Returns:
        dict[int, np.ndarray]: The arrays by sample id, each of shape
            (resolution, resolution).

    Raises:
        FileNotFoundError: If the folder has no parquet file of that kind.

    Examples:
        >>> samples = load_samples_from_parquet("dataset", "hologram", resolution=512)
        >>> hologram_0 = samples[0]  # Get first sample
        >>> # Or load ground truth
        >>> gt_samples = load_samples_from_parquet("dataset", "ground_truth", resolution=512)
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)

    # Find all parquet files matching the pattern
    pattern: str = str(dataset_dir / f"{data_type}_*.parquet")
    files: list[str] = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {pattern}")

    # Load all files and concatenate
    dfs: list[pd.DataFrame] = [pd.read_parquet(f) for f in files]
    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    samples: dict[int, np.ndarray] = {}

    for sample_id in df["sample_id"].unique():
        sample_df: pd.DataFrame = df[df["sample_id"] == sample_id]

        if "intensity" in sample_df.columns:
            # Intensity data
            samples[sample_id] = sample_df["intensity"].values.reshape(
                resolution, resolution
            )
        else:
            # Complex field data
            real: np.ndarray = sample_df["real"].values
            imag: np.ndarray = sample_df["imag"].values
            samples[sample_id] = (real + 1j * imag).reshape(resolution, resolution)

    return samples
