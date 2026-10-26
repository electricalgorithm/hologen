"""Saving and loading hologram datasets as parquet files."""

from pathlib import Path

import numpy as np
import pandas as pd


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
