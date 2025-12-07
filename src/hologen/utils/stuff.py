import pandas as pd
import numpy as np

from pathlib import Path
from glob import glob


def save_complex_field_to_parquet(data: np.ndarray, filepath: str | Path):
    """
    Save complex field as parquet with separate real and imaginary parts.

    Note: Data is flattened. To reconstruct, use:
    field = (real + 1j * imag).reshape(256, 256)
    
    Args:
        data: Complex numpy array to save
        filepath: Path to save the parquet file (str or Path)
    """
    
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    df = pd.DataFrame(
        {
            "real": data.real.flatten(),
            "imag": data.imag.flatten(),
        }
    )
    df.to_parquet(filepath, compression="snappy")


def save_intensity_to_parquet(data: np.ndarray, filepath: str | Path):
    """
    Save intensity (real-valued) as parquet.

    Note: Data is flattened. To reconstruct, use:
    intensity = data.reshape(256, 256)
    
    Args:
        data: Real-valued numpy array to save
        filepath: Path to save the parquet file (str or Path)
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    df = pd.DataFrame(
        {
            "intensity": data.flatten(),
        }
    )
    df.to_parquet(filepath, compression="snappy")


def load_samples_from_parquet(dataset_dir: str | Path, data_type: str = "hologram", resolution: int = 256) -> dict[int, np.ndarray]:
    """
    Load samples from parquet files (handles both single and multi-file datasets).
    
    Args:
        dataset_dir: Path to the dataset directory
        data_type: Type of data to load: "hologram", "ground_truth", or "reconstructed"
        resolution: Image resolution (assumes square images)
        
    Returns:
        Dictionary mapping sample_id to reconstructed arrays
        
    Example:
        >>> samples = load_samples_from_parquet("dataset", "hologram", resolution=512)
        >>> hologram_0 = samples[0]  # Get first sample
        >>> # Or load ground truth
        >>> gt_samples = load_samples_from_parquet("dataset", "ground_truth", resolution=512)
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
    
    # Find all parquet files matching the pattern
    pattern = str(dataset_dir / f"{data_type}_*.parquet")
    files = sorted(glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {pattern}")
    
    # Load all files and concatenate
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    samples = {}
    
    for sample_id in df["sample_id"].unique():
        sample_df = df[df["sample_id"] == sample_id]
        
        if "intensity" in sample_df.columns:
            # Intensity data
            samples[sample_id] = sample_df["intensity"].values.reshape(resolution, resolution)
        else:
            # Complex field data
            real = sample_df["real"].values
            imag = sample_df["imag"].values
            samples[sample_id] = (real + 1j * imag).reshape(resolution, resolution)
    
    return samples
