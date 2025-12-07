"""
Example: Loading data from consolidated parquet files.

This demonstrates how to load and filter samples from the consolidated
parquet format. The loading process is the same regardless of whether
the dataset was generated with batch writing or all at once.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Example 1: Load all samples using the utility function
from hologen.utils import load_samples_from_parquet

dataset_dir = Path("dataset")
resolution = 512  # Match your hologram config

# Load all holograms (automatically handles multiple parquet files)
holograms = load_samples_from_parquet(dataset_dir, "hologram", resolution=resolution)
print(f"Loaded {len(holograms)} hologram samples")
print(f"First hologram shape: {holograms[0].shape}")

# Example 2: Filter by configuration using pandas
# Load all parquet files for this data type
from glob import glob
parquet_files = sorted(glob(str(dataset_dir / "hologram_*.parquet")))
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

# Get unique configurations
configs = df["config_name"].unique()
print(f"\nAvailable configurations: {configs}")

# Filter samples from a specific config
clean_samples = df[df["config_name"] == "no_noise"]
sample_ids = clean_samples["sample_id"].unique()
print(f"\nSamples in 'no_noise' config: {len(sample_ids)}")

# Reconstruct a specific sample
sample_id = 0
sample_data = df[df["sample_id"] == sample_id]
hologram_array = sample_data["intensity"].values.reshape(resolution, resolution)
print(f"\nReconstructed sample {sample_id} shape: {hologram_array.shape}")

# Example 3: Load complex fields (ground truth or reconstructed)
gt_files = sorted(glob(str(dataset_dir / "ground_truth_*.parquet")))
gt_dfs = [pd.read_parquet(f) for f in gt_files]
gt_df = pd.concat(gt_dfs, ignore_index=True)
sample_data = gt_df[gt_df["sample_id"] == 0]
real = sample_data["real"].values
imag = sample_data["imag"].values
complex_field = (real + 1j * imag).reshape(resolution, resolution)
print(f"\nGround truth complex field shape: {complex_field.shape}")
print(f"Amplitude range: [{np.abs(complex_field).min():.3f}, {np.abs(complex_field).max():.3f}]")

# Example 4: Batch loading for ML training
def create_dataloader(dataset_dir: Path, data_type: str, config_name: str, resolution: int, batch_size: int = 32):
    """Create batches of samples for training."""
    # Load all parquet files for this data type
    parquet_files = sorted(glob(str(dataset_dir / f"{data_type}_*.parquet")))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    filtered = df[df["config_name"] == config_name]
    sample_ids = filtered["sample_id"].unique()
    
    for i in range(0, len(sample_ids), batch_size):
        batch_ids = sample_ids[i:i+batch_size]
        batch_data = filtered[filtered["sample_id"].isin(batch_ids)]
        
        # Reconstruct arrays for each sample in batch
        batch_arrays = []
        for sid in batch_ids:
            sample = batch_data[batch_data["sample_id"] == sid]
            if "intensity" in sample.columns:
                arr = sample["intensity"].values.reshape(resolution, resolution)
            else:
                real = sample["real"].values
                imag = sample["imag"].values
                arr = (real + 1j * imag).reshape(resolution, resolution)
            batch_arrays.append(arr)
        
        yield np.stack(batch_arrays)

# Use the dataloader
print("\nBatch loading example:")
for batch_idx, batch in enumerate(create_dataloader(dataset_dir, "hologram", "no_noise", resolution, batch_size=8)):
    print(f"Batch {batch_idx}: shape {batch.shape}")
    if batch_idx >= 2:  # Just show first 3 batches
        break
