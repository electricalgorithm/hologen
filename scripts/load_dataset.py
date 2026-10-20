"""
Example: loading a dataset written by hologen.

The images are stored in hologen format 2 (hologen.dataset.storage), one row
per hologram, and the per-hologram labels in metadata_XXXX.parquet. The images
join to the labels by sample_id.
"""

from pathlib import Path

import pandas as pd

from hologen.dataset.storage import load_samples

dataset_dir = Path("dataset")

# Every hologram (uint16 camera counts) and ground truth (complex64), by sample id.
holograms = load_samples(dataset_dir, "hologram")
truths = load_samples(dataset_dir, "ground_truth")
print(f"Loaded {len(holograms)} holograms of shape {holograms[0].shape}")

# The labels of every hologram: seed, group, object family, exposure, distances.
metadata = pd.concat(
    [pd.read_parquet(path) for path in sorted(dataset_dir.glob("metadata_*.parquet"))],
    ignore_index=True,
)
print(metadata.groupby("config_name").size())

# Holograms of one group, for example the red blood cells with laser speckle.
chosen = metadata[metadata["config_name"] == "camera_laser_speckle/red_blood_cells"]
subset = {sample_id: holograms[sample_id] for sample_id in chosen["sample_id"]}
print(f"{len(subset)} holograms in camera_laser_speckle/red_blood_cells")
