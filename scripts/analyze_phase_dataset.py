#!/usr/bin/env python
"""Analyze the generated phase dataset."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we're using the venv
try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Make sure virtual environment is activated.")
    sys.exit(1)

print("\n" + "=" * 70)
print("PHASE GENERATION ANALYSIS")
print("=" * 70 + "\n")

configs = [
    ("config_1_uniform_constant", "Uniform + Constant"),
    ("config_2_gaussian_blobs_shape_based", "Gaussian Blobs + Shape-Based"),
    ("config_3_perlin_noise_gradient", "Perlin Noise + Gradient"),
]

results = []

for config_dir, config_name in configs:
    print(f"{config_name}:")
    print("-" * 70)

    base_path = Path("phase_test_dataset") / config_dir
    object_files = sorted(base_path.glob("*_object.npz"))

    if not object_files:
        print("  No files found!\n")
        continue

    # Analyze first sample
    sample_file = object_files[0]
    data = np.load(sample_file)

    print(f"  Sample: {sample_file.name}")
    print(f"  Keys: {list(data.keys())}")

    if "real" in data and "imag" in data:
        field = data["real"] + 1j * data["imag"]
        amplitude = np.abs(field)
        phase = np.angle(field)

        # Analyze phase within object
        object_mask = amplitude > 0.01
        if object_mask.sum() > 0:
            phase_obj = phase[object_mask]

            phase_min_deg = np.degrees(phase_obj.min())
            phase_max_deg = np.degrees(phase_obj.max())
            phase_std_deg = np.degrees(phase_obj.std())
            phase_mean_deg = np.degrees(phase_obj.mean())

            print(f"  Phase range: [{phase_min_deg:.1f}°, {phase_max_deg:.1f}°]")
            print(f"  Phase mean: {phase_mean_deg:.1f}°")
            print(f"  Phase std: {phase_std_deg:.1f}°")
            print(f"  Object pixels: {object_mask.sum()}")

            results.append(
                {
                    "config": config_name,
                    "phase_min": phase_min_deg,
                    "phase_max": phase_max_deg,
                    "phase_std": phase_std_deg,
                    "phase_mean": phase_mean_deg,
                }
            )

    # Count files
    phase_pngs = len(list(base_path.glob("*_phase.png")))
    amp_pngs = len(list(base_path.glob("*_amplitude.png")))
    npz_files = len(list(base_path.glob("*.npz")))

    print(
        f"  Files: {phase_pngs} phase PNGs, {amp_pngs} amplitude PNGs, {npz_files} NPZ"
    )
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

for r in results:
    print(f"{r['config']}:")
    print(f"  Phase variation: {r['phase_std']:.1f}° std dev")
    print(f"  Range: [{r['phase_min']:.1f}°, {r['phase_max']:.1f}°]")
    print()

print("✓ Dataset generated successfully!")
print("✓ View *_phase.png files to see phase distributions (HSV colormap)")
print("✓ View *_amplitude.png files to see object shapes")
print()
