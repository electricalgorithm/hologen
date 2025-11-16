# Advanced Dataset Generation

This guide explains how to use the `generate_advanced_dataset.py` script to create synthetic hologram datasets with comprehensive physical simulation and per-sample randomization.

## Overview

The advanced dataset generator creates hologram samples with randomized physical parameters to maximize generalization for machine learning models. Each sample includes complete metadata describing all simulation parameters for reproducibility.

## Features

- **Per-sample randomization**: Each sample has unique physical parameters
- **Multi-wavelength support**: Simulate different illumination wavelengths
- **Diverse object types**: Intensity, amplitude, phase, and complex objects
- **Realistic noise modeling**: Sensor noise, speckle, and optical aberrations
- **Complete metadata**: All parameters logged in .npz files for reproducibility

## Usage

### Basic Usage

Generate 100 samples with default parameters:

```bash
python scripts/generate_advanced_dataset.py --samples 100
```

### Custom Configuration

```bash
python scripts/generate_advanced_dataset.py \
    --samples 1000 \
    --output my_dataset \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelengths "532e-9,633e-9,450e-9" \
    --seed 42
```

### Command-Line Arguments

- `--samples`: Number of samples to generate (default: 100)
- `--output`: Output directory path (default: dataset_advanced)
- `--seed`: Random seed for reproducibility (default: 42)
- `--height`: Image height in pixels (default: 512)
- `--width`: Image width in pixels (default: 512)
- `--pixel-pitch`: Pixel pitch in meters (default: 6.4e-6)
- `--wavelengths`: Comma-separated wavelengths in meters (default: "532e-9")
- `--no-preview`: Disable PNG preview generation

## Randomization Parameters

Each sample randomizes the following parameters:

### Optical Parameters

- **Wavelength**: Randomly selected from provided wavelengths
- **Propagation Distance**: Uniform random in [0.03, 0.15] meters (30-150 mm)
- **Object Type**: Complex only (amplitude + phase modulation)
- **Phase Shift**: Uniform random in [-π, π] radians

### Noise Parameters

- **Speckle Contrast**: Uniform random in [0.1, 0.9]
- **Sensor Read Noise**: Normal distribution, mean=5 e⁻, std=5 e⁻ (clipped to ≥0)
- **Sensor Shot Noise**: Always enabled (Poisson noise)
- **Sensor Dark Current**: Uniform random in [0.2, 1.0] e⁻
- **Sensor Bit Depth**: Uniform random integer in [8, 16] bits

### Aberration Parameters

- **Defocus**: Uniform random in [-30, +30] micrometers
- **Astigmatism (X, Y)**: Uniform random in [-0.5, +0.5] (Zernike coefficients)
- **Coma (X, Y)**: Uniform random in [-0.5, +0.5] (Zernike coefficients)

## Output Format

### File Structure

For each sample, three files are generated:

```
sample_00000_circle_object.npz          # Object domain
sample_00000_circle_hologram.npz        # Hologram
sample_00000_circle_reconstruction.npz  # Reconstruction
```

With PNG previews (if enabled):

```
sample_00000_circle_object.png
sample_00000_circle_hologram.png
sample_00000_circle_reconstruction.png
```

### Metadata in .npz Files

Each .npz file contains the field data plus complete metadata:

```python
import numpy as np

data = np.load('sample_00000_circle_hologram.npz')

# Field data
intensity = data['intensity']  # or 'amplitude', 'phase', 'real'/'imag'

# Metadata
wavelength = data['wavelength']                    # meters
propagation_distance = data['propagation_distance'] # meters
object_type = data['object_type']                  # str
phase_shift = data['phase_shift']                  # radians
speckle_contrast = data['speckle_contrast']        # 0-1
sensor_read_noise = data['sensor_read_noise']      # electrons
sensor_shot_noise = data['sensor_shot_noise']      # bool
sensor_dark_current = data['sensor_dark_current']  # electrons
sensor_bit_depth = data['sensor_bit_depth']        # bits
aberration_defocus_um = data['aberration_defocus_um']  # micrometers
aberration_astigmatism_x = data['aberration_astigmatism_x']
aberration_astigmatism_y = data['aberration_astigmatism_y']
aberration_coma_x = data['aberration_coma_x']
aberration_coma_y = data['aberration_coma_y']
grid_height = data['grid_height']                  # pixels
grid_width = data['grid_width']                    # pixels
pixel_pitch = data['pixel_pitch']                  # meters
```

## Example: Loading and Using the Dataset

```python
import numpy as np
from pathlib import Path

def load_sample(sample_path):
    """Load a sample with metadata."""
    data = np.load(sample_path)
    
    # Extract field
    if 'intensity' in data:
        field = data['intensity']
    elif 'amplitude' in data:
        field = data['amplitude']
    elif 'phase' in data:
        field = data['phase']
    elif 'real' in data and 'imag' in data:
        field = data['real'] + 1j * data['imag']
    
    # Extract metadata
    metadata = {k: data[k] for k in data.keys() if k not in 
                ['intensity', 'amplitude', 'phase', 'real', 'imag', 'representation']}
    
    return field, metadata

# Load a hologram sample
hologram, meta = load_sample('dataset_advanced/sample_00000_circle_hologram.npz')

print(f"Wavelength: {meta['wavelength']*1e9:.1f} nm")
print(f"Propagation: {meta['propagation_distance']*1000:.2f} mm")
print(f"Speckle contrast: {meta['speckle_contrast']:.3f}")
print(f"Bit depth: {meta['sensor_bit_depth']} bits")
```

## Multi-Wavelength Training

For multi-modal training with different wavelengths:

```bash
python scripts/generate_advanced_dataset.py \
    --samples 5000 \
    --wavelengths "405e-9,450e-9,532e-9,633e-9,780e-9" \
    --output dataset_multiwavelength
```

This creates a dataset with samples at 405nm (violet), 450nm (blue), 532nm (green), 633nm (red), and 780nm (near-IR).

## Performance Notes

- Generation time: ~0.5-1 second per sample (512×512 images)
- Disk space: ~6 MB per sample (with PNG previews), ~4 MB without
- Memory usage: ~100 MB peak for 512×512 images

For large datasets (>1000 samples), consider:
- Using `--no-preview` to save disk space and speed up generation
- Running on multiple cores by splitting the dataset into batches
- Using a faster storage device (SSD)

## Reproducibility

The random seed ensures reproducible datasets:

```bash
# Generate identical datasets
python scripts/generate_advanced_dataset.py --seed 42 --samples 100
python scripts/generate_advanced_dataset.py --seed 42 --samples 100
```

Each sample's metadata allows exact reproduction of the simulation parameters.
