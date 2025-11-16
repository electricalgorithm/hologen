# HoloGen Quick Start Guide

Get started with HoloGen in 5 minutes. This guide covers installation, your first dataset, and common use cases.

## Installation (3 Steps)

### 1. Clone or Download HoloGen
```bash
git clone https://github.com/yourusername/hologen.git
cd hologen
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

That's it! You're ready to generate datasets.

## Your First Dataset (5 Minutes)

Generate a basic hologram dataset with default settings:

```bash
python scripts/generate_dataset.py --samples 10
```

This creates a `dataset/` directory with:
- **NPZ files**: NumPy arrays containing object, hologram, and reconstruction data
- **PNG previews**: Visual representations of the generated samples

**What you get:**
- 10 samples with random circular objects
- Inline holography method
- 256×256 pixel resolution
- 532 nm wavelength (green laser)
- 20 mm propagation distance

**Output structure:**
```
dataset/
├── npz/
│   ├── sample_00000_circle.npz
│   ├── sample_00001_circle.npz
│   └── ...
└── preview/
    ├── object/
    │   ├── sample_00000_circle_object.png
    │   └── ...
    ├── hologram/
    │   ├── sample_00000_circle_hologram.png
    │   └── ...
    └── reconstruction/
        ├── sample_00000_circle_reconstruction.png
        └── ...
```

## Common Use Cases

### 1. High-Resolution Dataset
Generate larger, higher-quality samples:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --output ./dataset_highres
```

**Use case:** Training deep learning models that need more spatial detail.

### 2. Phase-Only Objects (Transparent Samples)
Generate transparent objects like biological cells:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --output ./dataset_phase
```

**Use case:** Quantitative phase imaging, biological cell imaging, transparent sample analysis.

**Learn more:** [Complex Fields Documentation](COMPLEX_FIELDS.md)

### 3. Off-Axis Holography
Generate off-axis holograms with carrier frequency:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --method off_axis \
    --carrier-x 1600 \
    --carrier-y 1600 \
    --output ./dataset_offaxis
```

**Use case:** Off-axis holographic microscopy, Fourier filtering applications.

**Learn more:** [Holography Methods Documentation](HOLOGRAPHY_METHODS.md) *(coming soon)*

### 4. Realistic Noisy Dataset
Add sensor noise and optical aberrations for realistic simulations:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --sensor-read-noise 3.0 \
    --sensor-shot-noise \
    --sensor-dark-current 0.5 \
    --sensor-bit-depth 12 \
    --speckle-contrast 0.8 \
    --output ./dataset_noisy
```

**Use case:** Training robust models that work with real experimental data.

**Learn more:** [Noise Simulation Documentation](NOISE_SIMULATION.md)

### 5. Large-Scale Dataset Generation
Generate a large dataset for production ML training:

```bash
python scripts/generate_dataset.py \
    --samples 10000 \
    --height 512 \
    --width 512 \
    --no-preview \
    --output ./dataset_large
```

**Tip:** Use `--no-preview` to skip PNG generation and speed up large dataset creation.

## Loading Your Data

### Python/NumPy
```python
import numpy as np

# Load a sample
data = np.load('dataset/npz/sample_00000_circle.npz')
object_img = data['object']
hologram = data['hologram']
reconstruction = data['reconstruction']

print(f"Object shape: {object_img.shape}")
print(f"Hologram shape: {hologram.shape}")
```

### PyTorch DataLoader
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class HologramDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = sorted(Path(data_dir).glob("*.npz"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        hologram = torch.from_numpy(data['hologram']).float()
        object_img = torch.from_numpy(data['object']).float()
        return hologram, object_img

# Create DataLoader
dataset = HologramDataset('dataset/npz')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for holograms, objects in dataloader:
    print(f"Batch shape: {holograms.shape}")
    break
```

### TensorFlow/Keras
```python
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_sample(file_path):
    data = np.load(file_path.numpy().decode())
    return data['hologram'].astype(np.float32), data['object'].astype(np.float32)

# Create dataset
file_paths = [str(p) for p in sorted(Path('dataset/npz').glob('*.npz'))]
dataset = tf.data.Dataset.from_tensor_slices(file_paths)
dataset = dataset.map(
    lambda x: tf.py_function(load_sample, [x], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

for holograms, objects in dataset:
    print(f"Batch shape: {holograms.shape}")
    break
```

## Key Command-Line Arguments

### Basic Arguments
- `--samples`: Number of samples to generate (default: 10)
- `--output`: Output directory path (default: ./dataset)
- `--seed`: Random seed for reproducibility (default: 42)
- `--no-preview`: Skip PNG preview generation (faster for large datasets)

### Grid Configuration
- `--height`: Image height in pixels (default: 256)
- `--width`: Image width in pixels (default: 256)
- `--pixel-pitch`: Physical pixel size in meters (default: 4.65e-6)

### Optical Configuration
- `--wavelength`: Illumination wavelength in meters (default: 532e-9)
- `--distance`: Propagation distance in meters (default: 0.02)

### Holography Method
- `--method`: Choose `inline` or `off_axis` (default: inline)
- `--carrier-x`: Off-axis carrier frequency x (default: 1600)
- `--carrier-y`: Off-axis carrier frequency y (default: 1600)

### Object Type (Complex Fields)
- `--object-type`: Choose `amplitude`, `phase`, or `complex` (default: amplitude)
- `--output-domain`: Choose `intensity`, `amplitude`, `phase`, or `complex` (default: intensity)
- `--phase-shift`: Phase shift in radians for phase objects (default: 1.5708 = π/2)

### Noise Parameters
- `--sensor-read-noise`: Read noise standard deviation (default: 0.0)
- `--sensor-shot-noise`: Enable Poisson shot noise (flag)
- `--sensor-dark-current`: Dark current mean value (default: 0.0)
- `--sensor-bit-depth`: ADC bit depth for quantization (default: None)
- `--speckle-contrast`: Speckle contrast ratio 0-1 (default: 0.0)
- `--aberration-defocus`: Defocus aberration coefficient (default: 0.0)

**Full list:** Run `python scripts/generate_dataset.py --help`

## Next Steps

### Learn More About Features
- **[Complex Fields](COMPLEX_FIELDS.md)**: Amplitude, phase, and complex field representations
- **[Complex Fields Quick Start](COMPLEX_FIELDS_QUICKSTART.md)**: Quick reference for complex field usage
- **[Noise Simulation](NOISE_SIMULATION.md)**: Realistic sensor noise and optical aberrations
- **[Master Documentation Index](README.md)**: Complete feature documentation *(coming soon)*

### Explore the Codebase
- **Shape Generators**: `src/hologen/shapes.py` - Create custom object patterns
- **Holography Strategies**: `src/hologen/holography/` - Inline and off-axis methods
- **Noise Models**: `src/hologen/noise/` - Sensor and optical noise simulation
- **Pipeline Components**: `src/hologen/converters.py` - Dataset generation pipeline

### Advanced Usage
- **Custom Shape Generators**: Implement your own object patterns
- **Custom Noise Models**: Add application-specific noise sources
- **Pipeline Customization**: Build custom generation workflows
- **Batch Processing**: Generate datasets in parallel

**API Documentation:** *(coming soon)*

## Troubleshooting

### Installation Issues

**Problem:** `pip install -e .` fails  
**Solution:** Ensure you're using Python 3.11+ and have activated your virtual environment

**Problem:** Import errors when running scripts  
**Solution:** Make sure you installed with `pip install -e .` (editable mode)

### Generation Issues

**Problem:** Out of memory errors  
**Solution:** Reduce `--height` and `--width`, or generate in smaller batches

**Problem:** Slow generation  
**Solution:** Use `--no-preview` to skip PNG generation, or reduce sample count

**Problem:** Phase-only objects look uniform  
**Solution:** This is correct! Phase objects are invisible in intensity. Check the hologram output for contrast.

### Data Loading Issues

**Problem:** NPZ files won't load  
**Solution:** Ensure you're using NumPy 1.20+ and the file path is correct

**Problem:** Complex field data structure different than expected  
**Solution:** Check the `representation` field in the NPZ file to determine the format

## Getting Help

- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: See `scripts/generate_visual_examples.py` for visualization examples
- **Issues**: Report bugs or request features on GitHub
- **Community**: Join discussions and share your use cases

## Quick Reference Card

```bash
# Basic dataset
python scripts/generate_dataset.py --samples 100

# High resolution
python scripts/generate_dataset.py --samples 100 --height 512 --width 512

# Phase objects
python scripts/generate_dataset.py --samples 100 --object-type phase --output-domain complex

# Off-axis
python scripts/generate_dataset.py --samples 100 --method off_axis

# With noise
python scripts/generate_dataset.py --samples 100 --sensor-read-noise 3.0 --sensor-shot-noise

# Large dataset (no previews)
python scripts/generate_dataset.py --samples 10000 --no-preview

# Custom output location
python scripts/generate_dataset.py --samples 100 --output ./my_dataset
```

---

**Ready to generate your first dataset?** Run the command above and explore the output in the `dataset/` directory!
