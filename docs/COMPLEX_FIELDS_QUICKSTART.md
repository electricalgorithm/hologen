# Complex Fields Quick Start Guide

This is a quick reference for using HoloGen's complex field support. For complete documentation, see [COMPLEX_FIELDS.md](COMPLEX_FIELDS.md).

## What Are Complex Fields?

Complex fields contain both **amplitude** and **phase** information, representing the complete optical wave:

```
E(x,y) = A(x,y) · exp(i·φ(x,y))
```

Traditional holography only records **intensity** (|E|²), losing phase information. Complex field support enables:
- Phase-only objects (transparent samples like biological cells)
- Full field reconstruction (amplitude + phase)
- Physics-aware ML models

## Quick Examples

### 1. Generate Phase-Only Objects (Transparent Samples)

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708
```

### 2. Generate Amplitude Objects with Complex Output

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --object-type amplitude \
    --output-domain complex
```

### 3. Traditional Intensity-Only (Default)

```bash
python scripts/generate_dataset.py --samples 100
```

## Field Representations

| Representation | What It Is | When to Use |
|---------------|------------|-------------|
| **Intensity** | \|E\|² | Default, backward compatible, camera output |
| **Amplitude** | \|E\| | Amplitude-based reconstruction methods |
| **Phase** | arg(E) | Quantitative phase imaging, phase retrieval |
| **Complex** | Real + Imag | Physics-aware models, full field information |

## Object Types

| Object Type | Physical Example | Amplitude | Phase |
|------------|------------------|-----------|-------|
| **Amplitude** | Stained cells, printed patterns | Varies | Zero |
| **Phase** | Unstained cells, transparent samples | Uniform (1.0) | Varies |

## Common Use Cases

### Biological Cell Imaging (Phase-Only)
```bash
python scripts/generate_dataset.py \
    --samples 1000 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --wavelength 532e-9
```

### Classical Holography (Intensity-Only)
```bash
python scripts/generate_dataset.py \
    --samples 1000 \
    --object-type amplitude \
    --output-domain intensity
```

### Physics-Aware ML Training
```bash
python scripts/generate_dataset.py \
    --samples 1000 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --sensor-shot-noise \
    --sensor-read-noise 3.0
```

## Loading Data in Python

### PyTorch
```python
import torch
import numpy as np

# Load complex hologram
data = np.load('sample_00000_circle_hologram.npz')
hologram = data['real'] + 1j * data['imag']

# Convert to 2-channel tensor
hologram_tensor = torch.stack([
    torch.from_numpy(data['real']).float(),
    torch.from_numpy(data['imag']).float()
], dim=0)
```

### NumPy
```python
import numpy as np

# Load complex field
data = np.load('sample_00000_circle_hologram.npz')
hologram = data['real'] + 1j * data['imag']

# Extract representations
intensity = np.abs(hologram) ** 2
amplitude = np.abs(hologram)
phase = np.angle(hologram)
```

## File Structure

```
output_dir/
├── npz/
│   ├── sample_00000_circle_object.npz      # Complex field data
│   ├── sample_00000_circle_hologram.npz
│   └── sample_00000_circle_reconstruction.npz
└── preview/
    ├── object/
    │   ├── sample_00000_circle_object_amplitude.png
    │   └── sample_00000_circle_object_phase.png
    ├── hologram/
    └── reconstruction/
```

## Key Parameters

### --object-type
- `amplitude`: Absorbing objects (default)
- `phase`: Transparent objects with phase modulation
- `complex`: Reserved for future mixed objects

### --output-domain
- `intensity`: |E|² only (default, backward compatible)
- `amplitude`: |E| only
- `phase`: arg(E) only
- `complex`: Full complex field (real + imaginary)

### --phase-shift
- Range: [0, 2π] radians
- Common values:
  - `0.7854` (π/4, 45°): Small phase shift
  - `1.5708` (π/2, 90°): Moderate phase shift (default)
  - `3.1416` (π, 180°): Large phase shift

## Next Steps

- **Full Documentation**: [COMPLEX_FIELDS.md](COMPLEX_FIELDS.md)
- **API Reference**: [COMPLEX_FIELDS.md#api-reference](COMPLEX_FIELDS.md#api-reference)
- **Visual Examples**: Run `python scripts/generate_visual_examples.py` (requires matplotlib)
- **Noise Modeling**: [NOISE_SIMULATION.md](NOISE_SIMULATION.md)

## Troubleshooting

**Q: Phase-only objects look uniform in intensity?**  
A: That's correct! Phase-only objects are invisible in intensity. The contrast appears in the hologram after propagation.

**Q: How do I visualize phase?**  
A: Use a cyclic colormap like 'twilight' or 'hsv'. Phase ranges from -π to π radians.

**Q: Can I mix object types in one dataset?**  
A: Currently, one object type per dataset. Generate separate datasets and combine them.

**Q: What's the difference between amplitude and intensity?**  
A: Intensity = Amplitude². Amplitude preserves more dynamic range.

**Q: Do I need complex fields for my application?**  
A: If you're working with transparent samples (cells, phase masks) or need phase information, yes. For traditional absorbing objects, intensity-only may suffice.
