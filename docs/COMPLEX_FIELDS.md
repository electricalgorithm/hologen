# Complex Field Support in HoloGen

HoloGen supports complex-valued optical fields throughout the hologram generation pipeline, enabling more realistic and physics-accurate synthetic data for machine learning applications. This document explains the different field representations, how to use them, and when each is appropriate.

## Field Representations Overview

Optical fields in holography are fundamentally complex-valued, containing both amplitude and phase information. HoloGen supports four different representations of these fields:

### 1. Intensity Representation

**What it is**: The squared magnitude of the complex field, representing the detected light power.

```
I = |E|² = A²
```

**When to use**:
- Classical holography ML applications where only intensity is recorded
- Backward compatibility with existing intensity-only workflows
- When phase information is not needed or available
- Training models for intensity-based reconstruction methods

**Characteristics**:
- Real-valued (non-negative)
- Phase information is lost
- Matches what typical cameras record
- Default representation for backward compatibility

**Example use case**: Training a CNN to reconstruct object shapes from inline hologram intensity patterns.

### 2. Amplitude Representation

**What it is**: The magnitude of the complex field, representing the field strength.

```
A = |E|
```

**When to use**:
- Amplitude-based reconstruction methods
- When you need field strength without phase
- Intermediate processing steps
- Visualizing field magnitude

**Characteristics**:
- Real-valued (non-negative)
- Phase information is lost
- Square root of intensity
- Preserves more dynamic range than intensity

**Example use case**: Training models for amplitude-based phase retrieval algorithms.

### 3. Phase Representation

**What it is**: The phase angle of the complex field in radians.

```
φ = arg(E) ∈ [-π, π]
```

**When to use**:
- Quantitative phase imaging (QPI) applications
- Phase-contrast microscopy simulations
- When amplitude is uniform or unimportant
- Training phase unwrapping or phase retrieval models

**Characteristics**:
- Real-valued (range: -π to π)
- Amplitude information is lost (assumed uniform)
- Critical for transparent sample imaging
- Wraps at ±π boundaries

**Example use case**: Generating training data for biological cell imaging where cells are transparent and only modulate phase.

### 4. Complex Representation

**What it is**: The full complex field with both amplitude and phase.

```
E = A·exp(iφ) = real + i·imag
```

**When to use**:
- Physics-aware ML models that process full optical fields
- When both amplitude and phase are important
- Holographic reconstruction algorithms
- Preserving complete field information through pipeline
- Advanced applications requiring full wave information

**Characteristics**:
- Complex-valued (real + imaginary components)
- Contains complete field information
- No information loss
- Enables full wave optics simulations
- Larger storage requirements (2x memory)

**Example use case**: Training neural networks for holographic autofocusing or aberration correction that operate on complex fields.

## Visual Comparison

The different representations capture different aspects of the optical field:

```
Original Complex Field: E = 0.8·exp(i·π/4)
├─ Intensity:  I = 0.64
├─ Amplitude:  A = 0.8
├─ Phase:      φ = 0.785 rad (45°)
└─ Complex:    E = 0.566 + 0.566i
```

For a simple circular object:

**Amplitude-only object** (absorbing circle):
- Intensity: Dark circle on bright background
- Amplitude: Smooth transition from 0 to 1
- Phase: Uniform (zero everywhere)
- Complex: Real-valued field

**Phase-only object** (transparent circle with phase shift):
- Intensity: Uniform (no contrast!)
- Amplitude: Uniform (1.0 everywhere)
- Phase: Step function (0 outside, π/2 inside)
- Complex: Pure phase modulation

**Mixed object** (partially absorbing with phase shift):
- Intensity: Partial contrast
- Amplitude: Varies with absorption
- Phase: Varies with optical path length
- Complex: Full amplitude-phase modulation

## Conversion Between Representations

HoloGen provides utilities to convert between representations:

```python
from hologen.utils.fields import complex_to_representation
from hologen.types import FieldRepresentation

# Convert complex field to different representations
intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)
phase = complex_to_representation(field, FieldRepresentation.PHASE)
```

**Important**: Converting from intensity, amplitude, or phase back to complex results in information loss:
- Intensity → Complex: Phase is assumed zero
- Amplitude → Complex: Phase is assumed zero
- Phase → Complex: Amplitude is assumed unity (1.0)

Only complex → complex conversion is lossless.

## Choosing the Right Representation

Use this decision tree to select the appropriate representation:

1. **Do you need phase information?**
   - No → Use Intensity (default, most compatible)
   - Yes → Continue to step 2

2. **Do you need amplitude information?**
   - No (uniform amplitude) → Use Phase
   - Yes → Continue to step 3

3. **Are you training physics-aware models?**
   - Yes → Use Complex (full information)
   - No → Use Amplitude or Intensity depending on your reconstruction method

4. **Storage and memory constraints?**
   - Tight constraints → Use Intensity or Amplitude (half the size)
   - No constraints → Use Complex (preserves all information)

## Performance Considerations

| Representation | Memory Usage | Information Content | Compatibility |
|---------------|--------------|---------------------|---------------|
| Intensity     | 1x (baseline) | Low (magnitude²) | High (legacy) |
| Amplitude     | 1x (baseline) | Medium (magnitude) | Medium |
| Phase         | 1x (baseline) | Medium (angle) | Low |
| Complex       | 2x (baseline) | High (complete) | Low (new) |

**Memory example** for 512×512 images:
- Intensity/Amplitude/Phase: ~2 MB (float64) or ~1 MB (float32)
- Complex: ~4 MB (complex128) or ~2 MB (complex64)

## Next Steps

- [Complex Object Generation](COMPLEX_FIELDS.md#complex-object-generation) - Learn how to generate phase-only and mixed objects
- [Complex Hologram Export](COMPLEX_FIELDS.md#complex-hologram-export) - Understand file formats and loading data
- [CLI Usage Examples](COMPLEX_FIELDS.md#cli-usage-examples) - See command-line examples
- [API Reference](COMPLEX_FIELDS.md#api-reference) - Detailed API documentation

## Complex Object Generation

HoloGen can generate three types of object domains: amplitude-only, phase-only, and complex (mixed amplitude-phase). This section explains how to generate each type and when to use them.

### Object Types

#### Amplitude-Only Objects

Amplitude-only objects modulate the amplitude of transmitted light while leaving phase unchanged (zero phase).

**Physical interpretation**: Absorbing or scattering samples (e.g., stained biological samples, printed patterns, metal particles)

**Mathematical representation**:
```
E(x,y) = A(x,y)·exp(i·0) = A(x,y)
```

**Code example**:
```python
from hologen.shapes import CircleGenerator
from hologen.types import GridSpec
import numpy as np

# Create grid and generator
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
generator = CircleGenerator(radius_range=(20e-6, 50e-6))
rng = np.random.default_rng(42)

# Generate amplitude-only object
complex_field = generator.generate_complex(
    grid=grid,
    rng=rng,
    mode="amplitude",
    phase_shift=0.0  # Not used for amplitude mode
)

# Result: complex field with varying amplitude, zero phase
print(f"Amplitude range: [{np.abs(complex_field).min():.2f}, {np.abs(complex_field).max():.2f}]")
print(f"Phase range: [{np.angle(complex_field).min():.2f}, {np.angle(complex_field).max():.2f}]")
# Output: Amplitude range: [0.00, 1.00], Phase range: [0.00, 0.00]
```

#### Phase-Only Objects

Phase-only objects modulate the phase of transmitted light while maintaining uniform amplitude.

**Physical interpretation**: Transparent samples with varying refractive index or thickness (e.g., biological cells, phase masks, transparent polymers)

**Mathematical representation**:
```
E(x,y) = 1.0·exp(i·φ(x,y))
```

**Code example**:
```python
from hologen.shapes import CircleGenerator
from hologen.types import GridSpec
import numpy as np

# Create grid and generator
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
generator = CircleGenerator(radius_range=(20e-6, 50e-6))
rng = np.random.default_rng(42)

# Generate phase-only object with π/2 phase shift
complex_field = generator.generate_complex(
    grid=grid,
    rng=rng,
    mode="phase",
    phase_shift=np.pi/2  # 90-degree phase shift inside circle
)

# Result: complex field with uniform amplitude, varying phase
print(f"Amplitude range: [{np.abs(complex_field).min():.2f}, {np.abs(complex_field).max():.2f}]")
print(f"Phase range: [{np.angle(complex_field).min():.2f}, {np.angle(complex_field).max():.2f}]")
# Output: Amplitude range: [1.00, 1.00], Phase range: [0.00, 1.57]
```

#### Complex (Mixed) Objects

Complex objects modulate both amplitude and phase simultaneously.

**Physical interpretation**: Samples with both absorption and refractive index variation (e.g., partially stained cells, composite materials)

**Note**: Full mixed mode is reserved for future extension. Current implementation supports amplitude-only and phase-only modes.

### Phase Shift Parameter

The `phase_shift` parameter controls the phase difference between the object and background for phase-only objects.

**Valid range**: [0, 2π] radians (0 to 6.28)

**Common values**:
- `π/4` (0.785 rad, 45°): Small phase shift, subtle contrast
- `π/2` (1.571 rad, 90°): Quarter-wave shift, good contrast
- `π` (3.142 rad, 180°): Half-wave shift, maximum contrast
- `3π/2` (4.712 rad, 270°): Three-quarter wave shift

**Physical meaning**: The phase shift represents the optical path difference between light passing through the object versus the background:

```
φ = (2π/λ) × (n₁ - n₀) × d
```

Where:
- λ = wavelength
- n₁ = refractive index of object
- n₀ = refractive index of background (typically 1.0 for air)
- d = object thickness

**Example**: For a biological cell in water:
- λ = 532 nm (green laser)
- n₁ = 1.38 (cell cytoplasm)
- n₀ = 1.33 (water)
- d = 5 μm (cell thickness)

Phase shift: φ = (2π/532e-9) × (1.38 - 1.33) × 5e-6 ≈ 0.59 rad ≈ π/5

**Choosing phase_shift**:
```python
# Subtle phase contrast (thin samples)
phase_shift = np.pi / 4  # 45 degrees

# Moderate phase contrast (typical cells)
phase_shift = np.pi / 2  # 90 degrees (default)

# Strong phase contrast (thick samples)
phase_shift = np.pi  # 180 degrees

# Very strong contrast (phase masks)
phase_shift = 3 * np.pi / 2  # 270 degrees
```

### Supported Shape Generators

All shape generators support complex field generation:

```python
from hologen.shapes import (
    CircleGenerator,
    RectangleGenerator,
    RingGenerator,
    CircleCheckerGenerator,
    RectangleCheckerGenerator,
    EllipseCheckerGenerator
)

# Each generator has generate_complex() method
generators = [
    CircleGenerator(radius_range=(20e-6, 50e-6)),
    RectangleGenerator(width_range=(30e-6, 60e-6), height_range=(30e-6, 60e-6)),
    RingGenerator(inner_radius_range=(15e-6, 25e-6), outer_radius_range=(30e-6, 50e-6)),
    CircleCheckerGenerator(radius_range=(40e-6, 80e-6), num_divisions=8),
    RectangleCheckerGenerator(width_range=(50e-6, 100e-6), height_range=(50e-6, 100e-6), num_divisions=8),
    EllipseCheckerGenerator(semi_major_range=(40e-6, 80e-6), semi_minor_range=(30e-6, 60e-6), num_divisions=8)
]

# Generate phase-only objects with each shape
for generator in generators:
    field = generator.generate_complex(
        grid=grid,
        rng=rng,
        mode="phase",
        phase_shift=np.pi/2
    )
```

### Complete Pipeline Example

Here's a complete example generating phase-only objects and complex holograms:

```python
from hologen import *
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter, HologramDatasetGenerator
from hologen.holography.inline import InlineHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, HolographyMethod,
    FieldRepresentation, OutputConfig
)
from hologen.utils.io import ComplexFieldWriter
from pathlib import Path
import numpy as np

# Configuration
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

# Output configuration for complex fields
output_config = OutputConfig(
    object_representation=FieldRepresentation.PHASE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# Create pipeline components
shape_generator = CircleGenerator(radius_range=(20e-6, 50e-6))
object_producer = ObjectDomainProducer(
    generator=shape_generator,
    phase_shift=np.pi/2,  # 90-degree phase shift
    mode="phase"  # Phase-only objects
)

strategy = InlineHolographyStrategy()
converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.INLINE: strategy},
    output_config=output_config
)

# Generate dataset
rng = np.random.default_rng(42)
dataset_generator = HologramDatasetGenerator(
    object_producer=object_producer,
    converter=converter,
    config=config
)

# Write to disk
writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
samples = dataset_generator.generate(count=10, rng=rng)
writer.save(samples, output_dir=Path("phase_objects_dataset"))
```

### Validation

HoloGen automatically validates phase values to ensure they're in the valid range [-π, π]:

```python
from hologen.utils.fields import validate_phase_range, PhaseRangeError

# This will raise PhaseRangeError if phase is out of range
try:
    validate_phase_range(phase_array)
except PhaseRangeError as e:
    print(f"Invalid phase values: {e}")
```

### Tips for Complex Object Generation

1. **Start with amplitude-only**: If you're new to complex fields, start with amplitude-only objects (default behavior) before moving to phase-only.

2. **Use moderate phase shifts**: For biological samples, π/2 to π is typically realistic. Larger shifts may not be physically meaningful.

3. **Consider your application**: 
   - Amplitude-only: Classical holography, absorbing samples
   - Phase-only: Quantitative phase imaging, transparent samples
   - Complex: Advanced applications requiring both

4. **Validate your data**: Always check that generated fields have expected properties (uniform amplitude for phase-only, zero phase for amplitude-only).

5. **Memory usage**: Complex fields use 2x memory compared to intensity. For large datasets, consider generating in batches.

## Complex Hologram Export

HoloGen exports complex field data in two formats: NumPy `.npz` archives for numerical processing and PNG images for visualization. This section explains the file structure and how to load data in ML pipelines.

### NumPy .npz File Structure

The `.npz` format stores arrays and metadata in a compressed archive. The structure varies by field representation:

#### Complex Representation

Stores real and imaginary components separately:

```python
# File: sample_00000_circle_hologram.npz
{
    'real': ndarray,              # Real component (float64, shape: [H, W])
    'imag': ndarray,              # Imaginary component (float64, shape: [H, W])
    'representation': 'complex',  # String identifier
    'wavelength': 532e-9,         # Optical wavelength in meters
    'propagation_distance': 0.05  # Propagation distance in meters
}
```

**Loading example**:
```python
import numpy as np

# Load complex hologram
data = np.load('sample_00000_circle_hologram.npz')
hologram = data['real'] + 1j * data['imag']

# Access metadata
wavelength = float(data['wavelength'])
distance = float(data['propagation_distance'])

print(f"Hologram shape: {hologram.shape}")
print(f"Hologram dtype: {hologram.dtype}")
print(f"Wavelength: {wavelength*1e9:.1f} nm")
```

#### Amplitude Representation

Stores magnitude only:

```python
# File: sample_00000_circle_hologram.npz
{
    'amplitude': ndarray,         # Amplitude (float64, shape: [H, W])
    'representation': 'amplitude',
    'wavelength': 532e-9,
    'propagation_distance': 0.05
}
```

**Loading example**:
```python
data = np.load('sample_00000_circle_hologram.npz')
amplitude = data['amplitude']
```

#### Phase Representation

Stores phase angle in radians:

```python
# File: sample_00000_circle_hologram.npz
{
    'phase': ndarray,            # Phase in radians (float64, shape: [H, W])
    'representation': 'phase',
    'wavelength': 532e-9,
    'propagation_distance': 0.05
}
```

**Loading example**:
```python
data = np.load('sample_00000_circle_hologram.npz')
phase = data['phase']  # Range: [-π, π]
```

#### Intensity Representation (Legacy)

Backward-compatible format for intensity-only data:

```python
# File: sample_00000_circle.npz
{
    'object': ndarray,           # Object intensity (float64, shape: [H, W])
    'hologram': ndarray,         # Hologram intensity (float64, shape: [H, W])
    'reconstruction': ndarray    # Reconstruction intensity (float64, shape: [H, W])
}
```

**Loading example**:
```python
data = np.load('sample_00000_circle.npz')
object_intensity = data['object']
hologram_intensity = data['hologram']
reconstruction = data['reconstruction']
```

### PNG Export

PNG files provide visual previews of the data. The export behavior depends on the field representation:

#### Complex Representation

Generates **two PNG files** per field:
- `*_amplitude.png`: Amplitude visualization (grayscale, 8-bit)
- `*_phase.png`: Phase visualization (colormap, 8-bit)

**File naming example**:
```
sample_00000_circle_object_amplitude.png
sample_00000_circle_object_phase.png
sample_00000_circle_hologram_amplitude.png
sample_00000_circle_hologram_phase.png
sample_00000_circle_reconstruction_amplitude.png
sample_00000_circle_reconstruction_phase.png
```

**Amplitude PNG encoding**:
- Amplitude values normalized to [0, 255]
- Linear mapping: `pixel = 255 * (amplitude / amplitude.max())`
- Grayscale 8-bit PNG

**Phase PNG encoding**:
- Phase values mapped from [-π, π] to [0, 255]
- Linear mapping: `pixel = 255 * (phase + π) / (2π)`
- Optional colormap applied (default: "twilight")
- 8-bit PNG (grayscale or RGB if colormap used)

#### Amplitude, Phase, or Intensity Representation

Generates **one PNG file** per field:
- Single grayscale image
- Values normalized to [0, 255]

### Loading Complex Data in ML Pipelines

#### PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class ComplexHologramDataset(Dataset):
    """PyTorch dataset for complex hologram data."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples = sorted(self.data_dir.glob("*_hologram.npz"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load hologram
        data = np.load(self.samples[idx])
        
        if 'real' in data and 'imag' in data:
            # Complex representation
            hologram = data['real'] + 1j * data['imag']
            
            # Convert to 2-channel tensor (real, imag)
            hologram_tensor = torch.stack([
                torch.from_numpy(data['real']).float(),
                torch.from_numpy(data['imag']).float()
            ], dim=0)
        else:
            # Intensity representation (legacy)
            hologram = data['hologram']
            hologram_tensor = torch.from_numpy(hologram).float().unsqueeze(0)
        
        # Load corresponding object
        object_path = self.samples[idx].parent / self.samples[idx].name.replace('_hologram', '_object')
        object_data = np.load(object_path)
        
        if 'real' in object_data and 'imag' in object_data:
            object_tensor = torch.stack([
                torch.from_numpy(object_data['real']).float(),
                torch.from_numpy(object_data['imag']).float()
            ], dim=0)
        else:
            object_tensor = torch.from_numpy(object_data['object']).float().unsqueeze(0)
        
        return hologram_tensor, object_tensor

# Usage
dataset = ComplexHologramDataset(Path("phase_objects_dataset/npz"))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

for hologram_batch, object_batch in dataloader:
    # hologram_batch shape: [batch_size, 2, height, width] for complex
    # or [batch_size, 1, height, width] for intensity
    print(f"Hologram batch shape: {hologram_batch.shape}")
    break
```

#### TensorFlow/Keras Dataset

```python
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_complex_sample(hologram_path, object_path):
    """Load a single complex hologram-object pair."""
    # Load hologram
    hologram_data = np.load(hologram_path.numpy().decode())
    if b'real' in hologram_data.files and b'imag' in hologram_data.files:
        hologram = np.stack([hologram_data['real'], hologram_data['imag']], axis=-1)
    else:
        hologram = hologram_data['hologram'][..., np.newaxis]
    
    # Load object
    object_data = np.load(object_path.numpy().decode())
    if b'real' in object_data.files and b'imag' in object_data.files:
        obj = np.stack([object_data['real'], object_data['imag']], axis=-1)
    else:
        obj = object_data['object'][..., np.newaxis]
    
    return hologram.astype(np.float32), obj.astype(np.float32)

# Create dataset
data_dir = Path("phase_objects_dataset/npz")
hologram_paths = sorted(data_dir.glob("*_hologram.npz"))
object_paths = [str(p).replace('_hologram', '_object') for p in hologram_paths]

dataset = tf.data.Dataset.from_tensor_slices((
    [str(p) for p in hologram_paths],
    object_paths
))

dataset = dataset.map(
    lambda h, o: tf.py_function(
        load_complex_sample,
        [h, o],
        [tf.float32, tf.float32]
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Usage
for hologram_batch, object_batch in dataset:
    # hologram_batch shape: [batch_size, height, width, 2] for complex
    # or [batch_size, height, width, 1] for intensity
    print(f"Hologram batch shape: {hologram_batch.shape}")
    break
```

#### NumPy/Scikit-learn Pipeline

```python
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_dataset(data_dir: Path):
    """Load entire dataset into memory."""
    data_dir = Path(data_dir)
    hologram_files = sorted(data_dir.glob("*_hologram.npz"))
    
    holograms = []
    objects = []
    
    for hologram_path in hologram_files:
        # Load hologram
        h_data = np.load(hologram_path)
        if 'real' in h_data and 'imag' in h_data:
            hologram = h_data['real'] + 1j * h_data['imag']
        else:
            hologram = h_data['hologram']
        
        # Load object
        object_path = hologram_path.parent / hologram_path.name.replace('_hologram', '_object')
        o_data = np.load(object_path)
        if 'real' in o_data and 'imag' in o_data:
            obj = o_data['real'] + 1j * o_data['imag']
        else:
            obj = o_data['object']
        
        holograms.append(hologram)
        objects.append(obj)
    
    return np.array(holograms), np.array(objects)

# Load and split
X, y = load_dataset(Path("phase_objects_dataset/npz"))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, dtype: {X_train.dtype}")
print(f"Test set: {X_test.shape}, dtype: {X_test.dtype}")
```

### Utility Function for Loading

HoloGen provides a utility function for loading samples:

```python
from hologen.utils.io import load_complex_sample

# Load a complex sample (auto-detects format)
sample = load_complex_sample(Path("phase_objects_dataset/npz/sample_00000_circle.npz"))

# Returns ComplexObjectSample or ObjectSample depending on format
if hasattr(sample, 'field'):
    # Complex sample
    print(f"Complex field shape: {sample.field.shape}")
    print(f"Representation: {sample.representation}")
else:
    # Legacy intensity sample
    print(f"Intensity shape: {sample.pixels.shape}")
```

### File Organization

ComplexFieldWriter organizes files in a structured directory:

```
output_dir/
├── npz/
│   ├── sample_00000_circle_object.npz
│   ├── sample_00000_circle_hologram.npz
│   ├── sample_00000_circle_reconstruction.npz
│   ├── sample_00001_ring_object.npz
│   ├── sample_00001_ring_hologram.npz
│   └── ...
└── preview/
    ├── object/
    │   ├── sample_00000_circle_object_amplitude.png
    │   ├── sample_00000_circle_object_phase.png
    │   └── ...
    ├── hologram/
    │   ├── sample_00000_circle_hologram_amplitude.png
    │   ├── sample_00000_circle_hologram_phase.png
    │   └── ...
    └── reconstruction/
        ├── sample_00000_circle_reconstruction_amplitude.png
        ├── sample_00000_circle_reconstruction_phase.png
        └── ...
```

### Best Practices

1. **Check representation type**: Always check the 'representation' field in .npz files to determine how to load the data.

2. **Handle both formats**: Write loaders that can handle both legacy intensity and new complex formats for backward compatibility.

3. **Normalize appropriately**: 
   - Amplitude: Normalize by max value or use fixed normalization
   - Phase: Already in [-π, π], may need wrapping for some applications
   - Intensity: Normalize by max or use percentile-based normalization

4. **Memory management**: Complex fields use 2x memory. For large datasets, use generators or load in batches.

5. **Validate loaded data**:
```python
# Check for NaN or Inf
assert np.isfinite(hologram).all(), "Hologram contains non-finite values"

# Check phase range
phase = np.angle(hologram)
assert np.all((-np.pi <= phase) & (phase <= np.pi)), "Phase out of range"

# Check amplitude is non-negative
amplitude = np.abs(hologram)
assert np.all(amplitude >= 0), "Amplitude is negative"
```

6. **Use appropriate dtypes**: 
   - `complex128` for high precision (default)
   - `complex64` for memory efficiency (half the size)
   - Convert after loading if needed: `hologram.astype(np.complex64)`

## CLI Usage Examples

The `generate_dataset.py` script supports complex field generation through command-line arguments. This section provides examples for common use cases.

### New Command-Line Arguments

```bash
--object-type {amplitude,phase,complex}
    Type of object domain representation
    Default: amplitude
    
--output-domain {intensity,amplitude,phase,complex}
    Type of hologram output representation
    Default: intensity (backward compatible)
    
--phase-shift FLOAT
    Phase shift in radians for phase-only objects
    Default: 1.5708 (π/2, 90 degrees)
    Valid range: [0, 2π]
```

### Basic Examples

#### Example 1: Amplitude Objects with Intensity Output (Default/Legacy)

Generate traditional intensity-only holograms from amplitude objects:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_intensity \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05
```

This is the default behavior and maintains backward compatibility.

#### Example 2: Phase-Only Objects with Complex Output

Generate phase-only objects (transparent samples) with full complex hologram output:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_phase_complex \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708
```

**Use case**: Quantitative phase imaging, biological cell imaging

#### Example 3: Phase-Only Objects with Intensity Output

Generate phase-only objects but export only intensity (for phase contrast imaging):

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_phase_intensity \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain intensity \
    --phase-shift 1.5708
```

**Use case**: Training models for phase contrast microscopy where only intensity is recorded

#### Example 4: Amplitude Objects with Complex Output

Generate amplitude objects with full complex hologram output:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_amplitude_complex \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type amplitude \
    --output-domain complex
```

**Use case**: Physics-aware models that need full field information even for absorbing samples

### Advanced Examples

#### Example 5: Off-Axis Holography with Complex Fields

Generate off-axis holograms with complex output:

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_offaxis_complex \
    --method off_axis \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708
```

**Use case**: Off-axis holography with phase objects

#### Example 6: Large Phase Shift for Strong Contrast

Generate phase objects with large phase shift (π radians):

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_large_phase \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 3.14159
```

**Use case**: Thick samples or large refractive index differences

#### Example 7: Small Phase Shift for Subtle Contrast

Generate phase objects with small phase shift (π/4 radians):

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_small_phase \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 0.7854
```

**Use case**: Thin samples or small refractive index differences

#### Example 8: Amplitude-Only Output

Generate holograms with amplitude-only output (no phase):

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_amplitude_only \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type amplitude \
    --output-domain amplitude
```

**Use case**: Amplitude-based reconstruction methods

#### Example 9: Phase-Only Output

Generate holograms with phase-only output (no amplitude):

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_phase_only \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain phase \
    --phase-shift 1.5708
```

**Use case**: Phase retrieval algorithms, quantitative phase imaging

### Combining with Noise Models

Complex fields work seamlessly with noise models:

#### Example 10: Phase Objects with Sensor Noise

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_phase_noisy \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --sensor-read-noise 3.0 \
    --sensor-shot-noise \
    --sensor-bit-depth 12
```

**Note**: Noise is applied to the intensity representation, then the complex field is reconstructed preserving phase.

#### Example 11: Phase Objects with Speckle Noise

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_phase_speckle \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --speckle-contrast 0.8
```

#### Example 12: Complete Realistic Simulation

Combine phase objects, complex output, and comprehensive noise:

```bash
python scripts/generate_dataset.py \
    --samples 1000 \
    --output ./dataset_realistic \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --speckle-contrast 0.8 \
    --sensor-read-noise 3.0 \
    --sensor-shot-noise \
    --sensor-dark-current 0.5 \
    --sensor-bit-depth 12 \
    --aberration-defocus 0.5 \
    --aberration-astigmatism 0.3
```

**Use case**: Most realistic simulation for training robust ML models

### Output Domain Combinations

Here's a matrix of valid combinations:

| Object Type | Output Domain | Use Case |
|-------------|---------------|----------|
| amplitude   | intensity     | Default, backward compatible |
| amplitude   | amplitude     | Amplitude-based reconstruction |
| amplitude   | phase         | Not recommended (phase is zero) |
| amplitude   | complex       | Physics-aware models |
| phase       | intensity     | Phase contrast imaging |
| phase       | amplitude     | Not recommended (amplitude is uniform) |
| phase       | phase         | Quantitative phase imaging |
| phase       | complex       | Full field QPI |

### Validation and Troubleshooting

#### Check Generated Files

```bash
# List generated files
ls -lh dataset_phase_complex/npz/

# Check .npz contents
python -c "import numpy as np; data = np.load('dataset_phase_complex/npz/sample_00000_circle_hologram.npz'); print(data.files)"

# Verify complex data
python -c "
import numpy as np
data = np.load('dataset_phase_complex/npz/sample_00000_circle_hologram.npz')
if 'real' in data and 'imag' in data:
    field = data['real'] + 1j * data['imag']
    print(f'Complex field shape: {field.shape}')
    print(f'Amplitude range: [{np.abs(field).min():.3f}, {np.abs(field).max():.3f}]')
    print(f'Phase range: [{np.angle(field).min():.3f}, {np.angle(field).max():.3f}]')
"
```

#### Verify Phase-Only Objects

```bash
# Check that phase-only objects have uniform amplitude
python -c "
import numpy as np
data = np.load('dataset_phase_complex/npz/sample_00000_circle_object.npz')
field = data['real'] + 1j * data['imag']
amplitude = np.abs(field)
print(f'Amplitude std: {amplitude.std():.6f}')  # Should be ~0
print(f'Amplitude mean: {amplitude.mean():.6f}')  # Should be ~1.0
"
```

#### Common Errors

**Error**: `ValueError: Invalid output-domain value`
**Solution**: Use one of: intensity, amplitude, phase, complex

**Error**: `PhaseRangeError: Phase values outside [-π, π]`
**Solution**: Check --phase-shift is in [0, 2π] range

**Error**: `FileNotFoundError: No such file or directory`
**Solution**: Ensure output directory exists or script will create it

### Performance Considerations

```bash
# Small dataset for testing (fast)
python scripts/generate_dataset.py --samples 10 --output ./test --object-type phase --output-domain complex

# Medium dataset for development (moderate)
python scripts/generate_dataset.py --samples 100 --output ./dev --object-type phase --output-domain complex

# Large dataset for training (slow, ~1-2 samples/sec)
python scripts/generate_dataset.py --samples 10000 --output ./train --object-type phase --output-domain complex

# Very large dataset (use batching)
for i in {0..9}; do
    python scripts/generate_dataset.py \
        --samples 1000 \
        --output ./train_batch_$i \
        --object-type phase \
        --output-domain complex
done
```

### Quick Reference

```bash
# Minimal command (defaults)
python scripts/generate_dataset.py

# Phase objects, complex output
python scripts/generate_dataset.py --object-type phase --output-domain complex

# Custom phase shift
python scripts/generate_dataset.py --object-type phase --phase-shift 3.14159

# With noise
python scripts/generate_dataset.py --object-type phase --output-domain complex --sensor-shot-noise

# Off-axis
python scripts/generate_dataset.py --method off_axis --object-type phase --output-domain complex

# Help
python scripts/generate_dataset.py --help
```

## API Reference

This section documents the new classes, functions, and protocols introduced for complex field support.

### Type Definitions (hologen.types)

#### FieldRepresentation

Enumeration of field representation types.

```python
from enum import StrEnum, auto, unique

@unique
class FieldRepresentation(StrEnum):
    """Enumeration of optical field representation types."""
    
    INTENSITY = auto()  # |E|² - squared magnitude
    AMPLITUDE = auto()  # |E| - magnitude
    PHASE = auto()      # arg(E) - phase angle in radians
    COMPLEX = auto()    # E = real + i*imag - full complex field
```

**Usage**:
```python
from hologen.types import FieldRepresentation

# Specify representation
rep = FieldRepresentation.COMPLEX
print(rep)  # Output: 'complex'

# Compare representations
if rep == FieldRepresentation.COMPLEX:
    print("Using complex representation")
```

#### ComplexObjectSample

Dataclass representing an object-domain sample with complex field.

```python
@dataclass(slots=True)
class ComplexObjectSample:
    """Object-domain sample with complex field representation.
    
    Attributes:
        name: Sample identifier (e.g., 'sample_00000_circle')
        field: Complex-valued 2D array representing the object field
        representation: Type of field representation
    """
    name: str
    field: ArrayComplex  # Complex128 array, shape: [H, W]
    representation: FieldRepresentation
```

**Usage**:
```python
from hologen.types import ComplexObjectSample, FieldRepresentation
import numpy as np

# Create a complex object sample
sample = ComplexObjectSample(
    name="my_sample",
    field=np.ones((512, 512), dtype=np.complex128),
    representation=FieldRepresentation.COMPLEX
)

# Access properties
print(f"Sample name: {sample.name}")
print(f"Field shape: {sample.field.shape}")
print(f"Field dtype: {sample.field.dtype}")
print(f"Representation: {sample.representation}")
```

#### ComplexHologramSample

Dataclass representing a complete hologram sample with complex fields.

```python
@dataclass(slots=True)
class ComplexHologramSample:
    """Hologram sample with complex field support.
    
    Attributes:
        object_sample: The original object-domain sample
        hologram_field: Complex hologram field at sensor plane
        hologram_representation: Representation type for hologram
        reconstruction_field: Reconstructed object field
        reconstruction_representation: Representation type for reconstruction
    """
    object_sample: ComplexObjectSample
    hologram_field: ArrayComplex
    hologram_representation: FieldRepresentation
    reconstruction_field: ArrayComplex
    reconstruction_representation: FieldRepresentation
```

**Usage**:
```python
from hologen.types import ComplexHologramSample

# Access sample components
print(f"Object name: {sample.object_sample.name}")
print(f"Hologram shape: {sample.hologram_field.shape}")
print(f"Hologram representation: {sample.hologram_representation}")
print(f"Reconstruction shape: {sample.reconstruction_field.shape}")

# Extract specific representations
hologram_intensity = np.abs(sample.hologram_field) ** 2
hologram_phase = np.angle(sample.hologram_field)
```

#### OutputConfig

Configuration for output field representations.

```python
@dataclass(slots=True)
class OutputConfig:
    """Configuration for output field representations.
    
    Attributes:
        object_representation: Desired representation for object field
        hologram_representation: Desired representation for hologram field
        reconstruction_representation: Desired representation for reconstruction
    
    Default values maintain backward compatibility (intensity-only).
    """
    object_representation: FieldRepresentation = FieldRepresentation.INTENSITY
    hologram_representation: FieldRepresentation = FieldRepresentation.INTENSITY
    reconstruction_representation: FieldRepresentation = FieldRepresentation.INTENSITY
```

**Usage**:
```python
from hologen.types import OutputConfig, FieldRepresentation

# Default configuration (backward compatible)
config = OutputConfig()
print(config.hologram_representation)  # Output: 'intensity'

# Custom configuration for complex output
config = OutputConfig(
    object_representation=FieldRepresentation.PHASE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# Use in converter
converter = ObjectToHologramConverter(
    strategy_mapping=strategies,
    output_config=config
)
```

### Field Utilities (hologen.utils.fields)

#### complex_to_representation()

Convert complex field to requested representation.

```python
def complex_to_representation(
    field: ArrayComplex,
    representation: FieldRepresentation
) -> ArrayFloat | ArrayComplex:
    """Convert complex field to requested representation.
    
    Args:
        field: Complex-valued field array
        representation: Target representation type
    
    Returns:
        Converted field (real-valued for intensity/amplitude/phase,
        complex-valued for complex representation)
    
    Raises:
        FieldRepresentationError: If representation is invalid
    
    Examples:
        >>> field = np.array([[1+1j, 2+0j], [0+1j, 1-1j]])
        >>> intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
        >>> amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)
        >>> phase = complex_to_representation(field, FieldRepresentation.PHASE)
    """
```

**Usage**:
```python
from hologen.utils.fields import complex_to_representation
from hologen.types import FieldRepresentation
import numpy as np

# Create complex field
field = np.exp(1j * np.linspace(0, 2*np.pi, 100).reshape(10, 10))

# Convert to different representations
intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)
phase = complex_to_representation(field, FieldRepresentation.PHASE)
complex_copy = complex_to_representation(field, FieldRepresentation.COMPLEX)

print(f"Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
print(f"Amplitude range: [{amplitude.min():.3f}, {amplitude.max():.3f}]")
print(f"Phase range: [{phase.min():.3f}, {phase.max():.3f}]")
```

#### amplitude_phase_to_complex()

Construct complex field from amplitude and phase arrays.

```python
def amplitude_phase_to_complex(
    amplitude: ArrayFloat,
    phase: ArrayFloat
) -> ArrayComplex:
    """Construct complex field from amplitude and phase.
    
    Args:
        amplitude: Amplitude array (non-negative)
        phase: Phase array in radians (typically [-π, π])
    
    Returns:
        Complex field: amplitude * exp(i * phase)
    
    Examples:
        >>> amplitude = np.ones((10, 10))
        >>> phase = np.random.uniform(-np.pi, np.pi, (10, 10))
        >>> field = amplitude_phase_to_complex(amplitude, phase)
    """
```

**Usage**:
```python
from hologen.utils.fields import amplitude_phase_to_complex
import numpy as np

# Create amplitude and phase separately
amplitude = np.random.rand(512, 512)
phase = np.random.uniform(-np.pi, np.pi, (512, 512))

# Combine into complex field
field = amplitude_phase_to_complex(amplitude, phase)

# Verify round-trip
recovered_amplitude = np.abs(field)
recovered_phase = np.angle(field)
assert np.allclose(amplitude, recovered_amplitude)
assert np.allclose(phase, recovered_phase)
```

#### validate_phase_range()

Validate that phase values are within valid range.

```python
def validate_phase_range(phase: ArrayFloat) -> None:
    """Validate phase values are in [-π, π] range.
    
    Args:
        phase: Phase array in radians
    
    Raises:
        PhaseRangeError: If any phase values are outside [-π, π]
    
    Examples:
        >>> phase = np.array([0, np.pi/2, np.pi, -np.pi])
        >>> validate_phase_range(phase)  # OK
        >>> 
        >>> invalid_phase = np.array([0, 4*np.pi])
        >>> validate_phase_range(invalid_phase)  # Raises PhaseRangeError
    """
```

**Usage**:
```python
from hologen.utils.fields import validate_phase_range, PhaseRangeError
import numpy as np

# Valid phase
phase = np.random.uniform(-np.pi, np.pi, (512, 512))
validate_phase_range(phase)  # No error

# Invalid phase
try:
    invalid_phase = np.array([0, 5*np.pi])
    validate_phase_range(invalid_phase)
except PhaseRangeError as e:
    print(f"Validation failed: {e}")
```

#### Exception Classes

```python
class FieldRepresentationError(ValueError):
    """Raised when field representation is invalid or incompatible."""
    pass

class PhaseRangeError(ValueError):
    """Raised when phase values are outside [-π, π] range."""
    pass
```

### Shape Generators (hologen.shapes)

#### BaseShapeGenerator.generate_complex()

Generate complex-valued object field.

```python
def generate_complex(
    self,
    grid: GridSpec,
    rng: Generator,
    phase_shift: float = 0.0,
    mode: str = "amplitude"
) -> ArrayComplex:
    """Generate complex-valued object field.
    
    Args:
        grid: Grid specification defining spatial dimensions
        rng: NumPy random number generator
        phase_shift: Phase modulation in radians for phase-only mode
        mode: Generation mode - "amplitude" or "phase"
    
    Returns:
        Complex field array with shape [grid.height, grid.width]
    
    Raises:
        ValueError: If mode is invalid
        PhaseRangeError: If phase_shift is outside [0, 2π]
    
    Examples:
        >>> generator = CircleGenerator(radius_range=(20e-6, 50e-6))
        >>> grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
        >>> rng = np.random.default_rng(42)
        >>> 
        >>> # Amplitude-only object
        >>> amp_field = generator.generate_complex(grid, rng, mode="amplitude")
        >>> 
        >>> # Phase-only object
        >>> phase_field = generator.generate_complex(
        ...     grid, rng, phase_shift=np.pi/2, mode="phase"
        ... )
    """
```

**Available generators**:
- `CircleGenerator`
- `RectangleGenerator`
- `RingGenerator`
- `CircleCheckerGenerator`
- `RectangleCheckerGenerator`
- `EllipseCheckerGenerator`

**Usage**:
```python
from hologen.shapes import CircleGenerator, RectangleGenerator
from hologen.types import GridSpec
import numpy as np

grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
rng = np.random.default_rng(42)

# Circle with amplitude modulation
circle_gen = CircleGenerator(radius_range=(20e-6, 50e-6))
amp_circle = circle_gen.generate_complex(grid, rng, mode="amplitude")

# Rectangle with phase modulation
rect_gen = RectangleGenerator(width_range=(30e-6, 60e-6), height_range=(30e-6, 60e-6))
phase_rect = rect_gen.generate_complex(grid, rng, phase_shift=np.pi/2, mode="phase")
```

### Converters (hologen.converters)

#### ObjectDomainProducer.generate_complex()

Generate complex object sample.

```python
def generate_complex(self, rng: Generator) -> ComplexObjectSample:
    """Generate a complex object-domain sample.
    
    Args:
        rng: NumPy random number generator
    
    Returns:
        ComplexObjectSample with generated field
    
    Examples:
        >>> producer = ObjectDomainProducer(
        ...     generator=CircleGenerator(radius_range=(20e-6, 50e-6)),
        ...     phase_shift=np.pi/2,
        ...     mode="phase"
        ... )
        >>> sample = producer.generate_complex(rng)
    """
```

#### ObjectToHologramConverter (Updated)

Converter now supports complex fields and OutputConfig.

```python
@dataclass(slots=True)
class ObjectToHologramConverter:
    """Convert object-domain samples to holograms with complex field support.
    
    Attributes:
        strategy_mapping: Mapping from holography method to strategy
        noise_model: Optional noise model to apply
        output_config: Configuration for output representations
    """
    strategy_mapping: dict[HolographyMethod, HolographyStrategy]
    noise_model: NoiseModel | None = None
    output_config: OutputConfig = field(default_factory=OutputConfig)
```

**Updated methods**:
```python
def create_hologram(
    self,
    sample: ComplexObjectSample,
    config: HolographyConfig,
    rng: Generator
) -> ArrayComplex:
    """Generate complex hologram from complex object.
    
    Args:
        sample: Complex object sample
        config: Holography configuration
        rng: Random number generator
    
    Returns:
        Complex hologram field
    """

def reconstruct(
    self,
    hologram: ArrayComplex,
    config: HolographyConfig
) -> ArrayComplex:
    """Reconstruct complex object field from complex hologram.
    
    Args:
        hologram: Complex hologram field
        config: Holography configuration
    
    Returns:
        Complex reconstruction field
    """
```

### I/O Utilities (hologen.utils.io)

#### ComplexFieldWriter

Writer for complex field data.

```python
@dataclass(slots=True)
class ComplexFieldWriter:
    """Write complex field data to NumPy archives and PNG previews.
    
    Attributes:
        save_preview: Whether to save PNG preview images
        phase_colormap: Matplotlib colormap name for phase visualization
    
    Examples:
        >>> writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
        >>> writer.save(samples, output_dir=Path("dataset"))
    """
    save_preview: bool = True
    phase_colormap: str = "twilight"
    
    def save(
        self,
        samples: Iterable[ComplexHologramSample],
        output_dir: Path
    ) -> None:
        """Write complex hologram samples to disk.
        
        Args:
            samples: Iterable of complex hologram samples
            output_dir: Output directory path
        
        Creates directory structure:
            output_dir/
            ├── npz/
            │   ├── *_object.npz
            │   ├── *_hologram.npz
            │   └── *_reconstruction.npz
            └── preview/
                ├── object/
                ├── hologram/
                └── reconstruction/
        """
```

**Usage**:
```python
from hologen.utils.io import ComplexFieldWriter
from pathlib import Path

# Create writer
writer = ComplexFieldWriter(
    save_preview=True,
    phase_colormap="twilight"  # or "hsv", "twilight_shifted", etc.
)

# Write samples
writer.save(samples, output_dir=Path("my_dataset"))
```

#### load_complex_sample()

Load sample with automatic format detection.

```python
def load_complex_sample(path: Path) -> ComplexObjectSample | ObjectSample:
    """Load sample with automatic format detection.
    
    Args:
        path: Path to .npz file
    
    Returns:
        ComplexObjectSample if file contains complex data,
        ObjectSample if file contains legacy intensity data
    
    Raises:
        ValueError: If file format is unrecognized
    
    Examples:
        >>> sample = load_complex_sample(Path("dataset/npz/sample_00000_circle_object.npz"))
        >>> if isinstance(sample, ComplexObjectSample):
        ...     print(f"Complex field: {sample.field.shape}")
        ... else:
        ...     print(f"Intensity field: {sample.pixels.shape}")
    """
```

**Usage**:
```python
from hologen.utils.io import load_complex_sample
from hologen.types import ComplexObjectSample, ObjectSample
from pathlib import Path

# Load sample (auto-detects format)
sample = load_complex_sample(Path("dataset/npz/sample_00000_circle_object.npz"))

# Handle both formats
if isinstance(sample, ComplexObjectSample):
    print(f"Complex sample: {sample.representation}")
    field = sample.field
else:
    print("Legacy intensity sample")
    intensity = sample.pixels
```

### Protocol Updates

#### HolographyStrategy (Updated)

Protocol for holography strategies now uses complex fields.

```python
class HolographyStrategy(Protocol):
    """Protocol for holography strategy implementations."""
    
    def create_hologram(
        self,
        object_field: ArrayComplex,  # Updated: was ArrayFloat
        config: HolographyConfig
    ) -> ArrayComplex:  # Updated: was ArrayFloat
        """Create complex hologram field from complex object field."""
        ...
    
    def reconstruct(
        self,
        hologram: ArrayComplex,  # Updated: was ArrayFloat
        config: HolographyConfig
    ) -> ArrayComplex:  # Updated: was ArrayFloat
        """Reconstruct complex object field from complex hologram."""
        ...
```

### Migration Guide

#### From Intensity-Only to Complex Fields

**Old code** (intensity-only):
```python
from hologen.converters import generate_dataset
from hologen.utils.io import NumpyDatasetWriter

# Generate intensity-only dataset
generate_dataset(
    count=100,
    config=config,
    rng=rng,
    writer=NumpyDatasetWriter(save_preview=True),
    output_dir=Path("dataset")
)
```

**New code** (complex fields):
```python
from hologen.converters import generate_dataset
from hologen.utils.io import ComplexFieldWriter
from hologen.types import OutputConfig, FieldRepresentation

# Configure complex output
output_config = OutputConfig(
    object_representation=FieldRepresentation.PHASE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# Generate complex dataset
generate_dataset(
    count=100,
    config=config,
    rng=rng,
    writer=ComplexFieldWriter(save_preview=True),
    output_dir=Path("dataset"),
    output_config=output_config
)
```

#### Backward Compatibility

All existing code continues to work without modification:

```python
# This still works (intensity-only, default behavior)
generate_dataset(
    count=100,
    config=config,
    rng=rng,
    writer=NumpyDatasetWriter(save_preview=True),
    output_dir=Path("dataset")
)
```

### Type Aliases

```python
from numpy.typing import NDArray
import numpy as np

# Array type aliases
ArrayFloat = NDArray[np.float64]      # Real-valued arrays
ArrayComplex = NDArray[np.complex128]  # Complex-valued arrays
```

### Complete API Example

```python
from hologen import *
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter
from hologen.holography.inline import InlineHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, HolographyMethod,
    FieldRepresentation, OutputConfig, ComplexObjectSample
)
from hologen.utils.fields import complex_to_representation, validate_phase_range
from hologen.utils.io import ComplexFieldWriter
from pathlib import Path
import numpy as np

# 1. Configure system
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

# 2. Configure output
output_config = OutputConfig(
    object_representation=FieldRepresentation.PHASE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# 3. Create components
generator = CircleGenerator(radius_range=(20e-6, 50e-6))
producer = ObjectDomainProducer(generator=generator, phase_shift=np.pi/2, mode="phase")
strategy = InlineHolographyStrategy()
converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.INLINE: strategy},
    output_config=output_config
)

# 4. Generate sample
rng = np.random.default_rng(42)
object_sample = producer.generate_complex(rng)

# 5. Validate
phase = np.angle(object_sample.field)
validate_phase_range(phase)

# 6. Create hologram
hologram = converter.create_hologram(object_sample, config, rng)

# 7. Reconstruct
reconstruction = converter.reconstruct(hologram, config)

# 8. Convert representations
hologram_intensity = complex_to_representation(hologram, FieldRepresentation.INTENSITY)
hologram_amplitude = complex_to_representation(hologram, FieldRepresentation.AMPLITUDE)
hologram_phase = complex_to_representation(hologram, FieldRepresentation.PHASE)

# 9. Save
writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
samples = [ComplexHologramSample(
    object_sample=object_sample,
    hologram_field=hologram,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_field=reconstruction,
    reconstruction_representation=FieldRepresentation.COMPLEX
)]
writer.save(samples, output_dir=Path("output"))
```

## Visual Examples

This section provides visual examples demonstrating the differences between field representations and object types.

### Generating Visual Examples

To generate visual comparison images, install matplotlib and run the provided script:

```bash
pip install matplotlib
python scripts/generate_visual_examples.py
```

This will create comparison images in `docs/examples/complex_fields/` showing:
- Amplitude-only vs phase-only objects
- Intensity vs complex field representations
- Complete hologram generation and reconstruction pipeline

### Field Representation Comparison

For a phase-only circular object, the four representations show:

**Intensity (|E|²)**:
- Uniform brightness (no contrast!)
- Phase information is lost
- Cannot distinguish object from background
- This is what a camera would record

**Amplitude (|E|)**:
- Uniform amplitude (value = 1.0 everywhere)
- Still no contrast for phase-only objects
- Square root of intensity

**Phase (arg(E))**:
- Clear circular pattern visible
- Phase shift of π/2 inside circle, 0 outside
- Values range from -π to π radians
- Contains the actual object information

**Complex (Real + Imaginary)**:
- Complete field information preserved
- Can be visualized as amplitude with phase color overlay
- Enables full wave optics processing
- Required for physics-aware ML models

### Object Type Comparison

#### Amplitude-Only Object (Absorbing Circle)

An amplitude-only object modulates light intensity through absorption:

- **Intensity**: Dark circle on bright background (high contrast)
- **Amplitude**: Smooth transition from 0 (inside) to 1 (outside)
- **Phase**: Uniform zero everywhere (no phase modulation)
- **Complex**: Real-valued field (imaginary part is zero)

**Physical example**: Stained biological sample, printed pattern, metal particle

#### Phase-Only Object (Transparent Circle)

A phase-only object modulates light phase without absorption:

- **Intensity**: Uniform brightness (NO contrast - invisible!)
- **Amplitude**: Uniform value of 1.0 everywhere
- **Phase**: Step function (0 outside, π/2 inside circle)
- **Complex**: Pure phase modulation (|E| = 1)

**Physical example**: Unstained biological cell, phase mask, transparent polymer

### Why Intensity-Only Fails for Phase Objects

For phase-only objects, intensity-based imaging provides no contrast:

```
Phase-only object: E = exp(iφ)
Intensity: I = |E|² = |exp(iφ)|² = 1 (uniform!)
```

The object is **invisible** in intensity! However, after propagation through holography:

```
After propagation: E' = F⁻¹[F[E] × H]
Intensity: I' = |E'|² (now shows interference pattern)
```

The hologram intensity shows interference fringes that encode the phase information.

### Complete Pipeline Example

A typical hologram generation pipeline for a phase-only object:

1. **Object Field**: Phase-only circle (uniform amplitude, varying phase)
   - Amplitude: 1.0 everywhere
   - Phase: 0 outside, π/2 inside

2. **Hologram Field**: After propagation to sensor plane
   - Amplitude: Interference pattern (fringes visible)
   - Phase: Complex phase distribution
   - Intensity: Shows characteristic hologram pattern

3. **Reconstruction Field**: After back-propagation
   - Amplitude: Recovered (close to 1.0)
   - Phase: Recovered (close to original 0 and π/2)
   - Quality depends on propagation distance and noise

### Intensity vs Complex Comparison

**Intensity-Only Workflow** (Legacy):
```
Phase Object → [Propagate] → Hologram Intensity → [ML Model] → Reconstruction
                              ↑
                              Phase information lost here!
```

**Complex Field Workflow** (New):
```
Phase Object → [Propagate] → Complex Hologram → [ML Model] → Complex Reconstruction
                              ↑
                              Full field information preserved!
```

### Key Observations

1. **Phase-only objects are invisible in intensity**: You need holographic propagation to create contrast

2. **Complex fields preserve all information**: Both amplitude and phase are available for processing

3. **Hologram patterns differ by object type**:
   - Amplitude objects: Fresnel diffraction pattern
   - Phase objects: Interference fringes
   - Mixed objects: Combination of both

4. **Reconstruction quality**: Complex field reconstruction can recover both amplitude and phase, while intensity-only reconstruction can only recover intensity

### Practical Implications for ML

**For intensity-only models**:
- Can learn hologram → intensity reconstruction
- Cannot recover phase information
- Limited to amplitude/intensity objects
- Simpler data format (1 channel)

**For complex field models**:
- Can learn hologram → full field reconstruction
- Can recover both amplitude and phase
- Works with all object types
- Richer data format (2 channels: real + imaginary)
- Enables physics-informed architectures

### Example Visualizations

The `generate_visual_examples.py` script creates the following comparison images:

1. **amplitude_only_object.png**: Shows all four representations of an absorbing circle
2. **phase_only_object.png**: Shows all four representations of a transparent circle
3. **example_object.png**: Phase-only object field
4. **example_hologram.png**: Hologram field after propagation
5. **example_reconstruction.png**: Reconstructed field after back-propagation
6. **intensity_vs_complex_comparison.png**: Side-by-side comparison showing why complex fields matter

These images demonstrate:
- Why phase-only objects need complex field support
- How information is preserved through the pipeline
- The difference between intensity-only and complex representations
- What ML models can learn from each representation type

### Colormap Reference

For phase visualization, common colormaps include:

- **twilight**: Cyclic colormap, good for phase (default)
- **hsv**: Classic phase colormap (hue = phase)
- **twilight_shifted**: Shifted version of twilight
- **cyclic**: Generic cyclic colormap

Example of setting colormap:
```python
writer = ComplexFieldWriter(phase_colormap="twilight")
```

Or for custom visualization:
```python
import matplotlib.pyplot as plt
phase = np.angle(field)
plt.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
plt.colorbar(label='Phase (radians)')
```
