# Practical Examples and Recipes

This document provides practical, copy-paste examples for common HoloGen use cases. Examples progress from basic to advanced, covering dataset generation, data loading, pipeline customization, and ML framework integration.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Intermediate Examples](#intermediate-examples)
- [Advanced Examples](#advanced-examples)
- [Research Examples](#research-examples)
- [Code Recipes](#code-recipes)

## Basic Examples

### Example 1: Generate a Simple Dataset

Generate a basic hologram dataset with default settings using the command-line interface.

**Use case**: Quick start for testing or learning HoloGen.

**Command**:
```bash
python scripts/generate_dataset.py --samples 100 --output ./my_first_dataset
```

**What you get**:
- 100 samples with random shapes (circles, rectangles, rings)
- Inline holography method
- 256×256 pixel resolution
- 532 nm wavelength (green laser)
- 20 mm propagation distance
- NPZ files with object, hologram, and reconstruction
- PNG preview images

**Output structure**:
```
my_first_dataset/
├── npz/
│   ├── sample_00000_circle.npz
│   ├── sample_00001_rectangle.npz
│   └── ...
└── preview/
    ├── object/
    ├── hologram/
    └── reconstruction/
```

**See also**: [Quickstart Guide](QUICKSTART.md)


### Example 2: Load and Visualize Samples

Load generated samples and visualize them using matplotlib.

**Use case**: Inspect dataset quality and verify generation parameters.

**Code**:
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load a sample
data_path = Path("my_first_dataset/npz/sample_00000_circle.npz")
data = np.load(data_path)

# Extract arrays
object_img = data['object']
hologram = data['hologram']
reconstruction = data['reconstruction']

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(object_img, cmap='gray')
axes[0].set_title('Object Domain')
axes[0].axis('off')

axes[1].imshow(hologram, cmap='gray')
axes[1].set_title('Hologram')
axes[1].axis('off')

axes[2].imshow(reconstruction, cmap='gray')
axes[2].set_title('Reconstruction')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Object shape: {object_img.shape}")
print(f"Hologram intensity range: [{hologram.min():.3f}, {hologram.max():.3f}]")
print(f"Reconstruction quality (MSE): {np.mean((object_img - reconstruction)**2):.6f}")
```

**Expected output**:
```
Object shape: (256, 256)
Hologram intensity range: [0.000, 1.000]
Reconstruction quality (MSE): 0.001234
```

**Tips**:
- Use `cmap='viridis'` for better contrast on holograms
- Check reconstruction MSE to verify holography quality
- Inspect hologram for expected interference patterns


### Example 3: Train a Basic Reconstruction Model

Train a simple CNN to reconstruct objects from holograms using PyTorch.

**Use case**: Baseline ML model for holographic reconstruction.

**Code**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# Define dataset
class HologramDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = sorted(Path(data_dir).glob("*.npz"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        hologram = torch.from_numpy(data['hologram']).float().unsqueeze(0)
        target = torch.from_numpy(data['object']).float().unsqueeze(0)
        return hologram, target

# Simple U-Net style model
class SimpleReconstructionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training setup
dataset = HologramDataset('my_first_dataset/npz')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = SimpleReconstructionNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(10):
    model.train()
    train_loss = 0.0
    for holograms, targets in train_loader:
        holograms, targets = holograms.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(holograms)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for holograms, targets in val_loader:
            holograms, targets = holograms.to(device), targets.to(device)
            outputs = model(holograms)
            val_loss += criterion(outputs, targets).item()
    
    print(f"Epoch {epoch+1}/10 - Train Loss: {train_loss/len(train_loader):.6f}, "
          f"Val Loss: {val_loss/len(val_loader):.6f}")

# Save model
torch.save(model.state_dict(), 'reconstruction_model.pth')
print("Model saved to reconstruction_model.pth")
```

**Expected output**:
```
Epoch 1/10 - Train Loss: 0.045123, Val Loss: 0.038456
Epoch 2/10 - Train Loss: 0.032145, Val Loss: 0.029876
...
Epoch 10/10 - Train Loss: 0.012345, Val Loss: 0.015678
Model saved to reconstruction_model.pth
```

**Tips**:
- Start with a small dataset (100-1000 samples) for quick iteration
- Monitor validation loss to detect overfitting
- Experiment with different architectures (U-Net, ResNet, Transformer)
- Use data augmentation (rotation, flip) to improve generalization


## Intermediate Examples

### Example 4: Custom Shape Generator

Create a custom shape generator for star-shaped objects.

**Use case**: Generate domain-specific object patterns not included in default generators.

**Code**:
```python
from hologen.shapes import BaseShapeGenerator
from hologen.types import GridSpec, ArrayFloat, ArrayComplex
from numpy.random import Generator
import numpy as np

class StarGenerator(BaseShapeGenerator):
    """Generate star-shaped objects with configurable points."""
    
    __slots__ = ("min_radius", "max_radius", "num_points")
    
    def __init__(self, name: str = "star", min_radius: float = 0.1, 
                 max_radius: float = 0.2, num_points: int = 5):
        super().__init__(name=name)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_points = num_points
    
    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        """Generate a star-shaped binary amplitude pattern."""
        canvas = self._empty_canvas(grid)
        
        # Random parameters
        outer_radius = rng.uniform(self.min_radius, self.max_radius) * min(grid.height, grid.width)
        inner_radius = outer_radius * 0.4
        center_y = rng.uniform(0.3, 0.7) * grid.height
        center_x = rng.uniform(0.3, 0.7) * grid.width
        rotation = rng.uniform(0, 2 * np.pi)
        
        # Create star vertices (alternating outer and inner radii)
        angles = np.linspace(0, 2*np.pi, 2*self.num_points, endpoint=False) + rotation
        radii = np.tile([outer_radius, inner_radius], self.num_points)
        
        vertices_y = center_y + radii * np.sin(angles)
        vertices_x = center_x + radii * np.cos(angles)
        
        # Fill star using polygon rasterization
        yy, xx = np.ogrid[:grid.height, :grid.width]
        
        # Simple point-in-polygon test (for each pixel, check if inside star)
        for i in range(len(vertices_y)):
            j = (i + 1) % len(vertices_y)
            # Create triangle from center to edge
            mask = self._point_in_triangle(
                xx, yy, 
                center_x, center_y,
                vertices_x[i], vertices_y[i],
                vertices_x[j], vertices_y[j]
            )
            canvas[mask] = 1.0
        
        return self._clamp(canvas)
    
    def _point_in_triangle(self, px, py, x1, y1, x2, y2, x3, y3):
        """Check if points (px, py) are inside triangle (x1,y1)-(x2,y2)-(x3,y3)."""
        def sign(px, py, x1, y1, x2, y2):
            return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
        
        d1 = sign(px, py, x1, y1, x2, y2)
        d2 = sign(px, py, x2, y2, x3, y3)
        d3 = sign(px, py, x3, y3, x1, y1)
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        
        return ~(has_neg & has_pos)

# Use custom generator in pipeline
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter, HologramDatasetGenerator
from hologen.converters import default_converter
from hologen.types import HolographyConfig, HolographyMethod, OpticalConfig
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path

# Create custom producer with star generator
star_gen = StarGenerator(name="star", min_radius=0.1, max_radius=0.2, num_points=5)
producer = ObjectDomainProducer(shape_generators=(star_gen,))

# Create pipeline
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

converter = default_converter()
generator = HologramDatasetGenerator(object_producer=producer, converter=converter)

# Generate dataset with star shapes
rng = np.random.default_rng(42)
samples = list(generator.generate(count=50, config=config, rng=rng))

# Save dataset
writer = NumpyDatasetWriter(save_preview=True)
writer.save(samples, output_dir=Path("star_dataset"))

print(f"Generated {len(samples)} star-shaped samples")
```

**Tips**:
- Inherit from `BaseShapeGenerator` for helper methods
- Use `_empty_canvas()` and `_clamp()` for consistent behavior
- Test your generator with various random seeds
- Validate output is binary (values in {0.0, 1.0})

**See also**: [Shape Generators Documentation](SHAPES.md)


### Example 5: Custom Noise Model

Create a custom noise model simulating atmospheric turbulence.

**Use case**: Add application-specific noise not covered by built-in models.

**Code**:
```python
from hologen.noise.base import BaseNoiseModel
from hologen.types import ArrayFloat, HolographyConfig
from numpy.random import Generator
import numpy as np
from scipy.ndimage import gaussian_filter

class AtmosphericTurbulenceModel(BaseNoiseModel):
    """Simulate atmospheric turbulence using phase screens."""
    
    __slots__ = ("turbulence_strength", "correlation_length")
    
    def __init__(self, name: str = "turbulence", 
                 turbulence_strength: float = 0.5,
                 correlation_length: float = 10.0):
        super().__init__(name=name)
        self.turbulence_strength = turbulence_strength
        self.correlation_length = correlation_length
    
    def apply(self, hologram: ArrayFloat, config: HolographyConfig, 
              rng: Generator) -> ArrayFloat:
        """Apply atmospheric turbulence to hologram intensity."""
        
        # Generate random phase screen
        phase_screen = rng.normal(0, 1, hologram.shape)
        
        # Apply spatial correlation
        phase_screen = gaussian_filter(phase_screen, sigma=self.correlation_length)
        
        # Scale by turbulence strength
        phase_screen *= self.turbulence_strength
        
        # Convert hologram intensity to complex field
        amplitude = np.sqrt(hologram)
        
        # Apply phase distortion
        distorted_field = amplitude * np.exp(1j * phase_screen)
        
        # Return distorted intensity
        return np.abs(distorted_field) ** 2

# Use custom noise model in pipeline
from hologen.converters import (
    ObjectDomainProducer, ObjectToHologramConverter, 
    HologramDatasetGenerator, default_object_producer
)
from hologen.holography.inline import InlineHolographyStrategy
from hologen.types import HolographyMethod
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path

# Create noise model
turbulence = AtmosphericTurbulenceModel(
    name="turbulence",
    turbulence_strength=0.3,
    correlation_length=15.0
)

# Create converter with custom noise
strategies = {HolographyMethod.INLINE: InlineHolographyStrategy()}
converter = ObjectToHologramConverter(
    strategy_mapping=strategies,
    noise_model=turbulence
)

# Create pipeline
producer = default_object_producer()
generator = HologramDatasetGenerator(object_producer=producer, converter=converter)

# Generate dataset with turbulence
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

rng = np.random.default_rng(42)
samples = list(generator.generate(count=100, config=config, rng=rng))

# Save dataset
writer = NumpyDatasetWriter(save_preview=True)
writer.save(samples, output_dir=Path("turbulent_dataset"))

print(f"Generated {len(samples)} samples with atmospheric turbulence")
```

**Tips**:
- Inherit from `BaseNoiseModel` for consistent interface
- Implement `apply()` method taking hologram, config, and RNG
- Use scipy filters for spatial correlations
- Validate noise doesn't create negative intensities
- Combine with built-in noise using `CompositeNoiseModel`

**See also**: [Noise Simulation Documentation](NOISE_SIMULATION.md)


### Example 6: Pipeline Customization

Customize the generation pipeline to add preprocessing and postprocessing steps.

**Use case**: Add custom transformations or filtering to generated samples.

**Code**:
```python
from hologen.converters import (
    ObjectDomainProducer, ObjectToHologramConverter, 
    HologramDatasetGenerator, default_object_producer, default_converter
)
from hologen.types import HologramSample, HolographyConfig
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from collections.abc import Iterable

class CustomHologramGenerator(HologramDatasetGenerator):
    """Extended generator with preprocessing and postprocessing."""
    
    def __init__(self, object_producer, converter, 
                 blur_sigma: float = 0.0,
                 normalize_output: bool = True):
        super().__init__(object_producer, converter)
        self.blur_sigma = blur_sigma
        self.normalize_output = normalize_output
    
    def generate(self, count: int, config: HolographyConfig, 
                 rng, **kwargs) -> Iterable[HologramSample]:
        """Generate samples with custom processing."""
        
        for sample in super().generate(count, config, rng, **kwargs):
            # Preprocessing: blur object slightly
            if self.blur_sigma > 0:
                blurred_object = gaussian_filter(
                    sample.object_sample.pixels, 
                    sigma=self.blur_sigma
                )
                sample.object_sample.pixels[:] = blurred_object
            
            # Postprocessing: normalize hologram to [0, 1]
            if self.normalize_output:
                hologram_min = sample.hologram.min()
                hologram_max = sample.hologram.max()
                if hologram_max > hologram_min:
                    sample.hologram[:] = (sample.hologram - hologram_min) / (hologram_max - hologram_min)
            
            yield sample

# Create custom pipeline
producer = default_object_producer()
converter = default_converter()

custom_generator = CustomHologramGenerator(
    object_producer=producer,
    converter=converter,
    blur_sigma=1.0,  # Slight blur on objects
    normalize_output=True  # Normalize holograms
)

# Generate dataset
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

rng = np.random.default_rng(42)
samples = list(custom_generator.generate(count=100, config=config, rng=rng))

# Save dataset
writer = NumpyDatasetWriter(save_preview=True)
writer.save(samples, output_dir=Path("custom_pipeline_dataset"))

print(f"Generated {len(samples)} samples with custom pipeline")
```

**Tips**:
- Extend `HologramDatasetGenerator` for custom generation logic
- Use `super().generate()` to leverage existing pipeline
- Modify samples in-place or create new instances
- Add validation checks for custom transformations
- Document custom parameters clearly

**See also**: [Pipeline Architecture Documentation](PIPELINE.md)


## Advanced Examples

### Example 7: Multi-Scale Datasets

Generate datasets at multiple resolutions for multi-scale training.

**Use case**: Train models that work across different image resolutions or scales.

**Code**:
```python
from hologen.converters import (
    HologramDatasetGenerator, default_object_producer, default_converter
)
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np

def generate_multiscale_dataset(scales: list[int], samples_per_scale: int, 
                                base_output_dir: Path):
    """Generate datasets at multiple resolutions."""
    
    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    rng = np.random.default_rng(42)
    
    for scale in scales:
        print(f"Generating {samples_per_scale} samples at {scale}x{scale} resolution...")
        
        # Create grid for this scale
        grid = GridSpec(height=scale, width=scale, pixel_pitch=6.4e-6)
        config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
        
        # Generate samples
        samples = list(generator.generate(count=samples_per_scale, config=config, rng=rng))
        
        # Save to scale-specific directory
        output_dir = base_output_dir / f"scale_{scale}"
        writer = NumpyDatasetWriter(save_preview=True)
        writer.save(samples, output_dir=output_dir)
        
        print(f"  Saved to {output_dir}")

# Generate multi-scale dataset
scales = [128, 256, 512, 1024]
generate_multiscale_dataset(
    scales=scales,
    samples_per_scale=50,
    base_output_dir=Path("multiscale_dataset")
)

print("Multi-scale dataset generation complete!")
```

**Output structure**:
```
multiscale_dataset/
├── scale_128/
│   ├── npz/
│   └── preview/
├── scale_256/
│   ├── npz/
│   └── preview/
├── scale_512/
│   ├── npz/
│   └── preview/
└── scale_1024/
    ├── npz/
    └── preview/
```

**Tips**:
- Keep pixel pitch constant across scales for physical consistency
- Use same random seed for reproducibility across scales
- Consider memory constraints for large resolutions (1024+)
- Train models with mixed-scale batches for scale invariance


### Example 8: Hybrid Object Types

Generate datasets mixing amplitude-only and phase-only objects.

**Use case**: Train models robust to different object types (absorbing vs transparent).

**Code**:
```python
from hologen.converters import (
    ObjectDomainProducer, ObjectToHologramConverter, 
    HologramDatasetGenerator, default_converter
)
from hologen.shapes import CircleGenerator, RectangleGenerator
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, HolographyMethod,
    OutputConfig, FieldRepresentation, ComplexHologramSample
)
from hologen.utils.io import ComplexFieldWriter
from pathlib import Path
import numpy as np

def generate_hybrid_dataset(amplitude_samples: int, phase_samples: int, 
                           output_dir: Path):
    """Generate dataset with mixed amplitude and phase objects."""
    
    # Setup
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
    
    # Create generators
    generators = (
        CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18),
        RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.35)
    )
    producer = ObjectDomainProducer(shape_generators=generators)
    
    # Output configuration for complex fields
    output_config = OutputConfig(
        object_representation=FieldRepresentation.COMPLEX,
        hologram_representation=FieldRepresentation.COMPLEX,
        reconstruction_representation=FieldRepresentation.COMPLEX
    )
    
    converter = default_converter()
    converter.output_config = output_config
    
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    
    rng = np.random.default_rng(42)
    all_samples = []
    
    # Generate amplitude-only objects
    print(f"Generating {amplitude_samples} amplitude-only samples...")
    amplitude_samples_list = list(generator.generate(
        count=amplitude_samples,
        config=config,
        rng=rng,
        mode="amplitude",
        use_complex=True
    ))
    all_samples.extend(amplitude_samples_list)
    
    # Generate phase-only objects
    print(f"Generating {phase_samples} phase-only samples...")
    phase_samples_list = list(generator.generate(
        count=phase_samples,
        config=config,
        rng=rng,
        mode="phase",
        phase_shift=np.pi/2,
        use_complex=True
    ))
    all_samples.extend(phase_samples_list)
    
    # Shuffle samples
    rng.shuffle(all_samples)
    
    # Save dataset
    writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
    writer.save(all_samples, output_dir=output_dir)
    
    print(f"Generated {len(all_samples)} hybrid samples")
    print(f"  - {amplitude_samples} amplitude-only")
    print(f"  - {phase_samples} phase-only")

# Generate hybrid dataset
generate_hybrid_dataset(
    amplitude_samples=50,
    phase_samples=50,
    output_dir=Path("hybrid_dataset")
)
```

**Tips**:
- Use `ComplexFieldWriter` for complex field datasets
- Shuffle samples to mix object types
- Track object type in metadata for analysis
- Train models with balanced batches of each type
- Use appropriate loss functions for complex fields

**See also**: [Complex Fields Documentation](COMPLEX_FIELDS.md)


### Example 9: Custom Reconstruction Algorithm

Implement a custom reconstruction algorithm using the generated holograms.

**Use case**: Test novel reconstruction methods or compare with traditional approaches.

**Code**:
```python
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from hologen.types import GridSpec, OpticalConfig

def angular_spectrum_propagation(field: np.ndarray, distance: float, 
                                 wavelength: float, pixel_pitch: float) -> np.ndarray:
    """Propagate complex field using angular spectrum method.
    
    Args:
        field: Complex field to propagate.
        distance: Propagation distance in meters.
        wavelength: Wavelength in meters.
        pixel_pitch: Pixel pitch in meters.
    
    Returns:
        Propagated complex field.
    """
    height, width = field.shape
    
    # Frequency coordinates
    fy = np.fft.fftfreq(height, pixel_pitch)
    fx = np.fft.fftfreq(width, pixel_pitch)
    FX, FY = np.meshgrid(fx, fy)
    
    # Wave number
    k = 2 * np.pi / wavelength
    
    # Transfer function
    kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
    H = np.exp(1j * kz * distance)
    
    # Propagate
    field_fft = fft2(field)
    propagated_fft = field_fft * H
    propagated = ifft2(propagated_fft)
    
    return propagated

def custom_reconstruction(hologram: np.ndarray, grid: GridSpec, 
                         optics: OpticalConfig) -> np.ndarray:
    """Custom reconstruction with preprocessing and filtering.
    
    Args:
        hologram: Hologram intensity.
        grid: Grid specification.
        optics: Optical configuration.
    
    Returns:
        Reconstructed object amplitude.
    """
    # Convert intensity to complex field (assume amplitude from sqrt)
    hologram_field = np.sqrt(hologram).astype(np.complex128)
    
    # Apply preprocessing: background subtraction
    background = np.median(hologram)
    hologram_field = hologram_field - background
    
    # Propagate back to object plane
    reconstructed = angular_spectrum_propagation(
        hologram_field,
        distance=-optics.propagation_distance,  # Negative for back-propagation
        wavelength=optics.wavelength,
        pixel_pitch=grid.pixel_pitch
    )
    
    # Apply postprocessing: Wiener filter
    amplitude = np.abs(reconstructed)
    
    # Frequency domain filtering
    amplitude_fft = fft2(amplitude)
    
    # Create low-pass filter
    fy = np.fft.fftfreq(grid.height, grid.pixel_pitch)
    fx = np.fft.fftfreq(grid.width, grid.pixel_pitch)
    FX, FY = np.meshgrid(fx, fy)
    freq_radius = np.sqrt(FX**2 + FY**2)
    cutoff = 1.0 / (10 * grid.pixel_pitch)  # Cutoff frequency
    filter_mask = np.exp(-(freq_radius / cutoff)**2)
    
    # Apply filter
    filtered_fft = amplitude_fft * filter_mask
    filtered_amplitude = np.abs(ifft2(filtered_fft))
    
    # Normalize
    filtered_amplitude = (filtered_amplitude - filtered_amplitude.min()) / \
                        (filtered_amplitude.max() - filtered_amplitude.min() + 1e-10)
    
    return filtered_amplitude

# Test custom reconstruction
from pathlib import Path

# Load sample
data = np.load("my_first_dataset/npz/sample_00000_circle.npz")
hologram = data['hologram']
ground_truth = data['object']

# Reconstruct using custom algorithm
grid = GridSpec(height=256, width=256, pixel_pitch=4.65e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.02)

reconstruction = custom_reconstruction(hologram, grid, optics)

# Evaluate quality
mse = np.mean((ground_truth - reconstruction)**2)
psnr = 10 * np.log10(1.0 / (mse + 1e-10))

print(f"Reconstruction MSE: {mse:.6f}")
print(f"Reconstruction PSNR: {psnr:.2f} dB")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(ground_truth, cmap='gray')
axes[0].set_title('Ground Truth')
axes[1].imshow(hologram, cmap='gray')
axes[1].set_title('Hologram')
axes[2].imshow(reconstruction, cmap='gray')
axes[2].set_title('Custom Reconstruction')
plt.tight_layout()
plt.savefig('custom_reconstruction.png', dpi=150)
plt.show()
```

**Tips**:
- Use angular spectrum method for accurate propagation
- Add preprocessing (background subtraction, normalization)
- Apply frequency domain filtering to reduce noise
- Compare with built-in reconstruction for validation
- Optimize parameters (filter cutoff, propagation distance)


### Example 10: ML Framework Integration

Integrate HoloGen datasets with popular ML frameworks (PyTorch, TensorFlow, JAX).

**Use case**: Seamless integration into existing ML training pipelines.

**PyTorch Integration**:
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class ComplexHologramDataset(Dataset):
    """PyTorch dataset for complex hologram data with augmentation."""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.samples = sorted(self.data_dir.glob("*_hologram.npz"))
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load hologram
        hologram_data = np.load(self.samples[idx])
        
        if 'real' in hologram_data and 'imag' in hologram_data:
            # Complex representation: stack real and imaginary as channels
            hologram = torch.stack([
                torch.from_numpy(hologram_data['real']).float(),
                torch.from_numpy(hologram_data['imag']).float()
            ], dim=0)
        else:
            # Intensity representation
            hologram = torch.from_numpy(hologram_data['hologram']).float().unsqueeze(0)
        
        # Load object
        object_path = str(self.samples[idx]).replace('_hologram', '_object')
        object_data = np.load(object_path)
        
        if 'real' in object_data and 'imag' in object_data:
            obj = torch.stack([
                torch.from_numpy(object_data['real']).float(),
                torch.from_numpy(object_data['imag']).float()
            ], dim=0)
        else:
            obj = torch.from_numpy(object_data['object']).float().unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            hologram = self.transform(hologram)
            obj = self.transform(obj)
        
        return hologram, obj

# Data augmentation
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
])

# Create DataLoader
dataset = ComplexHologramDataset('hybrid_dataset/npz', transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

print(f"Dataset size: {len(dataset)}")
for batch_idx, (holograms, objects) in enumerate(dataloader):
    print(f"Batch {batch_idx}: holograms {holograms.shape}, objects {objects.shape}")
    if batch_idx >= 2:
        break
```

**TensorFlow Integration**:
```python
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_complex_sample(hologram_path, object_path):
    """Load complex hologram-object pair."""
    # Load hologram
    h_data = np.load(hologram_path.numpy().decode())
    if b'real' in h_data.files and b'imag' in h_data.files:
        hologram = np.stack([h_data['real'], h_data['imag']], axis=-1)
    else:
        hologram = h_data['hologram'][..., np.newaxis]
    
    # Load object
    o_data = np.load(object_path.numpy().decode())
    if b'real' in o_data.files and b'imag' in o_data.files:
        obj = np.stack([o_data['real'], o_data['imag']], axis=-1)
    else:
        obj = o_data['object'][..., np.newaxis]
    
    return hologram.astype(np.float32), obj.astype(np.float32)

def augment(hologram, obj):
    """Apply data augmentation."""
    # Random flip
    if tf.random.uniform(()) > 0.5:
        hologram = tf.image.flip_left_right(hologram)
        obj = tf.image.flip_left_right(obj)
    
    if tf.random.uniform(()) > 0.5:
        hologram = tf.image.flip_up_down(hologram)
        obj = tf.image.flip_up_down(obj)
    
    # Random rotation (90 degree increments)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    hologram = tf.image.rot90(hologram, k=k)
    obj = tf.image.rot90(obj, k=k)
    
    return hologram, obj

# Create dataset
data_dir = Path("hybrid_dataset/npz")
hologram_paths = sorted(data_dir.glob("*_hologram.npz"))
object_paths = [str(p).replace('_hologram', '_object') for p in hologram_paths]

dataset = tf.data.Dataset.from_tensor_slices((
    [str(p) for p in hologram_paths],
    object_paths
))

dataset = dataset.map(
    lambda h, o: tf.py_function(load_complex_sample, [h, o], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print(f"Dataset created with {len(hologram_paths)} samples")
for batch_idx, (holograms, objects) in enumerate(dataset.take(3)):
    print(f"Batch {batch_idx}: holograms {holograms.shape}, objects {objects.shape}")
```

**JAX Integration**:
```python
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple

def load_dataset_jax(data_dir: Path) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load entire dataset into JAX arrays."""
    hologram_files = sorted(Path(data_dir).glob("*_hologram.npz"))
    
    holograms = []
    objects = []
    
    for h_path in hologram_files:
        h_data = np.load(h_path)
        o_path = str(h_path).replace('_hologram', '_object')
        o_data = np.load(o_path)
        
        # Load complex or intensity
        if 'real' in h_data and 'imag' in h_data:
            hologram = np.stack([h_data['real'], h_data['imag']], axis=0)
        else:
            hologram = h_data['hologram'][np.newaxis, ...]
        
        if 'real' in o_data and 'imag' in o_data:
            obj = np.stack([o_data['real'], o_data['imag']], axis=0)
        else:
            obj = o_data['object'][np.newaxis, ...]
        
        holograms.append(hologram)
        objects.append(obj)
    
    return jnp.array(holograms), jnp.array(objects)

def batch_iterator(X: jnp.ndarray, y: jnp.ndarray, 
                  batch_size: int, key: jax.random.PRNGKey) -> Iterator:
    """Create batched iterator with shuffling."""
    n_samples = X.shape[0]
    indices = jax.random.permutation(key, n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield X[batch_indices], y[batch_indices]

# Load dataset
holograms, objects = load_dataset_jax(Path("hybrid_dataset/npz"))
print(f"Loaded dataset: holograms {holograms.shape}, objects {objects.shape}")

# Create batches
key = jax.random.PRNGKey(42)
for batch_idx, (h_batch, o_batch) in enumerate(batch_iterator(holograms, objects, 32, key)):
    print(f"Batch {batch_idx}: holograms {h_batch.shape}, objects {o_batch.shape}")
    if batch_idx >= 2:
        break
```

**Tips**:
- Use framework-specific data augmentation for better performance
- Enable multi-worker data loading for faster training
- Pin memory (PyTorch) or prefetch (TensorFlow) for GPU efficiency
- Handle both complex and intensity formats in loaders
- Normalize data appropriately for your model architecture


## Research Examples

### Example 11: Reproducing Experimental Conditions

Generate synthetic data matching specific experimental parameters.

**Use case**: Create training data that matches your lab's holography setup.

**Code**:
```python
from hologen.converters import (
    HologramDatasetGenerator, default_object_producer, default_converter,
    create_noise_model
)
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, HolographyMethod,
    NoiseConfig, OffAxisCarrier
)
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np

def generate_experimental_dataset(output_dir: Path):
    """Generate dataset matching experimental setup.
    
    Experimental parameters:
    - Camera: Basler acA2040-90um (2048x2048, 5.5 μm pixels)
    - Laser: 632.8 nm HeNe
    - Distance: 150 mm
    - Off-axis angle: 2.5 degrees
    - Noise: Read noise 3.2 e-, 12-bit ADC, shot noise
    """
    
    # Grid matching camera sensor
    grid = GridSpec(
        height=2048,
        width=2048,
        pixel_pitch=5.5e-6  # 5.5 μm pixels
    )
    
    # Optical parameters matching setup
    optics = OpticalConfig(
        wavelength=632.8e-9,  # HeNe laser
        propagation_distance=0.150  # 150 mm
    )
    
    # Off-axis carrier (2.5 degree angle)
    # carrier_frequency = sin(angle) / wavelength
    angle_rad = np.deg2rad(2.5)
    carrier_freq = np.sin(angle_rad) / optics.wavelength
    
    carrier = OffAxisCarrier(
        frequency_x=carrier_freq,
        frequency_y=0.0,
        gaussian_width=carrier_freq * 0.3  # 30% of carrier for filtering
    )
    
    config = HolographyConfig(
        grid=grid,
        optics=optics,
        method=HolographyMethod.OFF_AXIS,
        carrier=carrier
    )
    
    # Noise matching camera characteristics
    noise_config = NoiseConfig(
        sensor_read_noise=3.2,  # 3.2 electrons RMS
        sensor_shot_noise=True,  # Poisson noise
        sensor_bit_depth=12,  # 12-bit ADC
        speckle_contrast=0.0,  # No speckle in this setup
        aberration_defocus=0.0  # Well-focused system
    )
    
    noise_model = create_noise_model(noise_config)
    
    # Create pipeline
    producer = default_object_producer()
    converter = default_converter()
    converter.noise_model = noise_model
    
    generator = HologramDatasetGenerator(
        object_producer=producer,
        converter=converter
    )
    
    # Generate dataset
    rng = np.random.default_rng(42)
    samples = list(generator.generate(count=500, config=config, rng=rng))
    
    # Save dataset
    writer = NumpyDatasetWriter(save_preview=True)
    writer.save(samples, output_dir=output_dir)
    
    print(f"Generated {len(samples)} samples matching experimental conditions")
    print(f"  Camera: 2048x2048, 5.5 μm pixels")
    print(f"  Laser: 632.8 nm HeNe")
    print(f"  Distance: 150 mm")
    print(f"  Off-axis angle: 2.5°")
    print(f"  Noise: Read 3.2e-, shot noise, 12-bit ADC")

# Generate experimental dataset
generate_experimental_dataset(Path("experimental_match_dataset"))
```

**Tips**:
- Measure your experimental parameters carefully
- Match camera sensor specifications (resolution, pixel size, bit depth)
- Calibrate noise parameters from dark frames and flat fields
- Validate synthetic data against real experimental data
- Document all parameters for reproducibility


### Example 12: Parameter Studies

Systematically vary parameters to study their effects on hologram formation.

**Use case**: Understand how optical parameters affect reconstruction quality.

**Code**:
```python
from hologen.converters import (
    HologramDatasetGenerator, default_object_producer, default_converter
)
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def parameter_study_distance(distances: list[float], samples_per_distance: int,
                            output_dir: Path):
    """Study effect of propagation distance on reconstruction quality."""
    
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    wavelength = 532e-9
    
    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    
    results = []
    
    for distance in distances:
        print(f"Testing distance: {distance*1000:.1f} mm")
        
        optics = OpticalConfig(wavelength=wavelength, propagation_distance=distance)
        config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
        
        # Generate samples
        rng = np.random.default_rng(42)  # Same seed for fair comparison
        samples = list(generator.generate(count=samples_per_distance, config=config, rng=rng))
        
        # Compute reconstruction quality metrics
        mse_values = []
        for sample in samples:
            mse = np.mean((sample.object_sample.pixels - sample.reconstruction)**2)
            mse_values.append(mse)
        
        avg_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        
        results.append({
            'distance': distance,
            'avg_mse': avg_mse,
            'std_mse': std_mse
        })
        
        print(f"  Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        
        # Save samples for this distance
        distance_dir = output_dir / f"distance_{int(distance*1000)}mm"
        writer = NumpyDatasetWriter(save_preview=True)
        writer.save(samples, output_dir=distance_dir)
    
    # Plot results
    distances_mm = [r['distance'] * 1000 for r in results]
    avg_mses = [r['avg_mse'] for r in results]
    std_mses = [r['std_mse'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(distances_mm, avg_mses, yerr=std_mses, marker='o', capsize=5)
    plt.xlabel('Propagation Distance (mm)')
    plt.ylabel('Reconstruction MSE')
    plt.title('Effect of Propagation Distance on Reconstruction Quality')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'distance_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nParameter study complete. Results saved to {output_dir}")
    return results

# Run parameter study
distances = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]  # 10mm to 200mm
results = parameter_study_distance(
    distances=distances,
    samples_per_distance=20,
    output_dir=Path("parameter_study_distance")
)

# Find optimal distance
optimal = min(results, key=lambda x: x['avg_mse'])
print(f"\nOptimal distance: {optimal['distance']*1000:.1f} mm (MSE: {optimal['avg_mse']:.6f})")
```

**Other parameter studies**:
```python
# Study wavelength effect
def parameter_study_wavelength(wavelengths: list[float], ...):
    """Study effect of illumination wavelength."""
    # Similar structure, vary wavelength instead of distance
    pass

# Study pixel pitch effect
def parameter_study_resolution(pixel_pitches: list[float], ...):
    """Study effect of sensor resolution."""
    # Similar structure, vary pixel_pitch instead of distance
    pass

# Study noise effect
def parameter_study_noise(noise_levels: list[float], ...):
    """Study effect of noise on reconstruction."""
    # Similar structure, vary noise parameters
    pass
```

**Tips**:
- Use same random seed across parameter values for fair comparison
- Generate enough samples for statistical significance (20-50 per condition)
- Plot results with error bars to show variability
- Save all datasets for later analysis
- Document parameter ranges and units clearly


### Example 13: Ablation Studies

Systematically remove or modify components to understand their contribution.

**Use case**: Determine which noise sources or pipeline components are most important.

**Code**:
```python
from hologen.converters import (
    HologramDatasetGenerator, default_object_producer, default_converter,
    create_noise_model
)
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod, NoiseConfig
from hologen.noise import SensorNoiseModel, SpeckleNoiseModel, AberrationNoiseModel, CompositeNoiseModel
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def ablation_study_noise(samples_per_condition: int, output_dir: Path):
    """Ablation study: effect of different noise sources."""
    
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
    
    producer = default_object_producer()
    
    # Define noise conditions
    conditions = {
        'no_noise': None,
        'sensor_only': SensorNoiseModel(
            name="sensor",
            read_noise=3.0,
            shot_noise=True,
            dark_current=0.5,
            bit_depth=12
        ),
        'speckle_only': SpeckleNoiseModel(
            name="speckle",
            contrast=0.8,
            correlation_length=1.0
        ),
        'aberration_only': AberrationNoiseModel(
            name="aberration",
            defocus=0.5,
            astigmatism_x=0.2,
            astigmatism_y=0.2,
            coma_x=0.0,
            coma_y=0.0
        ),
        'all_noise': CompositeNoiseModel(
            name="composite",
            models=(
                SensorNoiseModel(name="sensor", read_noise=3.0, shot_noise=True, 
                               dark_current=0.5, bit_depth=12),
                SpeckleNoiseModel(name="speckle", contrast=0.8, correlation_length=1.0),
                AberrationNoiseModel(name="aberration", defocus=0.5, 
                                   astigmatism_x=0.2, astigmatism_y=0.2)
            )
        )
    }
    
    results = {}
    
    for condition_name, noise_model in conditions.items():
        print(f"\nTesting condition: {condition_name}")
        
        # Create converter with this noise model
        converter = default_converter()
        converter.noise_model = noise_model
        
        generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
        
        # Generate samples
        rng = np.random.default_rng(42)  # Same seed for fair comparison
        samples = list(generator.generate(count=samples_per_condition, config=config, rng=rng))
        
        # Compute metrics
        mse_values = []
        snr_values = []
        
        for sample in samples:
            # Reconstruction quality
            mse = np.mean((sample.object_sample.pixels - sample.reconstruction)**2)
            mse_values.append(mse)
            
            # Signal-to-noise ratio
            signal_power = np.mean(sample.hologram**2)
            noise_power = np.var(sample.hologram)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_values.append(snr)
        
        results[condition_name] = {
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'snr_mean': np.mean(snr_values),
            'snr_std': np.std(snr_values)
        }
        
        print(f"  MSE: {results[condition_name]['mse_mean']:.6f} ± {results[condition_name]['mse_std']:.6f}")
        print(f"  SNR: {results[condition_name]['snr_mean']:.2f} ± {results[condition_name]['snr_std']:.2f} dB")
        
        # Save samples
        condition_dir = output_dir / condition_name
        writer = NumpyDatasetWriter(save_preview=True)
        writer.save(samples, output_dir=condition_dir)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    conditions_list = list(results.keys())
    mse_means = [results[c]['mse_mean'] for c in conditions_list]
    mse_stds = [results[c]['mse_std'] for c in conditions_list]
    snr_means = [results[c]['snr_mean'] for c in conditions_list]
    snr_stds = [results[c]['snr_std'] for c in conditions_list]
    
    x_pos = np.arange(len(conditions_list))
    
    ax1.bar(x_pos, mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions_list, rotation=45, ha='right')
    ax1.set_ylabel('Reconstruction MSE')
    ax1.set_title('Effect of Noise on Reconstruction Quality')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x_pos, snr_means, yerr=snr_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions_list, rotation=45, ha='right')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('Effect of Noise on Signal Quality')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAblation study complete. Results saved to {output_dir}")
    return results

# Run ablation study
results = ablation_study_noise(
    samples_per_condition=50,
    output_dir=Path("ablation_study_noise")
)

# Analyze results
print("\n=== Ablation Study Summary ===")
baseline_mse = results['no_noise']['mse_mean']
for condition, metrics in results.items():
    if condition != 'no_noise':
        mse_increase = (metrics['mse_mean'] - baseline_mse) / baseline_mse * 100
        print(f"{condition}: MSE increased by {mse_increase:.1f}%")
```

**Tips**:
- Use same random seed across conditions for fair comparison
- Test one component at a time to isolate effects
- Include a baseline (no noise) condition for reference
- Generate enough samples for statistical significance
- Visualize results with bar plots and error bars
- Document which components have largest impact

**See also**: [Noise Simulation Documentation](NOISE_SIMULATION.md)


## Code Recipes

### Recipe 1: Batch Processing

Generate large datasets efficiently using batch processing.

**Code**:
```python
from hologen.converters import (
    HologramDatasetGenerator, default_object_producer, default_converter
)
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm

def generate_large_dataset_batched(total_samples: int, batch_size: int, 
                                   output_dir: Path):
    """Generate large dataset in batches to manage memory."""
    
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
    
    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    
    writer = NumpyDatasetWriter(save_preview=False)  # Disable previews for speed
    rng = np.random.default_rng(42)
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"Generating {total_samples} samples in {num_batches} batches of {batch_size}")
    
    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        # Determine batch size (last batch may be smaller)
        current_batch_size = min(batch_size, total_samples - batch_idx * batch_size)
        
        # Generate batch
        samples = list(generator.generate(count=current_batch_size, config=config, rng=rng))
        
        # Save batch to separate directory
        batch_dir = output_dir / f"batch_{batch_idx:04d}"
        writer.save(samples, output_dir=batch_dir)
        
        # Clear memory
        del samples
    
    print(f"Dataset generation complete: {total_samples} samples in {output_dir}")

# Generate 10,000 samples in batches of 100
generate_large_dataset_batched(
    total_samples=10000,
    batch_size=100,
    output_dir=Path("large_dataset_batched")
)
```

**Tips**:
- Use batch size of 50-100 for good memory/speed tradeoff
- Disable preview generation for large datasets (use `save_preview=False`)
- Save batches to separate directories for easier management
- Use `tqdm` for progress tracking
- Clear memory between batches with `del samples`


### Recipe 2: Parallel Generation

Generate datasets in parallel using multiprocessing.

**Code**:
```python
from hologen.converters import (
    HologramDatasetGenerator, default_object_producer, default_converter
)
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod
from hologen.utils.io import NumpyDatasetWriter
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

def generate_batch_worker(batch_id: int, samples_per_batch: int, 
                         config: HolographyConfig, seed: int, 
                         output_dir: Path) -> int:
    """Worker function to generate one batch of samples."""
    
    # Create fresh instances for this worker
    producer = default_object_producer()
    converter = default_converter()
    generator = HologramDatasetGenerator(object_producer=producer, converter=converter)
    
    # Use unique seed for this batch
    rng = np.random.default_rng(seed + batch_id)
    
    # Generate samples
    samples = list(generator.generate(count=samples_per_batch, config=config, rng=rng))
    
    # Save batch
    batch_dir = output_dir / f"batch_{batch_id:04d}"
    writer = NumpyDatasetWriter(save_preview=False)
    writer.save(samples, output_dir=batch_dir)
    
    return len(samples)

def generate_dataset_parallel(total_samples: int, num_workers: int, 
                              output_dir: Path, seed: int = 42):
    """Generate dataset using parallel workers."""
    
    # Configuration
    grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
    optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
    config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)
    
    # Determine batch distribution
    samples_per_worker = total_samples // num_workers
    num_batches = num_workers
    
    print(f"Generating {total_samples} samples using {num_workers} parallel workers")
    print(f"Each worker generates {samples_per_worker} samples")
    
    # Create worker function with fixed arguments
    worker_func = partial(
        generate_batch_worker,
        samples_per_batch=samples_per_worker,
        config=config,
        seed=seed,
        output_dir=output_dir
    )
    
    # Run parallel generation
    with Pool(processes=num_workers) as pool:
        results = pool.map(worker_func, range(num_batches))
    
    total_generated = sum(results)
    print(f"Dataset generation complete: {total_generated} samples in {output_dir}")

# Generate 10,000 samples using all available CPU cores
num_workers = cpu_count()
generate_dataset_parallel(
    total_samples=10000,
    num_workers=num_workers,
    output_dir=Path("parallel_dataset"),
    seed=42
)
```

**Tips**:
- Use `cpu_count()` to automatically detect available cores
- Each worker needs unique random seed for diversity
- Disable preview generation for maximum speed
- Monitor memory usage with many workers
- Combine batches after generation if needed

**Combining batches**:
```python
def combine_batches(batched_dir: Path, output_dir: Path):
    """Combine batched datasets into single directory."""
    import shutil
    
    output_npz = output_dir / "npz"
    output_npz.mkdir(parents=True, exist_ok=True)
    
    batch_dirs = sorted(batched_dir.glob("batch_*"))
    
    sample_idx = 0
    for batch_dir in batch_dirs:
        npz_files = sorted((batch_dir / "npz").glob("*.npz"))
        for npz_file in npz_files:
            # Rename with sequential index
            new_name = f"sample_{sample_idx:05d}_{npz_file.stem.split('_', 2)[-1]}.npz"
            shutil.copy(npz_file, output_npz / new_name)
            sample_idx += 1
    
    print(f"Combined {sample_idx} samples into {output_dir}")

# Combine parallel batches
combine_batches(
    batched_dir=Path("parallel_dataset"),
    output_dir=Path("combined_dataset")
)
```


### Recipe 3: Memory-Efficient Loading

Load large datasets efficiently without loading everything into memory.

**Code**:
```python
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple

class MemoryEfficientDataLoader:
    """Memory-efficient data loader using generators."""
    
    def __init__(self, data_dir: Path, batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.sample_files = sorted(self.data_dir.glob("*.npz"))
        self.num_samples = len(self.sample_files)
    
    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches without loading entire dataset."""
        batch_holograms = []
        batch_objects = []
        
        for sample_file in self.sample_files:
            # Load single sample
            data = np.load(sample_file)
            
            if 'hologram' in data:
                # Legacy format
                hologram = data['hologram']
                obj = data['object']
            else:
                # Complex format
                hologram = data['real'] + 1j * data['imag']
                obj_file = str(sample_file).replace('_hologram', '_object')
                obj_data = np.load(obj_file)
                obj = obj_data['real'] + 1j * obj_data['imag']
            
            batch_holograms.append(hologram)
            batch_objects.append(obj)
            
            # Yield batch when full
            if len(batch_holograms) == self.batch_size:
                yield np.array(batch_holograms), np.array(batch_objects)
                batch_holograms = []
                batch_objects = []
        
        # Yield remaining samples
        if batch_holograms:
            yield np.array(batch_holograms), np.array(batch_objects)

# Usage example
loader = MemoryEfficientDataLoader(
    data_dir=Path("large_dataset_batched/batch_0000/npz"),
    batch_size=32
)

print(f"Dataset has {loader.num_samples} samples, {len(loader)} batches")

for batch_idx, (holograms, objects) in enumerate(loader):
    print(f"Batch {batch_idx}: holograms {holograms.shape}, objects {objects.shape}")
    
    # Process batch (e.g., train model)
    # ...
    
    if batch_idx >= 2:
        break
```

**Streaming from multiple directories**:
```python
class MultiDirectoryLoader:
    """Load from multiple batch directories sequentially."""
    
    def __init__(self, base_dir: Path, batch_size: int = 32):
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.batch_dirs = sorted(self.base_dir.glob("batch_*"))
    
    def __iter__(self):
        for batch_dir in self.batch_dirs:
            npz_dir = batch_dir / "npz"
            if npz_dir.exists():
                loader = MemoryEfficientDataLoader(npz_dir, self.batch_size)
                yield from loader

# Usage
multi_loader = MultiDirectoryLoader(
    base_dir=Path("large_dataset_batched"),
    batch_size=32
)

for batch_idx, (holograms, objects) in enumerate(multi_loader):
    print(f"Batch {batch_idx}: {holograms.shape}")
    if batch_idx >= 5:
        break
```

**Tips**:
- Use generators to avoid loading entire dataset into memory
- Process batches one at a time
- Close file handles properly (NumPy does this automatically)
- Consider using memory-mapped arrays for very large files
- Monitor memory usage during training


### Recipe 4: Data Augmentation

Apply data augmentation to increase dataset diversity.

**Code**:
```python
import numpy as np
from scipy.ndimage import rotate, gaussian_filter
from typing import Tuple

class HologramAugmenter:
    """Data augmentation for hologram-object pairs."""
    
    def __init__(self, rotation_range: float = 90.0,
                 flip_horizontal: bool = True,
                 flip_vertical: bool = True,
                 noise_std: float = 0.01,
                 blur_sigma: float = 0.5):
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma
    
    def augment(self, hologram: np.ndarray, obj: np.ndarray, 
                rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation to hologram-object pair."""
        
        # Random rotation (same for both)
        if self.rotation_range > 0:
            angle = rng.uniform(-self.rotation_range, self.rotation_range)
            hologram = rotate(hologram, angle, reshape=False, order=1)
            obj = rotate(obj, angle, reshape=False, order=1)
        
        # Random horizontal flip
        if self.flip_horizontal and rng.random() > 0.5:
            hologram = np.fliplr(hologram)
            obj = np.fliplr(obj)
        
        # Random vertical flip
        if self.flip_vertical and rng.random() > 0.5:
            hologram = np.flipud(hologram)
            obj = np.flipud(obj)
        
        # Add random noise to hologram only
        if self.noise_std > 0:
            noise = rng.normal(0, self.noise_std, hologram.shape)
            hologram = hologram + noise
            hologram = np.clip(hologram, 0, 1)
        
        # Random blur to hologram only
        if self.blur_sigma > 0 and rng.random() > 0.5:
            sigma = rng.uniform(0, self.blur_sigma)
            hologram = gaussian_filter(hologram, sigma=sigma)
        
        return hologram, obj

# Usage with PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class AugmentedHologramDataset(Dataset):
    """Dataset with on-the-fly augmentation."""
    
    def __init__(self, data_dir: Path, augmenter: HologramAugmenter = None):
        self.samples = sorted(Path(data_dir).glob("*.npz"))
        self.augmenter = augmenter
        self.rng = np.random.default_rng()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        hologram = data['hologram']
        obj = data['object']
        
        # Apply augmentation
        if self.augmenter is not None:
            hologram, obj = self.augmenter.augment(hologram, obj, self.rng)
        
        # Convert to tensors
        hologram = torch.from_numpy(hologram).float().unsqueeze(0)
        obj = torch.from_numpy(obj).float().unsqueeze(0)
        
        return hologram, obj

# Create augmented dataset
augmenter = HologramAugmenter(
    rotation_range=45.0,
    flip_horizontal=True,
    flip_vertical=True,
    noise_std=0.01,
    blur_sigma=0.5
)

dataset = AugmentedHologramDataset(
    data_dir=Path("my_first_dataset/npz"),
    augmenter=augmenter
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Visualize augmentation effect
import matplotlib.pyplot as plt

sample_idx = 0
original_data = np.load(dataset.samples[sample_idx])
original_hologram = original_data['hologram']

# Get augmented version
augmented_hologram, _ = dataset[sample_idx]
augmented_hologram = augmented_hologram.squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_hologram, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(augmented_hologram, cmap='gray')
axes[1].set_title('Augmented')
plt.tight_layout()
plt.savefig('augmentation_example.png', dpi=150)
plt.show()
```

**Tips**:
- Apply same geometric transforms (rotation, flip) to both hologram and object
- Apply noise and blur only to hologram (not object)
- Use moderate augmentation to avoid unrealistic samples
- Visualize augmented samples to verify quality
- Disable augmentation during validation/testing


### Recipe 5: Dataset Validation and Quality Checks

Validate generated datasets for quality and correctness.

**Code**:
```python
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

class DatasetValidator:
    """Validate hologram dataset quality."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.sample_files = sorted(self.data_dir.glob("*.npz"))
    
    def validate_all(self) -> Dict[str, any]:
        """Run all validation checks."""
        results = {
            'num_samples': len(self.sample_files),
            'file_integrity': self.check_file_integrity(),
            'value_ranges': self.check_value_ranges(),
            'reconstruction_quality': self.check_reconstruction_quality(),
            'shape_consistency': self.check_shape_consistency(),
            'statistics': self.compute_statistics()
        }
        return results
    
    def check_file_integrity(self) -> Dict[str, any]:
        """Check if all files can be loaded."""
        corrupted = []
        missing_keys = []
        
        for sample_file in self.sample_files:
            try:
                data = np.load(sample_file)
                required_keys = ['object', 'hologram', 'reconstruction']
                
                for key in required_keys:
                    if key not in data:
                        missing_keys.append((sample_file.name, key))
                
            except Exception as e:
                corrupted.append((sample_file.name, str(e)))
        
        return {
            'corrupted_files': corrupted,
            'missing_keys': missing_keys,
            'all_valid': len(corrupted) == 0 and len(missing_keys) == 0
        }
    
    def check_value_ranges(self) -> Dict[str, any]:
        """Check if values are in expected ranges."""
        issues = []
        
        for sample_file in self.sample_files[:10]:  # Check first 10
            data = np.load(sample_file)
            
            # Check for NaN or Inf
            for key in ['object', 'hologram', 'reconstruction']:
                if not np.isfinite(data[key]).all():
                    issues.append(f"{sample_file.name}: {key} contains NaN or Inf")
            
            # Check value ranges
            if data['object'].min() < 0 or data['object'].max() > 1:
                issues.append(f"{sample_file.name}: object values out of [0,1]")
            
            if data['hologram'].min() < 0:
                issues.append(f"{sample_file.name}: hologram has negative values")
        
        return {
            'issues': issues,
            'all_valid': len(issues) == 0
        }
    
    def check_reconstruction_quality(self) -> Dict[str, any]:
        """Check reconstruction quality metrics."""
        mse_values = []
        psnr_values = []
        
        for sample_file in self.sample_files:
            data = np.load(sample_file)
            obj = data['object']
            recon = data['reconstruction']
            
            mse = np.mean((obj - recon)**2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            mse_values.append(mse)
            psnr_values.append(psnr)
        
        return {
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'mse_min': np.min(mse_values),
            'mse_max': np.max(mse_values),
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'poor_quality_count': sum(1 for mse in mse_values if mse > 0.1)
        }
    
    def check_shape_consistency(self) -> Dict[str, any]:
        """Check if all samples have consistent shapes."""
        shapes = set()
        
        for sample_file in self.sample_files:
            data = np.load(sample_file)
            shapes.add(data['object'].shape)
        
        return {
            'unique_shapes': list(shapes),
            'consistent': len(shapes) == 1
        }
    
    def compute_statistics(self) -> Dict[str, any]:
        """Compute dataset statistics."""
        object_means = []
        hologram_means = []
        object_stds = []
        hologram_stds = []
        
        for sample_file in self.sample_files:
            data = np.load(sample_file)
            
            object_means.append(data['object'].mean())
            hologram_means.append(data['hologram'].mean())
            object_stds.append(data['object'].std())
            hologram_stds.append(data['hologram'].std())
        
        return {
            'object_mean': np.mean(object_means),
            'object_std': np.mean(object_stds),
            'hologram_mean': np.mean(hologram_means),
            'hologram_std': np.mean(hologram_stds)
        }
    
    def generate_report(self, output_path: Path):
        """Generate validation report with visualizations."""
        results = self.validate_all()
        
        # Print text report
        print("=" * 60)
        print("DATASET VALIDATION REPORT")
        print("=" * 60)
        print(f"\nDataset: {self.data_dir}")
        print(f"Number of samples: {results['num_samples']}")
        
        print("\n--- File Integrity ---")
        print(f"All files valid: {results['file_integrity']['all_valid']}")
        if results['file_integrity']['corrupted_files']:
            print(f"Corrupted files: {results['file_integrity']['corrupted_files']}")
        
        print("\n--- Value Ranges ---")
        print(f"All values valid: {results['value_ranges']['all_valid']}")
        if results['value_ranges']['issues']:
            for issue in results['value_ranges']['issues']:
                print(f"  - {issue}")
        
        print("\n--- Reconstruction Quality ---")
        print(f"Mean MSE: {results['reconstruction_quality']['mse_mean']:.6f} "
              f"± {results['reconstruction_quality']['mse_std']:.6f}")
        print(f"Mean PSNR: {results['reconstruction_quality']['psnr_mean']:.2f} "
              f"± {results['reconstruction_quality']['psnr_std']:.2f} dB")
        print(f"Poor quality samples (MSE > 0.1): "
              f"{results['reconstruction_quality']['poor_quality_count']}")
        
        print("\n--- Shape Consistency ---")
        print(f"Consistent shapes: {results['shape_consistency']['consistent']}")
        print(f"Shapes: {results['shape_consistency']['unique_shapes']}")
        
        print("\n--- Statistics ---")
        print(f"Object mean: {results['statistics']['object_mean']:.4f}")
        print(f"Object std: {results['statistics']['object_std']:.4f}")
        print(f"Hologram mean: {results['statistics']['hologram_mean']:.4f}")
        print(f"Hologram std: {results['statistics']['hologram_std']:.4f}")
        
        # Generate visualization
        self._plot_quality_distribution(output_path)
        
        print(f"\nValidation report saved to {output_path}")
    
    def _plot_quality_distribution(self, output_path: Path):
        """Plot reconstruction quality distribution."""
        mse_values = []
        
        for sample_file in self.sample_files:
            data = np.load(sample_file)
            mse = np.mean((data['object'] - data['reconstruction'])**2)
            mse_values.append(mse)
        
        plt.figure(figsize=(10, 6))
        plt.hist(mse_values, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction MSE')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Quality')
        plt.axvline(np.mean(mse_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(mse_values):.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'quality_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

# Run validation
validator = DatasetValidator(Path("my_first_dataset/npz"))
validator.generate_report(Path("my_first_dataset"))
```

**Tips**:
- Run validation after generating datasets
- Check for corrupted files, NaN/Inf values, and range violations
- Monitor reconstruction quality metrics (MSE, PSNR)
- Verify shape consistency across samples
- Generate reports for documentation
- Set quality thresholds for automated checks


## Summary

This document provided practical examples covering:

**Basic Examples**:
- Generating simple datasets with CLI
- Loading and visualizing samples
- Training basic reconstruction models

**Intermediate Examples**:
- Creating custom shape generators
- Implementing custom noise models
- Customizing the generation pipeline

**Advanced Examples**:
- Multi-scale dataset generation
- Hybrid object types (amplitude + phase)
- Custom reconstruction algorithms
- ML framework integration (PyTorch, TensorFlow, JAX)

**Research Examples**:
- Reproducing experimental conditions
- Parameter studies for optimization
- Ablation studies for component analysis

**Code Recipes**:
- Batch processing for large datasets
- Parallel generation for speed
- Memory-efficient loading strategies
- Data augmentation techniques
- Dataset validation and quality checks

## Next Steps

- **[Quickstart Guide](QUICKSTART.md)**: Get started quickly with basic usage
- **[API Reference](API_REFERENCE.md)**: Detailed API documentation
- **[Shape Generators](SHAPES.md)**: Learn about available shape generators
- **[Complex Fields](COMPLEX_FIELDS.md)**: Understand complex field representations
- **[Noise Simulation](NOISE_SIMULATION.md)**: Add realistic noise to datasets
- **[Pipeline Architecture](PIPELINE.md)**: Understand the generation pipeline
- **[CLI Reference](CLI_REFERENCE.md)**: Command-line interface documentation

## Contributing Examples

Have a useful example or recipe? Consider contributing:

1. Ensure code is well-documented and tested
2. Include use case description and tips
3. Add expected output or results
4. Follow the established format
5. Submit via pull request

## Getting Help

- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Report bugs or request features on GitHub
- **Community**: Join discussions and share your use cases

## See Also

- **[Quickstart Guide](QUICKSTART.md)** - Quick start before diving into advanced examples
- **[Shape Generators](SHAPES.md)** - Shape generator documentation for Example 4 (custom generators)
- **[Holography Methods](HOLOGRAPHY_METHODS.md)** - Holography methods for Example 9 (custom reconstruction)
- **[Pipeline Architecture](PIPELINE.md)** - Pipeline documentation for Example 6 (pipeline customization)
- **[Noise Simulation](NOISE_SIMULATION.md)** - Noise models for Example 5 (custom noise)
- **[Complex Fields](COMPLEX_FIELDS.md)** - Complex fields for Example 8 (hybrid objects)
- **[I/O Formats](IO_FORMATS.md)** - I/O formats for Examples 2, 3, 10 (data loading)
- **[CLI Reference](CLI_REFERENCE.md)** - CLI reference for Example 1 and batch scripts
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all examples
- **[Utilities](UTILITIES.md)** - Utility functions used in many examples
- **[Master Documentation Index](README.md)** - Complete feature documentation overview

---

**Happy hologram generation!** 🔬✨

