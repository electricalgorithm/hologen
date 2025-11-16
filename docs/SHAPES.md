# Shape Generators in HoloGen

HoloGen provides a suite of shape generators for creating synthetic object-domain patterns in holography datasets. These generators produce binary amplitude distributions and complex fields representing various geometric shapes with randomized parameters, enabling diverse training data for machine learning models.

## Overview

Shape generators are the foundation of HoloGen's dataset generation pipeline. They create object-domain patterns that are then propagated through holographic imaging systems to produce synthetic holograms. Each generator implements the `ObjectShapeGenerator` protocol and can produce both simple binary amplitude patterns and complex-valued fields with amplitude and phase modulation.

### Key Features

- **Randomized parameters**: Each generated shape has randomized size, position, and geometry within configurable ranges
- **Binary amplitude output**: Standard `generate()` method produces binary masks (values in {0.0, 1.0})
- **Complex field support**: `generate_complex()` method creates amplitude-only or phase-only complex fields
- **Reproducible generation**: Uses NumPy's random number generator for deterministic output with seeds
- **Protocol-based design**: All generators implement `ObjectShapeGenerator` protocol for extensibility

### When to Use Shape Generators

- **Training data generation**: Create large synthetic datasets for supervised learning
- **Data augmentation**: Supplement experimental hologram datasets with synthetic samples
- **Algorithm testing**: Validate reconstruction algorithms with known ground truth
- **Parameter studies**: Systematically explore how object properties affect hologram formation
- **Prototyping**: Quickly test holography pipelines before acquiring experimental data

## Available Shape Generators

HoloGen includes seven shape generators covering basic geometric primitives and textured patterns:

| Generator | Description | Use Cases |
|-----------|-------------|-----------|
| `CircleGenerator` | Filled circular discs | Particles, cells, droplets |
| `RectangleGenerator` | Filled rectangles | Apertures, structured objects |
| `RingGenerator` | Annular rings | Ring-shaped particles, hollow structures |
| `CircleCheckerGenerator` | Circles with checkerboard pattern | Textured particles, resolution targets |
| `RectangleCheckerGenerator` | Rectangles with checkerboard pattern | Resolution charts, structured patterns |
| `EllipseCheckerGenerator` | Ellipses with checkerboard pattern | Elongated particles, anisotropic objects |
| `TriangleCheckerGenerator` | Triangles with checkerboard pattern | Angular features, non-circular objects |

## Shape Generator Details

### CircleGenerator

Generates filled circular discs with randomized radius and position.

**Parameters**:
- `name` (str): Identifier for the generator (e.g., "circle")
- `min_radius` (float): Minimum radius as fraction of image dimension (0.0 to 1.0)
- `max_radius` (float): Maximum radius as fraction of image dimension (0.0 to 1.0)

**Valid Ranges**:
- `min_radius`: 0.01 to 0.5 (typical: 0.08 to 0.18)
- `max_radius`: Must be ≥ `min_radius` (typical: 0.08 to 0.25)
- Position: Center randomly placed in central 40% of image (0.3 to 0.7 normalized)

**Physical Interpretation**: Represents spherical particles, biological cells, droplets, or circular apertures.

**Code Example**:
```python
from hologen.shapes import CircleGenerator
from hologen.types import GridSpec
import numpy as np

# Create generator
generator = CircleGenerator(
    name="circle",
    min_radius=0.08,  # 8% of image dimension
    max_radius=0.18   # 18% of image dimension
)

# Generate binary amplitude pattern
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
rng = np.random.default_rng(42)
amplitude = generator.generate(grid, rng)

print(f"Shape: {amplitude.shape}")  # (512, 512)
print(f"Values: {np.unique(amplitude)}")  # [0. 1.]
print(f"Fill fraction: {amplitude.mean():.3f}")  # ~0.025 (2.5% of pixels)
```

### RectangleGenerator

Generates filled rectangles with randomized dimensions and position.

**Parameters**:
- `name` (str): Identifier for the generator (e.g., "rectangle")
- `min_scale` (float): Minimum half-dimension as fraction of image size (0.0 to 1.0)
- `max_scale` (float): Maximum half-dimension as fraction of image size (0.0 to 1.0)

**Valid Ranges**:
- `min_scale`: 0.05 to 0.5 (typical: 0.1 to 0.35)
- `max_scale`: Must be ≥ `min_scale` (typical: 0.1 to 0.5)
- Position: Center randomly placed in central 20% of image (0.4 to 0.6 normalized)

**Physical Interpretation**: Represents rectangular apertures, structured objects, or pixelated features.

**Code Example**:
```python
from hologen.shapes import RectangleGenerator

generator = RectangleGenerator(
    name="rectangle",
    min_scale=0.1,   # 10% half-dimension
    max_scale=0.35   # 35% half-dimension
)

amplitude = generator.generate(grid, rng)
```

### RingGenerator

Generates annular rings (hollow circles) with randomized outer radius, thickness, and position.

**Parameters**:
- `name` (str): Identifier for the generator (e.g., "ring")
- `min_radius` (float): Minimum outer radius as fraction of image dimension
- `max_radius` (float): Maximum outer radius as fraction of image dimension
- `min_thickness` (float): Minimum ring thickness as fraction of outer radius
- `max_thickness` (float): Maximum ring thickness as fraction of outer radius

**Valid Ranges**:
- `min_radius`: 0.05 to 0.5 (typical: 0.12 to 0.25)
- `max_radius`: Must be ≥ `min_radius` (typical: 0.12 to 0.35)
- `min_thickness`: 0.05 to 1.0 (typical: 0.1 to 0.3)
- `max_thickness`: Must be ≥ `min_thickness` (typical: 0.1 to 0.5)
- Position: Center randomly placed in central 20% of image (0.4 to 0.6 normalized)

**Physical Interpretation**: Represents ring-shaped particles, hollow structures, or annular apertures.

**Code Example**:
```python
from hologen.shapes import RingGenerator

generator = RingGenerator(
    name="ring",
    min_radius=0.12,
    max_radius=0.25,
    min_thickness=0.1,   # 10% of outer radius
    max_thickness=0.3    # 30% of outer radius
)

amplitude = generator.generate(grid, rng)
```

**Note**: Inner radius is computed as `inner_radius = max(outer_radius - thickness, 2.0)` to ensure minimum 2-pixel width.

### CircleCheckerGenerator

Generates filled circles with an internal checkerboard pattern for high-frequency content.

**Parameters**:
- `name` (str): Identifier for the generator (e.g., "circle_checker")
- `min_radius` (float): Minimum radius as fraction of image dimension
- `max_radius` (float): Maximum radius as fraction of image dimension
- `checker_size` (int): Size of each checker square in pixels (default: 8)

**Valid Ranges**:
- `min_radius`: 0.05 to 0.5 (typical: 0.1 to 0.2)
- `max_radius`: Must be ≥ `min_radius` (typical: 0.1 to 0.3)
- `checker_size`: 2 to 64 pixels (typical: 8 to 16)
- Position: Center randomly placed in central 40% of image (0.3 to 0.7 normalized)

**Physical Interpretation**: Represents textured particles, resolution test targets, or objects with internal structure.

**Code Example**:
```python
from hologen.shapes import CircleCheckerGenerator

generator = CircleCheckerGenerator(
    name="circle_checker",
    min_radius=0.1,
    max_radius=0.2,
    checker_size=16  # 16×16 pixel checkers
)

amplitude = generator.generate(grid, rng)
```

**Pattern Details**: The checkerboard pattern is aligned to a bounding box around the circle, ensuring stable patterns across generations. Only pixels inside the circular mask receive the checker pattern.

### RectangleCheckerGenerator

Generates filled rectangles with an internal checkerboard pattern.

**Parameters**:
- `name` (str): Identifier for the generator
- `min_scale` (float): Minimum half-dimension as fraction of image size
- `max_scale` (float): Maximum half-dimension as fraction of image size
- `checker_size` (int): Size of each checker square in pixels (default: 8)

**Valid Ranges**:
- `min_scale`: 0.05 to 0.5 (typical: 0.1 to 0.35)
- `max_scale`: Must be ≥ `min_scale` (typical: 0.1 to 0.5)
- `checker_size`: 2 to 64 pixels (typical: 8 to 16)
- Position: Center randomly placed in central 20% of image (0.4 to 0.6 normalized)

**Physical Interpretation**: Represents resolution charts, structured patterns, or textured rectangular objects.

**Code Example**:
```python
from hologen.shapes import RectangleCheckerGenerator

generator = RectangleCheckerGenerator(
    name="rectangle_checker",
    min_scale=0.1,
    max_scale=0.35,
    checker_size=16
)

amplitude = generator.generate(grid, rng)
```

### EllipseCheckerGenerator

Generates filled ellipses with an internal checkerboard pattern, supporting anisotropic shapes.

**Parameters**:
- `name` (str): Identifier for the generator
- `min_radius_y` (float): Minimum vertical semi-axis as fraction of image height
- `max_radius_y` (float): Maximum vertical semi-axis as fraction of image height
- `min_radius_x` (float): Minimum horizontal semi-axis as fraction of image width
- `max_radius_x` (float): Maximum horizontal semi-axis as fraction of image width
- `checker_size` (int): Size of each checker square in pixels (default: 8)

**Valid Ranges**:
- `min_radius_y`, `min_radius_x`: 0.05 to 0.5 (typical: 0.1 to 0.35)
- `max_radius_y`, `max_radius_x`: Must be ≥ corresponding min (typical: 0.1 to 0.5)
- `checker_size`: 2 to 64 pixels (typical: 8 to 16)
- Position: Center randomly placed in central 40% of image (0.3 to 0.7 normalized)

**Physical Interpretation**: Represents elongated particles, anisotropic objects, or elliptical apertures with texture.

**Code Example**:
```python
from hologen.shapes import EllipseCheckerGenerator

generator = EllipseCheckerGenerator(
    name="ellipse_checker",
    min_radius_y=0.1,
    max_radius_y=0.35,
    min_radius_x=0.1,
    max_radius_x=0.35,
    checker_size=16
)

amplitude = generator.generate(grid, rng)
```

**Ellipse Equation**: Points satisfy `((y - cy) / ry)² + ((x - cx) / rx)² ≤ 1`

### TriangleCheckerGenerator

Generates filled triangles with an internal checkerboard pattern, providing angular features.

**Parameters**:
- `name` (str): Identifier for the generator
- `min_scale` (float): Minimum triangle size as fraction of image dimension
- `max_scale` (float): Maximum triangle size as fraction of image dimension
- `checker_size` (int): Size of each checker square in pixels (default: 8)

**Valid Ranges**:
- `min_scale`: 0.1 to 0.5 (typical: 0.15 to 0.3)
- `max_scale`: Must be ≥ `min_scale` (typical: 0.2 to 0.5)
- `checker_size`: 2 to 64 pixels (typical: 8 to 16)
- Position: Center randomly placed in central 30% of image (0.35 to 0.65 normalized)

**Physical Interpretation**: Represents angular features, non-circular particles, or triangular apertures.

**Code Example**:
```python
from hologen.shapes import TriangleCheckerGenerator

generator = TriangleCheckerGenerator(
    name="triangle_checker",
    min_scale=0.15,
    max_scale=0.3,
    checker_size=16
)

amplitude = generator.generate(grid, rng)
```

**Triangle Generation**: Creates approximately equilateral triangles with:
- Random base rotation angle
- Three vertices at 120° intervals
- Small random jitter (±0.2 radians) per vertex
- Random radial variation (80-100% of scale) per vertex

## Complex Field Generation

All shape generators support complex field generation through the `generate_complex()` method, enabling amplitude-only and phase-only object creation.

### Method Signature

```python
def generate_complex(
    self,
    grid: GridSpec,
    rng: Generator,
    phase_shift: float = 0.0,
    mode: str = "amplitude"
) -> ArrayComplex:
    """Generate a complex-valued object field.
    
    Args:
        grid: Grid specification for output resolution.
        rng: Random number generator for stochastic parameters.
        phase_shift: Phase modulation in radians for phase-only objects.
        mode: Generation mode - "amplitude" or "phase".
    
    Returns:
        Complex-valued field with amplitude and phase components.
    
    Raises:
        ValueError: If mode is not "amplitude" or "phase".
        PhaseRangeError: If generated phase values are outside [-π, π].
    """
```

### Amplitude Mode

Creates amplitude-only objects where the shape modulates amplitude and phase is zero everywhere.

**Mathematical representation**: `E(x,y) = A(x,y) · exp(i·0) = A(x,y)`

**Physical interpretation**: Absorbing or scattering samples (stained cells, printed patterns, metal particles)

**Code Example**:
```python
# Generate amplitude-only complex field
complex_field = generator.generate_complex(
    grid=grid,
    rng=rng,
    mode="amplitude",
    phase_shift=0.0  # Not used in amplitude mode
)

# Verify properties
amplitude = np.abs(complex_field)
phase = np.angle(complex_field)

print(f"Amplitude range: [{amplitude.min():.2f}, {amplitude.max():.2f}]")  # [0.00, 1.00]
print(f"Phase range: [{phase.min():.2f}, {phase.max():.2f}]")  # [0.00, 0.00]
```

### Phase Mode

Creates phase-only objects where amplitude is uniform and the shape modulates phase.

**Mathematical representation**: `E(x,y) = 1.0 · exp(i·φ(x,y))`

**Physical interpretation**: Transparent samples with varying refractive index or thickness (biological cells, phase masks, transparent polymers)

**Code Example**:
```python
# Generate phase-only complex field with π/2 phase shift
complex_field = generator.generate_complex(
    grid=grid,
    rng=rng,
    mode="phase",
    phase_shift=np.pi/2  # 90-degree phase shift inside shape
)

# Verify properties
amplitude = np.abs(complex_field)
phase = np.angle(complex_field)

print(f"Amplitude range: [{amplitude.min():.2f}, {amplitude.max():.2f}]")  # [1.00, 1.00]
print(f"Phase range: [{phase.min():.2f}, {phase.max():.2f}]")  # [0.00, 1.57]
```

### Phase Shift Parameter

The `phase_shift` parameter controls the phase difference between the object and background for phase-only objects.

**Valid range**: [0, 2π] radians (0 to 6.28)

**Common values**:
- `π/4` (0.785 rad, 45°): Small phase shift, subtle contrast
- `π/2` (1.571 rad, 90°): Quarter-wave shift, good contrast (default)
- `π` (3.142 rad, 180°): Half-wave shift, maximum contrast
- `3π/2` (4.712 rad, 270°): Three-quarter wave shift

**Physical meaning**: Represents optical path difference between light passing through object vs background:

```
φ = (2π/λ) × (n₁ - n₀) × d
```

Where:
- λ = wavelength
- n₁ = refractive index of object
- n₀ = refractive index of background
- d = object thickness

**Example calculation** for biological cell in water:
```python
wavelength = 532e-9  # 532 nm green laser
n_cell = 1.38        # Cell cytoplasm
n_water = 1.33       # Water background
thickness = 5e-6     # 5 μm cell thickness

phase_shift = (2 * np.pi / wavelength) * (n_cell - n_water) * thickness
print(f"Phase shift: {phase_shift:.3f} rad = {np.degrees(phase_shift):.1f}°")
# Output: Phase shift: 0.590 rad = 33.8°
```

## Using Shape Generators in Pipelines

### Basic Usage

```python
from hologen.shapes import CircleGenerator, RectangleGenerator
from hologen.types import GridSpec
import numpy as np

# Create grid specification
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)

# Create generators
circle_gen = CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18)
rect_gen = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.35)

# Generate samples
rng = np.random.default_rng(42)
circle_amplitude = circle_gen.generate(grid, rng)
rect_amplitude = rect_gen.generate(grid, rng)
```

### Using Default Generator Suite

HoloGen provides a convenience function to get all default generators:

```python
from hologen.shapes import available_generators

# Get all default generators with pre-configured parameters
generators = list(available_generators())

print(f"Available generators: {len(generators)}")
for gen in generators:
    print(f"  - {gen.name}")

# Output:
# Available generators: 6
#   - circle
#   - rectangle
#   - ring
#   - circle_checker
#   - rectangle_checker
#   - ellipse_checker
```

### Integration with Dataset Generation

```python
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter
from hologen.holography.inline import InlineHolographyStrategy
from hologen.types import HolographyConfig, HolographyMethod, OpticalConfig
from hologen.shapes import CircleGenerator

# Create shape generator
shape_gen = CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18)

# Create object producer
object_producer = ObjectDomainProducer(
    generator=shape_gen,
    mode="amplitude"  # or "phase" for phase-only objects
)

# Create holography configuration
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

# Create converter
strategy = InlineHolographyStrategy()
converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.INLINE: strategy}
)

# Generate hologram sample
rng = np.random.default_rng(42)
object_sample = object_producer.produce(config, rng)
hologram_sample = converter.convert(object_sample, config)
```

## Creating Custom Shape Generators

You can create custom shape generators by subclassing `BaseShapeGenerator` or implementing the `ObjectShapeGenerator` protocol.

### Subclassing BaseShapeGenerator

```python
from hologen.shapes import BaseShapeGenerator
from hologen.types import GridSpec, ArrayFloat
from numpy.random import Generator
import numpy as np

class CustomStarGenerator(BaseShapeGenerator):
    """Generator producing star shapes."""
    
    __slots__ = ("min_radius", "max_radius", "num_points")
    
    def __init__(self, name: str, min_radius: float, max_radius: float, num_points: int = 5):
        super().__init__(name=name)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_points = num_points
    
    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        """Generate a star-shaped pattern."""
        canvas = self._empty_canvas(grid)
        
        # Random parameters
        outer_radius = rng.uniform(self.min_radius, self.max_radius) * min(grid.height, grid.width)
        inner_radius = outer_radius * 0.4
        center_y = rng.uniform(0.3, 0.7) * grid.height
        center_x = rng.uniform(0.3, 0.7) * grid.width
        
        # Create star vertices
        angles = np.linspace(0, 2*np.pi, 2*self.num_points, endpoint=False)
        radii = np.tile([outer_radius, inner_radius], self.num_points)
        
        vertices_y = center_y + radii * np.sin(angles)
        vertices_x = center_x + radii * np.cos(angles)
        
        # Fill polygon (simplified - use proper polygon fill in production)
        yy, xx = np.ogrid[:grid.height, :grid.width]
        # ... implement polygon filling logic ...
        
        return self._clamp(canvas)

# Usage
star_gen = CustomStarGenerator(name="star", min_radius=0.1, max_radius=0.2, num_points=5)
star_amplitude = star_gen.generate(grid, rng)
```

### Implementing the Protocol

```python
from hologen.types import ObjectShapeGenerator, GridSpec, ArrayFloat
from numpy.random import Generator
import numpy as np

class MinimalGenerator:
    """Minimal generator implementing the protocol."""
    
    @property
    def name(self) -> str:
        return "minimal"
    
    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        """Generate a simple pattern."""
        # Create a random binary pattern
        return (rng.random((grid.height, grid.width)) > 0.5).astype(np.float64)

# This works with any HoloGen pipeline component expecting ObjectShapeGenerator
minimal_gen: ObjectShapeGenerator = MinimalGenerator()
```

## Best Practices

### Parameter Selection

1. **Size ranges**: Choose ranges that produce visible features at your resolution
   - Too small: Features may be undersampled or invisible
   - Too large: May exceed field of view or create edge artifacts

2. **Checker size**: Match to your spatial resolution requirements
   - Smaller checkers (4-8 pixels): High-frequency content, tests resolution limits
   - Larger checkers (16-32 pixels): More stable patterns, easier reconstruction

3. **Position constraints**: Default ranges keep shapes away from edges
   - Prevents truncation artifacts
   - Ensures full shape is captured in hologram

### Reproducibility

Always use seeded random number generators for reproducible datasets:

```python
# Reproducible generation
rng = np.random.default_rng(42)
sample1 = generator.generate(grid, rng)

# Same seed produces same output
rng = np.random.default_rng(42)
sample2 = generator.generate(grid, rng)

assert np.array_equal(sample1, sample2)  # True
```

### Performance Considerations

1. **Memory efficiency**: Generators create arrays on-demand, no caching
2. **Vectorization**: All generators use NumPy vectorized operations
3. **Batch generation**: Generate multiple samples in a loop, not in parallel (RNG state)

```python
# Efficient batch generation
samples = []
for i in range(100):
    sample = generator.generate(grid, rng)
    samples.append(sample)
```

### Validation

Verify generated shapes have expected properties:

```python
amplitude = generator.generate(grid, rng)

# Check binary values
assert set(np.unique(amplitude)).issubset({0.0, 1.0}), "Non-binary values"

# Check fill fraction is reasonable
fill_fraction = amplitude.mean()
assert 0.001 < fill_fraction < 0.5, f"Unusual fill fraction: {fill_fraction}"

# Check shape is not empty
assert amplitude.sum() > 0, "Empty shape generated"
```

## API Reference

### BaseShapeGenerator

Base class for all shape generators providing common functionality.

**Attributes**:
- `name` (str): Canonical name used when recording generated samples

**Methods**:

#### `generate(grid: GridSpec, rng: Generator) -> ArrayFloat`

Create a binary object-domain image.

**Parameters**:
- `grid`: Grid specification describing the desired output resolution
- `rng`: Random number generator providing stochastic parameters

**Returns**:
- Binary amplitude image with values in {0.0, 1.0}

**Raises**:
- `NotImplementedError`: If the subclass does not override the method

#### `generate_complex(grid: GridSpec, rng: Generator, phase_shift: float = 0.0, mode: str = "amplitude") -> ArrayComplex`

Generate a complex-valued object field.

**Parameters**:
- `grid`: Grid specification describing the desired output resolution
- `rng`: Random number generator providing stochastic parameters
- `phase_shift`: Phase modulation in radians for phase-only objects (default: 0.0)
- `mode`: Generation mode - "amplitude" or "phase" (default: "amplitude")

**Returns**:
- Complex-valued field with amplitude and phase components

**Raises**:
- `ValueError`: If mode is not "amplitude" or "phase"
- `PhaseRangeError`: If generated phase values are outside [-π, π]

#### `_empty_canvas(grid: GridSpec) -> ArrayFloat`

Allocate a zero-initialized canvas matching the grid (protected method).

#### `_clamp(canvas: ArrayFloat) -> ArrayFloat`

Clamp canvas values to the range [0.0, 1.0] (protected method).

### ObjectShapeGenerator Protocol

Protocol defining the interface for shape generators.

**Required Attributes**:
- `name` (str): Return the canonical name of the generator

**Required Methods**:
- `generate(grid: GridSpec, rng: Generator) -> ArrayFloat`: Create a binary object-domain image

### available_generators()

Return the default suite of shape generators.

**Returns**:
- Iterable of `ObjectShapeGenerator` instances with pre-configured parameters

**Example**:
```python
from hologen.shapes import available_generators

generators = list(available_generators())
# Returns: [CircleGenerator, RectangleGenerator, RingGenerator, 
#           CircleCheckerGenerator, RectangleCheckerGenerator, EllipseCheckerGenerator]
```

## See Also

- **[Complex Field Support](COMPLEX_FIELDS.md)** - Learn about amplitude-only and phase-only object generation with shapes
- **[Holography Methods](HOLOGRAPHY_METHODS.md)** - Understand how shapes are propagated to create holograms using inline and off-axis methods
- **[Pipeline Architecture](PIPELINE.md)** - See how shape generators integrate into the full dataset generation pipeline
- **[CLI Reference](CLI_REFERENCE.md)** - Command-line usage for dataset generation with different shapes
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all shape generator classes
- **[Examples](EXAMPLES.md)** - Practical examples including custom shape generators (Example 4)
- **[Quickstart Guide](QUICKSTART.md)** - Get started quickly with shape-based dataset generation

