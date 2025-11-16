# API Reference

Complete technical reference for all public APIs in HoloGen.

## Table of Contents

- [Module Organization](#module-organization)
- [Type Definitions](#type-definitions)
- [Shape Generators](#shape-generators)
- [Converters](#converters)
- [Holography Strategies](#holography-strategies)
- [Noise Models](#noise-models)
- [Utility Modules](#utility-modules)
- [Exception Classes](#exception-classes)

---

## Module Organization

HoloGen is organized into the following modules:

- **hologen.types**: Core type definitions, protocols, and dataclasses
- **hologen.shapes**: Object-domain shape generators
- **hologen.converters**: Pipeline components for dataset generation
- **hologen.holography**: Holography strategy implementations
- **hologen.noise**: Noise and aberration models
- **hologen.utils.fields**: Field representation utilities
- **hologen.utils.io**: Dataset I/O operations
- **hologen.utils.math**: Mathematical utilities

---

## Type Definitions

### Enumerations

#### `HolographyMethod`

Enumeration of supported holography sampling approaches.

**Values:**
- `INLINE = "inline"`: Inline holography method
- `OFF_AXIS = "off_axis"`: Off-axis holography method

**Example:**
```python
from hologen import HolographyMethod

method = HolographyMethod.INLINE
print(method.value)  # "inline"
```

#### `FieldRepresentation`

Enumeration of field representation types.

**Values:**
- `INTENSITY`: Intensity representation (|field|²)
- `AMPLITUDE`: Amplitude representation (|field|)
- `PHASE`: Phase representation (∠field)
- `COMPLEX`: Complex representation (amplitude + phase)

**Example:**
```python
from hologen.types import FieldRepresentation

rep = FieldRepresentation.INTENSITY
print(rep.value)  # "intensity"
```

### Type Aliases

#### `ArrayFloat`

Type alias for NumPy arrays containing 64-bit floating-point values.

```python
ArrayFloat = npt.NDArray[np.float64]
```

#### `ArrayComplex`

Type alias for NumPy arrays containing 128-bit complex values.

```python
ArrayComplex = npt.NDArray[np.complex128]
```

### Dataclasses

#### `GridSpec`

Sampling definition for a two-dimensional grid.

**Parameters:**
- `height` (int): Number of pixels along the vertical axis
- `width` (int): Number of pixels along the horizontal axis
- `pixel_pitch` (float): Sampling interval between adjacent pixels in meters

**Example:**
```python
from hologen import GridSpec

grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
print(f"Grid: {grid.height}x{grid.width}, pitch={grid.pixel_pitch}")
```

#### `OpticalConfig`

Physical parameters for hologram generation and reconstruction.

**Parameters:**
- `wavelength` (float): Illumination wavelength in meters
- `propagation_distance` (float): Distance between object and sensor planes in meters

**Example:**
```python
from hologen import OpticalConfig

optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.1)
print(f"λ={optics.wavelength*1e9:.0f}nm, z={optics.propagation_distance*1e3:.0f}mm")
```

#### `OffAxisCarrier`

Carrier modulation parameters for off-axis holography.

**Parameters:**
- `frequency_x` (float): Spatial carrier frequency along horizontal axis in cycles per meter
- `frequency_y` (float): Spatial carrier frequency along vertical axis in cycles per meter
- `gaussian_width` (float): Standard deviation of Gaussian filter for reconstruction in cycles per meter

**Example:**
```python
from hologen import OffAxisCarrier

carrier = OffAxisCarrier(
    frequency_x=1000.0,
    frequency_y=1000.0,
    gaussian_width=500.0
)
```

#### `HolographyConfig`

Configuration bundle for holography strategies.

**Parameters:**
- `grid` (GridSpec): Spatial sampling description of the sensor plane
- `optics` (OpticalConfig): Optical parameters governing propagation
- `method` (HolographyMethod): Holography method to employ
- `carrier` (OffAxisCarrier | None): Optional carrier configuration for off-axis holography

**Example:**
```python
from hologen import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod

config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.INLINE
)
```

#### `OutputConfig`

Configuration for output field representations.

**Parameters:**
- `object_representation` (FieldRepresentation): Field representation for object domain (default: INTENSITY)
- `hologram_representation` (FieldRepresentation): Field representation for hologram (default: INTENSITY)
- `reconstruction_representation` (FieldRepresentation): Field representation for reconstruction (default: INTENSITY)

**Example:**
```python
from hologen.types import OutputConfig, FieldRepresentation

output_config = OutputConfig(
    object_representation=FieldRepresentation.AMPLITUDE,
    hologram_representation=FieldRepresentation.INTENSITY,
    reconstruction_representation=FieldRepresentation.AMPLITUDE
)
```

#### `ObjectSample`

Object-domain sample representation (legacy format).

**Parameters:**
- `name` (str): Identifier of the shape generator that produced the sample
- `pixels` (ArrayFloat): Binary amplitude distribution of the object domain

**Example:**
```python
from hologen.types import ObjectSample
import numpy as np

sample = ObjectSample(
    name="circle",
    pixels=np.ones((512, 512), dtype=np.float64)
)
```

#### `ComplexObjectSample`

Object-domain sample with complex field representation.

**Parameters:**
- `name` (str): Identifier of the shape generator
- `field` (ArrayComplex): Complex-valued field (amplitude + phase)
- `representation` (FieldRepresentation): Field representation type

**Example:**
```python
from hologen.types import ComplexObjectSample, FieldRepresentation
import numpy as np

sample = ComplexObjectSample(
    name="circle",
    field=np.ones((512, 512), dtype=np.complex128),
    representation=FieldRepresentation.AMPLITUDE
)
```

#### `HologramSample`

Full holography sample including forward and reconstructed domains (legacy format).

**Parameters:**
- `object_sample` (ObjectSample): Reference to the originating object-domain sample
- `hologram` (ArrayFloat): Intensity hologram generated from the object sample
- `reconstruction` (ArrayFloat): Recovered object-domain amplitude from the hologram

**Example:**
```python
from hologen.types import HologramSample, ObjectSample
import numpy as np

sample = HologramSample(
    object_sample=ObjectSample(name="circle", pixels=np.ones((512, 512))),
    hologram=np.ones((512, 512)),
    reconstruction=np.ones((512, 512))
)
```

#### `ComplexHologramSample`

Hologram sample with complex field support.

**Parameters:**
- `object_sample` (ComplexObjectSample): Reference to the originating complex object-domain sample
- `hologram_field` (ArrayComplex): Complex hologram field
- `hologram_representation` (FieldRepresentation): Field representation type for hologram
- `reconstruction_field` (ArrayComplex): Complex reconstruction field
- `reconstruction_representation` (FieldRepresentation): Field representation type for reconstruction

#### `NoiseConfig`

Configuration for hologram recording noise simulation.

**Parameters:**
- `sensor_read_noise` (float): Standard deviation of Gaussian read noise (default: 0.0)
- `sensor_shot_noise` (bool): Enable Poisson shot noise simulation (default: False)
- `sensor_dark_current` (float): Mean dark current in intensity units (default: 0.0)
- `sensor_bit_depth` (int | None): ADC bit depth for quantization (default: None)
- `speckle_contrast` (float): Speckle contrast ratio 0.0-1.0 (default: 0.0)
- `speckle_correlation_length` (float): Speckle correlation length in pixels (default: 1.0)
- `aberration_defocus` (float): Defocus aberration coefficient (default: 0.0)
- `aberration_astigmatism_x` (float): Astigmatism x coefficient (default: 0.0)
- `aberration_astigmatism_y` (float): Astigmatism y coefficient (default: 0.0)
- `aberration_coma_x` (float): Coma x coefficient (default: 0.0)
- `aberration_coma_y` (float): Coma y coefficient (default: 0.0)

**Example:**
```python
from hologen.types import NoiseConfig

noise_config = NoiseConfig(
    sensor_read_noise=0.01,
    sensor_shot_noise=True,
    speckle_contrast=0.3,
    aberration_defocus=0.5
)
```

### Protocols

#### `ObjectShapeGenerator`

Protocol for object-domain shape generators.

**Properties:**
- `name` (str): Return the canonical name of the generator

**Methods:**
- `generate(grid: GridSpec, rng: Generator) -> ArrayFloat`: Create a binary object-domain image

**Example:**
```python
from hologen.types import ObjectShapeGenerator, GridSpec
import numpy as np

class CustomGenerator:
    @property
    def name(self) -> str:
        return "custom"
    
    def generate(self, grid: GridSpec, rng: np.random.Generator) -> np.ndarray:
        return np.random.rand(grid.height, grid.width)

# CustomGenerator satisfies the ObjectShapeGenerator protocol
```

#### `HolographyStrategy`

Protocol describing hologram generation and reconstruction operations.

**Methods:**
- `create_hologram(object_field: ArrayComplex, config: HolographyConfig) -> ArrayComplex`: Create a hologram from an object-domain complex field
- `reconstruct(hologram: ArrayComplex, config: HolographyConfig) -> ArrayComplex`: Recover an object-domain complex field from a hologram

**Example:**
```python
from hologen.types import HolographyStrategy, HolographyConfig
import numpy as np

class CustomStrategy:
    def create_hologram(self, object_field, config):
        # Custom hologram generation logic
        return object_field
    
    def reconstruct(self, hologram, config):
        # Custom reconstruction logic
        return hologram

# CustomStrategy satisfies the HolographyStrategy protocol
```

#### `NoiseModel`

Protocol for hologram noise simulation models.

**Methods:**
- `apply(hologram: ArrayFloat, config: HolographyConfig, rng: Generator) -> ArrayFloat`: Apply noise to a hologram

**Parameters:**
- `hologram`: Perfect hologram intensity distribution
- `config`: Holography configuration containing grid and optical parameters
- `rng`: Random number generator for stochastic noise

**Returns:**
- Noisy hologram intensity distribution

#### `DatasetWriter`

Protocol for persisting generated holography samples.

**Methods:**
- `save(samples: Iterable[HologramSample], output_dir: Path) -> None`: Persist a sequence of hologram samples

#### `DatasetGenerator`

Protocol for dataset generation routines.

**Methods:**
- `generate(count: int, config: HolographyConfig, rng: Generator) -> Iterable[HologramSample]`: Produce holography samples

---

## Shape Generators

### `BaseShapeGenerator`

Abstract base for object-domain shape generators.

**Parameters:**
- `name` (str): Canonical name used when recording generated samples

**Methods:**

#### `generate(grid: GridSpec, rng: Generator) -> ArrayFloat`

Create a binary object-domain image.

**Parameters:**
- `grid`: Grid specification describing the desired output resolution
- `rng`: Random number generator providing stochastic parameters

**Returns:**
- Binary amplitude image with values in {0.0, 1.0}

**Raises:**
- `NotImplementedError`: If the subclass does not override the method

#### `generate_complex(grid: GridSpec, rng: Generator, phase_shift: float = 0.0, mode: str = "amplitude") -> ArrayComplex`

Generate a complex-valued object field.

**Parameters:**
- `grid`: Grid specification describing the desired output resolution
- `rng`: Random number generator providing stochastic parameters
- `phase_shift`: Phase modulation in radians for phase-only objects (default: 0.0)
- `mode`: Generation mode - "amplitude" (shape with zero phase) or "phase" (uniform amplitude with phase modulation)

**Returns:**
- Complex-valued field with amplitude and phase components

**Raises:**
- `ValueError`: If mode is not "amplitude" or "phase"
- `PhaseRangeError`: If generated phase values are outside [-π, π]

**Example:**
```python
from hologen.shapes import CircleGenerator
from hologen import GridSpec
import numpy as np

generator = CircleGenerator(name="circle", min_radius=0.1, max_radius=0.2)
grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
rng = np.random.default_rng(42)

# Generate amplitude-only object
amplitude_field = generator.generate_complex(grid, rng, mode="amplitude")

# Generate phase-only object with π/2 phase shift
phase_field = generator.generate_complex(grid, rng, phase_shift=np.pi/2, mode="phase")
```

### `CircleGenerator`

Generator producing filled discs.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_radius` (float): Minimum radius as fraction of grid dimension
- `max_radius` (float): Maximum radius as fraction of grid dimension

**Example:**
```python
from hologen import CircleGenerator, GridSpec
import numpy as np

generator = CircleGenerator(name="circle", min_radius=0.08, max_radius=0.18)
grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
rng = np.random.default_rng(42)

circle = generator.generate(grid, rng)
print(f"Generated circle: {circle.shape}, values in [{circle.min()}, {circle.max()}]")
```

### `RectangleGenerator`

Generator producing filled rectangles.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_scale` (float): Minimum size as fraction of grid dimension
- `max_scale` (float): Maximum size as fraction of grid dimension

**Example:**
```python
from hologen import RectangleGenerator, GridSpec
import numpy as np

generator = RectangleGenerator(name="rectangle", min_scale=0.1, max_scale=0.35)
grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
rng = np.random.default_rng(42)

rectangle = generator.generate(grid, rng)
```

### `RingGenerator`

Generator producing annular rings.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_radius` (float): Minimum outer radius as fraction of grid dimension
- `max_radius` (float): Maximum outer radius as fraction of grid dimension
- `min_thickness` (float): Minimum thickness as fraction of outer radius
- `max_thickness` (float): Maximum thickness as fraction of outer radius

**Example:**
```python
from hologen.shapes import RingGenerator
from hologen import GridSpec
import numpy as np

generator = RingGenerator(
    name="ring",
    min_radius=0.12,
    max_radius=0.25,
    min_thickness=0.1,
    max_thickness=0.3
)
grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
rng = np.random.default_rng(42)

ring = generator.generate(grid, rng)
```

### `CircleCheckerGenerator`

Generator producing filled discs with a checkerboard pattern inside.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_radius` (float): Minimum radius as fraction of grid dimension
- `max_radius` (float): Maximum radius as fraction of grid dimension
- `checker_size` (int): Size of checkerboard squares in pixels (default: 8)

**Example:**
```python
from hologen.shapes import CircleCheckerGenerator
from hologen import GridSpec
import numpy as np

generator = CircleCheckerGenerator(
    name="circle_checker",
    min_radius=0.1,
    max_radius=0.2,
    checker_size=16
)
grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
rng = np.random.default_rng(42)

checker_circle = generator.generate(grid, rng)
```

### `RectangleCheckerGenerator`

Generator producing filled rectangles with a checkerboard pattern inside.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_scale` (float): Minimum size as fraction of grid dimension
- `max_scale` (float): Maximum size as fraction of grid dimension
- `checker_size` (int): Size of checkerboard squares in pixels (default: 8)

### `EllipseCheckerGenerator`

Generator producing filled ellipses with a checkerboard pattern inside.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_radius_y` (float): Minimum vertical radius as fraction of grid height
- `max_radius_y` (float): Maximum vertical radius as fraction of grid height
- `min_radius_x` (float): Minimum horizontal radius as fraction of grid width
- `max_radius_x` (float): Maximum horizontal radius as fraction of grid width
- `checker_size` (int): Size of checkerboard squares in pixels (default: 8)

### `TriangleCheckerGenerator`

Generator producing filled triangles with a checkerboard pattern inside.

**Parameters:**
- `name` (str): Canonical name for the generator
- `min_scale` (float): Minimum size as fraction of grid dimension
- `max_scale` (float): Maximum size as fraction of grid dimension
- `checker_size` (int): Size of checkerboard squares in pixels (default: 8)

### `available_generators() -> Iterable[ObjectShapeGenerator]`

Return the default suite of shape generators.

**Returns:**
- Iterable of pre-configured shape generators including Circle, Rectangle, Ring, CircleChecker, RectangleChecker, and EllipseChecker

**Example:**
```python
from hologen.shapes import available_generators

generators = list(available_generators())
print(f"Available generators: {[g.name for g in generators]}")
# Output: ['circle', 'rectangle', 'ring', 'circle_checker', 'rectangle_checker', 'ellipse_checker']
```

---

## Converters

### `ObjectDomainProducer`

Generate object-domain samples using registered shape generators.

**Parameters:**
- `shape_generators` (tuple[ObjectShapeGenerator, ...]): Tuple of shape generator implementations to sample from

**Methods:**

#### `generate(grid: GridSpec, rng: Generator) -> ObjectSample`

Produce a new object-domain sample.

**Parameters:**
- `grid`: Grid specification describing the required output resolution
- `rng`: Random number generator providing stochastic parameters

**Returns:**
- ObjectSample containing the generated amplitude image

**Example:**
```python
from hologen.converters import ObjectDomainProducer, default_object_producer
from hologen import GridSpec
import numpy as np

producer = default_object_producer()
grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
rng = np.random.default_rng(42)

sample = producer.generate(grid, rng)
print(f"Generated {sample.name} with shape {sample.pixels.shape}")
```

#### `generate_complex(grid: GridSpec, rng: Generator, phase_shift: float = 0.0, mode: str = "amplitude") -> ComplexObjectSample`

Produce a new complex object-domain sample.

**Parameters:**
- `grid`: Grid specification describing the required output resolution
- `rng`: Random number generator providing stochastic parameters
- `phase_shift`: Phase modulation in radians for phase-only objects (default: 0.0)
- `mode`: Generation mode - "amplitude" or "phase"

**Returns:**
- ComplexObjectSample containing the generated complex field

### `ObjectToHologramConverter`

Convert object-domain amplitudes into hologram representations.

**Parameters:**
- `strategy_mapping` (dict[HolographyMethod, HolographyStrategy]): Mapping from holography methods to strategy implementations
- `noise_model` (NoiseModel | None): Optional noise model to apply to generated holograms
- `output_config` (OutputConfig): Configuration for output field representations

**Methods:**

#### `create_hologram(sample: ObjectSample | ComplexObjectSample, config: HolographyConfig, rng: Generator | None = None) -> ArrayFloat | ArrayComplex`

Generate a hologram for the provided object sample.

**Parameters:**
- `sample`: Object-domain sample to transform
- `config`: Holography configuration specifying physical parameters
- `rng`: Random number generator for noise application (optional)

**Returns:**
- Hologram field (intensity for legacy ObjectSample, complex for ComplexObjectSample)

**Example:**
```python
from hologen.converters import default_converter
from hologen import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod
from hologen.types import ObjectSample
import numpy as np

converter = default_converter()
config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.INLINE
)

sample = ObjectSample(name="circle", pixels=np.ones((512, 512)))
rng = np.random.default_rng(42)

hologram = converter.create_hologram(sample, config, rng)
print(f"Hologram shape: {hologram.shape}")
```

#### `reconstruct(hologram: ArrayFloat | ArrayComplex, config: HolographyConfig) -> ArrayFloat | ArrayComplex`

Reconstruct an object-domain field from a hologram.

**Parameters:**
- `hologram`: Hologram field (intensity for legacy, complex for new)
- `config`: Holography configuration specifying physical parameters

**Returns:**
- Reconstructed field (amplitude for legacy, complex for new)

### `HologramDatasetGenerator`

Generate full hologram samples from object-domain sources.

**Parameters:**
- `object_producer` (ObjectDomainProducer): Producer responsible for creating object samples
- `converter` (ObjectToHologramConverter): Converter performing hologram generation and reconstruction

**Methods:**

#### `generate(count: int, config: HolographyConfig, rng: Generator, phase_shift: float = 0.0, mode: str = "amplitude", use_complex: bool = False) -> Iterable[HologramSample | ComplexHologramSample]`

Yield hologram samples as an iterable sequence.

**Parameters:**
- `count`: Number of samples to generate
- `config`: Holography configuration applied to all samples
- `rng`: Random number generator used throughout the pipeline
- `phase_shift`: Phase modulation in radians for phase-only objects (default: 0.0)
- `mode`: Generation mode - "amplitude" or "phase" (default: "amplitude")
- `use_complex`: If True, generate ComplexHologramSample; if False, generate legacy HologramSample (default: False)

**Yields:**
- Sequential hologram samples containing object, hologram, and reconstruction data

**Example:**
```python
from hologen.converters import HologramDatasetGenerator, default_object_producer, default_converter
from hologen import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod
import numpy as np

generator = HologramDatasetGenerator(
    object_producer=default_object_producer(),
    converter=default_converter()
)

config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.INLINE
)
rng = np.random.default_rng(42)

samples = list(generator.generate(count=5, config=config, rng=rng))
print(f"Generated {len(samples)} samples")
```

### Factory Functions

#### `default_object_producer() -> ObjectDomainProducer`

Create the default object domain producer with built-in shapes.

**Returns:**
- ObjectDomainProducer configured with all available shape generators

**Example:**
```python
from hologen.converters import default_object_producer

producer = default_object_producer()
```

#### `default_converter(noise_model: NoiseModel | None = None) -> ObjectToHologramConverter`

Create the default converter with inline and off-axis strategies.

**Parameters:**
- `noise_model`: Optional noise model to apply to generated holograms

**Returns:**
- ObjectToHologramConverter configured with inline and off-axis strategies

**Example:**
```python
from hologen.converters import default_converter, create_noise_model
from hologen.types import NoiseConfig

noise_config = NoiseConfig(sensor_read_noise=0.01, speckle_contrast=0.3)
noise_model = create_noise_model(noise_config)
converter = default_converter(noise_model=noise_model)
```

#### `generate_dataset(count: int, config: HolographyConfig, rng: Generator, writer: DatasetWriter, generator: HologramDatasetGenerator | None = None, output_dir: Path | None = None) -> None`

Generate and persist a holography dataset using the pipeline.

**Parameters:**
- `count`: Number of samples to produce
- `config`: Holography configuration applied to all samples
- `rng`: Random number generator used for stochastic steps
- `writer`: Dataset writer responsible for persisting results
- `generator`: Optional pre-configured generator to reuse
- `output_dir`: Optional output directory override for writer

**Example:**
```python
from hologen.converters import generate_dataset
from hologen import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod, NumpyDatasetWriter
import numpy as np
from pathlib import Path

config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.INLINE
)
rng = np.random.default_rng(42)
writer = NumpyDatasetWriter(save_preview=True)

generate_dataset(
    count=10,
    config=config,
    rng=rng,
    writer=writer,
    output_dir=Path("my_dataset")
)
```

#### `create_noise_model(config: NoiseConfig) -> NoiseModel | None`

Create a composite noise model from configuration.

**Parameters:**
- `config`: Noise configuration specifying all noise parameters

**Returns:**
- Composite noise model or None if all noise is disabled

**Example:**
```python
from hologen.converters import create_noise_model
from hologen.types import NoiseConfig

noise_config = NoiseConfig(
    sensor_read_noise=0.01,
    sensor_shot_noise=True,
    speckle_contrast=0.3,
    aberration_defocus=0.5
)

noise_model = create_noise_model(noise_config)
if noise_model:
    print(f"Created noise model: {noise_model.name}")
```

---

## Holography Strategies

### `InlineHolographyStrategy`

Implement inline hologram generation and reconstruction.

**Methods:**

#### `create_hologram(object_field: ArrayComplex, config: HolographyConfig) -> ArrayComplex`

Generate an inline hologram from an object-domain complex field.

**Parameters:**
- `object_field`: Complex-valued object field
- `config`: Holography configuration

**Returns:**
- Complex hologram field after propagation

**Example:**
```python
from hologen.holography.inline import InlineHolographyStrategy
from hologen import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod
import numpy as np

strategy = InlineHolographyStrategy()
config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.INLINE
)

object_field = np.ones((512, 512), dtype=np.complex128)
hologram = strategy.create_hologram(object_field, config)
```

#### `reconstruct(hologram: ArrayComplex, config: HolographyConfig) -> ArrayComplex`

Reconstruct the object domain from an inline hologram.

**Parameters:**
- `hologram`: Complex hologram field
- `config`: Holography configuration

**Returns:**
- Reconstructed complex object field

### `OffAxisHolographyStrategy`

Implement off-axis hologram generation and reconstruction.

**Methods:**

#### `create_hologram(object_field: ArrayComplex, config: HolographyConfig) -> ArrayComplex`

Generate an off-axis hologram from an object-domain complex field.

**Parameters:**
- `object_field`: Complex-valued object field
- `config`: Holography configuration (must include carrier configuration)

**Returns:**
- Complex hologram field with carrier modulation

**Raises:**
- `ValueError`: If carrier configuration is not provided

**Example:**
```python
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod, OffAxisCarrier
import numpy as np

strategy = OffAxisHolographyStrategy()
config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.OFF_AXIS,
    carrier=OffAxisCarrier(frequency_x=1000.0, frequency_y=1000.0, gaussian_width=500.0)
)

object_field = np.ones((512, 512), dtype=np.complex128)
hologram = strategy.create_hologram(object_field, config)
```

#### `reconstruct(hologram: ArrayComplex, config: HolographyConfig) -> ArrayComplex`

Reconstruct the object domain from an off-axis hologram.

**Parameters:**
- `hologram`: Complex hologram field
- `config`: Holography configuration (must include carrier configuration)

**Returns:**
- Reconstructed complex object field

**Raises:**
- `ValueError`: If carrier configuration is not provided

### Propagation Utilities

#### `angular_spectrum_propagate(field: ArrayComplex, grid: GridSpec, optics: OpticalConfig, distance: float) -> ArrayComplex`

Propagate a complex optical field using the angular spectrum method.

The angular spectrum method models propagation by decomposing the source field into plane waves (its spatial frequency spectrum), applying the appropriate phase (or evanescent decay) for a given propagation distance, and recomposing the field in the observation plane.

**Parameters:**
- `field`: Complex field distribution sampled in the source plane with shape (grid.height, grid.width)
- `grid`: Spatial sampling specification for the field
- `optics`: Optical parameters describing the illumination
- `distance`: Propagation distance along the optical axis in meters (positive advances, negative back-propagates)

**Returns:**
- Complex field after propagation over the requested distance

**Raises:**
- `ValueError`: If the supplied field shape is incompatible with the grid or if Nyquist criterion is violated

**Example:**
```python
from hologen.holography.propagation import angular_spectrum_propagate
from hologen import GridSpec, OpticalConfig
import numpy as np

grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.1)
field = np.ones((512, 512), dtype=np.complex128)

propagated = angular_spectrum_propagate(field, grid, optics, distance=0.1)
print(f"Propagated field shape: {propagated.shape}")
```

---

## Noise Models

### `BaseNoiseModel`

Abstract base for noise model implementations.

**Parameters:**
- `name` (str): Identifier for the noise model

**Methods:**

#### `apply(hologram: ArrayFloat, config: HolographyConfig, rng: Generator) -> ArrayFloat`

Apply noise to a hologram.

**Parameters:**
- `hologram`: Perfect hologram intensity distribution
- `config`: Holography configuration containing grid and optical parameters
- `rng`: Random number generator for stochastic noise

**Returns:**
- Noisy hologram intensity distribution

**Raises:**
- `NotImplementedError`: If the subclass does not override the method

### `SensorNoiseModel`

Simulate sensor recording noise including read, shot, dark current, and quantization.

**Parameters:**
- `name` (str): Identifier for the noise model
- `read_noise` (float): Standard deviation of Gaussian read noise (default: 0.0)
- `shot_noise` (bool): Enable Poisson shot noise simulation (default: False)
- `dark_current` (float): Mean dark current in intensity units (default: 0.0)
- `bit_depth` (int | None): ADC bit depth for quantization (default: None)

**Example:**
```python
from hologen.noise import SensorNoiseModel
from hologen import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod
import numpy as np

noise_model = SensorNoiseModel(
    name="sensor",
    read_noise=0.01,
    shot_noise=True,
    dark_current=0.005,
    bit_depth=12
)

config = HolographyConfig(
    grid=GridSpec(height=512, width=512, pixel_pitch=5.5e-6),
    optics=OpticalConfig(wavelength=532e-9, propagation_distance=0.1),
    method=HolographyMethod.INLINE
)
rng = np.random.default_rng(42)

hologram = np.ones((512, 512))
noisy_hologram = noise_model.apply(hologram, config, rng)
```

### `SpeckleNoiseModel`

Simulate multiplicative speckle noise from coherent illumination.

**Parameters:**
- `name` (str): Identifier for the noise model
- `contrast` (float): Speckle contrast ratio 0.0-1.0 (default: 0.5)
- `correlation_length` (float): Speckle correlation length in pixels (default: 2.0)

**Example:**
```python
from hologen.noise import SpeckleNoiseModel
import numpy as np

noise_model = SpeckleNoiseModel(
    name="speckle",
    contrast=0.3,
    correlation_length=2.5
)

hologram = np.ones((512, 512))
rng = np.random.default_rng(42)
noisy_hologram = noise_model.apply(hologram, config, rng)
```

### `AberrationNoiseModel`

Simulate optical aberrations using Zernike polynomials.

**Parameters:**
- `name` (str): Identifier for the noise model
- `defocus` (float): Defocus coefficient (Zernike Z_2^0) (default: 0.0)
- `astigmatism_x` (float): Astigmatism x coefficient (Zernike Z_2^-2) (default: 0.0)
- `astigmatism_y` (float): Astigmatism y coefficient (Zernike Z_2^2) (default: 0.0)
- `coma_x` (float): Coma x coefficient (Zernike Z_3^-1) (default: 0.0)
- `coma_y` (float): Coma y coefficient (Zernike Z_3^1) (default: 0.0)

**Example:**
```python
from hologen.noise import AberrationNoiseModel
import numpy as np

noise_model = AberrationNoiseModel(
    name="aberration",
    defocus=0.5,
    astigmatism_x=0.2,
    coma_x=0.1
)

hologram = np.ones((512, 512))
rng = np.random.default_rng(42)
noisy_hologram = noise_model.apply(hologram, config, rng)
```

### `CompositeNoiseModel`

Combine multiple noise models in sequence.

**Parameters:**
- `name` (str): Identifier for the noise model
- `models` (tuple[NoiseModel, ...]): Tuple of noise models to apply in order

**Example:**
```python
from hologen.noise import CompositeNoiseModel, SensorNoiseModel, SpeckleNoiseModel
import numpy as np

composite = CompositeNoiseModel(
    name="composite",
    models=(
        SpeckleNoiseModel(name="speckle", contrast=0.3),
        SensorNoiseModel(name="sensor", read_noise=0.01, shot_noise=True)
    )
)

hologram = np.ones((512, 512))
rng = np.random.default_rng(42)
noisy_hologram = composite.apply(hologram, config, rng)
```

---

## Utility Modules

### Field Utilities (hologen.utils.fields)

#### `complex_to_representation(field: ArrayComplex, representation: FieldRepresentation) -> ArrayFloat | ArrayComplex`

Convert a complex field to the requested representation.

**Parameters:**
- `field`: Complex-valued optical field
- `representation`: Target representation type

**Returns:**
- Field in the requested representation (ArrayFloat for intensity/amplitude/phase, ArrayComplex for complex)

**Raises:**
- `FieldRepresentationError`: If the representation type is invalid

**Example:**
```python
from hologen.utils.fields import complex_to_representation
from hologen.types import FieldRepresentation
import numpy as np

field = np.array([[1+1j, 2+0j]], dtype=np.complex128)

intensity = complex_to_representation(field, FieldRepresentation.INTENSITY)
amplitude = complex_to_representation(field, FieldRepresentation.AMPLITUDE)
phase = complex_to_representation(field, FieldRepresentation.PHASE)

print(f"Intensity: {intensity}")
print(f"Amplitude: {amplitude}")
print(f"Phase: {phase}")
```

#### `amplitude_phase_to_complex(amplitude: ArrayFloat, phase: ArrayFloat) -> ArrayComplex`

Construct a complex field from separate amplitude and phase arrays.

Uses the formula: field = amplitude * exp(i * phase)

**Parameters:**
- `amplitude`: Amplitude values (non-negative)
- `phase`: Phase values in radians

**Returns:**
- Complex field with the specified amplitude and phase

**Example:**
```python
from hologen.utils.fields import amplitude_phase_to_complex
import numpy as np

amplitude = np.array([[1.0, 2.0]], dtype=np.float64)
phase = np.array([[0.0, np.pi/2]], dtype=np.float64)

field = amplitude_phase_to_complex(amplitude, phase)
print(f"Complex field: {field}")
```

#### `validate_phase_range(phase: ArrayFloat) -> None`

Validate that all phase values are within the valid [-π, π] range.

**Parameters:**
- `phase`: Phase array in radians

**Raises:**
- `PhaseRangeError`: If any phase values are outside [-π, π] or non-finite

**Example:**
```python
from hologen.utils.fields import validate_phase_range
import numpy as np

phase = np.array([[0.0, np.pi/2, -np.pi/2]], dtype=np.float64)
validate_phase_range(phase)  # No error

invalid_phase = np.array([[4.0]], dtype=np.float64)
try:
    validate_phase_range(invalid_phase)
except PhaseRangeError as e:
    print(f"Error: {e}")
```

### I/O Utilities (hologen.utils.io)

#### `NumpyDatasetWriter`

Persist holography samples in NumPy archives and optional PNG previews.

**Parameters:**
- `save_preview` (bool): Whether to generate PNG previews for each domain (default: True)

**Methods:**

##### `save(samples: Iterable[HologramSample], output_dir: Path) -> None`

Write hologram samples to disk.

**Parameters:**
- `samples`: Iterable of hologram samples produced by the pipeline
- `output_dir`: Target directory for serialized dataset artifacts

**Raises:**
- `IOError`: If the dataset cannot be written to the storage path

**Example:**
```python
from hologen import NumpyDatasetWriter
from hologen.types import HologramSample, ObjectSample
from pathlib import Path
import numpy as np

writer = NumpyDatasetWriter(save_preview=True)

samples = [
    HologramSample(
        object_sample=ObjectSample(name="circle", pixels=np.ones((512, 512))),
        hologram=np.ones((512, 512)),
        reconstruction=np.ones((512, 512))
    )
]

writer.save(samples, Path("output_dataset"))
```

#### `ComplexFieldWriter`

Persist complex holography samples in NumPy archives and optional PNG previews.

**Parameters:**
- `save_preview` (bool): Whether to generate PNG previews for each domain (default: True)
- `phase_colormap` (str): Matplotlib colormap name for phase visualization (default: "twilight")

**Methods:**

##### `save(samples: Iterable[ComplexHologramSample], output_dir: Path) -> None`

Write complex hologram samples to disk.

**Parameters:**
- `samples`: Iterable of complex hologram samples produced by the pipeline
- `output_dir`: Target directory for serialized dataset artifacts

**Raises:**
- `IOError`: If the dataset cannot be written to the storage path

**Example:**
```python
from hologen.utils.io import ComplexFieldWriter
from hologen.types import ComplexHologramSample, ComplexObjectSample, FieldRepresentation
from pathlib import Path
import numpy as np

writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")

samples = [
    ComplexHologramSample(
        object_sample=ComplexObjectSample(
            name="circle",
            field=np.ones((512, 512), dtype=np.complex128),
            representation=FieldRepresentation.AMPLITUDE
        ),
        hologram_field=np.ones((512, 512), dtype=np.complex128),
        hologram_representation=FieldRepresentation.INTENSITY,
        reconstruction_field=np.ones((512, 512), dtype=np.complex128),
        reconstruction_representation=FieldRepresentation.AMPLITUDE
    )
]

writer.save(samples, Path("complex_dataset"))
```

#### `load_complex_sample(path: Path) -> ComplexObjectSample | ObjectSample`

Load a sample from a NumPy archive with automatic format detection.

Detects whether the file contains complex field data (new format with 'real'/'imag' keys) or legacy intensity data ('object' key) and returns the appropriate sample type.

**Parameters:**
- `path`: Path to the .npz file

**Returns:**
- ComplexObjectSample if the file contains complex field data, ObjectSample if it contains legacy intensity data

**Raises:**
- `ValueError`: If the file format is not recognized
- `IOError`: If the file cannot be read

**Example:**
```python
from hologen.utils.io import load_complex_sample
from pathlib import Path

# Load complex field data
sample = load_complex_sample(Path("complex_object.npz"))
print(f"Loaded {sample.name}: {type(sample).__name__}")

# Load legacy intensity data
legacy_sample = load_complex_sample(Path("legacy_object.npz"))
print(f"Loaded {legacy_sample.name}: {type(legacy_sample).__name__}")
```

### Math Utilities (hologen.utils.math)

#### `normalize_image(image: ArrayFloat) -> ArrayFloat`

Normalize an image to the range [0.0, 1.0].

**Parameters:**
- `image`: Arbitrary floating-point image

**Returns:**
- Normalized image or zeros when the input is constant

**Example:**
```python
from hologen.utils.math import normalize_image
import numpy as np

image = np.array([[10, 20], [30, 40]], dtype=np.float64)
normalized = normalize_image(image)
print(f"Normalized: {normalized}")
# Output: [[0.0, 0.333...], [0.666..., 1.0]]
```

#### `FourierGrid`

Frequency-domain sampling constructed for a spatial grid.

**Parameters:**
- `fx` (NDArray[np.float64]): Two-dimensional array of spatial frequencies along the x axis
- `fy` (NDArray[np.float64]): Two-dimensional array of spatial frequencies along the y axis

#### `make_fourier_grid(grid: GridSpec) -> FourierGrid`

Create Fourier-domain sampling coordinates for a spatial grid.

**Parameters:**
- `grid`: Spatial grid specification defining the sampling resolution

**Returns:**
- FourierGrid containing spatial frequency meshes along both axes

**Example:**
```python
from hologen.utils.math import make_fourier_grid
from hologen import GridSpec

grid = GridSpec(height=512, width=512, pixel_pitch=5.5e-6)
fourier_grid = make_fourier_grid(grid)
print(f"Frequency grid shape: {fourier_grid.fx.shape}")
```

#### `gaussian_blur(image: ArrayFloat, sigma: float) -> ArrayFloat`

Apply an isotropic Gaussian blur to a two-dimensional image.

**Parameters:**
- `image`: Input image to filter
- `sigma`: Standard deviation of the Gaussian kernel in pixel units

**Returns:**
- Blurred image with identical shape to the input

**Example:**
```python
from hologen.utils.math import gaussian_blur
import numpy as np

image = np.random.rand(512, 512)
blurred = gaussian_blur(image, sigma=2.0)
print(f"Blurred image shape: {blurred.shape}")
```

---

## Exception Classes

### `FieldRepresentationError`

Raised when field representation is invalid or incompatible.

**Base Class:** `ValueError`

**Example:**
```python
from hologen.utils.fields import FieldRepresentationError

try:
    # Invalid operation
    raise FieldRepresentationError("Invalid representation type")
except FieldRepresentationError as e:
    print(f"Error: {e}")
```

### `PhaseRangeError`

Raised when phase values are outside the valid [-π, π] range.

**Base Class:** `ValueError`

**Example:**
```python
from hologen.utils.fields import PhaseRangeError, validate_phase_range
import numpy as np

try:
    invalid_phase = np.array([[4.0]], dtype=np.float64)
    validate_phase_range(invalid_phase)
except PhaseRangeError as e:
    print(f"Error: {e}")
```

---

## See Also

## See Also

- **[Quickstart Guide](QUICKSTART.md)** - Get started with HoloGen in 5 minutes using the API
- **[Shape Generators](SHAPES.md)** - Detailed documentation of shape generator classes and protocols
- **[Holography Methods](HOLOGRAPHY_METHODS.md)** - Inline and off-axis holography strategy implementations
- **[Pipeline Architecture](PIPELINE.md)** - Understanding the dataset generation pipeline components
- **[I/O Formats](IO_FORMATS.md)** - File formats and writer/loader classes
- **[Noise Simulation](NOISE_SIMULATION.md)** - Realistic noise and aberration model classes
- **[Complex Fields](COMPLEX_FIELDS.md)** - Working with complex-valued optical fields and representations
- **[CLI Reference](CLI_REFERENCE.md)** - Command-line interface that uses these APIs
- **[Utilities](UTILITIES.md)** - Utility functions for field conversions and I/O operations
- **[Examples](EXAMPLES.md)** - Practical code examples using the API (all examples)
- **[Master Documentation Index](README.md)** - Complete feature documentation overview
