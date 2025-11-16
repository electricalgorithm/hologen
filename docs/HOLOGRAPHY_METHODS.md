# Holography Methods in HoloGen

HoloGen supports two fundamental approaches to digital holography: inline and off-axis. Each method has distinct optical configurations, reconstruction algorithms, and use cases. This document explains the principles behind each method, their advantages and limitations, and how to use them effectively in your synthetic dataset generation.

## Introduction to Digital Holography

Digital holography is a technique for recording and reconstructing optical wavefronts using digital sensors. Unlike conventional imaging, holography captures both the amplitude and phase of light, enabling three-dimensional information recovery and quantitative phase measurements.

### The Holographic Process

Holography involves three main steps:

1. **Recording**: An object wave interferes with a reference wave at the sensor plane, creating an interference pattern (hologram)
2. **Propagation**: Light waves travel from the object plane to the sensor plane, governed by wave optics
3. **Reconstruction**: The object wave is numerically recovered from the hologram using computational algorithms

### Why Multiple Methods?

Different holography configurations offer different trade-offs:

- **Inline holography**: Simple optical setup, but reconstruction suffers from twin-image problem
- **Off-axis holography**: More complex setup with carrier frequency, but enables clean separation of diffraction orders

The choice depends on your application requirements, optical constraints, and reconstruction quality needs.

## Holography Methods Overview

### Inline Holography

**Optical configuration**: Object and reference waves propagate along the same optical axis

**Key characteristics**:
- Simplest optical setup (no beam splitter required)
- Reference wave is the unscattered illumination
- Hologram contains overlapping diffraction orders
- Reconstruction affected by twin-image artifact
- Suitable for sparse or weakly scattering objects

**Mathematical representation**:
```
Hologram intensity: I = |O + R|² = |O|² + |R|² + O*R + OR*
                                    ↑      ↑      ↑      ↑
                                  object  DC   +1 order -1 order
                                  term   term  (signal) (twin)
```

### Off-Axis Holography

**Optical configuration**: Reference wave arrives at an angle, introducing a spatial carrier frequency

**Key characteristics**:
- Requires beam splitter and angled reference
- Carrier frequency separates diffraction orders in Fourier space
- Clean reconstruction without twin-image
- Requires higher spatial resolution
- Suitable for dense or strongly scattering objects

**Mathematical representation**:
```
Hologram intensity: I = |O + R·exp(i·k·r)|²
                      = |O|² + |R|² + O*R·exp(i·k·r) + OR*·exp(-i·k·r)
                                        ↑                    ↑
                                   +1 order (signal)    -1 order (twin)
                                   shifted by +k        shifted by -k
```


## Inline Holography

### Physical Principles

Inline holography, also known as Gabor holography, uses the simplest optical configuration. The object is illuminated by a plane wave, and the transmitted (or scattered) light interferes with the unscattered portion of the illumination at the sensor plane.

**Optical setup**:
```
Plane wave → Object → Propagation → Sensor
             (O)      (distance z)   (I)
```

The object wave O and reference wave R (unscattered illumination) propagate together along the optical axis and interfere at the sensor.

**Interference pattern**:
```
I(x,y) = |O(x,y) + R|²
       = |O|² + |R|² + O*R + OR*
```

Where:
- `|O|²`: Object intensity (DC term)
- `|R|²`: Reference intensity (DC term)
- `O*R`: First-order diffraction term (desired signal)
- `OR*`: Conjugate term (twin image)

### Advantages

1. **Simple optical setup**: No beam splitter, mirrors, or alignment required
2. **Compact configuration**: Minimal optical components
3. **High light efficiency**: No light loss from beam splitting
4. **Easy to implement**: Straightforward experimental realization
5. **Suitable for sparse objects**: Works well when object occupies small portion of field

### Limitations

1. **Twin-image problem**: Conjugate term creates out-of-focus artifact in reconstruction
2. **DC terms**: Zero-order terms reduce contrast and dynamic range
3. **Limited to weak scattering**: Strong scattering objects create severe artifacts
4. **Overlapping orders**: All diffraction orders overlap spatially
5. **Reconstruction ambiguity**: Cannot cleanly separate signal from artifacts

### Configuration Parameters

```python
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod

# Grid specification
grid = GridSpec(
    height=512,           # Sensor height in pixels
    width=512,            # Sensor width in pixels
    pixel_pitch=6.4e-6    # Pixel size in meters (6.4 μm)
)

# Optical parameters
optics = OpticalConfig(
    wavelength=532e-9,           # Illumination wavelength in meters (532 nm, green)
    propagation_distance=0.05    # Object-to-sensor distance in meters (50 mm)
)

# Holography configuration
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.INLINE,
    carrier=None  # No carrier for inline holography
)
```

**Parameter guidelines**:

- **wavelength**: Typical values 400-700 nm (visible light), 1064 nm (IR laser)
- **propagation_distance**: 
  - Too small (< 10 mm): Minimal diffraction, poor hologram formation
  - Optimal (10-100 mm): Good fringe visibility and reconstruction
  - Too large (> 500 mm): Requires very high resolution, sampling issues
- **pixel_pitch**: Must satisfy Nyquist criterion (see Angular Spectrum Propagation section)

### Use Cases

**Best suited for**:
- Sparse objects (particles, cells, small features)
- Weakly scattering samples
- Compact optical setups
- Real-time holography applications
- Educational demonstrations
- Preliminary experiments

**Not recommended for**:
- Dense objects covering large field of view
- Strongly scattering samples
- Applications requiring high reconstruction quality
- Quantitative phase imaging of complex samples

### Code Example

```python
from hologen.holography.inline import InlineHolographyStrategy
from hologen.types import HolographyConfig, GridSpec, OpticalConfig, HolographyMethod
import numpy as np

# Create configuration
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

# Create strategy
strategy = InlineHolographyStrategy()

# Generate hologram from object field
object_field = np.ones((512, 512), dtype=np.complex128)  # Example object
hologram = strategy.create_hologram(object_field, config)

# Reconstruct object from hologram
reconstruction = strategy.reconstruct(hologram, config)

print(f"Hologram shape: {hologram.shape}")
print(f"Hologram dtype: {hologram.dtype}")
print(f"Reconstruction shape: {reconstruction.shape}")
```


## Off-Axis Holography

### Physical Principles

Off-axis holography introduces a spatial carrier frequency by angling the reference wave relative to the object wave. This angular offset shifts the diffraction orders in Fourier space, enabling clean separation during reconstruction.

**Optical setup**:
```
                    ┌─ Object → Propagation ─┐
Beam Splitter ──────┤                         ├─→ Sensor
                    └─ Reference (angled) ────┘
```

The reference wave arrives at an angle θ, creating a spatial carrier:
```
R(x,y) = exp(i·k·r) = exp(i·2π·(fx·x + fy·y))
```

Where:
- `k = (kx, ky)`: Carrier wave vector
- `fx, fy`: Spatial carrier frequencies (cycles per meter)
- Carrier frequency magnitude: `f = (2/λ)·sin(θ/2)`

**Interference pattern**:
```
I(x,y) = |O(x,y) + R·exp(i·k·r)|²
       = |O|² + |R|² + O*R·exp(i·k·r) + OR*·exp(-i·k·r)
```

In Fourier space, the three terms are spatially separated:
- DC terms: Centered at origin (0, 0)
- +1 order: Shifted to (+fx, +fy) - desired signal
- -1 order: Shifted to (-fx, -fy) - twin image

### Advantages

1. **Twin-image elimination**: Spatial separation enables clean reconstruction
2. **High quality**: No overlapping artifacts in reconstruction
3. **Quantitative phase**: Accurate phase measurements possible
4. **Dense objects**: Works well for complex, strongly scattering samples
5. **Fourier filtering**: Simple bandpass filter isolates signal

### Limitations

1. **Complex optical setup**: Requires beam splitter, mirrors, precise alignment
2. **Higher resolution required**: Carrier frequency demands finer sampling
3. **Reduced field of view**: Carrier consumes spatial bandwidth
4. **Alignment sensitivity**: Misalignment affects reconstruction quality
5. **Lower light efficiency**: Beam splitting reduces signal

### Configuration Parameters

```python
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, 
    HolographyMethod, OffAxisCarrier
)

# Grid specification (higher resolution than inline)
grid = GridSpec(
    height=512,
    width=512,
    pixel_pitch=3.2e-6    # Smaller pixels for carrier sampling
)

# Optical parameters
optics = OpticalConfig(
    wavelength=532e-9,
    propagation_distance=0.05
)

# Carrier configuration
carrier = OffAxisCarrier(
    frequency_x=1e6,        # Carrier frequency in x (cycles/m)
    frequency_y=0,          # Carrier frequency in y (cycles/m)
    gaussian_width=2e5      # Filter width (cycles/m)
)

# Holography configuration
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.OFF_AXIS,
    carrier=carrier
)
```

**Parameter guidelines**:

- **frequency_x, frequency_y**: Carrier spatial frequencies
  - Must satisfy: `sqrt(fx² + fy²) < 1/(2·pixel_pitch)` (Nyquist)
  - Typical: 10-50% of Nyquist frequency
  - Example: For pixel_pitch=3.2μm, max frequency ≈ 1.56e5 cycles/m
  - Common choice: `fx = 1e6 cycles/m` (well below Nyquist)

- **gaussian_width**: Fourier filter bandwidth
  - Too narrow: Signal loss, reduced resolution
  - Too wide: Includes DC or twin-image contamination
  - Typical: 20-40% of carrier frequency
  - Example: For `fx=1e6`, use `gaussian_width=2e5 to 4e5`

### Carrier Frequency Selection

The carrier frequency determines the spatial separation in Fourier space. Choose based on:

1. **Object bandwidth**: Carrier must exceed object's spatial frequency content
2. **Sampling constraint**: Carrier + object bandwidth must fit within Nyquist limit
3. **Separation margin**: Sufficient gap between orders to avoid overlap

**Rule of thumb**:
```
carrier_frequency ≥ 2 × max_object_frequency
carrier_frequency + max_object_frequency < Nyquist_frequency
```

**Example calculation**:
```python
import numpy as np

# System parameters
pixel_pitch = 3.2e-6  # meters
wavelength = 532e-9   # meters
grid_size = 512       # pixels

# Nyquist frequency
nyquist_freq = 1 / (2 * pixel_pitch)  # 1.56e5 cycles/m

# Object bandwidth (estimate from object size)
object_size = 50e-6  # 50 μm object
max_object_freq = 1 / object_size  # 2e4 cycles/m

# Carrier frequency (2x object bandwidth + margin)
carrier_freq = 2.5 * max_object_freq  # 5e4 cycles/m

# Verify constraint
assert carrier_freq + max_object_freq < nyquist_freq
print(f"Carrier frequency: {carrier_freq:.2e} cycles/m")
print(f"Nyquist frequency: {nyquist_freq:.2e} cycles/m")
print(f"Margin: {(nyquist_freq - carrier_freq - max_object_freq)/nyquist_freq * 100:.1f}%")
```

### Fourier Filtering

Reconstruction uses Gaussian filtering in Fourier space to isolate the +1 diffraction order:

```python
# Fourier transform of hologram intensity
spectrum = np.fft.fft2(hologram_intensity)

# Create Gaussian filter centered at carrier frequency
filter_mask = exp(-((fx - carrier_x)² + (fy - carrier_y)²) / (2·σ²))

# Apply filter and inverse transform
filtered_spectrum = spectrum * filter_mask
object_field = np.fft.ifft2(filtered_spectrum)
```

The Gaussian width σ controls the filter bandwidth:
- Narrow filter: High selectivity, but may clip object spectrum
- Wide filter: Preserves object spectrum, but may include noise

### Use Cases

**Best suited for**:
- Dense objects covering significant field of view
- Strongly scattering samples
- Quantitative phase imaging applications
- High-quality reconstruction requirements
- Research-grade holographic microscopy
- Applications where optical complexity is acceptable

**Not recommended for**:
- Compact or portable systems
- Real-time applications with limited computation
- Low-resolution sensors
- Applications requiring maximum field of view

### Code Example

```python
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.types import (
    HolographyConfig, GridSpec, OpticalConfig, 
    HolographyMethod, OffAxisCarrier
)
import numpy as np

# Create configuration with carrier
grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
carrier = OffAxisCarrier(
    frequency_x=1e6,
    frequency_y=0,
    gaussian_width=2e5
)
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.OFF_AXIS,
    carrier=carrier
)

# Create strategy
strategy = OffAxisHolographyStrategy()

# Generate hologram from object field
object_field = np.ones((512, 512), dtype=np.complex128)  # Example object
hologram = strategy.create_hologram(object_field, config)

# Reconstruct object from hologram
reconstruction = strategy.reconstruct(hologram, config)

print(f"Hologram shape: {hologram.shape}")
print(f"Carrier frequency: ({carrier.frequency_x:.2e}, {carrier.frequency_y:.2e}) cycles/m")
print(f"Reconstruction shape: {reconstruction.shape}")
```


## Comparison Matrix

| Feature | Inline Holography | Off-Axis Holography |
|---------|------------------|---------------------|
| **Optical Setup** | Simple (no beam splitter) | Complex (beam splitter + alignment) |
| **Reference Wave** | Unscattered illumination | Angled plane wave |
| **Carrier Frequency** | None | Required |
| **Twin Image** | Present (overlapping) | Separated (removable) |
| **Reconstruction Quality** | Lower (artifacts) | Higher (clean) |
| **Resolution Requirement** | Standard | Higher (for carrier) |
| **Field of View** | Maximum | Reduced (carrier bandwidth) |
| **Computation** | Simple back-propagation | Fourier filtering + back-propagation |
| **Best For** | Sparse objects | Dense objects |
| **Typical Applications** | Particle tracking, simple imaging | Quantitative phase imaging, microscopy |
| **Light Efficiency** | High (no splitting) | Lower (beam splitting) |
| **Alignment Sensitivity** | Low | High |
| **Real-time Capability** | Easier | More challenging |

### When to Choose Inline

Choose inline holography when:
- Optical setup must be simple and compact
- Objects are sparse or weakly scattering
- Twin-image artifacts are acceptable
- Maximum field of view is needed
- Real-time processing is required
- Educational or demonstration purposes
- Preliminary experiments or prototyping

### When to Choose Off-Axis

Choose off-axis holography when:
- High reconstruction quality is critical
- Objects are dense or strongly scattering
- Quantitative phase measurements are needed
- Twin-image artifacts are unacceptable
- Optical complexity is acceptable
- Sufficient spatial resolution is available
- Research-grade results are required
- Publication-quality images are needed

### Hybrid Approaches

Some applications benefit from combining both methods:

1. **Inline for screening, off-axis for detailed analysis**: Use inline for rapid screening, then off-axis for selected samples
2. **Dual-mode systems**: Switchable configuration supporting both methods
3. **Computational methods**: Use inline acquisition with computational twin-image removal


## Angular Spectrum Propagation

Both inline and off-axis holography use the angular spectrum method for wave propagation. This section explains the underlying physics and implementation.

### Physical Principle

The angular spectrum method decomposes an optical field into plane waves, propagates each plane wave independently, and recombines them at the observation plane.

**Mathematical formulation**:

1. **Decompose** field into plane waves (Fourier transform):
   ```
   Ã(fx, fy, z=0) = FFT[A(x, y, z=0)]
   ```

2. **Propagate** each plane wave by distance z:
   ```
   Ã(fx, fy, z) = Ã(fx, fy, 0) · H(fx, fy, z)
   ```
   
   Where the transfer function is:
   ```
   H(fx, fy, z) = exp(i·kz·z)
   ```
   
   And the longitudinal wave vector component is:
   ```
   kz = k·sqrt(1 - (λ·fx)² - (λ·fy)²)  for propagating waves
   kz = i·k·sqrt((λ·fx)² + (λ·fy)² - 1)  for evanescent waves
   ```

3. **Recompose** field in spatial domain (inverse Fourier transform):
   ```
   A(x, y, z) = IFFT[Ã(fx, fy, z)]
   ```

### Propagating vs Evanescent Waves

The angular spectrum distinguishes two types of plane waves:

**Propagating waves** (real kz):
- Condition: `(λ·fx)² + (λ·fy)² < 1`
- Behavior: Oscillate with phase `exp(i·kz·z)`
- Carry energy to observation plane
- Contribute to far-field pattern

**Evanescent waves** (imaginary kz):
- Condition: `(λ·fx)² + (λ·fy)² ≥ 1`
- Behavior: Decay exponentially with distance
- Do not propagate to far field
- Contain sub-wavelength information

### Nyquist Criterion

The angular spectrum method requires proper spatial sampling to avoid aliasing:

**Sampling constraint**:
```
pixel_pitch < λ / 2
```

More precisely, the maximum representable spatial frequency must be less than the physical limit:
```
fmax = 1 / (2·pixel_pitch) < 1 / λ
```

**Example validation**:
```python
wavelength = 532e-9  # 532 nm
pixel_pitch = 6.4e-6  # 6.4 μm

max_spatial_freq = 1 / (2 * pixel_pitch)  # 7.8e4 cycles/m
physical_limit = 1 / wavelength  # 1.88e6 cycles/m
normalized_freq = max_spatial_freq * wavelength  # 0.042

assert normalized_freq < 1.0, "Nyquist criterion violated!"
print(f"Sampling is valid: {normalized_freq:.3f} < 1.0")
```

### Implementation Details

HoloGen's `angular_spectrum_propagate()` function handles:

1. **Validation**: Checks field shape matches grid dimensions
2. **Short-circuit**: Returns input for zero distance
3. **Nyquist check**: Validates sampling criterion
4. **Frequency grid**: Computes spatial frequency coordinates
5. **Transfer function**: Calculates propagation kernel
6. **Evanescent handling**: Applies exponential decay for evanescent components
7. **FFT propagation**: Applies kernel in Fourier domain
8. **Type preservation**: Returns complex128 array

**Function signature**:
```python
def angular_spectrum_propagate(
    field: ArrayComplex,
    grid: GridSpec,
    optics: OpticalConfig,
    distance: float,
) -> ArrayComplex:
    """Propagate complex optical field using angular spectrum method.
    
    Args:
        field: Complex field at source plane, shape (grid.height, grid.width)
        grid: Spatial sampling specification
        optics: Optical parameters (wavelength)
        distance: Propagation distance in meters (positive = forward, negative = backward)
    
    Returns:
        Complex field at observation plane
    
    Raises:
        ValueError: If field shape doesn't match grid or Nyquist criterion violated
    """
```

### Forward and Backward Propagation

The angular spectrum method supports bidirectional propagation:

**Forward propagation** (positive distance):
- Object plane → Sensor plane
- Used in hologram generation
- Distance: `z = propagation_distance`

**Backward propagation** (negative distance):
- Sensor plane → Object plane
- Used in reconstruction
- Distance: `z = -propagation_distance`

**Example**:
```python
from hologen.holography.propagation import angular_spectrum_propagate
from hologen.types import GridSpec, OpticalConfig
import numpy as np

# Configuration
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)

# Object field
object_field = np.ones((512, 512), dtype=np.complex128)

# Forward propagation (object → sensor)
sensor_field = angular_spectrum_propagate(
    field=object_field,
    grid=grid,
    optics=optics,
    distance=optics.propagation_distance  # +0.05 m
)

# Backward propagation (sensor → object)
reconstructed_field = angular_spectrum_propagate(
    field=sensor_field,
    grid=grid,
    optics=optics,
    distance=-optics.propagation_distance  # -0.05 m
)

# Verify round-trip
error = np.abs(reconstructed_field - object_field).max()
print(f"Round-trip error: {error:.2e}")  # Should be very small
```

### Propagation Distance Guidelines

**Too short** (< 1 mm):
- Minimal diffraction
- Hologram looks like object
- Poor fringe formation
- Limited depth information

**Optimal** (10-100 mm):
- Good fringe visibility
- Clear diffraction patterns
- Reasonable sampling requirements
- Practical optical setup

**Too long** (> 500 mm):
- Very fine fringes
- High resolution required
- Sampling challenges
- Large optical setup

**Fresnel number** as guide:
```
F = a² / (λ·z)
```

Where:
- `a`: Object characteristic size
- `λ`: Wavelength
- `z`: Propagation distance

- `F >> 1`: Near field (geometric optics)
- `F ≈ 1`: Fresnel region (optimal for holography)
- `F << 1`: Far field (Fraunhofer diffraction)


## API Reference

### HolographyStrategy Protocol

The `HolographyStrategy` protocol defines the interface for holography implementations.

```python
from typing import Protocol
from hologen.types import ArrayComplex, HolographyConfig

class HolographyStrategy(Protocol):
    """Protocol for holography strategy implementations.
    
    Implementations must provide methods for hologram generation
    and reconstruction using specific holography techniques.
    """
    
    def create_hologram(
        self,
        object_field: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Generate hologram from object-domain complex field.
        
        Args:
            object_field: Complex field at object plane, shape (H, W)
            config: Holography configuration including grid, optics, method
        
        Returns:
            Complex field at sensor plane (hologram)
        
        Raises:
            ValueError: If configuration is invalid for this strategy
        """
        ...
    
    def reconstruct(
        self,
        hologram: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct object-domain field from hologram.
        
        Args:
            hologram: Complex field at sensor plane
            config: Holography configuration (must match creation config)
        
        Returns:
            Complex field at object plane (reconstruction)
        
        Raises:
            ValueError: If configuration is invalid for this strategy
        """
        ...
```

### InlineHolographyStrategy

Implementation of inline (Gabor) holography.

```python
from hologen.holography.inline import InlineHolographyStrategy

class InlineHolographyStrategy:
    """Inline holography implementation.
    
    Generates holograms by propagating the object field to the sensor plane
    without additional reference wave modulation. Reconstruction uses
    back-propagation to the object plane.
    
    This method produces holograms with overlapping diffraction orders,
    resulting in twin-image artifacts in reconstruction.
    """
    
    def create_hologram(
        self,
        object_field: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Generate inline hologram.
        
        Process:
        1. Propagate object field to sensor plane using angular spectrum
        2. Return complex field (interference with unscattered wave implicit)
        
        Args:
            object_field: Complex object field, shape (H, W)
            config: Configuration with grid, optics, method=INLINE
        
        Returns:
            Complex hologram field at sensor plane
        
        Example:
            >>> strategy = InlineHolographyStrategy()
            >>> hologram = strategy.create_hologram(object_field, config)
        """
    
    def reconstruct(
        self,
        hologram: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct object from inline hologram.
        
        Process:
        1. Back-propagate hologram field to object plane
        2. Return reconstructed complex field
        
        Note: Reconstruction contains twin-image artifacts due to
        overlapping diffraction orders.
        
        Args:
            hologram: Complex hologram field, shape (H, W)
            config: Configuration (must match creation config)
        
        Returns:
            Complex reconstructed field at object plane
        
        Example:
            >>> reconstruction = strategy.reconstruct(hologram, config)
        """
```

**Usage example**:
```python
from hologen.holography.inline import InlineHolographyStrategy
from hologen.types import GridSpec, OpticalConfig, HolographyConfig, HolographyMethod
import numpy as np

# Setup
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(grid=grid, optics=optics, method=HolographyMethod.INLINE)

# Create strategy
strategy = InlineHolographyStrategy()

# Generate and reconstruct
object_field = np.ones((512, 512), dtype=np.complex128)
hologram = strategy.create_hologram(object_field, config)
reconstruction = strategy.reconstruct(hologram, config)
```

### OffAxisHolographyStrategy

Implementation of off-axis holography with spatial carrier.

```python
from hologen.holography.off_axis import OffAxisHolographyStrategy

class OffAxisHolographyStrategy:
    """Off-axis holography implementation.
    
    Generates holograms by adding a tilted reference wave to the propagated
    object field. Reconstruction uses Fourier filtering to isolate the
    first-order diffraction term, followed by back-propagation.
    
    This method produces clean reconstructions without twin-image artifacts.
    """
    
    def create_hologram(
        self,
        object_field: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Generate off-axis hologram.
        
        Process:
        1. Propagate object field to sensor plane
        2. Generate tilted reference wave with carrier frequency
        3. Add reference to propagated object field
        4. Return complex interference pattern
        
        Args:
            object_field: Complex object field, shape (H, W)
            config: Configuration with carrier parameters
        
        Returns:
            Complex hologram field with carrier modulation
        
        Raises:
            ValueError: If config.carrier is None
        
        Example:
            >>> strategy = OffAxisHolographyStrategy()
            >>> hologram = strategy.create_hologram(object_field, config)
        """
    
    def reconstruct(
        self,
        hologram: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct object from off-axis hologram.
        
        Process:
        1. Convert hologram to intensity
        2. Fourier transform to frequency domain
        3. Apply Gaussian filter centered at carrier frequency
        4. Inverse Fourier transform
        5. Back-propagate to object plane
        
        Args:
            hologram: Complex hologram field, shape (H, W)
            config: Configuration with carrier parameters
        
        Returns:
            Complex reconstructed field (clean, no twin image)
        
        Raises:
            ValueError: If config.carrier is None
        
        Example:
            >>> reconstruction = strategy.reconstruct(hologram, config)
        """
```

**Usage example**:
```python
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, 
    HolographyMethod, OffAxisCarrier
)
import numpy as np

# Setup with carrier
grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
carrier = OffAxisCarrier(frequency_x=1e6, frequency_y=0, gaussian_width=2e5)
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.OFF_AXIS,
    carrier=carrier
)

# Create strategy
strategy = OffAxisHolographyStrategy()

# Generate and reconstruct
object_field = np.ones((512, 512), dtype=np.complex128)
hologram = strategy.create_hologram(object_field, config)
reconstruction = strategy.reconstruct(hologram, config)
```

### angular_spectrum_propagate()

Core propagation function used by both strategies.

```python
from hologen.holography.propagation import angular_spectrum_propagate

def angular_spectrum_propagate(
    field: ArrayComplex,
    grid: GridSpec,
    optics: OpticalConfig,
    distance: float,
) -> ArrayComplex:
    """Propagate complex optical field using angular spectrum method.
    
    Decomposes field into plane waves, applies appropriate phase shifts
    (or evanescent decay) for given propagation distance, and recomposes
    the field at the observation plane.
    
    Args:
        field: Complex field at source plane, shape (grid.height, grid.width)
        grid: Spatial sampling specification (height, width, pixel_pitch)
        optics: Optical parameters (wavelength)
        distance: Propagation distance in meters
                  Positive: forward propagation (object → sensor)
                  Negative: backward propagation (sensor → object)
    
    Returns:
        Complex field at observation plane, dtype complex128
    
    Raises:
        ValueError: If field shape doesn't match grid dimensions
        ValueError: If Nyquist criterion violated (pixel_pitch too large)
    
    Notes:
        - Assumes monochromatic illumination
        - Handles both propagating and evanescent waves
        - Validates sampling criterion: pixel_pitch < λ/2
        - Returns immediately for distance=0 (no-op)
    
    Example:
        >>> from hologen.holography.propagation import angular_spectrum_propagate
        >>> from hologen.types import GridSpec, OpticalConfig
        >>> import numpy as np
        >>> 
        >>> grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
        >>> optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
        >>> field = np.ones((512, 512), dtype=np.complex128)
        >>> 
        >>> # Forward propagation
        >>> propagated = angular_spectrum_propagate(field, grid, optics, 0.05)
        >>> 
        >>> # Backward propagation
        >>> reconstructed = angular_spectrum_propagate(propagated, grid, optics, -0.05)
    """
```

### Configuration Types

#### HolographyConfig

```python
@dataclass(slots=True)
class HolographyConfig:
    """Configuration bundle for holography strategies.
    
    Attributes:
        grid: Spatial sampling specification
        optics: Optical parameters (wavelength, distance)
        method: Holography method (INLINE or OFF_AXIS)
        carrier: Carrier configuration (required for OFF_AXIS, None for INLINE)
    """
    grid: GridSpec
    optics: OpticalConfig
    method: HolographyMethod
    carrier: OffAxisCarrier | None = None
```

#### OffAxisCarrier

```python
@dataclass(slots=True)
class OffAxisCarrier:
    """Carrier modulation parameters for off-axis holography.
    
    Attributes:
        frequency_x: Spatial carrier frequency along x-axis (cycles/m)
        frequency_y: Spatial carrier frequency along y-axis (cycles/m)
        gaussian_width: Standard deviation of Gaussian filter (cycles/m)
    
    Constraints:
        - sqrt(frequency_x² + frequency_y²) < Nyquist frequency
        - gaussian_width typically 20-40% of carrier frequency magnitude
    """
    frequency_x: float
    frequency_y: float
    gaussian_width: float
```


## Complete Pipeline Examples

### Example 1: Inline Holography Pipeline

Complete example generating inline holograms with phase-only objects:

```python
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

# 1. Configure system
grid = GridSpec(height=512, width=512, pixel_pitch=6.4e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.INLINE,
    carrier=None  # No carrier for inline
)

# 2. Configure output
output_config = OutputConfig(
    object_representation=FieldRepresentation.PHASE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# 3. Create pipeline components
shape_generator = CircleGenerator(radius_range=(20e-6, 50e-6))
object_producer = ObjectDomainProducer(
    generator=shape_generator,
    phase_shift=np.pi/2,
    mode="phase"
)

strategy = InlineHolographyStrategy()
converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.INLINE: strategy},
    output_config=output_config
)

dataset_generator = HologramDatasetGenerator(
    object_producer=object_producer,
    converter=converter,
    config=config
)

# 4. Generate dataset
rng = np.random.default_rng(42)
samples = dataset_generator.generate(count=100, rng=rng)

# 5. Write to disk
writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
writer.save(samples, output_dir=Path("inline_dataset"))

print("Inline holography dataset generated successfully!")
```

### Example 2: Off-Axis Holography Pipeline

Complete example generating off-axis holograms:

```python
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter, HologramDatasetGenerator
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.shapes import RectangleGenerator
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, HolographyMethod,
    OffAxisCarrier, FieldRepresentation, OutputConfig
)
from hologen.utils.io import ComplexFieldWriter
from pathlib import Path
import numpy as np

# 1. Configure system with carrier
grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)  # Smaller pixels
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
carrier = OffAxisCarrier(
    frequency_x=1e6,
    frequency_y=0,
    gaussian_width=2e5
)
config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.OFF_AXIS,
    carrier=carrier
)

# 2. Configure output
output_config = OutputConfig(
    object_representation=FieldRepresentation.AMPLITUDE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# 3. Create pipeline components
shape_generator = RectangleGenerator(
    width_range=(30e-6, 60e-6),
    height_range=(30e-6, 60e-6)
)
object_producer = ObjectDomainProducer(
    generator=shape_generator,
    mode="amplitude"
)

strategy = OffAxisHolographyStrategy()
converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.OFF_AXIS: strategy},
    output_config=output_config
)

dataset_generator = HologramDatasetGenerator(
    object_producer=object_producer,
    converter=converter,
    config=config
)

# 4. Generate dataset
rng = np.random.default_rng(42)
samples = dataset_generator.generate(count=100, rng=rng)

# 5. Write to disk
writer = ComplexFieldWriter(save_preview=True, phase_colormap="twilight")
writer.save(samples, output_dir=Path("offaxis_dataset"))

print("Off-axis holography dataset generated successfully!")
```

### Example 3: Comparing Methods

Generate datasets with both methods for comparison:

```python
from hologen.converters import ObjectDomainProducer, ObjectToHologramConverter, HologramDatasetGenerator
from hologen.holography.inline import InlineHolographyStrategy
from hologen.holography.off_axis import OffAxisHolographyStrategy
from hologen.shapes import CircleGenerator
from hologen.types import (
    GridSpec, OpticalConfig, HolographyConfig, HolographyMethod,
    OffAxisCarrier, FieldRepresentation, OutputConfig
)
from hologen.utils.io import ComplexFieldWriter
from pathlib import Path
import numpy as np

# Shared configuration
grid = GridSpec(height=512, width=512, pixel_pitch=3.2e-6)
optics = OpticalConfig(wavelength=532e-9, propagation_distance=0.05)
shape_generator = CircleGenerator(radius_range=(20e-6, 50e-6))
rng = np.random.default_rng(42)

output_config = OutputConfig(
    object_representation=FieldRepresentation.PHASE,
    hologram_representation=FieldRepresentation.COMPLEX,
    reconstruction_representation=FieldRepresentation.COMPLEX
)

# Inline configuration
inline_config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.INLINE
)

# Off-axis configuration
carrier = OffAxisCarrier(frequency_x=1e6, frequency_y=0, gaussian_width=2e5)
offaxis_config = HolographyConfig(
    grid=grid,
    optics=optics,
    method=HolographyMethod.OFF_AXIS,
    carrier=carrier
)

# Generate inline dataset
object_producer = ObjectDomainProducer(
    generator=shape_generator,
    phase_shift=np.pi/2,
    mode="phase"
)
inline_strategy = InlineHolographyStrategy()
inline_converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.INLINE: inline_strategy},
    output_config=output_config
)
inline_generator = HologramDatasetGenerator(
    object_producer=object_producer,
    converter=inline_converter,
    config=inline_config
)
inline_samples = inline_generator.generate(count=50, rng=rng)
writer = ComplexFieldWriter(save_preview=True)
writer.save(inline_samples, output_dir=Path("comparison/inline"))

# Generate off-axis dataset (same objects, different method)
rng = np.random.default_rng(42)  # Reset for same objects
offaxis_strategy = OffAxisHolographyStrategy()
offaxis_converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.OFF_AXIS: offaxis_strategy},
    output_config=output_config
)
offaxis_generator = HologramDatasetGenerator(
    object_producer=object_producer,
    converter=offaxis_converter,
    config=offaxis_config
)
offaxis_samples = offaxis_generator.generate(count=50, rng=rng)
writer.save(offaxis_samples, output_dir=Path("comparison/offaxis"))

print("Comparison datasets generated!")
print("Compare reconstruction quality in comparison/inline vs comparison/offaxis")
```

### Example 4: Custom Strategy Implementation

Implement a custom holography strategy:

```python
from hologen.types import ArrayComplex, HolographyConfig
from hologen.holography.propagation import angular_spectrum_propagate
import numpy as np

class CustomHolographyStrategy:
    """Custom holography strategy with modified reconstruction."""
    
    def create_hologram(
        self,
        object_field: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Standard inline hologram generation."""
        propagated = angular_spectrum_propagate(
            field=object_field,
            grid=config.grid,
            optics=config.optics,
            distance=config.optics.propagation_distance
        )
        return propagated
    
    def reconstruct(
        self,
        hologram: ArrayComplex,
        config: HolographyConfig
    ) -> ArrayComplex:
        """Custom reconstruction with additional processing."""
        # Standard back-propagation
        reconstructed = angular_spectrum_propagate(
            field=hologram,
            grid=config.grid,
            optics=config.optics,
            distance=-config.optics.propagation_distance
        )
        
        # Custom post-processing (example: phase unwrapping, filtering, etc.)
        # Add your custom processing here
        
        return reconstructed

# Use custom strategy
from hologen.converters import ObjectToHologramConverter
from hologen.types import HolographyMethod

custom_strategy = CustomHolographyStrategy()
converter = ObjectToHologramConverter(
    strategy_mapping={HolographyMethod.INLINE: custom_strategy},
    output_config=output_config
)
```


## CLI Usage

Generate datasets using different holography methods from the command line.

### Inline Holography

**Basic inline hologram generation**:
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./inline_dataset \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05
```

**Inline with phase-only objects**:
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./inline_phase \
    --method inline \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05
```

### Off-Axis Holography

**Basic off-axis hologram generation**:
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./offaxis_dataset \
    --method off_axis \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --gaussian-width 2e5 \
    --height 512 \
    --width 512 \
    --pixel-pitch 3.2e-6 \
    --wavelength 532e-9 \
    --distance 0.05
```

**Off-axis with phase objects**:
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./offaxis_phase \
    --method off_axis \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --gaussian-width 2e5 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --height 512 \
    --width 512 \
    --pixel-pitch 3.2e-6 \
    --wavelength 532e-9 \
    --distance 0.05
```

### Comparing Methods

Generate both inline and off-axis datasets for comparison:

```bash
# Inline dataset
python scripts/generate_dataset.py \
    --samples 50 \
    --output ./comparison/inline \
    --method inline \
    --seed 42 \
    --height 512 \
    --width 512 \
    --pixel-pitch 3.2e-6 \
    --wavelength 532e-9 \
    --distance 0.05

# Off-axis dataset (same seed for same objects)
python scripts/generate_dataset.py \
    --samples 50 \
    --output ./comparison/offaxis \
    --method off_axis \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --gaussian-width 2e5 \
    --seed 42 \
    --height 512 \
    --width 512 \
    --pixel-pitch 3.2e-6 \
    --wavelength 532e-9 \
    --distance 0.05
```

### Parameter Guidelines

**For inline holography**:
- Use standard pixel pitch (6.4 μm typical)
- No carrier parameters needed
- Suitable for sparse objects
- Faster generation

**For off-axis holography**:
- Use smaller pixel pitch (3.2 μm or less) to sample carrier
- Must specify carrier-frequency-x, carrier-frequency-y
- Must specify gaussian-width for filtering
- Suitable for dense objects
- Slower generation (Fourier filtering overhead)

### Common Argument Combinations

**Inline, intensity output (default)**:
```bash
python scripts/generate_dataset.py --method inline
```

**Inline, complex output**:
```bash
python scripts/generate_dataset.py \
    --method inline \
    --output-domain complex
```

**Off-axis, complex output**:
```bash
python scripts/generate_dataset.py \
    --method off_axis \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --gaussian-width 2e5 \
    --output-domain complex
```

**Off-axis with noise**:
```bash
python scripts/generate_dataset.py \
    --method off_axis \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --gaussian-width 2e5 \
    --sensor-read-noise 3.0 \
    --sensor-shot-noise \
    --speckle-contrast 0.8
```


## Best Practices

### Choosing Propagation Distance

1. **Start with Fresnel number ≈ 1**:
   ```python
   # For object size a = 50 μm, wavelength λ = 532 nm
   a = 50e-6
   wavelength = 532e-9
   z_optimal = a**2 / wavelength  # ≈ 4.7 mm
   ```

2. **Verify fringe visibility**: Generate test hologram and check for clear interference patterns

3. **Adjust based on application**:
   - Particle tracking: 10-50 mm
   - Microscopy: 1-10 mm
   - Large objects: 50-200 mm

### Optimizing Carrier Frequency

For off-axis holography:

1. **Estimate object bandwidth**:
   ```python
   object_size = 50e-6  # meters
   max_object_freq = 1 / object_size  # 2e4 cycles/m
   ```

2. **Choose carrier frequency**:
   ```python
   carrier_freq = 2.5 * max_object_freq  # 5e4 cycles/m
   ```

3. **Verify Nyquist constraint**:
   ```python
   nyquist_freq = 1 / (2 * pixel_pitch)
   assert carrier_freq + max_object_freq < nyquist_freq
   ```

4. **Set filter width**:
   ```python
   gaussian_width = 0.3 * carrier_freq  # 30% of carrier
   ```

### Sampling Considerations

1. **Inline holography**: Standard sampling (pixel_pitch ≈ 6.4 μm)
2. **Off-axis holography**: Finer sampling (pixel_pitch ≈ 3.2 μm or less)
3. **Always verify**: `pixel_pitch < wavelength / 2`

### Reconstruction Quality

**For inline holography**:
- Accept twin-image artifacts for sparse objects
- Consider computational twin-image removal for critical applications
- Use multiple propagation distances for depth information

**For off-axis holography**:
- Optimize carrier frequency for clean separation
- Adjust Gaussian filter width to balance resolution and noise
- Verify no overlap between diffraction orders in Fourier space

### Performance Optimization

1. **Batch generation**: Generate multiple samples in parallel
2. **Memory management**: Use generators for large datasets
3. **FFT optimization**: Ensure grid dimensions are powers of 2 for faster FFT
4. **Caching**: Reuse propagation kernels when possible

### Validation Checklist

Before generating large datasets:

- [ ] Verify Nyquist criterion satisfied
- [ ] Check Fresnel number in reasonable range (0.1 - 10)
- [ ] Generate test sample and inspect visually
- [ ] Verify reconstruction quality acceptable
- [ ] Check file sizes and storage requirements
- [ ] Validate parameter ranges for your application
- [ ] Test loading data in your ML framework

### Common Pitfalls

1. **Pixel pitch too large**: Violates Nyquist criterion, causes aliasing
2. **Carrier frequency too high**: Exceeds Nyquist limit, creates artifacts
3. **Gaussian width too narrow**: Clips object spectrum, reduces resolution
4. **Gaussian width too wide**: Includes DC or twin-image contamination
5. **Propagation distance too short**: Poor hologram formation
6. **Propagation distance too long**: Requires impractical resolution

### Troubleshooting

**Problem**: Hologram looks like object (no fringes)
- **Solution**: Increase propagation distance

**Problem**: Reconstruction is blurry
- **Solution**: Check Nyquist criterion, reduce pixel pitch

**Problem**: Off-axis reconstruction has artifacts
- **Solution**: Adjust carrier frequency or Gaussian width

**Problem**: ValueError: Nyquist criterion violated
- **Solution**: Reduce pixel pitch or increase wavelength

**Problem**: Reconstruction has twin-image (inline)
- **Solution**: This is expected; use off-axis or computational removal

**Problem**: Off-axis reconstruction is empty
- **Solution**: Check carrier frequency matches hologram generation


## See Also

- **[Complex Field Support](COMPLEX_FIELDS.md)**: Learn about field representations (intensity, amplitude, phase, complex) and how to generate phase-only objects
- **[Noise Simulation](NOISE_SIMULATION.md)**: Add realistic sensor noise, speckle, and aberrations to holograms
- **[Shape Generators](SHAPES.md)**: Explore available object-domain shape generators
- **[Quickstart Guide](QUICKSTART.md)**: Get started quickly with basic examples
- **[API Reference](API_REFERENCE.md)**: Complete API documentation for all classes and functions

### Related Topics

- **Angular Spectrum Method**: See `hologen.holography.propagation` module
- **Dataset Generation Pipeline**: See `hologen.converters` module
- **I/O and File Formats**: See `hologen.utils.io` module
- **Configuration Types**: See `hologen.types` module

### Further Reading

**Digital holography fundamentals**:
- Schnars, U., & Jüptner, W. (2005). *Digital Holography: Digital Hologram Recording, Numerical Reconstruction, and Related Techniques*. Springer.
- Kreis, T. (2005). *Handbook of Holographic Interferometry: Optical and Digital Methods*. Wiley-VCH.

**Angular spectrum method**:
- Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.). Roberts and Company Publishers.
- Matsushima, K., & Shimobaba, T. (2009). Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields. *Optics Express*, 17(22), 19662-19673.

**Off-axis holography**:
- Cuche, E., Marquet, P., & Depeursinge, C. (1999). Simultaneous amplitude-contrast and quantitative phase-contrast microscopy by numerical reconstruction of Fresnel off-axis holograms. *Applied Optics*, 38(34), 6994-7001.

**Inline holography and twin-image problem**:
- Gabor, D. (1948). A new microscopic principle. *Nature*, 161(4098), 777-778.
- Latychevskaia, T., & Fink, H. W. (2015). Solution to the twin image problem in holography. *Physical Review Letters*, 98(23), 233901.

### Example Datasets

HoloGen includes example scripts for generating comparison datasets:

```bash
# Generate visual examples
python scripts/generate_visual_examples.py

# Generate shape examples
python scripts/generate_shape_examples.py

# Generate full dataset
python scripts/generate_dataset.py --samples 1000
```

Check the `docs/examples/` directory for pre-generated visual examples demonstrating different holography methods.

## See Also

- **[Shape Generators](SHAPES.md)** - Learn about object-domain pattern generation that feeds into holography methods
- **[Complex Field Support](COMPLEX_FIELDS.md)** - Understand field representations used in holography (amplitude, phase, complex)
- **[Pipeline Architecture](PIPELINE.md)** - See how holography strategies integrate into the complete generation pipeline
- **[Noise Simulation](NOISE_SIMULATION.md)** - Add realistic noise to holograms after generation
- **[CLI Reference](CLI_REFERENCE.md)** - Command-line options for selecting holography methods (--method inline/off_axis)
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for InlineHolographyStrategy and OffAxisHolographyStrategy
- **[I/O Formats](IO_FORMATS.md)** - File formats for storing hologram data
- **[Examples](EXAMPLES.md)** - Practical examples including custom reconstruction algorithms (Example 9)
- **[Quickstart Guide](QUICKSTART.md)** - Quick start with inline and off-axis holography

