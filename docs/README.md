# HoloGen Documentation

Welcome to the comprehensive documentation for HoloGen, a synthetic hologram dataset generation toolkit for machine learning workflows. This documentation covers all features, from basic usage to advanced customization.

## Quick Navigation

### Getting Started
- [Installation & Setup](#installation--setup)
- [5-Minute Quickstart](#5-minute-quickstart)
- [Complex Fields Quick Start](COMPLEX_FIELDS_QUICKSTART.md) - Fast introduction to complex field support

### By User Role

#### Beginners
Start here if you're new to HoloGen or digital holography:
1. [Installation & Setup](#installation--setup)
2. [5-Minute Quickstart](#5-minute-quickstart)
3. [Complex Fields Quick Start](COMPLEX_FIELDS_QUICKSTART.md)
4. [Noise Simulation Guide](NOISE_SIMULATION.md)

#### Researchers
For experimental holography and ML research:
1. [Complex Field Support](COMPLEX_FIELDS.md) - Amplitude, phase, and complex representations
2. [Noise Simulation](NOISE_SIMULATION.md) - Realistic optical and sensor imperfections
3. [Holography Methods](HOLOGRAPHY_METHODS.md) - Inline vs off-axis strategies
4. [Examples & Recipes](EXAMPLES.md) - Research workflows and parameter studies

#### Developers
For extending HoloGen or building custom pipelines:
1. [Pipeline Architecture](PIPELINE.md) - Component design and data flow
2. [API Reference](API_REFERENCE.md) - Complete API documentation
3. [Shape Generators](SHAPES.md) - Creating custom shapes
4. [Utilities Reference](UTILITIES.md) - Helper functions

### By Feature

#### Core Features
- [Complex Field Support](COMPLEX_FIELDS.md) - **Complete** - Amplitude, phase, and complex representations
- [Noise Simulation](NOISE_SIMULATION.md) - **Complete** - Sensor noise, speckle, and aberrations
- [Shape Generators](SHAPES.md) - **Complete** - Object-domain pattern generation
- [Holography Methods](HOLOGRAPHY_METHODS.md) - **Complete** - Inline and off-axis strategies

#### Pipeline & I/O
- [Pipeline Architecture](PIPELINE.md) - **Complete** - Dataset generation workflow
- [I/O Formats](IO_FORMATS.md) - **Complete** - File formats and data loading
- [CLI Reference](CLI_REFERENCE.md) - **Complete** - Command-line interface guide

#### Reference
- [API Reference](API_REFERENCE.md) - **Complete** - Complete API documentation
- [Utilities Reference](UTILITIES.md) - **Complete** - Utility functions
- [Examples & Recipes](EXAMPLES.md) - **Complete** - Practical code examples

## Feature Matrix

| Feature | Status | Documentation | Description |
|---------|--------|---------------|-------------|
| **Complex Field Support** | ✅ Complete | [COMPLEX_FIELDS.md](COMPLEX_FIELDS.md) | Amplitude, phase, and complex representations |
| **Noise Simulation** | ✅ Complete | [NOISE_SIMULATION.md](NOISE_SIMULATION.md) | Sensor noise, speckle, aberrations |
| **Inline Holography** | ✅ Complete | [HOLOGRAPHY_METHODS.md](HOLOGRAPHY_METHODS.md) | On-axis hologram recording |
| **Off-Axis Holography** | ✅ Complete | [HOLOGRAPHY_METHODS.md](HOLOGRAPHY_METHODS.md) | Carrier-based hologram recording |
| **Shape Generators** | ✅ Complete | [SHAPES.md](SHAPES.md) | Circle, rectangle, ring, checkerboard patterns |
| **Dataset Writers** | ✅ Complete | [IO_FORMATS.md](IO_FORMATS.md) | NumPy .npz and PNG export |
| **Reconstruction Pipeline** | ✅ Complete | [PIPELINE.md](PIPELINE.md) | Object-domain recovery |
| **CLI Interface** | ✅ Complete | [CLI_REFERENCE.md](CLI_REFERENCE.md) | Command-line dataset generation |
| **PyTorch Integration** | ✅ Complete | [IO_FORMATS.md](IO_FORMATS.md#pytorch-dataloader) | DataLoader examples |
| **TensorFlow Integration** | ✅ Complete | [IO_FORMATS.md](IO_FORMATS.md#tensorflow-dataset) | Dataset examples |

## Installation & Setup

### Requirements
- Python 3.11 or higher
- NumPy 2.3.x
- Pillow 12.x
- SciPy 1.16.x

### Installation Steps

1. **Clone the repository** (or download the source):
   ```bash
   git clone <repository-url>
   cd hologen
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install HoloGen in editable mode**:
   ```bash
   pip install -e .
   ```

5. **Verify installation**:
   ```bash
   python -c "import hologen; print(hologen.__version__)"
   ```

### Optional Dependencies

For development and testing:
```bash
pip install pytest pytest-cov black ruff
```

For visualization examples:
```bash
pip install matplotlib
```

## 5-Minute Quickstart

### Generate Your First Dataset

Generate a simple dataset with default settings:

```bash
python scripts/generate_dataset.py
```

This creates 5 samples in the `dataset/` directory with:
- 512×512 pixel resolution
- Inline holography method
- Amplitude-only objects (circles, rectangles, rings)
- Intensity output (backward compatible)

### View the Results

The dataset includes:
```
dataset/
├── npz/                    # NumPy data files
│   ├── sample_00000_circle.npz
│   ├── sample_00001_ring.npz
│   └── ...
└── preview/                # PNG visualizations
    ├── object/
    ├── hologram/
    └── reconstruction/
```

### Common Use Cases

#### 1. Phase-Only Objects (Transparent Samples)
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708
```

**Use case**: Biological cell imaging, quantitative phase imaging

#### 2. Noisy Realistic Holograms
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --sensor-shot-noise \
    --sensor-read-noise 3.0 \
    --speckle-contrast 0.8
```

**Use case**: Training robust reconstruction models

#### 3. Off-Axis Holography
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --method off_axis \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0
```

**Use case**: Spatial frequency separation, single-shot holography

#### 4. Large-Scale Dataset
```bash
python scripts/generate_dataset.py \
    --samples 10000 \
    --output ./large_dataset \
    --no-preview \
    --seed 42
```

**Use case**: Training deep learning models

### Next Steps

- **Learn about complex fields**: [COMPLEX_FIELDS.md](COMPLEX_FIELDS.md)
- **Add realistic noise**: [NOISE_SIMULATION.md](NOISE_SIMULATION.md)
- **Explore all CLI options**: `python scripts/generate_dataset.py --help`

## Complete Documentation

All documentation is now complete! Here's what's available:

### Core Documentation
- **[Shape Generators](SHAPES.md)** - All 7 shape generators with examples and custom generator guide
- **[Holography Methods](HOLOGRAPHY_METHODS.md)** - Inline and off-axis strategies with detailed physics
- **[Pipeline Architecture](PIPELINE.md)** - Complete pipeline components and customization guide
- **[I/O Formats](IO_FORMATS.md)** - NPZ and PNG formats with ML framework integration
- **[CLI Reference](CLI_REFERENCE.md)** - Comprehensive command-line interface documentation
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all public classes
- **[Utilities Reference](UTILITIES.md)** - Field, I/O, and math utility functions
- **[Examples & Recipes](EXAMPLES.md)** - 10+ practical examples from basic to advanced

## Key Concepts

### Field Representations

HoloGen supports four representations of optical fields:

| Representation | Formula | Information | Use Case |
|---------------|---------|-------------|----------|
| **Intensity** | I = \|E\|² | Magnitude squared | Classical holography, backward compatibility |
| **Amplitude** | A = \|E\| | Magnitude only | Amplitude-based reconstruction |
| **Phase** | φ = arg(E) | Phase angle only | Quantitative phase imaging |
| **Complex** | E = A·exp(iφ) | Complete field | Physics-aware models, full information |

See [COMPLEX_FIELDS.md](COMPLEX_FIELDS.md) for detailed explanations.

### Object Types

- **Amplitude Objects**: Absorbing samples (stained cells, printed patterns)
- **Phase Objects**: Transparent samples (unstained cells, phase masks)
- **Complex Objects**: Mixed absorption and phase modulation *(future)*

### Holography Methods

- **Inline**: On-axis recording, requires twin-image removal
- **Off-Axis**: Carrier frequency separation, single-shot reconstruction

### Noise Models

- **Sensor Noise**: Read noise, shot noise, dark current, quantization
- **Speckle Noise**: Coherent illumination interference patterns
- **Aberrations**: Optical imperfections (defocus, astigmatism, coma)

See [NOISE_SIMULATION.md](NOISE_SIMULATION.md) for parameter details.

## Configuration Examples

### Minimal Configuration
```bash
python scripts/generate_dataset.py --samples 10
```

### Research-Grade Configuration
```bash
python scripts/generate_dataset.py \
    --samples 1000 \
    --output ./research_dataset \
    --method inline \
    --height 1024 \
    --width 1024 \
    --pixel-pitch 3.45e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708 \
    --sensor-shot-noise \
    --sensor-read-noise 2.5 \
    --sensor-bit-depth 12 \
    --speckle-contrast 0.7 \
    --speckle-correlation 2.5 \
    --aberration-defocus 0.15 \
    --seed 42
```

### Production Dataset Configuration
```bash
python scripts/generate_dataset.py \
    --samples 100000 \
    --output ./production_dataset \
    --method off_axis \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --carrier-frequency-x 1e6 \
    --carrier-frequency-y 0 \
    --object-type amplitude \
    --output-domain intensity \
    --no-preview \
    --seed 42
```

## Best Practices

### For Beginners
1. Start with default settings to understand the output
2. Use preview images to visualize results
3. Begin with small sample counts (10-100) for experimentation
4. Read the [Complex Fields Quick Start](COMPLEX_FIELDS_QUICKSTART.md) for modern features

### For Researchers
1. Always set `--seed` for reproducibility
2. Document all parameters used for dataset generation
3. Use realistic noise parameters matching your experimental setup
4. Generate separate clean and noisy datasets for ablation studies
5. Use complex output for physics-aware models

### For Developers
1. Use editable installation (`pip install -e .`) for development
2. Run tests with `pytest` before committing changes
3. Follow the protocol-based design patterns for extensions
4. Consult the API reference for implementation details *(coming soon)*

## Performance Tips

### Memory Optimization
- Use `--no-preview` for large datasets to reduce I/O
- Generate in batches if memory is limited
- Use intensity output instead of complex for 2x memory savings

### Speed Optimization
- Reduce grid resolution for faster generation
- Disable noise models when not needed
- Use multiple processes for parallel generation *(script modification required)*

### Storage Optimization
- Complex fields require 2x storage compared to intensity
- PNG previews add ~30% storage overhead
- Use compression for long-term storage

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'hologen'`  
**Solution**: Install the package with `pip install -e .`

**Issue**: Phase-only objects look uniform in intensity  
**Solution**: This is correct! Phase objects are invisible in intensity. Check the hologram or use complex output.

**Issue**: Out of memory errors  
**Solution**: Reduce `--samples`, `--height`, or `--width`, or use `--no-preview`

**Issue**: Holograms look too noisy  
**Solution**: Reduce noise parameters or disable noise models

**Issue**: No contrast in reconstructions  
**Solution**: Check propagation distance and wavelength parameters

### Getting Help

1. Check the relevant documentation section
2. Review the [Complex Fields documentation](COMPLEX_FIELDS.md) for field-related questions
3. Review the [Noise Simulation documentation](NOISE_SIMULATION.md) for noise-related questions
4. Check the examples in the documentation
5. Open an issue on the project repository

## Contributing

Contributions are welcome! Areas where documentation would benefit from community input:

- Additional examples and use cases
- Integration guides for specific ML frameworks
- Performance benchmarks
- Experimental validation studies
- Custom shape generator examples

## License

HoloGen is released under the MIT License. See the `LICENSE` file for details.

## Citation

If you use HoloGen in your research, please cite:

```bibtex
@software{hologen,
  title = {HoloGen: Synthetic Hologram Dataset Generation Toolkit},
  author = {[Author Names]},
  year = {2024},
  url = {[Repository URL]}
}
```

## Changelog

### Current Version
- ✅ Complex field support (amplitude, phase, complex representations)
- ✅ Comprehensive noise simulation (sensor, speckle, aberrations)
- ✅ Inline and off-axis holography methods
- ✅ Multiple shape generators (7 types)
- ✅ PyTorch and TensorFlow integration examples
- ✅ CLI interface with extensive configuration options

### Planned Features
- 📋 Additional shape generators (custom patterns, images)
- 📋 Advanced noise models (atmospheric turbulence, vibration)
- 📋 Multi-wavelength holography
- 📋 Polarization support
- 📋 GPU acceleration for large-scale generation

## Acknowledgments

HoloGen builds on principles from digital holography, computational imaging, and machine learning research. We acknowledge the contributions of the holography and ML communities.

---

**Last Updated**: 2024  
**Documentation Version**: 1.0  
**HoloGen Version**: [Current Version]
