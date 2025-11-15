# HoloGen: Synthetic Hologram Dataset Toolkit

The HoloGen toolkit generates paired object-domain images and their inline or off-axis holograms for machine learning workflows.

> [See the first dataset (100k, 157.4GB!) generated with the HoloGen in HuggingFace 🤗](https://huggingface.co/datasets/gokhankocmarli/inline-digital-holography/)

## Features
* Binary object-domain sample generation with diverse analytic shapes
* **Complex field support** for amplitude, phase, and mixed object types
* Strategy-based hologram creation supporting inline and off-axis methods
* Reconstruction pipeline for object-domain recovery from holograms
* Dataset writer for NumPy bundles and preview imagery with multiple field representations
* Realistic noise and aberration modeling for physically accurate simulations

## Quickstart
1. Create a virtual environment and install the package:

```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
```

2. Generate a sample dataset:

```bash
   python scripts/generate_dataset.py
```

The default dataset is written to the ``dataset/`` directory with both ``.npz`` tensors and PNG previews.

## Configuration
Key parameters reside in ``scripts/generate_dataset.py``. Adjust them to change:

* Grid resolution and pixel pitch
* Optical wavelength and propagation distance
* Holography method (inline or off-axis)
* Carrier parameters for off-axis holography
* Noise levels and aberration coefficients (see [docs](docs/NOISE_SIMULATION.md))

## Complex Field Support
HoloGen supports complex-valued optical fields with amplitude, phase, and mixed representations. Generate phase-only objects (transparent samples), amplitude-only objects (absorbing samples), or full complex fields. Export data as intensity, amplitude, phase, or complex representations. See [docs/COMPLEX_FIELDS.md](docs/COMPLEX_FIELDS.md) for detailed documentation.

#### Quick Example: Phase-Only Objects
```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --object-type phase \
    --output-domain complex \
    --phase-shift 1.5708
```

## Noise Simulation & Realistic Modeling
HoloGen includes comprehensive noise and aberration modeling to bridge the gap between synthetic and experimental holography. See [docs/NOISE_SIMULATION.md](docs/NOISE_SIMULATION.md) for detailed parameter explanations.

#### Basic Noise Configuration
```py
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_noisy \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --speckle-contrast 0.8 \
    --sensor-read-noise 3.0 \
    --sensor-shot-noise \
    --sensor-dark-current 0.5 \
    --sensor-bit-depth 12
```

## Licensing

Released under the MIT License. See ``LICENSE`` for details.
