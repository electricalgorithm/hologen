# HoloGen: Synthetic Hologram Dataset Toolkit

The HoloGen toolkit generates paired object-domain images and their inline or off-axis holograms for machine learning workflows.

Features
--------

* Binary object-domain sample generation with diverse analytic shapes
* Strategy-based hologram creation supporting inline and off-axis methods
* Reconstruction pipeline for object-domain recovery from holograms
* Dataset writer for NumPy bundles and preview imagery

Quickstart
----------

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

Configuration
-------------

Key parameters reside in ``scripts/generate_dataset.py``. Adjust them to change:

* Grid resolution and pixel pitch
* Optical wavelength and propagation distance
* Holography method (inline or off-axis)
* Carrier parameters for off-axis holography

Licensing
---------

Released under the MIT License. See ``LICENSE`` for details.
