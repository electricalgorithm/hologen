# HoloGen: Synthetic Hologram Dataset Toolkit

[![DOI](https://zenodo.org/badge/1092625374.svg)](https://doi.org/10.5281/zenodo.17622375)

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
Install the package and its `hologen` command:

```sh
uv tool install .            # or: pip install .
```

Generate the dataset with the default, cited recipe (25,000 holograms of 224 x 224
pixels in four noise configurations):

```sh
hologen generate --output dataset-224
```

Common options override the recipe; run `hologen generate --help` for all of them:

```sh
hologen generate --output small --samples 1000 --resolution 128 --seed 1
hologen generate --output laser --config ideal --config camera_laser_speckle
```

The folder also receives `recipe.toml`, the exact recipe the dataset was made from.

## Recipes
Every physical value (geometry, camera, speckle) is part of a recipe. Write the
default recipe, with the unit and source of each value, edit it and generate from it:

```sh
hologen recipe --output paper.toml
hologen generate --recipe paper.toml --output dataset-paper
```

Values missing from a recipe file keep their defaults, and unknown keys are rejected.
From Python, use `hologen.dataset.default_recipe()` and `load_recipe(path)`.

## Noise Model
Every recording passes through a model of a real camera (Sony IMX183 CMOS behind a 10x/0.45
objective, EMVA 1288):
shot noise, dark current, read noise and a 12-bit ADC. The light source adds speckle
(Goodman's model). The dataset has four configurations of equal size:

| Configuration | Noise | rms deviation from the noise-free hologram |
|---|---|---|
| `ideal` | none (reference) | 0.02% |
| `camera` | shot + dark + read + ADC | 1.4% |
| `camera_led_speckle` | camera + speckle contrast 0.014 | 2.0% |
| `camera_laser_speckle` | camera + speckle contrast 0.12 | 12.1% |

To see each noise source live, open an interactive explorer:

```sh
hologen explore speckle          # contrast and grain size
hologen explore shot-noise       # exposure and saturation
hologen explore dark-current     # temperature and exposure time
hologen explore readout          # read noise and ADC bit depth
hologen explore speckle my_photo.png --save speckle.png
```

## Licensing

Released under the MIT License. See ``LICENSE`` for details.
