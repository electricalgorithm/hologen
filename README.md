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

### Exposure calibration
Each object family is recorded at its own background exposure. The exposure is the
largest background at which the recipe's `exposure_percentile` (95 by default) of
noise-free laser-speckle holograms keeps its brightest point, plus three standard
deviations of shot noise, below the saturation capacity. The brightest point depends
on the geometry, the speckle, the saturation capacity and the objects. When you change
any of these, calibrate the recipe again before you generate from it:

```sh
hologen recipe --output paper.toml            # 1. write and edit the recipe
hologen calibrate --recipe paper.toml --write # 2. calibrate its exposures
hologen measure --recipe paper.toml -o groups.md  # 3. check noise and clipping
hologen generate --recipe paper.toml --output dataset-paper
```

`calibrate --write` stores the exposures in the recipe with their provenance and a
fingerprint of the settings they were calibrated for. If those settings change
afterwards, `generate` and `measure` print a warning. The fingerprint covers only
the recipe, so a change to the object models in the code needs a recalibration that
no warning will announce.

`measure` reports, for every group (noise configuration and object family), the
noise relative to the background, how far normalising by the mean or the median
moves the background, and how often holograms clip. These are the numbers to quote
about a dataset. Both commands use fixed seeds far from the dataset seeds, so their
results can be reproduced.

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
