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

Generate the dataset with the default, cited recipe (25,000 holograms of 1024 x 1024
pixels, about 103 GB, in four noise configurations for each of two cell families):

```sh
hologen generate --output dataset
```

Common options override the recipe; run `hologen generate --help` for all of them:

```sh
hologen generate --output small --samples 1000 --resolution 128 --seed 1
hologen generate --output laser --config ideal --config camera_laser_speckle
```

The folder also receives `recipe.toml`, the exact recipe the dataset was made from.

### Dataset files

Each batch of holograms is stored in four Parquet files with one row per hologram (format 2, `hologen.dataset.storage`):

| File | Content |
|---|---|
| `hologram_XXXX.parquet` | `sample_id`, `intensity`: the recorded hologram as 16-bit camera counts, a 2-D list |
| `ground_truth_XXXX.parquet` | `sample_id`, `real`, `imag`: the complex object field as float32 2-D lists |
| `metadata_XXXX.parquet` | `sample_id`, `seed`, `config_name`, `object_family`, objects placed, exposure, clipped fraction, `z_true`, `z_nominal` |
| `reconstructed_XXXX.parquet` | only with `--reconstructed`: the field back-propagated at the nominal distance |

Every hologram is its own row group, so one can be read without the rest of its file, and each image file records the format version, units and image shape in its Parquet metadata. `hologen.dataset.storage.load_samples(folder, "hologram")` loads the images by sample id, and `ImageFile(path)` reads single rows; `scripts/load_dataset.py` shows how to join them to the metadata. Datasets in the earlier one-pixel-per-row layout are rejected with a message to regenerate them.

## Recipes
Every physical value (geometry, camera, speckle) is part of a recipe. Write the
default recipe, with the unit and source of each value, edit it and generate from it:

```sh
hologen recipe --output paper.toml
hologen generate --recipe paper.toml --output dataset-paper
```

Values missing from a recipe file keep their defaults, and unknown keys are rejected.
From Python, use `hologen.dataset.default_recipe()` and `load_recipe(path)`.

### Recording-distance errors
A real setup rarely records at exactly the distance its user assumes. The geometry
values `z_offset` and `z_error` record each hologram at
`z_distance + z_offset + U(-z_error, z_error)`, while the stored reconstruction is
still back-propagated at `z_distance`. The distance of each hologram is saved as
`z_true` (and the assumed one as `z_nominal`) in `metadata_XXXX.parquet`. The random
error has its own random stream, so datasets made from the same seed hold the same
objects and speckle. Their camera noise has the same statistics but a different
realisation, because Poisson sampling depends on the intensity.

```sh
hologen generate --output train-zaug --z-error 6   # training set, z in 14-26 um
hologen generate --output sweep-m4 --z-offset -4 --seed 900000  # test set at 16 um
```

The exposure belongs to the setup, so it is calibrated at `z_distance` and neither
value needs a recalibration. Holograms recorded further away can clip more often,
as they would on a real camera, and `clipped_fraction` in the metadata shows it.

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

| Configuration | Noise | rms deviation from the noise-free hologram, red blood cells / adherent cells |
|---|---|---|
| `ideal` | none (reference) | 0.04% / 0.05% |
| `camera` | shot + dark + read + ADC | 1.9% / 2.3% |
| `camera_led_speckle` | camera + speckle contrast 0.014 | 2.4% / 2.7% |
| `camera_laser_speckle` | camera + speckle contrast 0.12 | 12.9% / 12.3% |

The deviations are in percent of the background, measured with `hologen measure` on 50 holograms per group. The two families differ because each is recorded at its own calibrated exposure (2735 and 1855 e- per pixel).

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
