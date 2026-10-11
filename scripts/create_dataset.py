# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""
This script creates a dataset of inline holograms.

"""

from hologen import DatasetConfig, DatasetGenerator, HologramConfig, NoiseConfig

# Dataset Configurations
BATCH_SIZE = 250
NUM_SAMPLES = 3125  # 25000 / 8 = 3125

# Noise Configurations
# Speckle contrast (std/mean of the speckle intensity) of a laser-illuminated
# in-line hologram: 0.12 for a HeNe laser vs 0.014 for an LED (Hughes,
# arXiv:2006.09296, 2020; order-of-magnitude anchor measured through a fibre
# bundle). C = sigma / <I> as in Bianco et al., Light Sci. Appl. 7:48 (2018).
SPECKLE_CONTRAST = 0.12
# Mean grain size d_s = wavelength * z / D (Goodman, Speckle Phenomena in Optics,
# 2007) for scatterers at the object plane seen over the sensor field of view:
# 0.532 um * 20000 um / (224 * 4.65 um) = 10.2 um, about 2.2 pixels.
SPECKLE_GRAIN_SIZE = 10.2  # microns
READ_NOISE_SIGMA = 10.0
DARK_NOISE_MEAN = 20.0


def main():
    # Create hologram configuration
    holo_config = HologramConfig(
        resolution=224,
        pixel_size=4.65,  # microns
        wavelength=0.532,  # microns
        z_distance=20000,  # microns
        bit_depth=12,
        num_cells=5,
    )

    # Create dataset configuration
    config = DatasetConfig(
        output_dir="dataset-224",
        base_seed=10000,
        hologram_config=holo_config,
    )

    # Create generator
    generator = DatasetGenerator(config, batch_size=BATCH_SIZE)
    generator.add_config(
        "no_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=False,
            read=False,
            dark=False,
        ),
    )
    generator.add_config(
        "speckle_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=False,
            read=False,
            dark=False,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
        ),
    )
    generator.add_config(
        "shot_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=True,
            read=False,
            dark=False,
        ),
    )
    generator.add_config(
        "speckle_shot_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=True,
            read=False,
            dark=False,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
        ),
    )
    generator.add_config(
        "read_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=False,
            read=True,
            dark=False,
            read_noise_sigma=READ_NOISE_SIGMA,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )
    generator.add_config(
        "dark_current_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=False,
            shot=False,
            read=False,
            dark=True,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )
    generator.add_config(
        "speckle_shot_read_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=True,
            read=True,
            dark=False,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
            read_noise_sigma=READ_NOISE_SIGMA,
        ),
    )
    generator.add_config(
        "speckle_shot_read_dark_noise",
        num_samples=NUM_SAMPLES,
        noise=NoiseConfig(
            speckle=True,
            shot=True,
            read=True,
            dark=True,
            speckle_contrast=SPECKLE_CONTRAST,
            speckle_grain_size=SPECKLE_GRAIN_SIZE,
            read_noise_sigma=READ_NOISE_SIGMA,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )

    # Generate the dataset
    stats = generator.generate()

    print("Generation complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Configurations: {stats['configurations']}")
    print(f"Output directory: {stats['output_dir']}")


if __name__ == "__main__":
    main()
