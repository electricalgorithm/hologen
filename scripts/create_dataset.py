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
NUM_SAMPLES = 1500

# Noise Configurations
SPECKLE_STRENGTH = 0.15
SPECKLE_ROUGHNESS = 1.0
READ_NOISE_SIGMA = 10.0
DARK_NOISE_MEAN = 20.0


def main():
    # Create hologram configuration
    holo_config = HologramConfig(
        resolution=512,
        pixel_size=4.65,  # microns
        wavelength=0.532,  # microns
        z_distance=20000,  # microns
        bit_depth=12,
        num_cells=5,
    )

    # Create dataset configuration
    config = DatasetConfig(
        output_dir="inline-holo-dataset-v2",
        base_seed=42,
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
            speckle_strength=SPECKLE_STRENGTH,
            speckle_roughness=SPECKLE_ROUGHNESS,
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
            speckle_strength=SPECKLE_STRENGTH,
            speckle_roughness=SPECKLE_ROUGHNESS,
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
            speckle_strength=SPECKLE_STRENGTH,
            speckle_roughness=SPECKLE_ROUGHNESS,
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
            speckle_strength=SPECKLE_STRENGTH,
            speckle_roughness=SPECKLE_ROUGHNESS,
            read_noise_sigma=READ_NOISE_SIGMA,
            dark_noise_mean=DARK_NOISE_MEAN,
        ),
    )

    # Generate the dataset
    stats = generator.generate()

    print(f"\nGeneration complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Configurations: {stats['configurations']}")
    print(f"Output directory: {stats['output_dir']}")


if __name__ == "__main__":
    main()
