"""
Dataset generator for creating hologram datasets with various noise configurations.

This module provides a scalable and flexible API for generating hologram datasets
with different noise configurations, making it easy for consumers to create
custom datasets for machine learning or research purposes.
"""

from typing import Literal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hologen.inline_simulator import InlineHologramSimulator
from hologen.objects import add_random_cells
from hologen.objects.families import ObjectFamily
from hologen.utils import RandomNumberGenerator
from hologen.noises.bit_depth import adc_gain
from hologen.propagator import backwards_propagate
from .configs import DatasetConfig, NoiseConfig


class DatasetGenerator:
    """
    A scalable and flexible dataset generator for holograms.

    This class provides a clean API for generating hologram datasets with
    various noise configurations. Supports batch writing for memory-efficient
    generation of large datasets.

    Examples:
        >>> recipe = default_recipe()
        >>> config = DatasetConfig(output_dir="dataset", base_seed=42, hologram_config=recipe.hologram_config())
        >>> # For large datasets, use smaller batch_size to limit memory usage
        >>> generator = DatasetGenerator(config, batch_size=50)
        >>> generator.add_config("camera", num_samples=10, noise=recipe.camera_noise())
        >>> generator.generate()
    """

    def __init__(self, config: DatasetConfig, batch_size: int = 100):
        """
        Initialize the dataset generator.

        Args:
            config (DatasetConfig): Output folder, base seed and hologram settings.
            batch_size (int): Number of samples to accumulate before writing to
                disk. Smaller values use less memory but may be slower.
        """
        self.config = config
        self.batch_size = batch_size
        self._configs: dict[str, tuple[int, NoiseConfig, ObjectFamily | None]] = {}

        # Accumulate samples for batch writing
        self._ground_truths: list[np.ndarray] = []
        self._holograms: list[np.ndarray] = []
        self._reconstructed: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._sample_counter = 0  # Track total samples written

    def add_config(
        self,
        name: str,
        num_samples: int,
        noise: NoiseConfig,
        objects: ObjectFamily | None = None,
    ) -> "DatasetGenerator":
        """
        Add a noise configuration to generate.

        Args:
            name (str): Unique name of the group, stored as config_name.
            num_samples (int): Number of holograms to generate for this group.
            noise (NoiseConfig): Noise to apply to the recorded holograms.
            objects (ObjectFamily | None): Object family, with its placement and
                exposure. None places the first-version ellipses with the
                HologramConfig exposure.

        Returns:
            DatasetGenerator: The generator itself, for method chaining.

        Raises:
            ValueError: If a group with this name was already added.

        Examples:
            >>> generator = DatasetGenerator(config)
            >>> generator.add_config("clean", 100, NoiseConfig(...))
            >>> generator.add_config("noisy", 100, NoiseConfig(speckle=True, shot=True, ...))
        """
        if name in self._configs:
            raise ValueError(f"Configuration '{name}' already exists")

        self._configs[name] = (num_samples, noise, objects)
        return self

    def generate(self, verbose: bool = True) -> dict[str, Any]:
        """
        Generate the complete dataset.

        Args:
            verbose (bool): Whether to print progress information.

        Returns:
            dict[str, Any]: Generation statistics, with the total number of
                samples, the number of groups and the absolute output folder.

        Raises:
            ValueError: If no configurations have been added.
        """
        if not self._configs:
            raise ValueError(
                "No configurations added. Use add_config() to add configurations before generating."
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        total_samples: Literal[0] | int = sum(
            entry[0] for entry in self._configs.values()
        )
        global_idx = 0

        if verbose:
            print(f"Generating dataset with {len(self._configs)} configurations...")
            print(f"Total samples: {total_samples}")

        holo_config = self.config.hologram_config

        for config_name, (num_samples, noise_config, objects) in self._configs.items():
            if verbose:
                noise_desc = self._describe_noise(noise_config)
                print(
                    f"  Generating {num_samples} samples: {config_name} ({noise_desc})..."
                )

            for i in range(num_samples):
                # Create simulator with unique seed
                RandomNumberGenerator.set_seed(self.config.base_seed + global_idx)
                sim = InlineHologramSimulator(
                    resolution=holo_config.resolution,
                    pixel_size=holo_config.pixel_size,
                    wavelength=holo_config.wavelength,
                    numerical_aperture=holo_config.numerical_aperture,
                )

                # Generate hologram
                ground_truth, hologram, reconstructed, details = (
                    self._generate_single_hologram(sim, noise_config, objects)
                )

                # Accumulate samples
                self._ground_truths.append(ground_truth)
                self._holograms.append(hologram)
                self._reconstructed.append(reconstructed)
                self._metadata.append(
                    {
                        "config_name": config_name,
                        "sample_idx": i,
                        "global_idx": global_idx,
                        **details,
                    }
                )

                # Save first example for visualization
                if i == 0:
                    examples[config_name] = (ground_truth, hologram, reconstructed)

                global_idx += 1

                # Write batch if we've accumulated enough samples
                if len(self._ground_truths) >= self.batch_size:
                    if verbose:
                        print(
                            f"  Writing batch of {len(self._ground_truths)} samples..."
                        )
                    self._write_batch()

        # Write any remaining samples
        if self._ground_truths:
            if verbose:
                print(f"\nWriting final batch of {len(self._ground_truths)} samples...")
            self._write_batch()

        if verbose:
            print(f"Dataset generation complete! Total: {total_samples} holograms")
            print(f"Files saved to: {output_dir.absolute()}")

        # Generate visualization if requested
        if self.config.generate_visualization:
            self._generate_visualization(examples, verbose)

        return {
            "total_samples": total_samples,
            "configurations": len(self._configs),
            "output_dir": str(output_dir.absolute()),
        }

    def _generate_single_hologram(
        self,
        sim: InlineHologramSimulator,
        noise: NoiseConfig,
        objects: ObjectFamily | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Generate a single hologram with the given noise configuration.

        Args:
            sim (InlineHologramSimulator): Simulator for this hologram, seeded
                and still empty.
            noise (NoiseConfig): Noise configuration to apply.
            objects (ObjectFamily | None): Object family to place and its
                exposure, or None for the first-version ellipses with the
                HologramConfig exposure.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: The
                complex ground truth, the recorded intensity in digital numbers,
                the back-propagated field normalised by the background, and the
                metadata of the hologram (object family, objects placed, exposure
                and clipped pixel fraction).
        """
        holo_config = self.config.hologram_config

        # Reset and create object
        sim.reset_object_field()
        if objects is None:
            add_random_cells(sim, num_cells=holo_config.num_cells)
            placed: int = holo_config.num_cells
            family: str = "ellipses"
            background_electrons: float = holo_config.background_electrons
        else:
            placed = objects.place(sim)
            family = objects.name
            background_electrons = objects.background_electrons

        # Save ground truth object (before any noise or propagation)
        ground_truth = sim.object_field.copy()

        # Generate hologram with noise
        hologram_intensity = sim.generate_hologram(
            z_distance=holo_config.z_distance,
            speckle_grain_size=noise.speckle_grain_size if noise.speckle else None,
            speckle_contrast=noise.speckle_contrast if noise.speckle else 0.0,
            shot_noise=noise.shot,
            dark_noise_mean=noise.dark_noise_mean if noise.dark else 0.0,
            read_noise_sigma=noise.read_noise_sigma if noise.read else 0.0,
            bit_depth=holo_config.bit_depth,
            background_electrons=background_electrons,
            saturation_electrons=holo_config.saturation_electrons,
        )

        # Reconstruct complex field from intensity (realistic back-propagation)
        # In real inline holography, we only have intensity and no phase information
        # The hologram is in digital numbers; the background is K * background_e.
        background_dn: float = background_electrons * adc_gain(
            holo_config.bit_depth, holo_config.saturation_electrons
        )
        amplitude: np.ndarray = np.sqrt(hologram_intensity / background_dn)
        phase: np.ndarray = np.zeros_like(amplitude)  # Zero phase assumption
        reconstructed_complex_field_hologram = amplitude * np.exp(1j * phase)

        # Back-propagate to object plane
        reconstructed: np.ndarray[np.complex128] = backwards_propagate(
            reconstructed_complex_field_hologram,
            holo_config.z_distance,
            sim.wavelength,
            sim.pixel_size,
            sim.resolution,
        )

        # Pixels at full scale: the exposure lets a few percent of holograms clip
        # at the foci of strongly refracting cells (hologen.dataset.exposure).
        details: dict[str, Any] = {
            "object_family": family,
            "objects_placed": placed,
            "background_electrons": background_electrons,
            "clipped_fraction": float(
                np.mean(hologram_intensity >= 2**holo_config.bit_depth - 1)
            ),
        }
        return ground_truth, hologram_intensity, reconstructed, details

    def _write_batch(self) -> None:
        """
        Write accumulated samples to separate parquet files and clear buffers.

        Creates separate parquet files for each batch (e.g., ground_truth_0000.parquet,
        ground_truth_0001.parquet, etc.). This enables true memory-efficient batch writing
        for large datasets without ever loading the full dataset into memory.
        """
        if not self._ground_truths:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get array shape from first sample
        shape = self._ground_truths[0].shape
        num_pixels = shape[0] * shape[1]
        batch_size = len(self._ground_truths)

        # Generate sample IDs for this batch
        sample_ids = range(self._sample_counter, self._sample_counter + batch_size)

        # Calculate batch number for filename
        batch_num = self._sample_counter // self.batch_size

        # Ground truth (complex field)
        gt_data = {
            "sample_id": np.repeat(list(sample_ids), num_pixels),
            "config_name": np.repeat(
                [m["config_name"] for m in self._metadata], num_pixels
            ),
            "global_idx": np.repeat(
                [m["global_idx"] for m in self._metadata], num_pixels
            ),
            "real": np.concatenate([arr.real.flatten() for arr in self._ground_truths]),
            "imag": np.concatenate([arr.imag.flatten() for arr in self._ground_truths]),
        }
        pd.DataFrame(gt_data).to_parquet(
            output_dir / f"ground_truth_{batch_num:04d}.parquet",
            compression="snappy",
            index=False,
        )

        # Hologram (intensity)
        holo_data = {
            "sample_id": np.repeat(list(sample_ids), num_pixels),
            "config_name": np.repeat(
                [m["config_name"] for m in self._metadata], num_pixels
            ),
            "global_idx": np.repeat(
                [m["global_idx"] for m in self._metadata], num_pixels
            ),
            "intensity": np.concatenate([arr.flatten() for arr in self._holograms]),
        }
        pd.DataFrame(holo_data).to_parquet(
            output_dir / f"hologram_{batch_num:04d}.parquet",
            compression="snappy",
            index=False,
        )

        # Reconstructed (complex field)
        rec_data = {
            "sample_id": np.repeat(list(sample_ids), num_pixels),
            "config_name": np.repeat(
                [m["config_name"] for m in self._metadata], num_pixels
            ),
            "global_idx": np.repeat(
                [m["global_idx"] for m in self._metadata], num_pixels
            ),
            "real": np.concatenate([arr.real.flatten() for arr in self._reconstructed]),
            "imag": np.concatenate([arr.imag.flatten() for arr in self._reconstructed]),
        }
        pd.DataFrame(rec_data).to_parquet(
            output_dir / f"reconstructed_{batch_num:04d}.parquet",
            compression="snappy",
            index=False,
        )

        # One row per hologram: object family, exposure and clipping.
        pd.DataFrame(
            [
                {"sample_id": sid, **meta}
                for sid, meta in zip(sample_ids, self._metadata)
            ]
        ).to_parquet(
            output_dir / f"metadata_{batch_num:04d}.parquet",
            compression="snappy",
            index=False,
        )

        # Update counter and clear buffers
        self._sample_counter += batch_size
        self._ground_truths.clear()
        self._holograms.clear()
        self._reconstructed.clear()
        self._metadata.clear()

    def _describe_noise(self, noise: NoiseConfig) -> str:
        """
        Describe a noise configuration in a few words.

        Args:
            noise (NoiseConfig): The noise configuration.

        Returns:
            str: The enabled noise sources joined by "+", or "no noise".
        """
        parts: list[str] = []
        if noise.speckle:
            parts.append("speckle")
        if noise.shot:
            parts.append("shot")
        if noise.read:
            parts.append("read")
        if noise.dark:
            parts.append("dark")
        return "+".join(parts) if parts else "no noise"

    def _generate_visualization(
        self,
        examples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        verbose: bool = True,
    ) -> None:
        """
        Generate visualization figure showing examples from each configuration.

        Args:
            examples (dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]): One
                example per group, as (ground truth, hologram, reconstruction).
            verbose (bool): Whether to print progress.
        """
        if verbose:
            print("\nGenerating visualization figure...")

        num_configs = len(examples)
        fig, axes = plt.subplots(num_configs, 5, figsize=(20, 5 * num_configs))

        # Handle single configuration case
        if num_configs == 1:
            axes = axes.reshape(1, -1)

        for row, (config_name, (ground_truth, hologram, reconstructed)) in enumerate(
            examples.items()
        ):
            # Ground Truth Object (Amplitude)
            im1 = axes[row, 0].imshow(
                np.abs(ground_truth), cmap="gray", origin="lower", vmin=0.8, vmax=1.0
            )
            axes[row, 0].set_title(f"{config_name}\nGround Truth Object (Amplitude)")
            axes[row, 0].axis("off")
            plt.colorbar(im1, ax=axes[row, 0])

            # Ground Truth Object (Phase)
            phase_gt = np.angle(ground_truth)
            phase_range = np.max(np.abs(phase_gt))
            if phase_range == 0:
                phase_range = 0.1
            im2 = axes[row, 1].imshow(
                phase_gt,
                cmap="twilight",
                origin="lower",
                vmin=-phase_range,
                vmax=phase_range,
            )
            axes[row, 1].set_title(f"{config_name}\nGround Truth Object (Phase)")
            axes[row, 1].axis("off")
            plt.colorbar(im2, ax=axes[row, 1])

            # Hologram (Intensity)
            im3 = axes[row, 2].imshow(hologram, cmap="gray", origin="lower")
            axes[row, 2].set_title(f"{config_name}\nHologram (Intensity)")
            axes[row, 2].axis("off")
            plt.colorbar(im3, ax=axes[row, 2])

            # Reconstructed Object (Amplitude)
            im4 = axes[row, 3].imshow(
                np.abs(reconstructed), cmap="gray", origin="lower", vmin=0.8, vmax=1.0
            )
            axes[row, 3].set_title(
                f"{config_name}\nReconstructed Object (Amplitude)\n(Artifacts from zero-phase assumption)"
            )
            axes[row, 3].axis("off")
            plt.colorbar(im4, ax=axes[row, 3])

            # Reconstructed Object (Phase)
            phase_rec = np.angle(reconstructed)
            phase_range_rec = np.max(np.abs(phase_rec))
            if phase_range_rec == 0:
                phase_range_rec = 0.1
            im5 = axes[row, 4].imshow(
                phase_rec,
                cmap="twilight",
                origin="lower",
                vmin=-phase_range_rec,
                vmax=phase_range_rec,
            )
            axes[row, 4].set_title(
                f"{config_name}\nReconstructed Object (Phase)\n(Artifacts from zero-phase assumption)"
            )
            axes[row, 4].axis("off")
            plt.colorbar(im5, ax=axes[row, 4])

        plt.tight_layout()
        output_path = Path(self.config.output_dir) / "dataset_examples.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"Visualization saved to: {output_path}")
