# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "hologen",
#     "matplotlib",
#     "numpy",
# ]
# ///
import matplotlib.pyplot as plt
import numpy as np

import hologen


def main() -> None:
    # Example: Initialize with a seed for reproducibility
    hologen.RandomNumberGenerator.set_seed(42)

    sim = hologen.InlineHologramSimulator(resolution=512, pixel_size=0.5)

    # 1. Create Objects
    hologen.Objects.add_random_cells(sim, num_cells=10)

    # 2. Add Aberrations (Simulating imperfect illumination/optics)
    # (2,0): Defocus, (2,2): Astigmatism, (3,1): Coma
    # print("Adding Aberrations (Astigmatism & Coma)...")
    # sim.add_aberrations(coeffs={(2, 2): 1.5, (3, 1): 0.5})

    # 3. Add Speckle (Simulating coherence/roughness)
    # print("Adding Speckle noise...")
    sim.add_speckle_noise(strength=0.1, roughness_scale=1.0)

    # 4. Generate Hologram with Sensor Noise
    # This simulates a noisy, imperfect real-world camera
    hologram = sim.generate_hologram(
        z_distance=200,
        shot_noise=True,  # Quantum noise
        # dark_noise_mean=10,     # Thermal noise
        # read_noise_sigma=5,     # Electronic noise
        # bit_depth=8             # 8-bit camera
    )

    # Plotting
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Show Phase (What we want to recover)
    im1 = ax[0].imshow(np.angle(sim.field), cmap="twilight", origin="lower")
    ax[0].set_title("Distorted Object Phase\n(Cells + Aberrations + Speckle)")
    plt.colorbar(im1, ax=ax[0])

    # Show Noisy Hologram (What the camera sees)
    im2 = ax[1].imshow(hologram, cmap="gray", origin="lower")
    ax[1].set_title("Final Noisy Hologram\n(Shot+Dark+Read Noise)")
    plt.colorbar(im2, ax=ax[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
