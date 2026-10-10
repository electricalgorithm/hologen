import numpy as np

import hologen.utils as utils
import hologen.noises as noises
import hologen.propagator as propagator


class InlineHologramSimulator:
    def __init__(
        self, resolution: int = 512, pixel_size: float = 0.1, wavelength: float = 0.532
    ) -> None:
        """
        Initialize the simulation environment.

        Args:
            resolution (int): Number of pixels (NxN grid).
            pixel_size (float): Physical size of one pixel in microns.
            wavelength (float): Wavelength of the light in microns (e.g., 0.532 for green).
            seed (int, optional): Seed for the random number generator for reproducibility.
        """
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.wavelength = wavelength

        self._rng: np.random.Generator = utils.RandomNumberGenerator()

        # Create coordinate grid (Spatial Domain)
        coordinate: utils.CoordinateGrid = utils.create_coordinate_grid(
            self.resolution, self.pixel_size
        )
        self.X, self.Y = coordinate.X, coordinate.Y

        # Polar coordinates for aberrations (rho normalized to 1 at edge).
        polar: utils.PolarGrid = utils.create_polar_grid(
            self.resolution, self.pixel_size
        )
        self.R, self.Theta, self.Rho_norm = polar.R, polar.Theta, polar.Rho_norm

        # Create frequency grid (Fourier Domain)
        freq: utils.FrequencyGrid = utils.create_frequency_grid(
            self.resolution, self.pixel_size
        )
        self.FX, self.FY = freq.FX, freq.FY

        # Initialize the object field (Complex plane).
        self.object_field = np.ones(
            (self.resolution, self.resolution), dtype=np.complex128
        )

    def add_object(
        self, mask: np.ndarray, amplitude: float = 1.0, phase_shift: float = 0.0
    ) -> None:
        """Adds an object to the current field based on a boolean mask."""
        self.object_field[mask] *= amplitude
        self.object_field[mask] *= np.exp(1j * phase_shift)

    def reset_object_field(self) -> None:
        """Resets the object field to a clear, transparent slide."""
        self.object_field = np.ones(
            (self.resolution, self.resolution), dtype=np.complex128
        )

    def add_aberrations(self, coeffs: dict[tuple[int, int], float]) -> None:
        """
        Adds optical aberrations to the field using Zernike polynomials.

        Args:
            coeffs (dict): Dictionary of { (n, m): weight }.
                           e.g., {(2,0): 0.5} for Defocus.
        """
        total_phase_error = np.zeros((self.resolution, self.resolution))

        for (n, m), weight in coeffs.items():
            z = utils.get_zernike_polynomial(self.Rho_norm, self.Theta, n, m)
            total_phase_error += weight * z

        # Apply aberration as a phase mask to the field (Source/System imperfection)
        self.object_field *= np.exp(1j * total_phase_error)

    def generate_hologram(
        self,
        z_distance: float,
        speckle_grain_size: float | None = None,
        speckle_contrast: float = 0.0,
        shot_noise: bool = False,
        dark_noise_mean: float = 0.0,
        read_noise_sigma: float = 0.0,
        bit_depth: int | None = None,
    ) -> np.ndarray:
        """
        Simulates the sensor recording with advanced noise models.

        Args:
            z_distance (float): Propagation distance.
            speckle_grain_size (float): Mean speckle grain diameter (in microns).
            speckle_contrast (float): Speckle contrast std/mean; 0 disables speckle.
            shot_noise (bool): If True, applies Poisson shot noise.
            dark_noise_mean (float): Mean thermal electrons (Poisson dist).
            read_noise_sigma (float): Std dev of Gaussian read noise.
            bit_depth (int): If set (e.g., 8, 12), simulates quantization.
        """
        complex_field = propagator.forwards_propagate(
            self.object_field,
            z_distance,
            self.wavelength,
            self.pixel_size,
            self.resolution,
        )

        # Ideal Intensity (Normalized 0-1 usually, but let's assume photon count)
        # We assume a baseline photon count for "bright" areas to make Shot Noise meaningful
        base_photon_count = 1000.0
        holo_intensity = np.abs(complex_field) ** 2

        # Speckle modulates the light reaching the sensor, so it acts on the ideal
        # intensity before photons are counted.
        if speckle_contrast > 0:
            if speckle_grain_size is None:
                raise ValueError("speckle_grain_size is required when speckle is on.")
            holo_intensity = noises.add_speckle_noise(
                holo_intensity, self.pixel_size, speckle_grain_size, speckle_contrast
            )
        holo_intensity = holo_intensity * base_photon_count

        # Add noises.
        if dark_noise_mean > 0:
            holo_intensity = noises.add_dark_current_noise(
                holo_intensity, dark_noise_mean
            )
        if shot_noise:
            holo_intensity = noises.add_shot_noise(holo_intensity)
        if read_noise_sigma > 0:
            holo_intensity = noises.add_read_noise(holo_intensity, read_noise_sigma)
        if bit_depth is not None:
            holo_intensity = noises.add_bit_depth_noise(holo_intensity, bit_depth)

        return holo_intensity
