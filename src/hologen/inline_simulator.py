import numpy as np

import hologen.utils as utils
import hologen.noises as noises
import hologen.propagator as propagator

# Mean photo-electrons per pixel from the unit-intensity illumination (|U|^2 = 1).
# The modelled sensor is the Sony ICX267 CCD (4.65 um pixels) with a saturation
# capacity of 11.0 ke- (Basler EMVA 1288 data overview, scout scA1390-17gm). With
# speckle the brightest fringe reaches 2.53x the background over 300 holograms, so
# 4000 e- keeps it below saturation including 3 sigma of shot noise
# (2.53 * 4000 + 300 = 10.4 ke-): the exposure is as bright as possible without
# clipping.
DEFAULT_BACKGROUND_ELECTRONS: float = 4000.0


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
        background_electrons: float = DEFAULT_BACKGROUND_ELECTRONS,
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
            background_electrons (float): Mean photo-electrons (e-) a pixel
                collects from the unit-intensity illumination (|U|^2 = 1).
        """
        complex_field = propagator.forwards_propagate(
            self.object_field,
            z_distance,
            self.wavelength,
            self.pixel_size,
            self.resolution,
        )

        holo_intensity = np.abs(complex_field) ** 2

        # Speckle modulates the light reaching the sensor, so it acts on the ideal
        # intensity before photons are counted.
        if speckle_contrast > 0:
            if speckle_grain_size is None:
                raise ValueError("speckle_grain_size is required when speckle is on.")
            holo_intensity = noises.add_speckle_noise(
                holo_intensity, self.pixel_size, speckle_grain_size, speckle_contrast
            )

        # Convert the normalised intensity into expected photo-electrons per pixel.
        holo_intensity = holo_intensity * background_electrons

        # Photo-electrons and dark electrons are independent Poisson processes, so
        # each is sampled exactly once: shot noise on the signal, then the dark
        # electrons are added. Sampling shot noise after adding the dark signal
        # would count the dark noise twice (variance N + 2d instead of N + d).
        if shot_noise:
            holo_intensity = noises.add_shot_noise(holo_intensity)
        if dark_noise_mean > 0:
            holo_intensity = noises.add_dark_current_noise(
                holo_intensity, dark_noise_mean
            )
        if read_noise_sigma > 0:
            holo_intensity = noises.add_read_noise(holo_intensity, read_noise_sigma)
        if bit_depth is not None:
            holo_intensity = noises.add_bit_depth_noise(holo_intensity, bit_depth)

        return holo_intensity
