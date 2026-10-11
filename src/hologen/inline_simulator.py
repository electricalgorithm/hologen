import numpy as np

import hologen.utils as utils
import hologen.noises as noises
import hologen.propagator as propagator

# Saturation capacity of the modelled sensor, the Sony IMX183 CMOS (2.4 um pixels)
# of the magnified in-line setup of Wdowiak et al., arXiv:2508.15419 (2025):
# 14,837 e- (FLIR BFS-U3-200S6M EMVA 1288 results, 20 C; 14,775 e- for LUCID
# Triton 20 MP). The ADC maps it to full scale.
DEFAULT_SATURATION_ELECTRONS: float = 14837.0

# Mean photo-electrons per pixel from the unit-intensity illumination (|U|^2 = 1):
# the largest background for which the brightest fringe (peak x background) plus
# 3 sigma of shot noise stays below saturation. PROVISIONAL: the first-version
# ellipses in the magnified geometry peak at 2.66x the background
# (scripts/measure_dataset.py), so 2.66 * 5400 + 3 sqrt(2.66 * 5400) = 14.5 ke-;
# re-derive once the HOLO-002 cell objects exist.
DEFAULT_BACKGROUND_ELECTRONS: float = 5400.0


class InlineHologramSimulator:
    def __init__(
        self,
        resolution: int = 512,
        pixel_size: float = 0.1,
        wavelength: float = 0.532,
        numerical_aperture: float | None = None,
    ) -> None:
        """
        Initialize the simulation environment.

        Args:
            resolution (int): Number of pixels (NxN grid).
            pixel_size (float): Physical size of one pixel in microns, in the object
                plane (sensor pixel divided by the magnification when magnified).
            wavelength (float): Wavelength of the light in microns (e.g., 0.532 for green).
            numerical_aperture (float | None): NA of the imaging objective of a
                magnified setup; None for a lensless setup without a pupil.
        """
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.numerical_aperture = numerical_aperture

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

    def apply_transmission(self, amplitude: np.ndarray, phase: np.ndarray) -> None:
        """
        Multiply the object field by a transmission map amplitude * exp(i phase),
        e.g. a cell with a continuous thickness profile. Overlapping objects
        multiply their transmissions, so their phases add.
        """
        self.object_field = self.object_field * (amplitude * np.exp(1j * phase))

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
        saturation_electrons: float = DEFAULT_SATURATION_ELECTRONS,
    ) -> np.ndarray:
        """
        Simulates the sensor recording with advanced noise models.

        Args:
            z_distance (float): Propagation distance.
            speckle_grain_size (float): Mean speckle grain diameter (in microns).
            speckle_contrast (float): Speckle contrast std/mean; 0 disables speckle.
            shot_noise (bool): If True, applies Poisson shot noise.
            dark_noise_mean (float): Mean dark electrons per pixel per exposure.
            read_noise_sigma (float): Std dev of Gaussian read noise (in e-).
            bit_depth (int): If set (e.g., 8, 12), converts electrons to digital
                numbers with an ADC of this many bits.
            background_electrons (float): Mean photo-electrons (e-) a pixel
                collects from the unit-intensity illumination (|U|^2 = 1).
            saturation_electrons (float): Saturation capacity (e-), mapped to the
                largest digital value by the ADC.

        Returns:
            np.ndarray: Electrons per pixel, or digital numbers (DN) if bit_depth
                is set.
        """
        complex_field = propagator.forwards_propagate(
            self.object_field,
            z_distance,
            self.wavelength,
            self.pixel_size,
            self.resolution,
            numerical_aperture=self.numerical_aperture,
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

        # Readout (EMVA 1288 Eq. 14): read noise in electrons, then the ADC converts
        # to digital numbers with gain K, clips at 2^k - 1 and rounds.
        if read_noise_sigma > 0:
            holo_intensity = noises.add_read_noise(holo_intensity, read_noise_sigma)
        if bit_depth is not None:
            holo_intensity = noises.add_bit_depth_noise(
                holo_intensity, bit_depth, saturation_electrons
            )

        return holo_intensity
