"""Inline holography strategy implementation.

This module implements inline (Gabor) holography, a technique where the reference
and object beams propagate along the same optical axis. A fundamental characteristic
of inline holography is that detectors can only record intensity |E|², not the full
complex field E = A·exp(iφ). This physical limitation has important consequences
for hologram recording and reconstruction.

Physical Process
----------------
In inline holography, the recording process follows these steps:

1. **Object Wave Propagation**: The object field E_obj(x,y) propagates from the
   object plane to the detector plane, creating a complex field E_det(x,y) with
   both amplitude and phase information.

2. **Intensity Recording**: The detector records only the intensity:
   I(x,y) = |E_det(x,y)|² = A_det²(x,y)

   The phase information φ_det(x,y) is lost during this measurement, as real
   detectors (CCD, CMOS sensors) respond only to photon flux, not phase.

3. **Reconstruction**: To reconstruct the object, we back-propagate from the
   recorded intensity. Since phase is unknown, we assume zero phase:
   E_recon_input(x,y) = √I(x,y) · exp(i·0) = √I(x,y)

Twin Image Artifact
-------------------
The loss of phase information during intensity recording creates the characteristic
**twin image artifact** in inline holography. This artifact appears because:

- A real-valued field √I can be decomposed as: √I = (E + E*)/2
- Back-propagating this gives: Propagate(E) + Propagate(E*)
- The first term Propagate(E) produces the desired focused object image
- The second term Propagate(E*) produces an out-of-focus conjugate "twin image"

The twin image is a fundamental consequence of the phase problem in inline holography
and cannot be avoided without additional constraints or measurements. It appears as
a defocused, overlapping artifact in the reconstruction plane.

Physical Principles
-------------------
The intensity-only recording constraint arises from quantum mechanics and detector
physics:

- Photons carry energy proportional to intensity |E|², not field amplitude E
- Photodetectors (CCD/CMOS) count photons, measuring integrated intensity
- Phase information requires interferometric measurement with a known reference
- In inline geometry, the reference and object waves are not spatially separated

This differs from off-axis holography, where a spatial carrier frequency allows
phase information to be encoded in the intensity pattern and recovered through
Fourier filtering.

References
----------
- Gabor, D. (1948). "A new microscopic principle". Nature, 161(4098), 777-778.
  Original description of inline holography and twin image problem.

- Goodman, J. W. (2005). "Introduction to Fourier Optics" (3rd ed.). Roberts & Co.
  Chapter 9: Holography - comprehensive treatment of inline and off-axis methods.

- Schnars, U., & Jüptner, W. (2005). "Digital Holography". Springer.
  Modern treatment of digital holography including twin image suppression techniques.

Implementation Notes
--------------------
This implementation simulates the physical intensity recording process to generate
realistic inline holograms with twin image artifacts. This is essential for creating
training data that matches experimental holography conditions.
"""

from __future__ import annotations

import numpy as np

from hologen.holography.propagation import angular_spectrum_propagate
from hologen.types import ArrayComplex, ArrayFloat, HolographyConfig, HolographyStrategy


def _object_to_complex(object_field: ArrayFloat) -> ArrayComplex:
    """Convert a real amplitude field into a complex representation."""

    return object_field.astype(np.complex128)


def _field_to_intensity(field: ArrayComplex) -> ArrayFloat:
    """Convert a complex field to its intensity distribution."""

    return np.abs(field) ** 2


class InlineHolographyStrategy(HolographyStrategy):
    """Implement inline hologram generation and reconstruction with intensity recording.

    This strategy simulates realistic inline (Gabor) holography where only intensity
    is recorded at the detector plane. This physical constraint—the inability of
    detectors to measure phase—creates the characteristic twin image artifact during
    reconstruction.

    Physical Process
    ----------------
    Inline holography follows a three-step process:

    1. **Propagation**: The object field E_obj(x,y) = A_obj·exp(iφ_obj) propagates
       from the object plane to the detector plane, creating a complex field
       E_det(x,y) = A_det·exp(iφ_det) at the hologram plane.

    2. **Intensity Recording**: The detector records only the intensity pattern:

       I(x,y) = |E_det(x,y)|² = A_det²(x,y)

       The phase information φ_det(x,y) is irreversibly lost during this measurement.
       Real photodetectors (CCD, CMOS sensors) respond to photon flux (intensity),
       not to the electromagnetic field phase.

    3. **Reconstruction**: To recover the object, we back-propagate from the recorded
       intensity. Since phase is unknown, we assume zero phase:

       E_recon_input(x,y) = √I(x,y) · exp(i·0) = √I(x,y)

       This real-valued field is back-propagated to the object plane to form the
       reconstruction.

    Twin Image Artifact
    -------------------
    The loss of phase information during intensity recording creates the twin image
    artifact, which is a fundamental limitation of inline holography. The artifact
    arises because:

    - A real-valued field √I can be mathematically decomposed as: √I = (E + E*)/2
      where E is the true complex field and E* is its complex conjugate

    - Back-propagating this decomposition gives:
      Reconstruction = Propagate(√I) = Propagate((E + E*)/2)
                     = Propagate(E) + Propagate(E*)

    - The first term Propagate(E) produces the desired **real image** (focused object)

    - The second term Propagate(E*) produces the unwanted **twin image** (out-of-focus
      conjugate artifact)

    The twin image appears as a defocused, overlapping artifact in the reconstruction
    plane. It cannot be removed without additional measurements or constraints because
    the phase information needed to distinguish E from E* was lost during recording.

    The strength of the twin image depends on the phase content of the original object:

    - **Amplitude-only objects**: Minimal twin image (phase was already zero/constant)
    - **Phase objects**: Strong twin image (significant phase information was lost)
    - **Complex objects**: Twin image strength proportional to phase variation

    Inline vs Off-Axis Holography
    ------------------------------
    This implementation differs fundamentally from off-axis holography:

    **Inline Holography** (this class):
    - Reference and object beams propagate along the same axis
    - Only intensity |E|² is recorded (phase is lost)
    - Twin image artifact is unavoidable in reconstruction
    - Simpler optical setup, but lower reconstruction quality
    - Matches Gabor's original holography technique (1948)

    **Off-Axis Holography** (OffAxisHolographyStrategy):
    - Reference beam at an angle creates spatial carrier frequency
    - Phase information encoded in intensity fringe pattern
    - Fourier filtering can separate real and twin images
    - No twin image artifact in properly filtered reconstruction
    - More complex setup, but higher reconstruction quality

    The key difference is that off-axis holography encodes phase information in the
    spatial frequency of interference fringes, allowing phase recovery from intensity
    measurements. Inline holography has no such encoding mechanism, making phase loss
    irreversible.

    Physical Basis
    --------------
    The intensity-only recording constraint arises from fundamental physics:

    - Photodetectors measure photon arrival rate (intensity ∝ |E|²)
    - Individual photons carry energy, not phase information
    - Phase is a property of the electromagnetic wave field
    - Measuring phase requires interference with a known reference
    - In inline geometry, object and reference are not spatially separated

    This is not a limitation of the detector technology—it is a fundamental consequence
    of quantum mechanics and the nature of light-matter interaction.

    Implementation Purpose
    ----------------------
    This implementation simulates the physical intensity recording process to generate
    realistic inline holograms with twin image artifacts. This is essential for:

    - Creating training data that matches experimental holography conditions
    - Testing ML models' ability to handle twin image artifacts
    - Studying twin image suppression and phase retrieval algorithms
    - Bridging the gap between synthetic and real holography data

    For applications requiring twin-image-free data, use OffAxisHolographyStrategy
    instead, which simulates the spatial carrier frequency approach.

    Examples
    --------
    Generate an inline hologram with twin image artifacts:

    >>> from hologen.holography.inline import InlineHolographyStrategy
    >>> from hologen.types import HolographyConfig, GridSpec, OpticsConfig
    >>> import numpy as np
    >>>
    >>> # Create configuration
    >>> grid = GridSpec(shape=(512, 512), pixel_pitch=1e-6)
    >>> optics = OpticsConfig(wavelength=532e-9, propagation_distance=0.01)
    >>> config = HolographyConfig(grid=grid, optics=optics)
    >>>
    >>> # Create complex object with phase
    >>> object_field = np.ones((512, 512), dtype=np.complex128)
    >>> object_field[200:300, 200:300] *= np.exp(1j * np.pi/2)  # Phase object
    >>>
    >>> # Generate hologram (intensity recording)
    >>> strategy = InlineHolographyStrategy()
    >>> hologram = strategy.create_hologram(object_field, config)
    >>>
    >>> # Reconstruct (will contain twin image)
    >>> reconstruction = strategy.reconstruct(hologram, config)
    >>> # reconstruction now contains both real and twin images

    See Also
    --------
    OffAxisHolographyStrategy : Off-axis holography without twin images
    angular_spectrum_propagate : Wave propagation method used internally

    References
    ----------
    .. [1] Gabor, D. (1948). "A new microscopic principle". Nature, 161(4098),
           777-778. Original description of inline holography and twin image problem.
    .. [2] Goodman, J. W. (2005). "Introduction to Fourier Optics" (3rd ed.).
           Roberts & Co. Chapter 9: Comprehensive treatment of holography.
    .. [3] Schnars, U., & Jüptner, W. (2005). "Digital Holography". Springer.
           Modern treatment including twin image suppression techniques.

    Notes
    -----
    The twin image artifact is a feature, not a bug. It represents the physical
    reality of inline holography and is essential for creating realistic synthetic
    datasets that match experimental conditions.
    """

    def create_hologram(
        self, object_field: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Generate an inline hologram from an object-domain complex field.

        This method simulates the physical process of inline holography recording,
        which consists of three steps:

        1. **Forward Propagation**: The object field propagates from the object plane
           to the detector plane using the angular spectrum method. This creates a
           complex field E_det(x,y) = A_det·exp(iφ_det) at the hologram plane.

        2. **Intensity Recording**: The detector records only the intensity pattern:
           I(x,y) = |E_det(x,y)|² = A_det²(x,y)

           This step simulates the fundamental limitation of real photodetectors
           (CCD, CMOS sensors), which respond to photon flux and cannot measure
           phase directly. The phase information φ_det(x,y) is irreversibly lost
           during this measurement process.

        3. **Complex Field Conversion**: The recorded intensity is converted back
           to a complex field representation with zero phase:
           E_hologram(x,y) = √I(x,y) · exp(i·0) = √I(x,y)

           This represents the information available for reconstruction. The
           assumption of zero phase is what creates the twin image artifact during
           back-propagation.

        Physical Background
        -------------------
        Real detectors cannot measure phase because:
        - Photodetectors count photons, which carry energy ∝ |E|²
        - Phase is a property of the electromagnetic wave, not individual photons
        - Measuring phase requires interference with a known reference wave
        - In inline geometry, object and reference are not spatially separated

        Twin Image Generation
        ---------------------
        The loss of phase information during intensity recording is what creates
        the characteristic twin image artifact in inline holography. When we
        back-propagate from the intensity-only hologram (assuming zero phase),
        the reconstruction contains:

        - **Real Image**: The focused object at the correct depth
        - **Twin Image**: An out-of-focus conjugate image (artifact)

        The twin image appears because a real-valued field √I can be decomposed
        as (E + E*)/2, and back-propagating this creates both the desired image
        Propagate(E) and the unwanted conjugate Propagate(E*).

        This behavior matches experimental inline holography and is essential for
        generating realistic training data for machine learning models.

        Parameters
        ----------
        object_field : ArrayComplex
            Complex object field with amplitude and phase information. The phase
            information will be lost during intensity recording, simulating the
            physical measurement process.
        config : HolographyConfig
            Holography configuration containing grid specifications and optical
            parameters (wavelength, propagation distance, pixel pitch).

        Returns
        -------
        ArrayComplex
            Complex field representing the recorded hologram. This field has:
            - Amplitude: √I where I is the recorded intensity
            - Phase: Zero (phase information was lost during recording)

            This intensity-only representation will produce twin image artifacts
            during reconstruction, matching physical inline holography behavior.

        See Also
        --------
        reconstruct : Back-propagates from the hologram to reconstruct the object
        angular_spectrum_propagate : Performs wave propagation calculations

        Notes
        -----
        For amplitude-only object fields (legacy ObjectSample), this method
        maintains backward compatibility. The intensity recording process still
        applies, but since the object has no phase information to lose, the
        behavior is similar to the previous implementation.

        References
        ----------
        .. [1] Gabor, D. (1948). "A new microscopic principle". Nature, 161(4098),
               777-778. Original description of inline holography.
        .. [2] Goodman, J. W. (2005). "Introduction to Fourier Optics" (3rd ed.).
               Roberts & Co. Chapter 9 covers the twin image problem.
        """
        # Step 1: Propagate object field to hologram plane
        # This simulates the physical propagation of light from object to detector
        propagated = angular_spectrum_propagate(
            field=object_field,
            grid=config.grid,
            optics=config.optics,
            distance=config.optics.propagation_distance,
        )

        # Step 2: Record intensity (simulate physical detector)
        # Real detectors (CCD, CMOS) can only measure intensity |E|², not phase
        # This is the fundamental limitation that creates twin image artifacts
        intensity = np.abs(propagated) ** 2

        # Step 3: Convert intensity back to complex field with zero phase
        # This represents the information available for reconstruction
        # Phase information is lost, which will cause twin image artifacts
        amplitude = np.sqrt(intensity)
        hologram_field = amplitude.astype(np.complex128)  # Zero phase

        return hologram_field

    def reconstruct(
        self, hologram: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct the object domain from an inline hologram.

        This method back-propagates from the intensity-only hologram to recover
        the object field. Because phase information was lost during intensity
        recording (in the create_hologram step), the reconstruction will contain
        both the desired focused object image and an unwanted out-of-focus twin
        image artifact.

        Twin Image Generation
        ---------------------
        The twin image artifact is a fundamental consequence of the phase problem
        in inline holography. It arises because:

        1. **Phase Loss During Recording**: The hologram was recorded as intensity
           |E|², discarding phase information φ(x,y). This is a physical limitation
           of real photodetectors.

        2. **Zero Phase Assumption**: For reconstruction, we assume the hologram
           has zero phase: E_hologram = √I · exp(i·0) = √I (real-valued field).

        3. **Conjugate Ambiguity**: A real-valued field √I can be decomposed as:
           √I = (E + E*)/2
           where E is the true complex field and E* is its conjugate.

        4. **Dual Image Formation**: Back-propagating this field creates:
           - Propagate(E) → **Real Image** (focused object at correct depth)
           - Propagate(E*) → **Twin Image** (out-of-focus conjugate artifact)

        The twin image appears as a defocused, overlapping artifact in the
        reconstruction plane. Its characteristics include:
        - Conjugate symmetry relative to the real image
        - Defocused appearance (wrong depth)
        - Cannot be removed without additional information or constraints

        Physical Basis
        --------------
        The twin image problem was first identified by Dennis Gabor in his 1948
        paper introducing holography. It is an inherent limitation of inline
        holography that arises from:

        - Detectors measuring intensity (photon flux) rather than field amplitude
        - Loss of phase information during the measurement process
        - Inability to distinguish between E and E* from intensity alone

        This differs from off-axis holography, where a spatial carrier frequency
        allows the real and twin images to be separated in the Fourier domain.

        Reconstruction Quality
        ----------------------
        The presence of twin images affects reconstruction quality:

        - **Amplitude-only objects**: Minimal twin image artifacts (phase was
          already zero or constant in the object)
        - **Phase objects**: Strong twin image artifacts (significant phase
          information was lost during recording)
        - **Complex objects**: Twin image strength depends on the phase variation
          in the original object field

        This behavior matches experimental inline holography and is essential for
        generating realistic training data for machine learning models that must
        handle twin image artifacts.

        Parameters
        ----------
        hologram : ArrayComplex
            Complex field from create_hologram representing the recorded hologram.
            This field has amplitude √I and zero phase, where I is the recorded
            intensity. The zero phase is what causes twin image artifacts.
        config : HolographyConfig
            Holography configuration containing grid specifications and optical
            parameters. The propagation distance is negated for back-propagation.

        Returns
        -------
        ArrayComplex
            Complex reconstruction containing both the real object image and the
            twin image artifact. The reconstruction will have:
            - **Real Image**: Focused object at the reconstruction plane
            - **Twin Image**: Out-of-focus conjugate artifact overlapping the real image

            The relative strength of the twin image depends on the phase content
            of the original object field.

        See Also
        --------
        create_hologram : Creates the intensity-only hologram (where phase is lost)
        angular_spectrum_propagate : Performs wave propagation calculations

        Notes
        -----
        The twin image artifact is a feature, not a bug. It represents the physical
        reality of inline holography and is essential for creating realistic
        synthetic datasets that match experimental conditions.

        For applications requiring twin-image-free reconstructions, consider:
        - Using off-axis holography (OffAxisHolographyStrategy)
        - Implementing iterative phase retrieval algorithms
        - Using multiple measurements with different constraints

        References
        ----------
        .. [1] Gabor, D. (1948). "A new microscopic principle". Nature, 161(4098),
               777-778. First description of the twin image problem.
        .. [2] Goodman, J. W. (2005). "Introduction to Fourier Optics" (3rd ed.).
               Roberts & Co. Section 9.4 discusses twin image formation.
        .. [3] Schnars, U., & Jüptner, W. (2005). "Digital Holography". Springer.
               Chapter 4 covers twin image suppression techniques.
        """

        reconstructed = angular_spectrum_propagate(
            field=hologram,
            grid=config.grid,
            optics=config.optics,
            distance=-config.optics.propagation_distance,
        )
        return reconstructed
