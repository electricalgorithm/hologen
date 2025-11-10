
# HoloGen Toolkit: Generating Realistic Holographic Datasets with Controlled Noise and Aberrations

Holographic imaging systems, like any optical setup, are inevitably influenced by various noise sources and optical aberrations. To design algorithms that are both robust and physically meaningful, it is essential to **simulate these imperfections** accurately in synthetic datasets.

This document describes how to generate holographic datasets with different noise levels using the provided script, and explains the **purpose and impact** of each flag for **realistic holographic reconstruction research**.

---

## 1. Basic Usage

```bash
python scripts/generate_dataset.py \
    --samples 100 \
    --output ./dataset_noisy \
    --method inline \
    --height 512 \
    --width 512 \
    --pixel-pitch 6.4e-6 \
    --wavelength 532e-9 \
    --distance 0.05 \
    --speckle-contrast 0.8 \
    --sensor-read-noise 3.0 \
    --sensor-shot-noise \
    --sensor-dark-current 0.5 \
    --sensor-bit-depth 12
```

This command generates **100 holograms** of size **512×512**, using an **inline configuration**, a **green laser (532 nm)**, and a **realistic sensor model** with combined read, shot, and dark noise.

---

## 2. Optical Setup Parameters

| Parameter                     | Description                                                 | Typical Values         | Physical Relevance                                                                                                                                                                             |
| ----------------------------- | ----------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--method {inline, off_axis}` | Defines the holographic recording method.                   | `inline` or `off_axis` | *Inline* holography captures on-axis interference (requires twin-image removal), while *off-axis* introduces a carrier frequency to separate object and reference waves in the Fourier domain. |
| `--pixel-pitch`               | Distance between adjacent sensor pixels (m).                | 4–8 µm                 | Determines spatial sampling and field-of-view; directly affects reconstruction resolution.                                                                                                     |
| `--wavelength`                | Illumination wavelength (m).                                | 450e-9 to 633e-9       | Defines propagation and phase behavior; shorter wavelengths yield higher resolution.                                                                                                           |
| `--distance`                  | Propagation distance from object to sensor (m).             | 0.01–0.1               | Longer distances produce higher fringe density, affecting reconstruction complexity.                                                                                                           |
| `--carrier-x`, `--carrier-y`  | Carrier frequencies (cycles/m) for off-axis reference wave. | ~1e5–1e6               | Introduce angular offset for spatial frequency separation.                                                                                                                                     |
| `--carrier-sigma`             | Gaussian filter width for off-axis reconstruction.          | 1–10                   | Controls the bandwidth of carrier filtering; too small may cut signal, too large may retain DC noise.                                                                                          |

---

## 3. Sensor Noise Models

Real image sensors introduce stochastic variations during image capture. These are simulated through several independent but interacting noise components.

### **3.1 Read Noise**

```
--sensor-read-noise <σ>
```

* **Definition:** Electronic noise added during pixel readout, modeled as Gaussian with standard deviation σ.
* **Typical Range:** 1–10 electrons.
* **Physical Origin:** Thermal fluctuations in the sensor amplifier or ADC electronics.
* **Impact:** Limits sensitivity in low-light holography and affects phase retrieval at low signal intensities.

---

### **3.2 Shot Noise**

```
--sensor-shot-noise
```

* **Definition:** Poisson-distributed fluctuation proportional to signal intensity.
* **Physical Origin:** Quantum nature of light; randomness in photon arrival rate.
* **Impact:** Dominant in high-intensity regions; essential for realistic photon-limited reconstructions and phase-denoising model evaluation.

---

### **3.3 Dark Current**

```
--sensor-dark-current <mean>
```

* **Definition:** Mean thermal electrons accumulated per pixel even without illumination.
* **Typical Range:** 0.1–5 electrons/pixel/s.
* **Physical Origin:** Thermal excitation within the photodiode array.
* **Impact:** Adds a constant bias and noise floor, especially visible in long exposures or high temperatures.

---

### **3.4 Quantization Noise**

```
--sensor-bit-depth <bits>
```

* **Definition:** Simulates analog-to-digital conversion limits.
* **Typical Values:** 8–16 bits.
* **Impact:** Reduces gray-level precision; critical when testing phase retrieval algorithms that rely on fine amplitude variations.

---

## 4. Speckle Simulation

Speckle is an interference phenomenon inherent to coherent illumination and is **essential to simulate** for optical realism.

### **4.1 Speckle Contrast**

```
--speckle-contrast <ratio>
```

* **Definition:** Ratio of speckle standard deviation to mean intensity (0–1).
* **Physical Relevance:**

  * `1.0`: Fully developed speckle (laser-illuminated rough surface).
  * `<0.5`: Partial coherence or multi-look averaging conditions.
* **Impact:** Determines texture roughness and affects reconstruction fidelity under diffuse illumination.

### **4.2 Speckle Correlation Length**

```
--speckle-correlation <pixels>
```

* **Definition:** Spatial extent (in pixels) over which speckle grains remain correlated.
* **Physical Origin:** Numerical aperture, illumination divergence, and object roughness.
* **Impact:** Controls speckle grain size; crucial for comparing denoising or speckle reduction algorithms.

---

## 5. Aberration Simulation

Optical aberrations simulate imperfections in lenses or wavefronts, influencing focus and image quality.
Each aberration coefficient corresponds to **Zernike polynomial terms** applied to the reconstructed wavefront.

| Flag                                                       | Aberration Type | Description                                                        | Typical Coefficient Range |
| ---------------------------------------------------------- | --------------- | ------------------------------------------------------------------ | ------------------------- |
| `--aberration-defocus`                                     | Defocus         | Simulates axial misalignment or incorrect reconstruction distance. | ±0.1–0.5                  |
| `--aberration-astigmatism-x`, `--aberration-astigmatism-y` | Astigmatism     | Models cylindrical lens effects or sensor tilt.                    | ±0.05–0.2                 |
| `--aberration-coma-x`, `--aberration-coma-y`               | Coma            | Simulates asymmetric distortion from off-axis lens elements.       | ±0.05–0.3                 |

**Practical Use:**
These parameters are critical when designing learning-based reconstruction pipelines intended for **microscopy**, **digital holographic interferometry**, or **display calibration**, where real optics introduce unavoidable phase distortions.

---

## 6. Reproducibility and Randomization

```
--seed <integer>
```

Fixes the random seed for reproducibility. For publication-quality experiments, always log the seed to ensure that synthetic and noisy datasets can be exactly regenerated for benchmarking.

---

## 7. Dataset Preview Control

```
--no-preview
```

Disables generation of PNG preview files. Recommended for large datasets to reduce I/O load, but keep enabled for visual inspection during experimental parameter tuning.

---

## 8. Recommendations for Academic Experiments

| Objective                            | Recommended Noise Configuration                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **Algorithm benchmarking (clean)**   | Disable all noise and aberration flags.                                                           |
| **Photon-limited simulation**        | Enable `--sensor-shot-noise` with high intensity variation.                                       |
| **Low-end camera simulation**        | Combine `--sensor-read-noise 5`, `--sensor-bit-depth 8`, and `--sensor-dark-current 1.0`.         |
| **Real optical microscope modeling** | Include `--speckle-contrast 0.8`, `--speckle-correlation 3`, and mild `--aberration-defocus 0.2`. |
| **System calibration study**         | Vary one aberration coefficient per dataset while fixing all other parameters.                    |

---

## 9. Example for Progressive Noise Levels

```bash
# Low noise
python scripts/generate_dataset.py --samples 100 --output data_low --sensor-read-noise 1

# Medium noise
python scripts/generate_dataset.py --samples 100 --output data_mid --sensor-read-noise 3 --sensor-shot-noise

# High noise with aberrations
python scripts/generate_dataset.py --samples 100 --output data_high \
    --sensor-read-noise 5 --sensor-shot-noise --sensor-dark-current 2 \
    --speckle-contrast 0.9 --aberration-defocus 0.3
```

---

## 10. Conclusion

Accurate noise and aberration modeling is crucial for **bridging the gap between synthetic and experimental holography**. By controlling the parameters described above, researchers can produce datasets that emulate realistic imaging conditions, ensuring that trained models or reconstruction methods generalize to **true optical setups**.