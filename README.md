# aafragpy

Python implementation of the **AAFrag** secondary particle production model (based on QGSJET-II-04m) with adaptive curve-morphing interpolation. The package provides fast, vectorized reconstruction of differential production cross-sections and spectra from high-energy hadronic interactions while preserving the original tabulated Monte Carlo precision.

---

### Features

* **Secondary Products:** Photons (γ), electrons and positrons (e±), neutrinos (ν_e, ν̄_e, ν_μ, ν̄_μ, ν_all), nucleons (p, p̄, n, n̄), and light antinuclei (d̄, ³Hē).
* **Collision Systems:** p–p, p–He, He–p, He–He, heavy projectile ions (C, Al, Fe) on proton targets, and antiproton-induced channels (p̄–p, p̄–He).
* **Broad Energy Range:** Tabulated interactions from kinematic thresholds (hundreds of MeV / low GeV) up to hundreds of PeV / EeV.
* **Integrated Alternative Models:** Built-in parameterizations for p–p interactions from Kamae et al. (2006) and Kafexhiu et al. (2014).

---

### Installation

Install `aafragpy` via `pip`:

```bash
pip install aafragpy
```

---

### Interactive Tutorial

A step-by-step walkthrough of the package API and spectral calculations is available in the Jupyter Notebook tutorial.

You can launch and run the interactive tutorial directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aafragpy/aafragpy/blob/main/aafragpy_tutorial.ipynb)

---

### Citation Policy

> **Important Notice on Citations:**  
> If you use `aafragpy` in your scientific work, **please cite both** the `aafragpy` package paper **and** the corresponding original AAFrag / Monte Carlo data publications. Proper attribution to the original cross-section calculations ensures continued support and development of hadronic interaction models.

#### 1. Python Package Citation

* S. Koldobskiy, M. Kachelrieß, A. Lskavyan, A. Neronov, S. Ostapchenko, and D. V. Semikoz,  
  *“Energy spectra of secondaries in proton-proton interactions,”*  
  [Phys. Rev. D 104, 123027 (2021)](https://doi.org/10.1103/PhysRevD.104.123027), [arXiv:2110.00496](https://arxiv.org/abs/2110.00496).

#### 2. Original AAFrag Model & Data Calculations

* **AAfrag 2.0 (v2.02 Data Tables & Antinuclei):**  
  M. Kachelrieß, S. Ostapchenko, and J. Tjemsland,  
  *“AAfrag 2.01: Interpolation routines for Monte Carlo results on secondary production including light antinuclei in hadronic interactions,”*  
  [Comput. Phys. Commun. 287, 108698 (2023)](https://doi.org/10.1016/j.cpc.2023.108698), [arXiv:2206.00998](https://arxiv.org/abs/2206.00998).

* **AAfrag 1.0 (Original QGSJET-II-04m Tabulations):**  
  M. Kachelrieß, I. V. Moskalenko, and S. Ostapchenko,  
  *“AAfrag: Interpolation routines for Monte Carlo results on secondary production in proton-proton, proton-nucleus and nucleus-nucleus interactions,”*  
  [Comput. Phys. Commun. 245, 106846 (2019)](https://doi.org/10.1016/j.cpc.2019.08.001), [arXiv:1904.05129](https://arxiv.org/abs/1904.05129).

---

### Additional References for Alternative Models

If you utilize the optional alternative models included in the package, please also cite the respective source papers:

* **Kamae et al. (2006) + Erratum:**  
  * T. Kamae, N. Karlsson, T. Mizuno, T. Abe, and T. Koi,  
    *“Parameterization of γ, e±, and Neutrino Spectra Produced by p-p Interaction in Astronomical Environments,”*  
    [Astrophys. J. 647, 692–708 (2006)](https://doi.org/10.1086/505189), [arXiv:astro-ph/0605581](https://arxiv.org/abs/astro-ph/0605581).  
  * T. Kamae, N. Karlsson, T. Mizuno, T. Abe, and T. Koi,  
    *“Erratum: 'Parameterization of γ, e±, and Neutrino Spectra Produced by p-p Interaction in Astronomical Environments',”*  
    [Astrophys. J. 662, 779 (2007)](https://doi.org/10.1086/519449).

* **Kafexhiu et al. (2014):**  
  E. Kafexhiu, F. Aharonian, A. M. Taylor, and G. S. Vila,  
  *“Parametrization of gamma-ray production cross-sections for p-p interactions in a broad proton energy range from the kinematic threshold to PeV energies,”*  
  [Phys. Rev. D 90, 123014 (2014)](https://doi.org/10.1103/PhysRevD.90.123014), [arXiv:1406.7369](https://arxiv.org/abs/1406.7369).