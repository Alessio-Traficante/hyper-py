---
title: 'HYPER-PY: HYbrid Photometry and Extraction Routine in PYthon'
tags:
  - Python
  - astronomy
  - photometry
  - astronomy tools
authors:
  - name: Alessio Traficante
    orcid: 0000-0003-1665-6402
    affiliation: 1 
    corresponding: true
  - name: Fabrizio De Angelis
    affiliation: 1
  - name: Alice Nucara
    affiliation: 1
  - name: Milena Benedettini
    affiliation: 1
affiliations:
  - name: INAF-IAPS, Via Fosso del Cavaliere, 100, 00133 Rome (IT)
    index: 1
date: 02 September 2025
repository: https://github.com/Alessio-Traficante/hyper-py
bibliography: paper.bib
---

 
 
# Summary
 
Source extraction and photometry of compact objects are key tasks in observational astronomy. Numerous tools have been developed to tackle the complexities of astronomical data, especially for precise background estimation and source deblending, which are essential for reliable flux measurements across wavelengths (e.g., Cutex: @Molinari11; getsources: @Menshinkov12; Fellwalker: @Berry15; Astrodendro). These challenges are particularly significant in star-forming regions, best observed in the far-infrared (FIR), sub-millimeter, and millimeter bands, where cold, dense compact sources emit strongly. To address this, several software packages have been designed for handling the structured backgrounds and blended sources typical of observations by instruments like Herschel (70–500 μm) and ALMA (1–3 mm). These tools differ in detection and flux estimation approaches. Within this context, we developed HYPER (HYbrid Photometry and Extraction Routine, @Traficante15), originally implemented in IDL, aiming to deliver robust and reproducible photometry of compact sources in FIR/sub-mm/mm maps. HYPER combines source detection via high-pass filtering, background estimation through local polynomial fitting, and source modeling with 2D elliptical Gaussians, simultaneously fitting multiple Gaussians to deblend overlapping sources.

Aperture photometry in **HYPER** is performed on background- and companion-subtracted images, using footprints defined by the source’s 2D Gaussian models, ensuring robust flux measurements in crowded or structured environments (@Traficante15; @Traficante23). The hybrid approach combines parametric Gaussian modeling with classical aperture photometry.
Here, we present **Hyper-Py**, a fully restructured and extended Python implementation of **HYPER**. **Hyper-Py** preserves the original logic while offering improvements in performance, configurability, and background modeling capabilities, making it a flexible modern tool for source extraction and photometry across diverse datasets. Notably, **Hyper-Py** enables background estimation and subtraction across individual slices of 3D datacubes, allowing consistent background modeling along the spectral axis for line or continuum studies in spectrally resolved observations.

 
 
# Statement of need
*Hyper-Py* is an open-source Python package freely available to the community. This new implementation builds upon and improves the original IDL **HYPER** by incorporating several major advancements:

**Parallel execution for multi-map analysis**. *Hyper-Py* employs built-in parallelization where each input map is independently assigned to a processing core on multi-core systems. This allows concurrent execution of the complete photometric pipeline on different maps simultaneously. This parallel framework dramatically increases computational efficiency without altering individual map results.

**Native support for FITS datacubes**. The software treats each slice along the third axis as an independent 2D map, compatible with parallel processing, allowing simultaneous background subtraction per slice. The output is a 3D background cube matching the input cube’s shape, configurable for targeted regions or full spatial coverage through a user-friendly configuration file. This capability provides flexibility for both line-specific and broader continuum background modeling.

**Improved source detection reliability**. Source detection has been enhanced with a robust sigma-clipping algorithm that iteratively estimates the root mean square (*rms*) noise of input maps, excluding outliers to characterize background fluctuations accurately—even with bright sources or structures present. This *rms* serves as a threshold reference for detecting compact sources exceeding a configurable significance level (*n_sigma* × *rms*), settable via the config file. Such refinement increases detection reliability and reproducibility across heterogeneous datasets.

**Advanced background estimation strategy**. Unlike the original IDL implementation @Traficante15, which modeled background separately from source fitting, *Hyper-Py* supports multiple statistical fitting techniques—least-squares, Huber, and Theil–Sen regressions—applied to masked cutouts around each source. Least-squares performs well in regions dominated by Gaussian noise; Huber regression balances L2 and L1 losses to reduce outlier effects via a tunable parameter ε (huber_epsilons in the config file); and Theil–Sen is a non-parametric, robust method ideal for non-Gaussian noise or contamination. When multiple methods are enabled, *Hyper-Py* selects the best model by minimizing residuals, ensuring accurate background reconstruction even with gradients or faint extended emission. Furthermore, an optional joint fit of background and 2D Gaussian models with L2 (ridge) regularization stabilizes fits in regions with strong gradients, preventing background overfitting at the expense of source flux.

**Gaussian plus background model optimization strategy**. *Hyper-py* utilizes the Levenberg–Marquardt algorithm through the lmfit package’s “least_squares” minimizer, allowing control over the cost function’s residual weighting by selecting different loss models. The default “cauchy” loss diminishes outlier influence, improving robustness to data artifacts, unmasked sources, or non-Gaussian noise. Alternatives like “soft_l1” and “huber” are also available for specific dataset optimization.

**Model selection criteria for background fitting**. Background model selection criteria are configurable, offering Normalized Mean Squared Error (NMSE), reduced chi-squared ($\chi^2_\nu$), or Bayesian Information Criterion (BIC) via the config file. NMSE is default due to its robust, scale- and weighting-independent nature, ideal under varying masking or pixel weighting. Reduced chi-squared depends on noise models and pixel counts, potentially biasing selection. BIC penalizes model complexity, favoring simplicity when residuals are comparable. These options allow users to select criteria best suited to their scientific aims and data noise properties.

**Improved user configurability**. *Hyper-Py* is designed to be more user-friendly, featuring a clear and well-documented configuration file. This allows users to adapt the full photometric workflow to a wide range of observational conditions and scientific goals by modifying only a minimal set of parameters. 
 
We assessed *Hyper-Py* performance using extensive simulations. Starting from a noise-only map based on ALMA program #2022.1.0917.S, we generated two maps with reference headers and superimposed varying backgrounds plus 500 synthetic 2D Gaussian sources. These sources mimic real compact objects with integrated fluxes spanning 8–20 times the map rms (peak fluxes ~1–1.5 × *rms*) and FWHM sizes of 0.5–1.5 times the beam size to include both unresolved and moderately extended sources (@Elia21). Random position angles and a minimum 30% overlap ensured realistic blending, thus providing a rigorous test of the code under realistic and challenging conditions.
We compared the original IDL **HYPER** and *Hyper-Py* under equivalent configurations, with the latter benefiting from improved background estimation, optional regularization, and parallel processing. The key results are presented in **Table 1**, detailing differences in source identification and false positives between the codes.


| Catalog | Source Type | Total | Matched | False | False Percentage |
|--------:|:------------|------:|--------:|------:|------------------:|
|       1 | *Hyper-py*    |   500 |     490 |     4 |          0.8%   |
|       1 | **HYPER (IDL)**   |   500 |     493 |    73 |         12.9%   |
|       2 | *Hyper-py*    |   500 |     487 |     4 |          0.8%   |
|       2 | **HYPER (IDL)**   |   500 |     487 |    46 |          8.6%   |

In addition, Figure 1 and Figure 2 show the differences between the peak fluxes and the integrated fluxes of the sources with respect to the reference values of the simulated input sources as estimated by *Hyper-Py* and HYPER, respectively.

*Hyper-Py* is freely available for download via its [GitHub repository](https://github.com/Alessio-Traficante/hyper-py), and can also be installed using pip, as described in the accompanying README file.

![histogram of the differences between the peak fluxes of the sources as recovered by *Hyper-Py* and HYPER, respectively, with respect to the reference values of the simulated input sources](Figures/Flux_Diff_Histogram_Peak.png)

![histogram of the differences between the peak fluxes of the sources as recovered by *Hyper-Py* and HYPER, respectively, with respect to the reference values of the simulated input sources](Figures/Flux_Diff_Histogram_Int.png)