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
date: 13 August 2017
bibliography: Hyper_py.bib
---

 
 
# Summary
 
Source extraction and photometry of compact objects are fundamental tasks in observational astronomy. Over the years, various tools have been developed to address the inherent complexity of astronomical data—particularly for accurate background estimation and removal, and for deblending nearby sources to ensure reliable flux measurements across multiple wavelengths (@Molinari11). These challenges are especially pronounced in star-forming regions, which are best observed in the far-infrared (FIR), sub-millimeter, and millimeter regimes, where the cold, dense envelopes of compact sources emit most strongly.
To address these needs, several software packages have been designed to handle the structured backgrounds and blended source populations typical of observations with instruments such as Herschel (70–500 μm) and ALMA (1–3 mm). These packages differ significantly in their detection strategies and flux estimation methods (CIT.). Among them, we developed HYPER (HYbrid Photometry and Extraction Routine, @Traficante2015), originally implemented in IDL, with the goal of providing robust and reproducible photometry of compact sources in FIR/sub-mm/mm maps. HYPER combines: (1) source detection via high-pass filtering; (2) background estimation and removal through local polynomial fitting; and (3) source modeling using 2D elliptical Gaussians. For blended regions, HYPER fits multiple Gaussians simultaneously to deblend overlapping sources, subtracting companions before performing photometry.
Aperture photometry in HYPER is then carried out on the background-subtracted, companion-subtracted images, using the footprint defined by each source’s 2D Gaussian model. This ensures a consistent and robust integrated flux measurement, even in crowded or strongly structured environments (CIT.).
The hybrid nature of HYPER lies in this combined approach: using 2D Gaussian modeling, as done in methods such as CIT., while retaining classical aperture photometry techniques, as in CIT.
In this work, we present Hyper-py, a fully restructured and extended version of HYPER developed entirely in Python. Hyper-py not only replicates the core logic of the original IDL implementation, but introduces multiple improvements in performance, configurability, and background modeling capabilities—making it a modern and flexible tool for source extraction and photometric analysis across a wide range of datasets. Notably, Hyper-py also introduces the ability to estimate and subtract the background emission across individual slices of 3D datacubes, enabling consistent background modeling along the spectral axis for line or continuum studies in spectrally resolved observations.

 
 
# Statement of need
Hyper-py is a Python package freely accessible to the community. This new Python implementation is a conversion of the IDL version HYPER which includes several
improvements to the original package:
Parallel execution for multi-map analysis: Hyper-Py introduces built-in parallelization for the analysis of multiple input maps. On multi-core systems, each map is independently assigned to a dedicated core, allowing concurrent execution of the full photometric pipeline—source detection, background fitting, Gaussian modeling, and aperture photometry—for each map. This parallel framework substantially improves computational efficiency without altering the scientific output for individual maps.

Native support for the analysis of FITS datacubes, enabling users to treat each slice along the third axis as an independent 2D map. This functionality is fully compatible with the existing parallelization framework, allowing multiple slices to be processed simultaneously across different cores. The primary goal of this mode is to estimate the spatially-varying background emission on a per-slice basis. The final output is a reconstructed 3D background datacube with the same shape as the input cube. This background cube can either be focused on a specific region or line of sight—leaving all other voxels as NaNs—or computed across the full spatial extent of the cube. The extraction region and related parameters are configurable via the config.yaml file, offering flexibility for both targeted and global background modeling.

Advanced background estimation strategy. Hyper-py introduces a robust and flexible background estimation framework designed to improve subtraction accuracy in complex or blended regions. Unlike the original IDL version, which estimated and subtracted the background independently of source modeling [CIT.], Hyper-py supports multiple statistical methods for background fitting, applied to masked cutouts around each source. Users can select from least-squares regression, Huber regression, and Theil–Sen regression, either individually or in combination. The least-squares method is optimal in regions dominated by Gaussian noise. The Huber regressor provides robustness against outliers by interpolating between L2 and L1 loss functions, with the tuning parameter ε (huber_epsilons in the config file) controlling the transition. The Theil–Sen estimator is a non-parametric, highly robust approach particularly suited for non-Gaussian noise or residual contamination. When multiple methods are enabled, Hyper-py evaluates all and selects the background model that minimizes the residuals within the unmasked region, ensuring accurate reconstruction even in the presence of variable gradients or faint extended emission. In addition, Hyper-py offers an optional joint fit of the background and 2D elliptical Gaussians, which may improve stability or convergence in specific cases. When this combined fit is used, the background polynomial terms can be regularized using L2 (ridge) regression, helping suppress unphysically large coefficients. This constraint enhances the robustness of the background model in regions with strong intensity gradients or spatially variable emission, reducing the risk of overfitting.


Model selection criteria for background fitting. In Hyper-py, the optimal background model—i.e., the best combination of box size and polynomial order—is determined by evaluating the residuals between the observed data and the fitted model. The framework offers a configurable choice of model selection criteria, specified via the config.yaml file. Users can select from three statistical metrics: Normalized Mean Squared Error (NMSE), reduced chi-squared ($\chi^2_\nu$), or the Bayesian Information Criterion (BIC). By default, Hyper-py adopts NMSE, a robust, unitless, and scale-independent metric that quantifies the fraction of residual power relative to the total signal power in the cutout. Unlike reduced chi-squared, NMSE is not sensitive to changes in pixel weighting schemes or the number of valid (unmasked) pixels, making it particularly reliable when background fits are performed under varying masking conditions, inverse-RMS weighting, or SNR-based weighting. In contrast, the reduced chi-squared statistic depends directly on the assumed noise model and weighting, and may therefore be biased when the number or distribution of contributing pixels changes. The BIC criterion offers an alternative that penalizes overfitting by incorporating the number of model parameters and the sample size, favoring simpler models when multiple fits achieve similar residuals. This flexibility allows users to tailor the model selection strategy to the scientific context or noise characteristics of their data, ensuring that the chosen background model is both statistically sound and physically meaningful.

Improved user configurability: Hyper-Py is designed to be more user-friendly, featuring a clear and well-documented configuration file. This allows users to adapt the full photometric workflow to a wide range of observational conditions and scientific goals by modifying only a minimal set of parameters. The modular structure of the configuration also enhances transparency and reproducibility in all stages of the analysis.
 
We assessed the performance of the Hyper-py pipeline using a dedicated suite of simulations. Specifically, we adopted a noise-only map derived from the ALMA program #2022.1.0917.S and generated five independent realizations by injecting 100 synthetic 2D Gaussian sources into each map, for a total of 500 sources. These sources were designed to emulate the properties of real compact astronomical objects: integrated fluxes ranged between 10 and 20 times the map rms (corresponding to peak fluxes of approximately 1–1.5 times the rms), and FWHMs spanned from 0.5 to 1.5 times the beam size, as computed from the FITS header, to simulate both unresolved and moderately extended sources (e.g., CIT.). The source position angles were randomly assigned, and a minimum overlap fraction of 30% was imposed to ensure a significant level of blending, thus providing a rigorous test of the code under realistic and challenging conditions.We then compared the performance of the original IDL implementation of Hyper with the new Python-based Hyper-py version. Both codes were run using equivalent configurations, with Hyper-py additionally benefiting from its extended capabilities—such as improved background estimation, optional L2 regularization, and multi-core parallel processing. The main results of this comparison are summarized in Table 1.
 
 
HYPER-py can be downloaded through standard pip installation
via the command "pip install hyper_py" or through the GitHUB repository ….

