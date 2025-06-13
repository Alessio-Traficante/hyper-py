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
affiliations:
  - name: INAF-IAPS, Via Fosso del Cavaliere, 100, 00133 Rome (IT)
    index: 1
bibliography: Hyper_py.bib
---


# Summary

Source extraction and photometry of compact objects is a crucial task for observational astronomers. Several tools have been designed during the years to account for the complexity of the data, with particular attention to e.g. the estimation and removal of complex background, or to de-blend the contamination from nearby sources in order to obtain the most reliable flux estimation of the compact, point-like objects under investigation at the various wavelengths (Traficante15) 
These contributions are particularly relevant in star-forming regions, primarily observed in the Far-Infrared (FIR), sub-mm and mm domains where there is the peak emission of their cold, dense envelopes. Therefore, several software have been developed to deal with the complex background and blended regions observed with instruments such as Herschel (70-500mu) or ALMA (principally 1-3 mm). 
They substantially differ for the different philosophy chosen to detect objects and to estimate their peak and integrated fluxes. Among them, we developed Hyper HYbrid Photometry and Extraction Routine, originally developed in IDL to provide reliable and robust photometry of compact objects in FIR/sub-mm/mm images by: 1) detecting compact objects using a high-pass filter; 2) estimating and removing polynomial backgrounds from the images; 3) use 2D-Guassian fittings to model the identified sources, including group of sources with multiple 2D-Gaussian fittings in case of blended clusters. In this case, Hyper uses the 2D-Gaussian models of the group to subtract and deblend the companion sources from the image.

The aperture photometry in Hyper is finally performed in the background-subtracted, companion-removed images within the area of integration determined by the 2D-Gaussian footprint of each source, to obtain a robust photometry also in the most complex regions (CIT.).

The hybrid approach of Hyper comes from this combination of using the 2D-Gaussian models, as it has done e.g. in CIT., and of performing a classical aperture photometry such as in e.g. CIT.

Here we present Hyper-py, a revised, more extended version of the Hyper software now fully developed in Python.


# Statement of need
Hyper-py is a Python package freely accessible to the community. This new Python implementation is a conversion of the IDL version HYPER which includes several
improvements to the original package:
-	Parallel execution for multi-map analysis: Hyper-Py introduces built-in parallelization for the analysis of multiple input maps. On multi-core systems, each map is independently assigned to a dedicated core, allowing concurrent execution of the full photometric pipeline—source detection, background fitting, Gaussian modeling, and aperture photometry—for each map. This parallel framework substantially improves computational efficiency without altering the scientific output for individual maps.
-	Advanced background estimation strategy: The code now supports multiple background subtraction modes to improve the accuracy of background removal, especially in complex or blended regions. While the original IDL version estimated and subtracted the background independently of the source modeling [CIT.], Hyper-Py extends this approach by optionally performing an additional joint fit of the background together with the 2D elliptical Gaussians. This combined fitting strategy allows for a more accurate reconstruction of the local background structure, particularly in the presence of overlapping sources or inclined background gradients.
-	L2 regularization in background fitting: The estimation of the background polynomial terms now optionally includes L2 regularization (ridge regression), which acts to suppress unphysically large coefficients. This constraint helps stabilize the background solution in regions with strong intensity gradients or highly variable emission, reducing the risk of overfitting and improving the robustness of the background model.
-	In HYPER-py, we use the Normalized Mean Squared Error (NMSE) to identify the best fit configuration (i.e., box size and polynomial order), rather than relying solely on the reduced chi-squared value. This choice is motivated by the fact that the reduced chi-squared depends directly on the assumed noise model and pixel weighting, which can vary depending on masking, inverse RMS weights, or SNR-based weights. As a result, the $\chi^2_\nu$ value may not provide a fair comparison across different background fitting trials if the number of contributing pixels changes due to masking. In contrast, the NMSE is a robust, unitless, and scale-independent metric that quantifies the fraction of residual power relative to the total signal power in the cutout. It remains consistent regardless of weighting or the specific distribution of valid pixels, making it more reliable for comparing different model configurations under varying conditions. Therefore, NMSE serves as a more stable and uniform criterion for identifying the optimal fit in the context of masked and variable-size background fitting.
-	Improved user configurability: Hyper-Py is designed to be more user-friendly, featuring a clear and well-documented configuration file. This allows users to adapt the full photometric workflow to a wide range of observational conditions and scientific goals by modifying only a minimal set of parameters. The modular structure of the configuration also enhances transparency and reproducibility in all stages of the analysis.

We assessed the performance of the Hyper-py pipeline using a dedicated suite of simulations. Specifically, we adopted a noise-only map derived from the ALMA program #2022.1.0917.S and generated five independent realizations by injecting 100 synthetic 2D Gaussian sources into each map, for a total of 500 sources. These sources were designed to emulate the properties of real compact astronomical objects: integrated fluxes ranged between 10 and 20 times the map rms (corresponding to peak fluxes of approximately 1–1.5 times the rms), and FWHMs spanned from 0.5 to 1.5 times the beam size, as computed from the FITS header, to simulate both unresolved and moderately extended sources (e.g., CIT.). The source position angles were randomly assigned, and a minimum overlap fraction of 30% was imposed to ensure a significant level of blending, thus providing a rigorous test of the code under realistic and challenging conditions.We then compared the performance of the original IDL implementation of Hyper with the new Python-based Hyper-py version. Both codes were run using equivalent configurations, with Hyper-py additionally benefiting from its extended capabilities—such as improved background estimation, optional L2 regularization, and multi-core parallel processing. The main results of this comparison are summarized in Table 1.


HYPER-py can be downloaded through standard pip installation 
via the command "pip install hyper_py" or through the GitHUB repository ….


# Citations



