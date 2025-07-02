import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture

def masked_background_single_sources(cutout_masked, cutout, cutout_header, x0, y0, external_sources, max_fwhm_extent, pol_orders_separate, suffix, source_id, config, logger_file_only):
    """
    Estimate polynomial background on a cutout image, masking the source region(s),
    and optionally save the result as FITS and/or 3D plot.

    Parameters
    ----------
    cutout : 2D np.ndarray
        Image cutout to analyze.
    x0, y0 : float
        Centroid position (in pixels) of the main source.
    external_sources : list of (x, y)
        Additional centroids to mask (e.g., neighboring sources).
    max_fwhm_extent : float
        Mask radius (in pixels) around each source.
    pol_orders_separate : list of int
        Polynomial orders to consider (only max used here).
    suffix : str
        Label suffix for output naming (e.g., map ID).
    source_id : int
        Index of the current source (0-based).
    config : ConfigReader
        Configuration object to read parameters.
    logger_file_only : Logger
        Logger object for writing progress/status.
    """
    logger_file_only.info("[INFO] Estimating background separately on masked cutout...")

    # Create mask: True = valid pixel
    mask_bg = np.ones_like(cutout_masked, dtype=bool)
    r_mask = max_fwhm_extent
    
    # Also mask NaN values in the cutout
    mask_bg[np.isnan(cutout_masked)] = False


    #- Look at main centroids -#
    xcen_cut_bg = np.array([x0])
    ycen_cut_bg = np.array([y0])


    # Mask circular areas around all source centroids
    for xc, yc in zip(xcen_cut_bg, ycen_cut_bg):
        aperture = CircularAperture((xc, yc), r=r_mask)
        mask_obj = aperture.to_mask(method="center")
        mdata = mask_obj.to_image(cutout_masked.shape)
        if mdata is not None:
            mask_bg[mdata > 0] = False


    # Fit polynomial to valid background pixels
    y_bg, x_bg = np.where(mask_bg)
    z_bg = cutout_masked[y_bg, x_bg]
    
    valid = ~np.isnan(z_bg)
    x_bg_valid = x_bg[valid]
    y_bg_valid = y_bg[valid]
    z_bg_valid = z_bg[valid]
    
    # Build the design matrix for 2D polynomial
    max_order = max(pol_orders_separate)
    terms = []
    param_names = []
    
    for dx in range(max_order + 1):
        for dy in range(max_order + 1 - dx):
            terms.append((x_bg_valid**dx) * (y_bg_valid**dy))
            param_names.append(f"c{dx}_{dy}")
    
    A = np.vstack(terms).T  # Shape: (N, num_terms)
    
    # Solve linear least squares: A @ coeffs ≈ z
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_bg_valid, rcond=None)
    
    # Map coefficients to names
    poly_params = dict(zip(param_names, coeffs))
    
    # Build 2D background model
    yy, xx = np.indices(cutout_masked.shape)
    bg_model = np.zeros_like(cutout_masked)
    
    for pname, val in poly_params.items():
        dx, dy = map(int, pname[1:].split("_"))
        bg_model += val * (xx ** dx) * (yy ** dy)
    

    # Subtract background from cutout
    cutout -= bg_model
    logger_file_only.info("[INFO] Background subtracted from cutout.")


    return cutout, bg_model, poly_params
