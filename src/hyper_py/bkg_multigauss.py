import numpy as np
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def estimate_masked_background(cutout, cutout_header, xcen_cut, ycen_cut, aper_sup, max_fwhm_extent, orders, suffix,
                                count_source_blended_indexes=None, config=None, logger=None, logger_file_only=None):
    """
    Estimate a polynomial background from a cutout map, masking elliptical/circular regions
    around sources using a radius = aper_sup.

    Parameters
    ----------
    cutout : 2D ndarray
        The input image cutout.
    xcen_cut, ycen_cut : array_like
        Source positions in the cutout (already shifted).
    aper_sup : float
        Size of the mask radius, in pixels.
    orders : list of int
        List of polynomial orders to try (the maximum is used).
    group_id : tuple, optional
        (i,j) tuple for labeling output file.
    config : HyperConfig, optional
        Full configuration dictionary (needed for output path, visualization flag).

    Returns
    -------
    cutout_bs : 2D ndarray
        Cutout with polynomial background subtracted.
    bg_poly : 2D ndarray
        The estimated background model.
    """

    # print("[INFO] Estimating initial background on masked cutout...")
    mask_bg = np.ones_like(cutout, dtype=bool)
    r_mask = max_fwhm_extent

    for xc, yc in zip(xcen_cut, ycen_cut):
        aperture = CircularAperture((xc, yc), r=r_mask)
        mask_obj = aperture.to_mask(method="center")
        mdata = mask_obj.to_image(cutout.shape)
        if mdata is not None:
            mask_bg[mdata > 0] = False

    y_bg, x_bg = np.where(mask_bg)
    z_bg = cutout[y_bg, x_bg]
    yy, xx = np.indices(cutout.shape)
    

    try:
        # Select only valid (non-NaN) pixels
        valid = ~np.isnan(z_bg)
        x_bg_valid = x_bg[valid]
        y_bg_valid = y_bg[valid]
        z_bg_valid = z_bg[valid]
    
        # Build the design matrix
        max_order = max(orders)
        terms = []
        param_names = []
    
        for dx in range(max_order + 1):
            for dy in range(max_order + 1 - dx):
                terms.append((x_bg_valid ** dx) * (y_bg_valid ** dy))
                param_names.append(f"c{dx}_{dy}")
    
        A = np.vstack(terms).T  # shape: (N_pixels, N_terms)
    
        # Solve least squares: A @ coeffs ≈ z
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z_bg_valid, rcond=None)
    
        # Map coefficients to names
        poly_params = dict(zip(param_names, coeffs))
    
        # Build the background model
        bg_poly = np.zeros_like(cutout, dtype=float)
        for pname, coeff in poly_params.items():
            dx, dy = map(int, pname[1:].split("_"))
            bg_poly += coeff * (xx ** dx) * (yy ** dy)
    
        # Subtract background
        cutout_bs = cutout - bg_poly
        logger_file_only.info(f"[INFO] Masked background polynomial (order ≤ {max_order}) subtracted.")
    
    except Exception as e:
        logger_file_only.info(f"[WARNING] Background estimation failed: {e}")
        cutout_bs = cutout.copy()
        bg_poly = np.zeros_like(cutout)










    # === Optional 3D visualization ===
    if config is not None:
        try:
            visualize_bg = config.get("visualization", "visualize_bg_separate", False)
        except:
            pass


    # --- save separated background estimation in fits format --- #
    try:
        fits_bg_separate = config.get("fits_output", "fits_bg_separate", False)
        dir_comm =  config.get("paths", "dir_comm")
        fits_output_dir_bg_separate = dir_comm + config.get("fits_output", "fits_output_dir_bg_separate", "Fits/Bg_separate")  
    except:
        fits_bg_separate = False


    if fits_bg_separate:
        # Ensure the output directory exists
        os.makedirs(fits_output_dir_bg_separate, exist_ok=True)
        
        label_name = f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}"
        filename = f"{fits_output_dir_bg_separate}/{label_name}_bg_masked3D.fits"
    
        # Create a PrimaryHDU object and write the array into the FITS file
        convert_mjy=config.get("units", "convert_mJy")

        hdu = fits.PrimaryHDU(data=bg_poly, header=cutout_header)
        if convert_mjy:
            hdu.header['BUNIT'] = 'mJy/pixel'
        else: hdu.header['BUNIT'] = 'Jy/pixel'    
        hdu.writeto(filename, overwrite=True)


    # --- Visualize separated background estimation in png format --- #
    if visualize_bg:
        logger_file_only.info("[INFO] Plotting 3D background model from masked map subtraction...")  
        
        dir_comm =  config.get("paths", "dir_comm")
        output_dir_vis = dir_comm + config.get("visualization", "output_dir_bg_separate")
        os.makedirs(output_dir_vis, exist_ok=True)
                
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, bg_poly, cmap="viridis", linewidth=0, antialiased=True)

        ax.set_xlabel("X (pix)", fontsize=8, fontweight="bold")
        ax.set_ylabel("Y (pix)", fontsize=8, fontweight="bold")
        ax.set_zlabel("Flux (Jy)", fontsize=8, fontweight="bold")
        for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            label.set_fontsize(8)
            label.set_fontweight("bold")

        ax.set_title("Initial Background Model from Masked Map", fontsize=10, fontweight="bold")
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.12)
        label_str = f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}" if count_source_blended_indexes is not None else "group"
        outname = os.path.join(output_dir_vis, f"{label_str}_bg_masked3D.png")
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close()


    return cutout_bs, bg_poly