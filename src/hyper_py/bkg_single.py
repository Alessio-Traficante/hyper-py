import warnings

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.modeling import fitting, models
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, TheilSenRegressor


def masked_background_single_sources(
    minimize_method, 
    image,
    header,
    xcen,
    ycen,
    nx, ny, 
    all_sources_xcen,
    all_sources_ycen,
    max_fwhm_extent,
    box_sizes,
    pol_orders_separate,
    suffix,
    source_id,
    config,
    logger,
    logger_file_only
):
    """
    Estimate and subtract a polynomial background from a masked cutout image of a single source.
    Loops over multiple box sizes and polynomial orders, and selects the background model
    that minimizes the residual scatter and residual mean.

    Parameters
    ----------
    cutout_masked : 2D array
        Cutout with source region masked (used for fitting background).
    cutout : 2D array
        Original (unmasked) cutout where background will be subtracted.
    cutout_header : fits.Header
        Header of the cutout (not used in this routine, but passed for I/O).
    x0, y0 : float
        Sub-pixel centroid of the main source.
    external_sources : list of (x, y)
        Additional source positions to be masked.
    max_fwhm_extent : float
        Radius (in pixels) used to mask each source.
    pol_orders_separate : list of int
        Polynomial orders to try (e.g. [0, 1, 2]).
    suffix : str
        String identifying the map or slice.
    source_id : int
        Index of the current source (for logging).
    config : ConfigReader
        Configuration reader with fitting options.
    logger_file_only : Logger
        Logger to print status and warnings to log file only.
    """

    logger_file_only.info("[INFO] Estimating background separately on masked cutout...")
    
    
    # ---------- SELECT WHICH FITTERS TO USE ----------
    bg_fitters = config.get("fit_options", "bg_fitters", ["least_squares"])
    huber_epsilons = config.get("fit_options", "huber_epsilons", [1.35])
    
    fitters = []
    if "least_squares" in bg_fitters:
        fitters.append(("LeastSquares", None, None))  # Will use np.linalg.lstsq
    
    if "huber" in bg_fitters:
        for eps in huber_epsilons:
            reg = HuberRegressor(fit_intercept=False, max_iter=100, epsilon=eps)
            fitters.append((f"Huber_{eps}", eps, reg))
    
    if "theilsen" in bg_fitters:
        reg = TheilSenRegressor(fit_intercept=False, max_subpopulation=1e4, random_state=42)
        fitters.append(("TheilSen", None, reg))
        
    
    # --- identify if trun on whole map (fix_min_box = 0) or not --- #
    fix_min_box = config.get("background", "fix_min_box", 3)     # minimum padding value (multiple of FWHM)

    
    # - Initialize parameters - #
    best_params = {}
    best_order = None
    best_min = np.inf

    cutout_reference_mask = None


    # ------------------ Loop over box sizes ------------------ #
    for box in box_sizes:
        if fix_min_box != 0:
            half_box = box // 2 -1
            xmin = max(0, int(np.min(xcen)) - half_box)
            xmax = min(nx, int(np.max(xcen)) + half_box + 1)
            ymin = max(0, int(np.min(ycen)) - half_box)
            ymax = min(ny, int(np.max(ycen)) + half_box + 1)
            
            cutout = image[ymin:ymax, xmin:xmax].copy()
        else:
            xmin = 0
            xmax = box_sizes[0]
            ymin = 0
            ymax = box_sizes[1]
            cutout = image
                
        
        if cutout.size == 0 or np.isnan(cutout).all():
            logger.warning("[WARNING] Empty or invalid cutout. Skipping.")
            continue

        # - first regularize mean background - #
        valid_cutout = ~np.isnan(cutout)        
        mean_valid_cutout, median_valid_cutout, std_valid_cutout = sigma_clipped_stats(cutout[valid_cutout], sigma=3.0, maxiters=5)
        cutout = cutout - median_valid_cutout
                

        yy, xx = np.indices(cutout.shape)
        x0 = xcen - xmin
        y0 = ycen - ymin
        
        
        # ---Initialize mask: True = valid pixel for background fitting --- #
        # mask_bg = np.ones_like(cutout, dtype=bool)
        mask_bg = np.isfinite(cutout)

        all_sources_to_mask = []
        all_sources_to_mask.append((x0, y0))
        
        main_sources = []
        main_sources.append((x0, y0))
        
        external_sources = []

        #--- Identify external sources inside box and add to main source - only if the background is not estimated over the whole map ---#
        if fix_min_box != 0:
            for i in range(len(all_sources_xcen)):                 
                 if (all_sources_xcen[i]-xmin != x0) and (all_sources_ycen[i]-ymin != y0):    
                     sx = all_sources_xcen[i]
                     sy = all_sources_ycen[i]
                    
                     if xmin <= sx <= xmax and ymin <= sy <= ymax:            
                         ex = sx - xmin
                         ey = sy - ymin
                         all_sources_to_mask.append((ex, ey))
                         external_sources.append((ex, ey))
                         
                
    
            # --- Mask all external sources using simple 2D Gaussian fitting --- #
            cut_local = cutout
            for xc, yc in external_sources:
                xc_int = int(round(xc))
                yc_int = int(round(yc))
                
                # Define small cutout around each source (e.g. max_fwhm_extent)
                fit_size = round(max_fwhm_extent/2.)  # half-size
                xfit_min = max(0, xc_int - fit_size)
                xfit_max = min(cut_local.shape[1], xc_int + fit_size + 1)
                yfit_min = max(0, yc_int - fit_size)
                yfit_max = min(cut_local.shape[0], yc_int + fit_size + 1)
                
                data_fit = cut_local[yfit_min:yfit_max, xfit_min:xfit_max]
                if data_fit.size < max_fwhm_extent*2 or np.all(np.isnan(data_fit)) or np.nanmax(data_fit) <= 0:
                    continue  # skip this source if empty or invalid
            
                yy_sub, xx_sub = np.mgrid[yfit_min:yfit_max, xfit_min:xfit_max]
            
                # Define and fit elliptical Gaussian
                g_init = models.Gaussian2D(
                    amplitude=np.nanmax(data_fit),
                    x_mean=xc,
                    y_mean=yc,
                    x_stddev=max_fwhm_extent,
                    y_stddev=max_fwhm_extent,
                    theta=0.0,
                    bounds={'x_stddev': (max_fwhm_extent/4., max_fwhm_extent*2), 'y_stddev': (max_fwhm_extent/4., max_fwhm_extent*2), 'theta': (-np.pi/2, np.pi/2)}
                )
            
                fit_p = fitting.LevMarLSQFitter()
                try:
                    g_fit = fit_p(g_init, xx_sub, yy_sub, data_fit)
                except Exception:
                    continue  # skip if fit fails
            
                # Evaluate fitted model over full local cutout
                yy_full, xx_full = np.indices(cut_local.shape)
                model_vals = g_fit(xx_full, yy_full)
            
                # Mask pixels above 1-FWHM threshold for external sopurces (≈ 0.1353 × peak)
                threshold = g_fit.amplitude.value * 0.1353 
                mask_bg[model_vals > threshold] = False
        
        
        ### --- From now on, all photometry and background estimation is done on cutout_masked from external sources --- ###
        # --- Apply external sources mask → set masked pixels to np.nan --- #
        cutout_masked = np.copy(cutout)
        cutout_masked[~mask_bg] = np.nan
        
        
        
        # # ---- interpolate NaNs at the edges of the maps --- #
        # --- Count NaNs in edge pixels ---
        edge_thickness = round(max_fwhm_extent)  # pixels to define the edge region
        # ny_masked, nx_masked = cutout_masked.shape
        
        edge_mask = np.zeros_like(cutout_masked, dtype=bool)
        edge_mask[:edge_thickness, :] = True  # top edge
        edge_mask[-edge_thickness:, :] = True  # bottom edge
        edge_mask[:, :edge_thickness] = True  # left edge
        edge_mask[:, -edge_thickness:] = True  # right edge
        
        n_edge_total = np.sum(edge_mask)
        n_edge_nan = np.sum(edge_mask & ~np.isfinite(cutout_masked))
        nan_fraction = n_edge_nan / n_edge_total
        
        # --- Only interpolate if edge NaNs < threshold ---
        nan_threshold = 0.3  # allow up to 30% NaNs in edge region
        
        if nan_fraction < nan_threshold:
            sigma = 2.0
            kernel = Gaussian2DKernel(x_stddev=sigma)
            interpolated_map = interpolate_replace_nans(cutout_masked, kernel)
            
            cutout_masked = interpolated_map
            mask_bg = np.isfinite(cutout)
        
            if np.any(~np.isfinite(interpolated_map)):
                logger_file_only.warning("⚠️ Some NaNs remain after interpolation!")
        else:
            logger_file_only.warning(f"⚠️ Too many NaNs at edges (fraction: {nan_fraction:.2f}) — interpolation skipped.")
    


        # --- Mask all main sources using simple 2D Gaussian fitting for background estimation purposes --- #
        mask_bg_all = np.copy(mask_bg)

        cut_local = cutout_masked
        for xc, yc in main_sources:
            xc_int = int(round(xc))
            yc_int = int(round(yc))
            
            # Define small cutout around each source (e.g. 2*max_fwhm_extent)
            fit_size = round(max_fwhm_extent/2.)  # half-size
            xfit_min = max(0, xc_int - fit_size)
            xfit_max = min(cut_local.shape[1], xc_int + fit_size + 1)
            yfit_min = max(0, yc_int - fit_size)
            yfit_max = min(cut_local.shape[0], yc_int + fit_size + 1)
            
            data_fit = cut_local[yfit_min:yfit_max, xfit_min:xfit_max]
            if data_fit.size < max_fwhm_extent*2 or np.all(np.isnan(data_fit)) or np.nanmax(data_fit) <= 0:
                continue  # skip this source if empty or invalid
        
            yy_sub, xx_sub = np.mgrid[yfit_min:yfit_max, xfit_min:xfit_max]
        
            # Define and fit elliptical Gaussian
            g_init = models.Gaussian2D(
                amplitude=np.nanmax(data_fit),
                x_mean=xc,
                y_mean=yc,
                x_stddev=max_fwhm_extent,
                y_stddev=max_fwhm_extent,
                theta=0.0,
                bounds={'x_stddev': (max_fwhm_extent/4., max_fwhm_extent*2), 'y_stddev': (max_fwhm_extent/4., max_fwhm_extent*2), 'theta': (-np.pi/2, np.pi/2)}
            )
        
            fit_p = fitting.LevMarLSQFitter()
            try:
                g_fit = fit_p(g_init, xx_sub, yy_sub, data_fit)
            except Exception:
                continue  # skip if fit fails
        
            # Evaluate fitted model over full local cutout
            yy_full, xx_full = np.indices(cut_local.shape)
            model_vals = g_fit(xx_full, yy_full)
        
            # Mask pixels above 2-FWHM threshold for main sources (≈ 0.1353 × peak)
            threshold = g_fit.amplitude.value * 0.1353 
            mask_bg_all[model_vals > threshold] = False

        # --- Apply main sources mask → set masked pixels to np.nan --- #
        cutout_masked_all = np.copy(cutout_masked)
        cutout_masked_all[~mask_bg_all] = np.nan        
               

        # - Estimate good pixels in cutout_masked_all - #
        y_bg, x_bg = np.where(mask_bg_all)
        z_bg = cutout_masked_all[y_bg, x_bg]
        
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        clipped = sigma_clip(z_bg)
        valid = ~clipped.mask
        
        x_valid = x_bg[valid]
        y_valid = y_bg[valid]
        z_valid = clipped.data[valid]
          

        # - identify the reference mask to estimate best_min from the first run - #
        if cutout_reference_mask is None:
            cutout_reference_mask = np.copy(cutout_masked_all)
            ref_ny, ref_nx = cutout_reference_mask.shape
            ref_box_size = box
            


        # ------------------ Loop over polynomial orders ------------------
        for order in pol_orders_separate:
            # Build design matrix
            terms = []
            param_names = []
            for dx in range(order + 1):
                for dy in range(order + 1 - dx):
                    terms.append((x_valid ** dx) * (y_valid ** dy))
                    param_names.append(f"c{dx}_{dy}")
        
            A = np.vstack(terms).T
            add_intercept = False
            if "c0_0" not in param_names:
                A = np.column_stack([np.ones_like(z_valid), A])
                param_names = ["c0_0"] + param_names
                add_intercept = True
        

            # --- run chosen fitter algorithm --- #        
            for method_name, eps, reg in fitters:
                try:
                    if reg is None:
                        # Least-squares case
                        coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
                    else:
                        reg.fit(A, z_valid)
                        coeffs = reg.coef_
                        if add_intercept:
                            coeffs[0] = reg.intercept_
                except Exception as e:
                    logger_file_only.warning(f"[FAIL] {method_name} fit failed (order={order}, ε={eps}): {e}")
                    continue
            
                # Rebuild coeff_dict
                coeff_dict = dict(zip(param_names, coeffs))
                
                               
                # --- Estimate best_min on common mask size for all runs --- #
                if fix_min_box != 0:
                    half_ref_box = ref_box_size // 2 -1
                    
                    x_start = max(0, int((x0)) - half_ref_box)
                    x_end   = min(nx, int(x0) + half_ref_box +1)
                    y_start = max(0, int((y0)) - half_ref_box)
                    y_end   = min(ny, int(y0) + half_ref_box +1)
                    
                    # --- Check bounds ---
                    if (x_start < 0 or y_start < 0):
                        x_start = 0
                        y_start = 0
                        logger_file_only.warning(f"[SKIP] Box size {box} cannot be cropped to match reference.")
                        continue  # this cutout is too small to extract the reference region               
                    if (x_end > cutout_masked_all.shape[1]):
                        x_end = cutout_masked_all.shape[1]
    
                    if (y_end > cutout_masked_all.shape[0]):
                        y_end = cutout_masked_all.shape[0]
                    cutout_eval = cutout_masked_all[y_start:y_end, x_start:x_end]
                else:
                    x_start = 0
                    x_end = box_sizes[0]
                    y_start = 0
                    y_end = box_sizes[1]
                    cutout_eval = cutout_masked_all


                # --- Crop current cutout to match reference size ---
                shared_valid_mask = np.isfinite(cutout_reference_mask) & np.isfinite(cutout_eval)
                                                        
                if np.count_nonzero(shared_valid_mask) < 10:
                    continue  # Not enough shared pixels
    
                                            
                yy_best_min, xx_best_min = np.where(shared_valid_mask)
                z_valid_best_min = cutout_eval[yy_best_min, xx_best_min]
                x_valid_best_min = xx_best_min
                y_valid_best_min = yy_best_min
                

                bg_model_local_valid_best_min = np.zeros_like(z_valid_best_min)
                for pname, val in coeff_dict.items():
                    dx, dy = map(int, pname[1:].split("_"))
                    bg_model_local_valid_best_min += val * (x_valid_best_min ** dx) * (y_valid_best_min ** dy)
                    
                # Then compute your residual and metric
                residual_valid_best_min = bg_model_local_valid_best_min - z_valid_best_min
                                
                
                mse = np.mean(residual_valid_best_min ** 2)
                norm = np.mean(z_valid ** 2) + 1e-12
                nmse = mse / norm
            
                k_params = len(coeff_dict)
                n_points = len(z_valid)
                bic = n_points * np.log(mse) + k_params * np.log(n_points)
                
                std_res = np.nanstd(residual_valid_best_min)
                std_res = std_res if std_res > 0 else 1e-10
                redchi = np.sum((residual_valid_best_min / std_res) ** 2) / (n_points - k_params)
            
                # Evaluate metric
                if minimize_method == "nmse":
                    my_min = nmse 
                elif minimize_method == "bic":
                    my_min = bic 
                elif minimize_method == "redchi":
                    my_min = redchi 
                else:
                    my_min = nmse  # fallback
                    
                    
           
                if my_min < best_min:
                    # Evaluate full model only once now
                    bg_model_full = np.zeros_like(xx, dtype=np.float64)
                    for pname, val in coeff_dict.items():
                        dx, dy = map(int, pname[1:].split("_"))
                        bg_model_full += val * (xx ** dx) * (yy ** dy)
                        
                    #- save cutout header -#
                    cutout_wcs = WCS(header).deepcopy()
                    cutout_wcs.wcs.crpix[0] -= xmin  # CRPIX1
                    cutout_wcs.wcs.crpix[1] -= ymin  # CRPIX2
                    cutout_header = cutout_wcs.to_header()
                    #- preserve other non-WCS cards (e.g. instrument, DATE-OBS) -#
                    cutout_header.update({k: header[k] for k in header if k not in cutout_header and k not in ['COMMENT', 'HISTORY']})
                                  
                    best_cutout = cutout
                    best_cutout_masked = cutout_masked
                    best_cutout_masked_full = cutout_masked_all
                    best_bg_model = bg_model_full
                    best_median_cutout = median_valid_cutout
                    best_header = cutout_header
                    best_mask_bg = mask_bg
                    best_x0 = x0
                    best_y0 = y0
                    best_xx = xx
                    best_yy = yy
                    best_xmin = xmin
                    best_xmax = xmax
                    best_ymin = ymin
                    best_ymax = ymax
                    best_params = coeff_dict
                    best_order = order
                    best_box_sizes = [box]
                    best_method = method_name
                    best_eps = eps
                    
                    best_min = my_min
            
                    

    # ------------------ Final background subtraction ------------------
    if best_order is None:
        # If no valid background was found, return unmodified cutout
        logger_file_only.warning("[WARNING] Background fit failed; returning original cutout.")        
        return cutout_masked, np.zeros_like(cutout), None, np.zeros_like(cutout), np.zeros_like(cutout), 0, 0, 0, 0, 0, 0, 0, 0, [box], 0, {}

    else:
        # Subtract background from the original cutout
        best_cutout -= best_bg_model
        best_cutout_masked -= best_bg_model
        best_bg_model = best_bg_model + best_median_cutout
        
        logger_file_only.info(f"[INFO] Background subtracted using order {best_order} polynomial.")
 
        return best_cutout_masked, best_cutout_masked_full, best_header, best_bg_model, best_mask_bg, best_x0, best_y0, best_xx, best_yy, best_xmin, best_xmax, best_ymin, best_ymax, best_box_sizes, best_order, best_params