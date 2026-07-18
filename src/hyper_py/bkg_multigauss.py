import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from scipy.ndimage import find_objects, label


def multigauss_background(minimize_method, image, header, xcen, ycen, nx, ny, all_sources_xcen, all_sources_ycen, aper_sup, max_fwhm_extent, 
                          box_sizes, pol_orders_separate, suffix, group_id, count_source_blended_indexes=None, config=None, logger=None, logger_file_only=None):
    """
    Estimate polynomial background in masked cutout, looping over box sizes and polynomial orders.
    """


    # ---------- SELECT WHICH FITTERS TO USE ----------
    bg_fitters = config.get("fit_options", "bg_fitters", ["least_squares"])
    huber_epsilons = config.get("fit_options", "huber_epsilons", [1.35])

    # Initial sigma for the per-source masking Gaussian.
    # aper_sup is the *maximum* allowed sigma; using it as the initial value
    # leaves only ~1 sigma visible in the fit window, making LevMarLSQ very
    # slow.  Use beam_pix (= aper_sup / aper_sup_config) instead — it is the
    # natural starting size and lets the optimizer converge in far fewer iters.
    _aper_sup_config = config.get("photometry", "aper_sup", 2.0)
    mask_gauss_stddev = aper_sup / max(_aper_sup_config, 1e-6)  # = beam_pix
    
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


    for box in box_sizes:  
        if fix_min_box != 0:
            half_box = box // 2 -1
            xmin = max(0, int(np.min(xcen)) - half_box)
            xmax = min(nx, int(np.max(xcen)) + half_box + 1)
            ymin = max(0, int(np.min(ycen)) - half_box)
            ymax = min(ny, int(np.max(ycen)) + half_box + 1)
            
            cutout = np.array(image[ymin:ymax, xmin:xmax], dtype=np.float64)        
        else:
            xmin = 0
            xmax = box_sizes[0]
            ymin = 0
            ymax = box_sizes[1]
            cutout = np.array(image)

        if cutout.size == 0 or np.isnan(cutout).all():
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
        all_sources_to_mask.extend(zip(x0, y0))
        
        main_sources = []
        main_sources.extend(zip(x0, y0))
        
        external_sources = []
        
        #--- Identify external sources inside box and add to main source - only if the background is not estimated over the whole map ---#
        mask_bg = np.ones_like(cutout, dtype=bool)
        
        if fix_min_box != 0:                    
            # Convert reference Gaussians to a set of (x0, y0) pixel tuples
            x0_int = np.array(x0).astype(int)
            y0_int = np.array(y0).astype(int)
            reference_positions = set(zip(x0_int, y0_int))
            
            # Loop over all other sources
            for i in range(len(all_sources_xcen)):
                sx = all_sources_xcen[i]
                sy = all_sources_ycen[i]
            
                ex = int(sx - xmin)
                ey = int(sy - ymin)
            
                if (ex, ey) not in reference_positions:
                    if xmin <= sx <= xmax and ymin <= sy <= ymax:
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
                # Use beam_pix (mask_gauss_stddev) as the initial sigma so the
                # Gaussian spans several sigma in the fit window and LevMarLSQ
                # converges quickly.  Wide bounds still allow elongated sources.
                g_init = models.Gaussian2D(
                    amplitude=np.nanmax(data_fit),
                    x_mean=xc,
                    y_mean=yc,
                    x_stddev=mask_gauss_stddev,
                    y_stddev=mask_gauss_stddev,
                    theta=0.0,
                    bounds={'x_stddev': (aper_sup/4., aper_sup), 'y_stddev': (aper_sup/4., aper_sup), 'theta': (-np.pi/2, np.pi/2)}
                )
            
                fit_p = fitting.LevMarLSQFitter()
                try:
                    g_fit = fit_p(g_init, xx_sub, yy_sub, data_fit)
                except Exception:
                    continue  # skip if fit fails
            
                # Evaluate fitted model over full local cutout
                yy_full, xx_full = np.indices(cut_local.shape)
                model_vals = g_fit(xx_full, yy_full)
            
                # Guard: if amplitude is non-positive the threshold loop below would never
                # converge (threshold ≤ 0 → mask covers everything → threshold_factor
                # keeps doubling forever).  Skip masking for this source.
                if g_fit.amplitude.value <= 0 or np.nanmax(model_vals) <= 0:
                    continue

                # # Mask pixels above 2-FWHM threshold for external sources (≈ 0.1353 × peak) - or below it if too many NaNs
                threshold_factor = 0.1353
                max_fraction = 0.3
                max_extent_pix = max_fwhm_extent *2  # in pixels
                
                # Loop to find acceptable threshold
                while True:
                    threshold = g_fit.amplitude.value * threshold_factor
                    mask_above = model_vals > threshold
                    frac_above_thresh = np.sum(mask_above) / model_vals.size
                
                    if frac_above_thresh <= max_fraction:
                        # Check the maximum spatial extent of the mask
                        labeled, _ = label(mask_above)
                        slices = find_objects(labeled)
                
                        max_extent = 0
                        for slc in slices:
                            if slc is None:
                                continue
                            dy = slc[0].stop - slc[0].start
                            dx = slc[1].stop - slc[1].start
                            extent = np.sqrt(dx**2 + dy**2)
                            max_extent = max(max_extent, extent)
                
                        if max_extent <= max_extent_pix:
                            break  # both conditions passed
                        else:
                            # increase threshold to shrink extent
                            threshold_factor *= 2
                    else:
                        # too many pixels above threshold
                        threshold_factor *= 2

                    # Safety exit: threshold now exceeds peak → mask would be empty
                    if threshold_factor > 1.0:
                        threshold = np.nanmax(model_vals) + 1.0  # guarantees empty mask
                        break
                
                # Apply the final mask
                mask_bg[model_vals > threshold] = False           
                
        
        
        ### --- From now on, all photometry and background estimation is done on cutout_masked from external sources --- ###
        # --- Apply external sources mask → set masked pixels to np.nan --- #
        cutout_masked = np.copy(cutout)
        cutout_masked[~mask_bg] = np.nan
        
        
        
        # --- Interpolate NaN regions (masked external sources) for better background estimation ---
        # Count edge NaN fraction for diagnostic purposes only.
        edge_thickness = round(max_fwhm_extent)
        edge_mask = np.zeros_like(cutout_masked, dtype=bool)
        edge_mask[:edge_thickness, :] = True
        edge_mask[-edge_thickness:, :] = True
        edge_mask[:, :edge_thickness] = True
        edge_mask[:, -edge_thickness:] = True
        n_edge_total = np.sum(edge_mask)
        n_edge_nan = np.sum(edge_mask & ~np.isfinite(cutout_masked))
        nan_fraction = n_edge_nan / n_edge_total if n_edge_total > 0 else 0.0
        if nan_fraction > 0.0:
            logger_file_only.info(f"[INFO] Edge NaN fraction before interpolation: {nan_fraction:.2f}")

        # Always attempt interpolation with an adaptive growing kernel.
        sigma = 2.0
        max_sigma = 10.0
        success = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyUserWarning)

            while sigma <= max_sigma:
                kernel = Gaussian2DKernel(x_stddev=sigma)
                interpolated_map = interpolate_replace_nans(cutout_masked, kernel)

                if np.all(np.isfinite(interpolated_map)):
                    success = True
                    break
                else:
                    sigma += 1.0  # Increase kernel size and retry

        if success:
            cutout_masked = interpolated_map
            mask_bg = np.ones_like(cutout_masked, dtype=bool)
        else:
            logger_file_only.warning(f"⚠️ NaNs remain after interpolation with max kernel (sigma={max_sigma}); edge NaN fraction: {nan_fraction:.2f}")

        
        

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
                x_stddev=aper_sup,
                y_stddev=aper_sup,
                theta=0.0,
                bounds={'x_stddev': (aper_sup/4., aper_sup*2), 'y_stddev': (aper_sup/4., aper_sup*2), 'theta': (-np.pi/2, np.pi/2)}
            )
        
            fit_p = fitting.LevMarLSQFitter()
            try:
                g_fit = fit_p(g_init, xx_sub, yy_sub, data_fit)
            except Exception:
                continue  # skip if fit fails
        
            # Evaluate fitted model over full local cutout
            yy_full, xx_full = np.indices(cut_local.shape)
            model_vals = g_fit(xx_full, yy_full)

            # Guard: if amplitude is non-positive the threshold loop below would never
            # converge (threshold ≤ 0 → mask covers everything → threshold_factor
            # keeps doubling forever).  Skip masking for this source.
            if g_fit.amplitude.value <= 0 or np.nanmax(model_vals) <= 0:
                continue
        
            # # Mask pixels above 2-FWHM threshold for external sources (≈ 0.1353 × peak) - or below it if too many NaNs
            threshold_factor = 0.1353
            max_fraction = 0.3
            max_extent_pix = max_fwhm_extent *2  # in pixels
            
            # Loop to find acceptable threshold
            while True:
                threshold = g_fit.amplitude.value * threshold_factor
                mask_above = model_vals > threshold
                frac_above_thresh = np.sum(mask_above) / model_vals.size
            
                if frac_above_thresh <= max_fraction:
                    # Check the maximum spatial extent of the mask
                    labeled, _ = label(mask_above)
                    slices = find_objects(labeled)
            
                    max_extent = 0
                    for slc in slices:
                        if slc is None:
                            continue
                        dy = slc[0].stop - slc[0].start
                        dx = slc[1].stop - slc[1].start
                        extent = np.sqrt(dx**2 + dy**2)
                        max_extent = max(max_extent, extent)
            
                    if max_extent <= max_extent_pix:
                        break  # both conditions passed
                    else:
                        # increase threshold to shrink extent
                        threshold_factor *= 2
                else:
                    # too many pixels above threshold
                    threshold_factor *= 2

                # Safety exit: threshold now exceeds peak → mask would be empty
                if threshold_factor > 1.0:
                    threshold = np.nanmax(model_vals) + 1.0  # guarantees empty mask
                    break
            
            # Apply the final mask
            mask_bg_all[model_vals > threshold] = False
                        
                   

        # --- Apply main sources mask → set masked pixels to np.nan --- #
        cutout_masked_all = np.copy(cutout_masked)
        cutout_masked_all[~mask_bg_all] = np.nan
        
        
        
        # - Estimate good pixels for background estimation only in cutout_masked_all - #
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
                    half_ref_box_x = ref_nx // 2 -1 
                    half_ref_box_y = ref_ny // 2 -1
                    
                    x_start = max(0, round(np.mean(x0)) - half_ref_box_x) 
                    x_end   = x_start + ref_nx #max(ref_nx, round(np.mean(x0)) + half_ref_box_x +1) 
                    y_start = max(0, round(np.mean(y0)) - half_ref_box_y)  
                    y_end   = y_start + ref_ny #max(ref_ny, round(np.mean(y0)) + half_ref_box_y +1)
                            
                    if (x_end > cutout_masked_all.shape[1]):
                        x_start = x_start - (x_end - cutout_masked_all.shape[1])
                        x_end = cutout_masked_all.shape[1]

                    if (y_end > cutout_masked_all.shape[0]):
                        y_start = y_start - (y_end - cutout_masked_all.shape[0])
                        y_end = cutout_masked_all.shape[0]
                   
                    cutout_eval = cutout_masked_all[y_start:y_end, x_start:x_end]

                else:
                    xmin = 0
                    xmax = box_sizes[0]
                    ymin = 0
                    ymax = box_sizes[1]
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
        try:
            # xx, yy, cutout_masked, cutout_masked_all, x0, y0, mask_bg, xmin/xmax/ymin/ymax
            # are all defined if at least one box iteration passed the empty/NaN check.
            cutout_wcs = WCS(header).deepcopy()
            cutout_wcs.wcs.crpix[0] -= xmin
            cutout_wcs.wcs.crpix[1] -= ymin
            fb_header = cutout_wcs.to_header()
            fb_header.update({k: header[k] for k in header if k not in fb_header and k not in ['COMMENT', 'HISTORY']})
            return (cutout, cutout_masked_all, fb_header, np.zeros_like(cutout),
                    mask_bg, x0, y0, xx, yy, xmin, xmax, ymin, ymax, [box], 0, {})
        except NameError:
            # All box sizes were empty or all-NaN; fall back to the full image.
            fb_arr = np.array(image, dtype=np.float64)
            fb_yy, fb_xx = np.indices(fb_arr.shape)
            fb_x0 = np.atleast_1d(np.asarray(xcen, dtype=float))
            fb_y0 = np.atleast_1d(np.asarray(ycen, dtype=float))
            return (fb_arr, fb_arr, None, np.zeros_like(fb_arr), np.isfinite(fb_arr),
                    fb_x0, fb_y0, fb_xx, fb_yy, 0, nx, 0, ny, [box_sizes[0] if box_sizes else nx], 0, {})

    else:
        # Build the photometry cutout: original unmasked data minus fitted background.
        # This is a new array so there is no aliasing with best_cutout / cutout.
        cutout_bg_subtracted = best_cutout - best_bg_model
        best_cutout_masked -= best_bg_model
        best_bg_model = best_bg_model + best_median_cutout

        logger_file_only.info(f"[INFO] Background subtracted using order {best_order} polynomial.")
        # Return the original (unmasked) cutout minus background as position 0 so that
        # photometry is never done on interpolated pixels.
        return cutout_bg_subtracted, best_cutout_masked_full, best_header, best_bg_model, best_mask_bg, best_x0, best_y0, best_xx, best_yy, best_xmin, best_xmax, best_ymin, best_ymax, best_box_sizes, best_order, best_params