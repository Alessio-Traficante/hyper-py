import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.wcs import WCS
from lmfit import minimize, Parameters
from photutils.aperture import CircularAperture
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from scipy.ndimage import gaussian_filter

from hyper_py.visualization import plot_fit_summary
from .bkg_multigauss import multigauss_background

from scipy.spatial.distance import pdist


def _moment_init(data, xc, yc, win_pix):
    """
    Estimate initial sigma_x, sigma_y and rotation angle from image 2nd-order
    moments inside a window of radius ~2.5*win_pix around (xc, yc).

    Returns (sx, sy, theta) where sx >= sy, or (None, None, None) on failure.
    theta is in radians: angle of the major (sx) axis from the x-axis,
    matching the sign convention of the 2D Gaussian model used here.
    """
    ny, nx = data.shape
    r = max(int(np.ceil(win_pix * 2.5)), 3)
    y_lo = max(0, int(yc) - r);  y_hi = min(ny, int(yc) + r + 1)
    x_lo = max(0, int(xc) - r);  x_hi = min(nx, int(xc) + r + 1)
    sub  = data[y_lo:y_hi, x_lo:x_hi]
    yg, xg = np.indices(sub.shape)
    xg = xg + x_lo;  yg = yg + y_lo
    valid = np.isfinite(sub) & (sub > 0)
    if valid.sum() < 6:
        return None, None, None
    w = np.maximum(sub[valid], 0.0)
    W = w.sum()
    if W <= 0:
        return None, None, None
    mx  = np.sum(w * xg[valid]) / W
    my  = np.sum(w * yg[valid]) / W
    Mxx = np.sum(w * (xg[valid] - mx)**2) / W
    Myy = np.sum(w * (yg[valid] - my)**2) / W
    Mxy = np.sum(w * (xg[valid] - mx) * (yg[valid] - my)) / W
    trace = Mxx + Myy
    disc  = max(0.0, (trace / 2)**2 - (Mxx * Myy - Mxy**2))
    lam1  = trace / 2 + np.sqrt(disc)        # larger eigenvalue → major axis
    lam2  = max(trace / 2 - np.sqrt(disc), 0.0)
    sx    = np.sqrt(max(lam1, 1e-3))
    sy    = np.sqrt(max(lam2, 1e-3))
    theta = 0.5 * np.arctan2(2.0 * Mxy, Mxx - Myy)
    theta = float(np.clip(theta, -np.pi / 2, np.pi / 2))
    return sx, sy, theta
import matplotlib.pyplot as plt


def fit_group_with_background(image, xcen, ycen, all_sources_xcen, all_sources_ycen, group_indices, map_struct, config, 
                              suffix, logger, logger_file_only, group_id, count_source_blended_indexes):
    
    header = map_struct['header']
    ny, nx = image.shape

    # --- Load config parameters ---
    dir_root = config.get("paths", "output")["dir_root"]
    beam_pix = map_struct['beam_dim']/map_struct['pix_dim']/2.3548      # beam sigma size in pixels    
    fwhm_beam_pix = map_struct['beam_dim']/map_struct['pix_dim']      # beam FWHM size in pixels    
    aper_inf = config.get("photometry", "aper_inf", 1.0) * beam_pix
    aper_sup = config.get("photometry", "aper_sup", 2.0) * beam_pix
    max_fwhm_extent = aper_sup * 2.3548  # twice major FWHM in pixels


    convert_mjy=config.get("units", "convert_mJy")

    
    fit_cfg = config.get("fit_options", {})
    minimize_method = config.get("fit_options", "min_method", "redchi")
    weight_choice = fit_cfg.get("weights", None)
    weight_power_snr = fit_cfg.get("power_snr", 1.0)

    use_l2 = fit_cfg.get("use_l2_regularization", False)
    lambda_l2 = fit_cfg.get("lambda_l2", 1e-3)

    fit_gauss_and_bg_together = config.get("background", "fit_gauss_and_bg_together", False)
    fix_min_box = config.get("background", "fix_min_box", 3)     # minimum padding value (multiple of FWHM)
    fix_max_box = config.get("background", "fix_max_box", 5)     # maximum padding value (multiple of FWHM)
    orders = config.get("background", "polynomial_orders", [0, 1, 2]) if fit_gauss_and_bg_together else [0]
    fit_separately = config.get("background", "fit_gauss_and_bg_separately", False)
    pol_orders_separate = config.get("background", "pol_orders_separate", [0])


    try:
        lambda_l2 = float(lambda_l2)
    except Exception as e:
        logger.warning(f"[WARNING] lambda_l2 is not a float: {lambda_l2} → {e}")
        lambda_l2 = 1e-3  # fallback
        
    
    # === Determine box size === #
    if fix_min_box == 0:
        # Use entire map size directly
        box_sizes = list((ny, nx))
    else:
        positions = np.column_stack([xcen, ycen])
        max_dist = np.max(pdist(positions)) if len(positions) > 1 else 0.0
        # box size is a multiplicative factor of the fwhm_beam_pix + maximum source size: max_fwhm_extent*2 + distance between common sources (max_dist)
        dynamic_min_box = int(np.ceil(fix_min_box*fwhm_beam_pix)*2 + max_fwhm_extent*2 + max_dist)
        dynamic_max_box = int(np.ceil(fix_max_box*fwhm_beam_pix)*2 + max_fwhm_extent*2 + max_dist)
        box_sizes = list(range(dynamic_min_box + 1, dynamic_max_box + 2, 2))  # ensure odd

 

    # - initialize map and header - #    
    header=map_struct['header']
    ny, nx = image.shape

    
    # - initialize params - #     
    best_result = None
    best_min  = np.inf
    best_cutout = None
    best_header = None
    best_slice = None
    best_order = None
    best_box = None
    
    
    
    #=== Estimate separated background masking also external sources  ===#
    if fit_separately:
        cutout_after_bg, cutout_full_with_bg, cutout_header, bg_model, mask_bg, xcen_cut, ycen_cut, xx, yy, xmin, xmax, ymin, ymax, box_sizes_after_bg, back_order, poly_params = multigauss_background(
            minimize_method=minimize_method, 
            image=image,
            header=header,
            xcen=xcen,
            ycen=ycen,
            nx=nx,
            ny=ny,
            all_sources_xcen=all_sources_xcen,
            all_sources_ycen=all_sources_ycen,
            aper_sup=aper_sup,
            max_fwhm_extent=max_fwhm_extent,
            box_sizes=box_sizes,
            pol_orders_separate=pol_orders_separate,
            suffix=suffix,
            group_id=group_id, 
            count_source_blended_indexes=count_source_blended_indexes,
            config=config,
            logger=logger,
            logger_file_only=logger_file_only
        )
        
        # - save original map without background - #
        cutout = np.copy(cutout_after_bg)
        cutout_masked = cutout_after_bg
        cutout_masked_full = cutout_full_with_bg
        box_sizes = box_sizes_after_bg
    else:
        bg_model = None


    
    # --- Run over the various box sizes (if fit_separately = True this is the best size identified in the background fit) --- #
    for box in box_sizes:
        
        if not fit_separately:
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
                continue
                   
            #- save cutout header -#
            cutout_wcs = WCS(header).deepcopy()
            cutout_wcs.wcs.crpix[0] -= xmin  # CRPIX1
            cutout_wcs.wcs.crpix[1] -= ymin  # CRPIX2
            cutout_header = cutout_wcs.to_header()
            #- preserve other non-WCS cards (e.g. instrument, DATE-OBS) -#
            cutout_header.update({k: header[k] for k in header if k not in cutout_header and k not in ['COMMENT', 'HISTORY']})
    
            yy, xx = np.indices(cutout.shape)
    
            
            #--- estimate cutout rms and weighting scheme ---#         
            xcen_cut = xcen - xmin
            ycen_cut = ycen - ymin
                    

            
            #--- Identify external sources inside box ---#
            mask = np.ones_like(cutout, dtype=bool)  # True = valid, False = masked
            external_sources = []
            for i in range(len(all_sources_xcen)):
                if i in group_indices:
                    continue  # skip sources belonging to current group
                sx = all_sources_xcen[i]
                sy = all_sources_ycen[i]
                
                if xmin <= sx <= xmax and ymin <= sy <= ymax and fix_min_box != 0:
                    ex = sx - xmin
                    ey = sy - ymin
                    external_sources.append((ex, ey))  # local cutout coords
        
                    # Define a bounding box around the source, clipped to cutout size
                    masking_radius = max_fwhm_extent/2.   # radius
                    masking_radius_pix=np.round(masking_radius) 
                            
                    xmin_box = max(0, int(ex - masking_radius_pix))
                    xmax_box = min(nx, int(ex + masking_radius_pix + 1))
                    ymin_box = max(0, int(ey - masking_radius_pix))
                    ymax_box = min(ny, int(ey + masking_radius_pix + 1))
                    
                    # Create coordinate grid for the local region
                    mask[ymin_box:ymax_box, xmin_box:xmax_box] = False 
                
                
                
            #--- Apply external sources mask → set masked pixels to np.nan ---#
            cutout_masked = np.copy(cutout)
            mask_bg = np.ones_like(cutout_masked, dtype=bool)
            mask_bg[~np.isfinite(cutout_masked)] = False        
            mask_bg[~mask] = False  # mask external sources etc.
                
            ### --- From now on, all photometry and background estimation is done on cutout_masked from external sources --- ###
            cutout_masked[~mask_bg] = np.nan
            cutout_masked_full = cutout_masked
         
                   
               
        # Mask NaNs and Infs before computing stats
        valid = np.isfinite(cutout_masked)        
        mean_bg, median_bg, std_bg = sigma_clipped_stats(cutout_masked[valid], sigma=3.0, maxiters=10)
        
        # Create rms map and propagate NaNs
        cutout_rms = np.full_like(cutout_masked, std_bg)
        cutout_rms[~valid] = np.nan 
        

        weights = None
        if weight_choice == "inverse_rms":
            weights = 1.0 / (cutout_rms + np.abs(mean_bg) + 1e-30)
        elif weight_choice == "snr":
            # SNR = pixel value / noise level (std of background pixels)
            weights = np.maximum(cutout_masked / std_bg, 0.0)
        elif weight_choice == "power_snr":
            snr_clipped = np.maximum(cutout_masked / std_bg, 0.0)
            weights = snr_clipped ** weight_power_snr
        elif weight_choice == "spatial":
            # Gaussian kernel centred on each source in the group (union of
            # per-source kernels via max). Best for weak/blended sources.
            spatial_sigma = beam_pix * fit_cfg.get("spatial_weight_sigma", 1.5)
            weights = np.zeros_like(cutout_masked, dtype=float)
            for xc_s, yc_s in zip(xcen_cut, ycen_cut):
                dist2 = (xx - xc_s) ** 2 + (yy - yc_s) ** 2
                weights = np.maximum(weights,
                                     np.exp(-0.5 * dist2 / spatial_sigma ** 2))
            weights[~valid] = 0.0
        elif weight_choice == "map":
            weights = cutout_masked
        elif weight_choice == "mask":
            mask_stats = ~SigmaClip(sigma=3.0)(cutout_masked).mask
            weights = mask_stats.astype(float)
                       
                
        for order in orders:
            try:
                vary = config.get("fit_options", "vary", True)
                params = Parameters()
                src_sigma_list = []  # per-source geometric-mean sigma from moments

                # --- Add Gaussian components ---
                for i, (xc, yc) in enumerate(zip(xcen_cut, ycen_cut)):

                    # --- Local peak near (xc, yc) in cutout_masked ---
                    prefix = f"g{i}_"
                    yc_i = int(round(yc))
                    xc_i = int(round(xc))
                    y_lo = max(0, yc_i - 1)
                    y_hi = min(cutout_masked.shape[0], yc_i + 2)
                    x_lo = max(0, xc_i - 1)
                    x_hi = min(cutout_masked.shape[1], xc_i + 2)
                    window = cutout_masked[y_lo:y_hi, x_lo:x_hi]
                    finite_in_window = window[np.isfinite(window)]
                    if len(finite_in_window) > 0:
                        local_peak = float(np.max(finite_in_window))
                    else:
                        # Fallback: global finite maximum of the cutout
                        finite_all = cutout_masked[np.isfinite(cutout_masked)]
                        local_peak = float(np.max(finite_all)) if len(finite_all) > 0 else 1.0
                    
                    # - peak in cutout masked is well-defined after background subtraction (fit_separately = True) - #
                    if fit_separately:
                        params.add(f"{prefix}amplitude", value=local_peak, min=0.4*local_peak, max=1.3*local_peak)
                    else:
                        params.add(f"{prefix}amplitude", value=local_peak, min=0.2*local_peak, max=1.5*local_peak)
                        
                    if vary == True:
                        params.add(f"{prefix}x0", value=xc, min=xc - 1, max=xc + 1)
                        params.add(f"{prefix}y0", value=yc, min=yc - 1, max=yc + 1)
                       
                    if vary == False:
                        params.add(f"{prefix}x0", value=xc, vary=False)
                        params.add(f"{prefix}y0", value=yc, vary=False) 

                    # --- Moment-based initialization: breaks the sx=sy degeneracy ---
                    sx_m, sy_m, th_m = _moment_init(cutout_masked, xc, yc, beam_pix)
                    if sx_m is not None:
                        sx_max_i = max(aper_sup, 1.2 * sx_m)
                        sy_max_i = max(aper_sup, 1.2 * sy_m)
                        sx_init_i = float(np.clip(sx_m, aper_inf, sx_max_i))
                        sy_init_i = float(np.clip(sy_m, aper_inf, sy_max_i))
                        th_init_i = th_m
                    else:
                        mid = (aper_inf + aper_sup) / 2.0
                        sx_init_i, sy_init_i = mid * 1.1, mid * 0.9
                        sx_max_i, sy_max_i   = aper_sup, aper_sup
                        th_init_i = 0.0
                    src_sigma_list.append(np.sqrt(sx_init_i * sy_init_i))
                    params.add(f"{prefix}sx", value=sx_init_i, min=aper_inf, max=sx_max_i)
                    params.add(f"{prefix}sy", value=sy_init_i, min=aper_inf, max=sy_max_i)
                    params.add(f"{prefix}theta", value=th_init_i, min=-np.pi/2, max=np.pi/2)
                   
                                        
 
                # --- Add full 2D polynomial background (including cross terms) ---
                if fit_gauss_and_bg_together:
                    max_order_all = max(orders)

                    for dx in range(max_order_all + 1):
                        for dy in range(max_order_all + 1 - dx):
                            pname = f"c{dx}_{dy}"
                            val = median_bg if (dx == 0 and dy == 0) else 1e-5
                            params.add(pname, value=val, vary=(dx + dy <= order))
                            

                def model_fn(p, x, y):
                    n_src = len(xcen_cut)
                    # Extract all per-source parameters into numpy arrays once per
                    # call.  This eliminates the Python loop from the optimizer's
                    # hot path and lets numpy evaluate all N Gaussians in one
                    # vectorised (N × M) operation — results are bit-identical to
                    # the original sequential loop.
                    A_v  = np.array([float(p[f"g{i}_amplitude"]) for i in range(n_src)])
                    x0_v = np.array([float(p[f"g{i}_x0"])        for i in range(n_src)])
                    y0_v = np.array([float(p[f"g{i}_y0"])         for i in range(n_src)])
                    sx_v = np.array([float(p[f"g{i}_sx"])         for i in range(n_src)])
                    sy_v = np.array([float(p[f"g{i}_sy"])         for i in range(n_src)])
                    th_v = np.array([float(p[f"g{i}_theta"])      for i in range(n_src)])

                    cos_th  = np.cos(th_v)
                    sin_th  = np.sin(th_v)
                    sin2_th = np.sin(2.0 * th_v)
                    a_v = cos_th**2  / (2.0 * sx_v**2) + sin_th**2  / (2.0 * sy_v**2)  # (N,)
                    b_v = sin2_th    / (4.0 * sx_v**2) - sin2_th    / (4.0 * sy_v**2)   # (N,)
                    c_v = sin_th**2  / (2.0 * sx_v**2) + cos_th**2  / (2.0 * sy_v**2)  # (N,)

                    # Ravel to 1-D for (N × M) broadcasting; reshape back at the end
                    # so the function works for both 1-D (during fitting) and 2-D
                    # (post-fit evaluation on np.indices grids) inputs.
                    orig_shape = x.shape
                    xf = x.ravel()
                    yf = y.ravel()
                    dx_mat = xf[None, :] - x0_v[:, None]   # (N, M)
                    dy_mat = yf[None, :] - y0_v[:, None]   # (N, M)
                    exponent = -(a_v[:, None] * dx_mat**2
                                 + 2.0 * b_v[:, None] * dx_mat * dy_mat
                                 + c_v[:, None] * dy_mat**2)
                    model = np.dot(A_v, np.exp(exponent))   # (N,)·(N,M) → (M,)

                    if fit_gauss_and_bg_together:
                        max_order_all = max(orders)
                        for dx in range(max_order_all + 1):
                            for dy in range(max_order_all + 1 - dx):
                                pname = f"c{dx}_{dy}"
                                model = model + float(p[pname]) * (xf ** dx) * (yf ** dy)

                    model = np.where(np.isfinite(model), model, 0.0)
                    return model.reshape(orig_shape)
                

                def residual(params, x, y, data, weights=None):
                    model = model_fn(params, x, y)
                    resid = np.asarray(model - data, dtype=np.float64)  # Ensure float array
                
                    if weights is not None:
                        resid *= weights
                
                    # Ensure residual is a clean float64 array
                    resid = np.asarray(resid, dtype=np.float64).ravel()
                    
                    if use_l2 and fit_gauss_and_bg_together:
                        penalty_values = [
                            float(params[p].value)
                            for p in params if p.startswith("c")
                            ]
    
                        if penalty_values:
                            penalty_resid = lambda_l2 * np.array(penalty_values, dtype=np.float64)
                            return np.concatenate([resid.ravel(), penalty_resid.ravel()])                          
                    return resid
 

                # --- Extract extra minimize kwargs from config ---
                fit_cfg = config.get("fit_options", {})
                minimize_keys = ["max_nfev", "xtol", "ftol", "gtol", "calc_covar", "loss", "f_scale"]
                minimize_kwargs = {}
                
                for key in minimize_keys:
                    val = fit_cfg.get(key)
                    if val is not None:
                        if key == "calc_covar":
                            minimize_kwargs[key] = bool(val)
                        elif key == "max_nfev":
                            minimize_kwargs[key] = int(val)
                        elif key in ["loss"]:  # must be string
                            minimize_kwargs[key] = str(val)
                        else:
                            minimize_kwargs[key] = float(val)
                            
                                           
                        
               # --- Restrict fitting pixels to union of per-source apertures ---
                # Each source gets its own circle: radius = moment-estimated sigma
                # (geometric mean of sx_init, sy_init) + fit_aperture_sigma * beam_pix
                # margin to capture the wings. Set fit_aperture_sigma to null to disable.
                valid_fit = np.isfinite(cutout_masked)
                fit_aperture_extra = fit_cfg.get("fit_aperture_sigma", 1.0)
                if fit_aperture_extra is not None:
                    ap_mask = np.zeros_like(cutout_masked, dtype=bool)
                    for (xc_s, yc_s), src_sig in zip(zip(xcen_cut, ycen_cut), src_sigma_list):
                        ap_r2_i = (src_sig + fit_aperture_extra * beam_pix) ** 2
                        ap_mask |= ((xx - xc_s) ** 2 + (yy - yc_s) ** 2) <= ap_r2_i
                    valid_fit &= ap_mask
                x_valid = xx.ravel()[valid_fit.ravel()]
                y_valid = yy.ravel()[valid_fit.ravel()]
                data_valid = cutout_masked.ravel()[valid_fit.ravel()]
                weights_valid = weights.ravel()[valid_fit.ravel()] if weights is not None else None
                if weights_valid is not None:
                    weights_valid = np.where(np.isfinite(weights_valid), weights_valid, 0.0)

                result = minimize(
                    residual,
                    params,
                    args=(x_valid.ravel(), y_valid.ravel(), data_valid),
                    kws={'weights': weights_valid},
                    method=fit_cfg.get("fit_method", "leastsq"),
                    **minimize_kwargs
                )     
 
                # --- Evaluate reduced chi**2, BIC and NMSE (Normalized Mean Squared Error) statistics --- #
                if result.success:
                    # Evaluate model on grid #
                    model_eval = model_fn(result.params, xx, yy)
                
                    # Compute normalized mean squared error only on valid pixels
                    valid_mask = np.isfinite(cutout_masked) & np.isfinite(model_eval)
                    residual = (model_eval - cutout_masked)[valid_mask]
                    mse = np.mean(residual**2)
                    
                    norm = np.mean(cutout_masked[valid_mask]**2) + 1e-12
                    nmse = mse / norm

                    # Compute redchi and BIC from data residuals only (excludes L2 penalty terms)
                    n_valid = np.sum(valid_mask)
                    n_varys = sum(1 for name in result.params if result.params[name].vary)
                    chi2 = np.sum((residual / std_bg)**2)  # dimensionless chi-squared
                    nfree = n_valid - n_varys
                    redchi = chi2 / nfree if nfree > 0 else np.nan
                    bic = chi2 + n_varys * np.log(n_valid) if n_valid > 0 else np.nan
                                        
                    if minimize_method == "redchi" : my_min = redchi
                    if minimize_method == "nmse"   : my_min = nmse
                    if minimize_method == "bic"    : my_min = bic
                    logger_file_only.info(f"[SUCCESS] Fit (box={cutout_masked.shape[1], cutout_masked.shape[0]}, order={order}) → reduced chi² = {redchi:.5f}, NMSE = {nmse:.2e}, BIC = {bic:.2e}")
                else:
                    nmse = np.nan
                    redchi = np.nan
                    bic = np.nan
                    my_min = np.nan
                    logger_file_only.error(f"[FAILURE] Fit failed (box={cutout_masked.shape[1], cutout_masked.shape[0]}, order={order})")
                    

        
                if my_min < best_min:
                    best_result = result
                    best_nmse = nmse
                    best_redchi = redchi
                    best_bic = bic
                    if fit_separately:
                        best_order = back_order
                    else:
                        best_order = order    
                    best_cutout = cutout
                    best_cutout_masked_full = cutout_masked_full
                    best_header = cutout_header
                    
                    best_bg_model = np.where(np.isfinite(cutout_masked), bg_model if bg_model is not None else 0.0, np.nan)

                    best_slice = (slice(ymin, ymax), slice(xmin, xmax))
                    bg_mean = median_bg
                    best_box = (cutout_masked.shape[1], cutout_masked.shape[0])
                    best_min = my_min

            except Exception as e:
                logger.error(f"[ERROR] Fit failed (box={cutout_masked.shape[1], cutout_masked.shape[0]}, order={order}): {e}")
                continue



    if best_result is not None:
        fit_status = 1  # 1 if True, 0 if False

        yy, xx = np.indices(best_cutout.shape)
        bg_vals = model_fn(best_result.params, xx, yy)
        gauss_vals = np.zeros_like(bg_vals)

        for i in range(len(xcen)):
            prefix = f"g{i}_"
            A = best_result.params[f"{prefix}amplitude"]
            x0 = best_result.params[f"{prefix}x0"]
            y0 = best_result.params[f"{prefix}y0"]
            sx = best_result.params[f"{prefix}sx"]
            sy = best_result.params[f"{prefix}sy"]
            th = best_result.params[f"{prefix}theta"]
            a = (np.cos(th)**2)/(2*sx**2) + (np.sin(th)**2)/(2*sy**2)
            b = np.sin(2*th)/(4*sx**2) - np.sin(2*th)/(4*sy**2)
            c = (np.sin(th)**2)/(2*sx**2) + (np.cos(th)**2)/(2*sy**2)
            gauss_vals += A * np.exp(- (a*(xx - x0)**2 + 2*b*(xx - x0)*(yy - y0) + c*(yy - y0)**2))

        bg_component = bg_vals - gauss_vals if fit_gauss_and_bg_together else np.zeros_like(bg_vals)
        if fit_gauss_and_bg_together:
            bg_mean = np.mean(bg_component)
        # else: bg_mean already set to median_bg of the best-fit iteration


        model_eval = model_fn(best_result.params, xx, yy)
        residual_map = best_cutout - model_eval
        

        # --- save best fit in fits format --- #
        try:
            fits_fitting = config.get("fits_output", "fits_fitting", False)
            fits_output_dir_fitting = os.path.join(dir_root, config.get("fits_output", "fits_output_dir_fitting", "fits/fitting"))  
        except:
            fits_fitting = False

        if fits_fitting:
            def save_fits(array, output_dir, label_name, extension_name, header=None):
                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)

                 # Create the FITS filename based on the label and extension type
                filename = f"{output_dir}/{label_name}_{extension_name}.fits"
        
                # Create a PrimaryHDU object and write the array into the FITS file
                hdu = fits.PrimaryHDU(data=array, header=header)
                if convert_mjy:
                    hdu.header['BUNIT'] = 'mJy/pixel'
                else: hdu.header['BUNIT'] = 'Jy/pixel'    
                hdul = fits.HDUList([hdu])
                
                # Write the FITS file
                hdul.writeto(filename, overwrite=True)
            
            save_fits(best_cutout, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "cutout", header=best_header)
            save_fits(best_cutout_masked_full, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "cutout_masked_full", header=best_header)
            save_fits(model_eval, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "model", header=best_header)
            save_fits(residual_map, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "residual", header=best_header)

        # --- visualize best fit in png format --- #
        try:
            visualize = config.get("visualization", "visualize_fitting")
        except:
            visualize = False

        try:
            output_dir_vis = os.path.join(dir_root, config.get("visualization", "output_dir_fitting", "images/fitting"))
        except:
            output_dir_vis = "Images/Fitting"
            
        if visualize:
            logger_file_only.info("2D and 3D visualization of the Gaussian fits and residual ON")
                        
            plot_fit_summary(
                cutout=best_cutout,
                cutout_masked_full=best_cutout_masked_full,
                model=model_eval,
                residual=residual_map,
                output_dir=output_dir_vis,
                label_name=f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}" if group_id is not None else "group",
                box_size=best_box,
                poly_order=best_order,
                nmse=best_nmse,
                bg_subtracted=fit_separately
            )
            

            
        # --- Optionally save separated background model as FITS --- #
        try:
            fits_bg_separate = config.get("fits_output", "fits_bg_separate", False)
            fits_output_dir_bg_separate = os.path.join(dir_root, config.get("fits_output", "fits_output_dir_bg_separate", "fits/bg_separate"))
        except:
            fits_bg_separate = False


        if fits_bg_separate:
            # Ensure the output directory exists
            os.makedirs(fits_output_dir_bg_separate, exist_ok=True)
            
            label_name = f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}"
            filename = f"{fits_output_dir_bg_separate}/{label_name}_bg_masked3D.fits"
        
            # Create a PrimaryHDU object and write the array into the FITS file
            convert_mjy=config.get("units", "convert_mJy")

            hdu = fits.PrimaryHDU(data=bg_model, header=cutout_header)
            if convert_mjy:
                hdu.header['BUNIT'] = 'mJy/pixel'
            else: hdu.header['BUNIT'] = 'Jy/pixel'    
            hdu.writeto(filename, overwrite=True)


        # --- Visualize 3D separated background estimation in png format --- #
        try:
            visualize_bg = config.get("visualization", "visualize_bg_separate", False)
        except Exception:
            visualize_bg = False

        if visualize_bg:
            logger_file_only.info("[INFO] Plotting 3D background model from masked map subtraction...")  

            output_dir_vis = os.path.join(dir_root, config.get("visualization", "output_dir_bg_separate", "plots/bg_separate"))
            os.makedirs(output_dir_vis, exist_ok=True)
                    
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xx, yy, best_bg_model, cmap="viridis", linewidth=0, antialiased=True)

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
            
            
            

        return fit_status, best_result, model_fn, best_order, best_cutout, best_cutout_masked_full, best_slice, best_header, bg_mean, best_bg_model, best_box, best_nmse, best_redchi, best_bic

    else:
        # Ensure return is always complete
        return 0, None, None, None, cutout_masked, None, (None, None), None, None, None, None, None, None, None