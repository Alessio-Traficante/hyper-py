import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from lmfit import minimize, Parameters
from photutils.aperture import CircularAperture

from hyper_py.visualization import plot_fit_summary
from .bkg_single import masked_background_single_sources

def fit_isolated_gaussian(image, xcen, ycen, all_sources_xcen, all_sources_ycen, source_id, map_struct, suffix, config, logger, logger_file_only):
    """
    Fit a single 2D elliptical Gaussian + polynomial background to an isolated source.

    Parameters
    ----------
    image : 2D numpy array
        The full input map.
    xcen, ycen : float
        Pixel coordinates of the source center.
    config : HyperConfig object
        Configuration settings from YAML file.

    Returns
    -------
    result : lmfit MinimizerResult
        Best-fit parameters.
    best_order : int
        Polynomial order used for background.
    cutout : 2D numpy array
        Local image patch around the source.
    cutout_slice : tuple of slices
        Slices used to extract the cutout.
    bg_mean : float
        Mean value of the fitted background over the cutout.
    """
    
        
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

    fix_min_box = config.get("background", "fix_min_box", 3)     # minimum padding value (multiple of FWHM)
    fix_max_box = config.get("background", "fix_max_box", 5)     # maximum padding value (multiple of FWHM)

    fit_gauss_and_bg_together = config.get("background", "fit_gauss_and_bg_together", False)
    fit_separately = config.get("background", "fit_gauss_and_bg_separately", False)
    orders = config.get("background", "polynomial_orders", [0, 1, 2]) if fit_gauss_and_bg_together else [0]
    pol_orders_separate = config.get("background", "pol_orders_separate", [0])  # only if fit_separately


    use_l2 = fit_cfg.get("use_l2_regularization", False)
    lambda_l2 = fit_cfg.get("lambda_l2", 1e-3)
    
    try:
        lambda_l2 = float(lambda_l2)
    except Exception as e:
        logger.warning(f"[WARNING] lambda_l2 is not a float: {lambda_l2} → {e}")
        lambda_l2 = 1e-3  # fallback


    # - initialize map and header - #    
    header=map_struct['header']
    ny, nx = image.shape


    # === Determine box size ===
    if fix_min_box == 0:
        # Use entire map size directly
        box_sizes = list((ny, nx))
    else:
        # Standard logic for square box sizes (in pixels)
        dynamic_min_box = int(np.ceil(fix_min_box * fwhm_beam_pix) * 2 + max_fwhm_extent * 2)
        dynamic_max_box = int(np.ceil(fix_max_box * fwhm_beam_pix) * 2 + max_fwhm_extent * 2)
        box_sizes = list(range(dynamic_min_box + 1, dynamic_max_box + 2, 2))  # ensure odd

    

    # - initialize params - #
    best_result = None
    best_min = np.inf
    best_cutout = None
    best_header = None
    best_slice = None
    best_order = None
    bg_mean = 0.0
    best_box = None
    

    # --- Background estimation on cutout masked (optional) --- #
    # cutout_ref = np.copy(cutout)
    if fit_separately:
        cutout_after_bg, cutout_full_with_bg, cutout_header, bg_model, mask_bg, x0, y0, xx, yy, xmin, xmax, ymin, ymax, box_sizes_after_bg, back_order, poly_params = masked_background_single_sources(
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
                logger.warning("[WARNING] Empty or invalid cutout. Skipping.")
                continue
          
            #- save cutout header -#
            cutout_wcs = WCS(header).deepcopy()
            cutout_wcs.wcs.crpix[0] -= xmin  # CRPIX1
            cutout_wcs.wcs.crpix[1] -= ymin  # CRPIX2
            cutout_header = cutout_wcs.to_header()
            #- preserve other non-WCS cards (e.g. instrument, DATE-OBS) -#
            cutout_header.update({k: header[k] for k in header if k not in cutout_header and k not in ['COMMENT', 'HISTORY']})
                          
            yy, xx = np.indices(cutout.shape)
            x0 = xcen - xmin
            y0 = ycen - ymin
            
           
            #--- Identify external sources inside box ---#
            mask = np.ones_like(cutout, dtype=bool)  # True = valid, False = masked
            external_sources = []
            for i in range(len(all_sources_xcen)):
                if i == source_id:
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
            mask_bg[np.isnan(cutout_masked)] = False        
            mask_bg[~mask] = False  # mask external sources etc.
            
            ### --- From now on, all photometry and background estimation is done on cutout_masked from external sources --- ###
            cutout_masked[~mask_bg] = np.nan
            
            
        # --- Fit single 2D elliptical Gaussian (+ background) --- #
        # Mask NaNs before computing stats
        valid = ~np.isnan(cutout_masked)        
        mean_bg, median_bg, std_bg = sigma_clipped_stats(cutout_masked[valid], sigma=3.0, maxiters=10)
                
        
        # Create rms map and propagate NaNs
        cutout_rms = np.full_like(cutout_masked, std_bg)
        cutout_rms[~valid] = np.nan  
        
        weights = None
        if weight_choice == "inverse_rms":
            weights = 1.0 / (cutout_rms + mean_bg)
        elif weight_choice == "snr":
            weights = (cutout_masked / (cutout_rms + mean_bg))
        elif weight_choice == "power_snr":
            weights = ((cutout_masked / (cutout_rms + mean_bg)))**weight_power_snr
        elif weight_choice == "map":
            weights = cutout_masked
        elif weight_choice == "mask":
            mask_stats = ~SigmaClip(sigma=3.0)(cutout_masked).mask
            weights = mask_stats.astype(float)


        for order in orders:
            try:
                vary = config.get("fit_options", "vary", True)              
                params = Parameters()
                local_peak = np.nanmax(cutout_masked[int(y0)-1:int(y0)+1, int(x0)-1:int(x0)+1])
                
                # - peak in cutout masked is well-defined after background subtraction (fit_separately = True) - #
                if fit_separately:
                    params.add("g_amplitude", value=local_peak, min=0.8*local_peak, max=1.3*local_peak)
                else:
                    params.add("g_amplitude", value=local_peak, min=0.4*local_peak, max=1.5*local_peak)
                    
                if vary == True:
                    params.add("g_centerx", value=x0, min=x0 - 0.5, max=x0 + 0.5)
                    params.add("g_centery", value=y0, min=y0 - 0.5, max=y0 + 0.5)
                if vary == False:
                    params.add("g_centerx", value=x0, vary=False)
                    params.add("g_centery", value=y0, vary=False)
     

                params.add("g_sigmax", value=(aper_inf+aper_sup)/2., min=aper_inf, max=aper_sup)
                params.add("g_sigmay", value=(aper_inf+aper_sup)/2., min=aper_inf, max=aper_sup)
                
                params.add("g_theta", value=0.0, min=-np.pi/2, max=np.pi/2)


                # --- Add full 2D polynomial background (including cross terms) ---
                if fit_gauss_and_bg_together:
                    max_order_all = max(orders)

                    for dx in range(max_order_all + 1):
                        for dy in range(max_order_all + 1 - dx):
                            pname = f"c{dx}_{dy}"
                            val = median_bg if (dx == 0 and dy == 0) else 1e-5
                            params.add(pname, value=val, vary=(dx + dy <= order))


                def model_fn(p, x, y):
                    A = p["g_amplitude"]
                    x0 = p["g_centerx"]
                    y0 = p["g_centery"]
                    sx = p["g_sigmax"]
                    sy = p["g_sigmay"]
                    th = p["g_theta"]
                    a = (np.cos(th)**2)/(2*sx**2) + (np.sin(th)**2)/(2*sy**2)
                    b = -np.sin(2*th)/(4*sx**2) + np.sin(2*th)/(4*sy**2)
                    c = (np.sin(th)**2)/(2*sx**2) + (np.cos(th)**2)/(2*sy**2)
                    model = A * np.exp(- (a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))

                    if fit_gauss_and_bg_together:
                        max_order_all = max(orders)

                        for dx in range(max_order_all + 1):
                            for dy in range(max_order_all + 1 - dx):
                                pname = f"c{dx}_{dy}"
                                val = median_bg if (dx == 0 and dy == 0) else 1e-5
                                params.add(pname, value=val, vary=(dx + dy <= order))
                                
                    # Final check
                    model = np.where(np.isfinite(model), model, 0.0)
                    return model


                def residual(p, x, y, data, weights=None):
                    model = model_fn(p, x, y)
                    resid = (model - data).ravel().astype(np.float64)
                    
                    if weights is not None:
                        resid *= weights

                    if use_l2 and fit_gauss_and_bg_together:
                        penalty_values = [
                            float(p[name].value)
                            for name in p if name.startswith("c")
                            ]   
    
                        if penalty_values:
                            penalty_resid = lambda_l2 * np.array(penalty_values, dtype=np.float64)
                            return np.concatenate([resid.ravel(), penalty_resid.ravel()])
                          
                    return resid


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
                         
                            
           
                # --- Call minimize with dynamic kwargs ONLY across good pixels (masked sources from external sources within each box) --- # 
                valid = mask_bg.ravel()
                x_valid = xx.ravel()[valid]
                y_valid = yy.ravel()[valid]
                data_valid = cutout_masked.ravel()[valid]
                weights_valid = weights.ravel()[valid] if weights is not None else None   
                
                             
                result = minimize(
                    residual,
                    params,
                    args=(x_valid.ravel(), y_valid.ravel(), data_valid),
                    kws={'weights': weights_valid},
                    method=fit_cfg.get("fit_method", "least_squares"),
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
                    
                    redchi = result.redchi
                    bic = result.bic 
                                        
                    if minimize_method == "redchi" : my_min = redchi
                    if minimize_method == "nmse"   : my_min = nmse
                    if minimize_method == "bic"    : my_min = bic
                    logger_file_only.info(f"[SUCCESS] Fit (box={cutout.shape[1], cutout.shape[0]}, order={order}) → reduced chi² = {result.redchi:.5f}, NMSE = {nmse:.2e}, BIC = {bic:.2e}")
                else:
                    nmse = np.nan
                    redchi = np.nan
                    bic = np.nan
                    my_min = np.nan
                    logger_file_only.error(f"[FAILURE] Fit failed (box={cutout.shape[1], cutout.shape[0]}, order={order})")
        
                if my_min < best_min:
                    best_result = result
                    best_nmse = nmse
                    best_redchi = redchi
                    best_bic = bic
                    if fit_separately:
                        best_order = back_order
                    else:
                        best_order = order    
                    best_cutout = cutout_masked
                    best_cutout_masked_full = cutout_masked_full
                    best_header = cutout_header
                    
                    bg_model = np.where(np.isfinite(cutout_masked), bg_model, np.nan)
                    best_bg_model = bg_model
                    
                    best_slice = (slice(ymin, ymax), slice(xmin, xmax))
                    bg_mean = median_bg
                    best_box = (cutout_masked.shape[1], cutout_masked.shape[0])
                    best_min = my_min

                    
            except Exception as e:
                logger.error(f"[ERROR] Fit failed (box={cutout.shape[1], cutout.shape[0]}, order={order}): {e}")
                continue
            
                
    if best_result is not None:
        fit_status = 1  # 1 if True, 0 if False
        
        yy, xx = np.indices(best_cutout.shape)

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
            
            save_fits(best_cutout, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "cutout", header = best_header)
            save_fits(best_cutout_masked_full, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "cutout masked full", header = best_header)
            save_fits(model_eval, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "model", header = best_header)
            save_fits(residual_map, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "residual", header = best_header)
            

        # --- visualize best fit in png format --- #
        try:
            visualize = config.get("visualization", "visualize_fitting")
        except:
            visualize = False

        try:
            output_dir_vis = os.path.join(dir_root, config.get("visualization", "output_dir_fitting", "plots/fitting"))
        except:
            output_dir_vis = "Images/Fitting"

        if visualize:
            logger_file_only.info("2D and 3D visualization of the Gaussian fits and residual ON")
            plot_fit_summary(
                cutout=best_cutout,
                cutout_masked_full = best_cutout_masked_full,
                model=model_eval,
                residual=residual_map,
                output_dir=output_dir_vis,
                label_name=f"HYPER_MAP_{suffix}_ID_{source_id+1}" if source_id is not None else "source",
                box_size=best_box,
                poly_order=best_order,
                nmse=best_nmse
           )


        # --- Optionally save separated background model as FITS --- #
        try:
            fits_bg_separate = config.get("fits_output", "fits_bg_separate", False)
            fits_output_dir = os.path.join(dir_root, config.get("fits_output", "fits_output_dir_bg_separate", "fits/bg_separate"))
        except Exception:
            fits_bg_separate = False

        if fits_bg_separate:
            os.makedirs(fits_output_dir, exist_ok=True)
            label_name = f"HYPER_MAP_{suffix}_ID_{source_id + 1}"
            filename = f"{fits_output_dir}/{label_name}_bg_masked3D.fits"
            
            convert_mjy=config.get("units", "convert_mJy")

            hdu = fits.PrimaryHDU(data=best_bg_model, header=best_header)
            if convert_mjy:
                hdu.header['BUNIT'] = 'mJy/pixel'
            else: hdu.header['BUNIT'] = 'Jy/pixel'    
            hdu.writeto(filename, overwrite=True)
            
            
        # --- Optionally save separated background 3D visualization as png format --- #
        try:
            visualize_bg = config.get("visualization", "visualize_bg_separate", False)
            output_dir = os.path.join(dir_root, config.get("visualization", "output_dir_bg_separate", "plots/bg_separate"))
        except Exception:
            visualize_bg = False

        if visualize_bg:
            os.makedirs(output_dir, exist_ok=True)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(xx, yy, best_bg_model, cmap="viridis", linewidth=0, antialiased=True)
            ax.set_xlabel("X (pix)", fontsize=8, fontweight="bold")
            ax.set_ylabel("Y (pix)", fontsize=8, fontweight="bold")
            ax.set_zlabel("Flux (Jy)", fontsize=8, fontweight="bold")
            ax.set_title("Initial Background (Isolated)", fontsize=10, fontweight="bold")

            label_str = f"HYPER_MAP_{suffix}_ID_{source_id + 1}"
            outname = os.path.join(output_dir, f"{label_str}_bg_masked3D.png")
            plt.savefig(outname, dpi=300, bbox_inches="tight")
            plt.close()      

        
        return fit_status, best_result, model_fn, best_order, best_cutout, best_slice, bg_mean, best_bg_model, best_header, best_nmse, best_redchi, best_bic
    else:   
        return 0, None, None, None, cutout_masked, (None, None), None, None, None, None, None, None
