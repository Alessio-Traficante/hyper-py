import numpy as np
from lmfit import minimize, Parameters
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import CircularAperture
from hyper_py.visualization import plot_fit_summary
import matplotlib.pyplot as plt
import os

from bkg_single import masked_background_single_sources



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
    beam_pix = map_struct['beam_dim']/map_struct['pix_dim']/2.3548      # beam sigma size in pixels    
    aper_inf = config.get("photometry", "aper_inf", 1.0) * beam_pix
    aper_sup = config.get("photometry", "aper_sup", 2.0) * beam_pix
    
    convert_mjy=config.get("units", "convert_mJy")

    fit_cfg = config.get("fit_options", {})
    weight_choice = fit_cfg.get("weights", None)
    weight_power_snr = fit_cfg.get("power_snr", 1.0)

    fix_min_box = config.get("background", "fix_min_box", 15)     # padding value
    fix_max_box = config.get("background", "fix_max_box", 25)     # range above padding
    fix_box = config.get("background", "fix_box", False)          # use fixed size instead?
    
    no_background = config.get("background", "no_background", False)
    fit_separately = config.get("background", "fit_gauss_and_bg_separately", False)
    orders = config.get("background", "polynomial_orders", [0, 1, 2]) if not no_background else [0]
    pol_orders_separate = config.get("background", "pol_orders_separate", [1])  # only if fit_separately

    use_l2 = fit_cfg.get("use_l2_regularization", False)
    lambda_l2 = fit_cfg.get("lambda_l2", 1e-3)
    
    try:
        lambda_l2 = float(lambda_l2)
    except Exception as e:
        logger.warning(f"[WARNING] lambda_l2 is not a float: {lambda_l2} → {e}")
        lambda_l2 = 1e-3  # fallback


    # === Determine box size ===
    header=map_struct['header']
    ny, nx = image.shape

    if fix_box:
        box_sizes = [fix_min_box]
    else:
        max_fwhm_extent = aper_sup * 2.3548  # twice major FWHM in pixels
        dynamic_min_box = int(np.ceil(max_fwhm_extent) + fix_min_box)
        dynamic_max_box = dynamic_min_box + fix_max_box
        box_sizes = list(range(dynamic_min_box + 1, dynamic_max_box + 2, 2))  # ensure odd

    best_result = None
    min_nmse = np.inf
    best_cutout = None
    best_header = None
    best_slice = None
    best_order = None
    bg_mean = 0.0
    best_box = None


    for box in box_sizes:
        half_box = box // 2 - 1
        xmin = max(0, int(np.min(xcen)) - half_box)
        xmax = min(nx, int(np.max(xcen)) + half_box + 1)
        ymin = max(0, int(np.min(ycen)) - half_box)
        ymax = min(ny, int(np.max(ycen)) + half_box + 1)

        cutout = image[ymin:ymax, xmin:xmax].copy()
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
        external_sources = []
        for i in range(len(all_sources_xcen)):
            if i == source_id:
                continue  # skip sources belonging to current group
            sx = all_sources_xcen[i]
            sy = all_sources_ycen[i]
            if xmin <= sx <= xmax and ymin <= sy <= ymax:
                external_sources.append((sx - xmin, sy - ymin))  # local cutout coords
    
    
        # #--- Create mask ---#
        mask = np.ones_like(cutout, dtype=bool)  # True = valid, False = masked
        masking_radius = max_fwhm_extent #config.get("background", "external_mask_radius", 1.5) * beam_pix  # or set a fixed value
    
        for ex, ey in external_sources:
            aperture = CircularAperture((ex, ey), r=masking_radius)
            mask_data = aperture.to_mask(method="center").to_image(cutout.shape)
            if mask_data is not None:
                mask[mask_data > 0] = False  # Mask external source region
    
        #--- Apply mask → set masked pixels to np.nan ---#
        cutout_masked = np.copy(cutout)
        mask_bg = np.ones_like(cutout_masked, dtype=bool)
        mask_bg[np.isnan(cutout_masked)] = False
        mask_bg[~mask] = False  # mask external sources etc.
                

        # --- Background estimation on masked cutout (optional) --- #
        cutout_ref = np.copy(cutout)
        if fit_separately:
            cutout_after_bg, bg_model, poly_params = masked_background_single_sources(
                cutout_masked,
                cutout_ref, 
                cutout_header,
                x0,
                y0,
                external_sources,
                max_fwhm_extent,
                pol_orders_separate,
                suffix,
                source_id,
                config,
                logger_file_only
            )
            
            #--- Apply mask → set masked pixels to np.nan ---#
            cutout_masked = np.copy(cutout_after_bg)
            mask_bg = np.ones_like(cutout_masked, dtype=bool)
            mask_bg[np.isnan(cutout_masked)] = False
            mask_bg[~mask] = False  # mask external sources etc.

        # - save masked from nearby sources cutout - #
        cutout = cutout_masked
            
            
        # --- Fit single 2D elliptical Gaussian (+ background) ---
        # Mask NaNs before computing stats
        valid = ~np.isnan(cutout)        
        mean_bg, median_bg, std_bg = sigma_clipped_stats(cutout[valid], sigma=3.0, maxiters=5)
        
        # Create RMS map and propagate NaNs
        cutout_rms = np.full_like(cutout, std_bg)
        cutout_rms[~valid] = np.nan  
        
        weights = None
        if weight_choice == "inverse_rms":
            weights = 1.0 / (cutout_rms + mean_bg)
        elif weight_choice == "snr":
            weights = (cutout / (cutout_rms + mean_bg))
        elif weight_choice == "power_snr":
            weights = ((cutout / (cutout_rms + mean_bg)))**weight_power_snr
        elif weight_choice == "map":
            weights = cutout
        elif weight_choice == "mask":
            mask_stats = ~SigmaClip(sigma=3.0)(cutout).mask
            weights = mask_stats.astype(float)

        for order in orders:
            try:
                vary = config.get("fit_options", "vary", True)
                
                params = Parameters()
                local_peak = np.nanmax(cutout[int(y0)-1:int(y0)+1, int(x0)-1:int(x0)+1])
                params.add("g_amplitude", value=local_peak, min=0.8*local_peak, max=1.2*local_peak)


                if vary == True:
                    params.add("g_centerx", value=x0, min=x0 - 1, max=x0 + 1)
                    params.add("g_centery", value=y0, min=y0 - 1, max=y0 + 1)

                if vary == False:
                    params.add("g_centerx", value=x0, vary=False)
                    params.add("g_centery", value=y0, vary=False)

                params.add("g_sigmax", value=aper_inf, min=aper_inf, max=aper_sup)
                params.add("g_sigmay", value=aper_sup, min=aper_inf, max=aper_sup)
                params.add("g_theta", value=0.0, min=-np.pi/2, max=np.pi/2)


                # --- Add full 2D polynomial background (including cross terms) ---
                if not no_background:
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

                    if not no_background:
                        for dx in range(order + 1):
                            for dy in range(order + 1 - dx):
                                pname = f"c{dx}_{dy}"
                                model += p[pname].value * (x ** dx) * (y ** dy)
                    # Final check
                    model = np.where(np.isfinite(model), model, 0.0)
                    return model


                def residual(p, x, y, data, weights=None):
                    model = model_fn(p, x, y)
                    resid = (model - data).ravel().astype(np.float64)
                    
                    if weights is not None:
                        resid *= weights

                    if use_l2 and not no_background:
                        penalty_values = [
                            float(p[name].value)
                            for name in p if name.startswith("c")
                            ]   
    
                        if penalty_values:
                            penalty_resid = lambda_l2 * np.array(penalty_values, dtype=np.float64)
                            return np.concatenate([resid.ravel(), penalty_resid.ravel()])
                          
                    return resid


                fit_cfg = config.get("fit_options", {})
                minimize_keys = [ "max_nfev", "xtol", "ftol", "gtol", "calc_covar"]
                minimize_kwargs = {}
                
                for key in minimize_keys:
                    val = fit_cfg.get(key)
                    if val is not None:
                        if key == "calc_covar":
                            minimize_kwargs[key] = bool(val)
                        elif key == "max_nfev":
                            minimize_kwargs[key] = int(val)
                        else:
                            minimize_kwargs[key] = float(val) 
                            
           
                # --- Call minimize with dynamic kwargs ONLY across good pixels (masked sources within each box) --- # 
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
                    method=fit_cfg.get("fit_method", "leastsq"),
                    **minimize_kwargs
                )     
      
                             
                # --- Evaluate reduced chi**2 and NMSE (Normalized Mean Squared Error) ---
                if result.success:
                    redchi = result.redchi

                    # Evaluate model on grid
                    model_eval = model_fn(result.params, xx, yy)
                
                    # Compute normalized mean squared error
                    numerator = np.sum((model_eval - cutout)**2)
                    denominator = np.sum(cutout**2) + 1e-12  # avoid division by zero
                    nmse = numerator / denominator
                
                    logger_file_only.info(f"[SUCCESS] Fit (box={cutout.shape[1], cutout.shape[0]}, order={order}) → reduced chi² = {result.redchi:.5f}, NMSE = {nmse:.2e}")
                else:
                    logger_file_only.error(f"[FAILURE] Fit failed (box={cutout.shape[1], cutout.shape[0]}, order={order})")
                    nmse = np.nan
        
                if nmse < min_nmse:
                    best_result = result
                    best_nmse = nmse
                    best_order = order
                    best_cutout = cutout
                    best_header = cutout_header
                    best_slice = (slice(ymin, ymax), slice(xmin, xmax))
                    bg_mean = median_bg
                    best_box = (cutout.shape[1], cutout.shape[0])
                    min_nmse = nmse

            except Exception as e:
                logger.error(f"[ERROR] Fit failed (box={cutout.shape[1], cutout.shape[0]}, order={order}): {e}")
                continue
            
        
            
    if best_result is not None:
        fit_status = 1  # 1 if True, 0 if False
        
        yy, xx = np.indices(best_cutout.shape)

        model_eval = model_fn(best_result.params, xx, yy)
        residual_map = best_cutout - model_eval
        

        # --- visualize best fit in png format --- #
        try:
            visualize = config.get("visualization", "visualize_fitting")
        except:
            visualize = False

        try:
            dir_comm =  config.get("paths", "dir_comm")
            output_dir_vis = dir_comm + config.get("visualization", "output_dir_fitting")
        except:
            output_dir_vis = "Images/Fitting"

        if visualize:
            logger_file_only.info("2D and 3D visualization of the Gaussian fits and residual ON")
            plot_fit_summary(
                cutout=best_cutout,
                model=model_eval,
                residual=residual_map,
                output_dir=output_dir_vis,
                label_name=f"HYPER_MAP_{suffix}_ID_{source_id+1}" if source_id is not None else "source",
                box_size=best_box,
                poly_order=best_order,
                nmse=best_nmse
           )
       
        
        # --- save best fit in fits format --- #
        try:
            fits_fitting = config.get("fits_output", "fits_fitting", False)
            dir_comm =  config.get("paths", "dir_comm")
            fits_output_dir_fitting = dir_comm + config.get("fits_output", "fits_output_dir_fitting", "Fits/Fitting")  
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
            save_fits(model_eval, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "model", header = best_header)
            save_fits(residual_map, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "residual", header = best_header)


      
        
        return fit_status, best_result, model_fn, best_order, best_cutout, best_slice, bg_mean, best_nmse
    else:
        return 0, None, None, None, None, None, None, None