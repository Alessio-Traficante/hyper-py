import numpy as np
from lmfit import minimize, Parameters
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from scipy.spatial.distance import pdist
from hyper_py.visualization import plot_fit_summary
from bkg_multigauss import estimate_masked_background

from photutils.aperture import CircularAperture

import os


def fit_group_with_background(image, xcen, ycen, all_sources_xcen, all_sources_ycen, group_indices, map_struct, config, 
                              suffix, logger, logger_file_only, group_id, count_source_blended_indexes):
    
    header=map_struct['header']
    ny, nx = image.shape


    # --- Load config parameters ---
    beam_pix = map_struct['beam_dim']/map_struct['pix_dim']/2.3548      # beam sigma size in pixels    
    aper_inf = config.get("photometry", "aper_inf", 1.0) * beam_pix
    aper_sup = config.get("photometry", "aper_sup", 2.0) * beam_pix
    
    fit_cfg = config.get("fit_options", {})
    weight_choice = fit_cfg.get("weights", None)
    weight_power_snr = fit_cfg.get("power_snr", 1.0)

    use_l2 = fit_cfg.get("use_l2_regularization", False)
    lambda_l2 = fit_cfg.get("lambda_l2", 1e-3)

    no_background = config.get("background", "no_background", False)
    fix_min_box = config.get("background", "fix_min_box", 15)     # padding value
    fix_max_box = config.get("background", "fix_max_box", 25)     # range above padding
    fix_box = config.get("background", "fix_box", False)          # use fixed size instead?
    orders = config.get("background", "polynomial_orders", [0, 1, 2]) if not no_background else [0]
    fit_separately = config.get("background", "fit_gauss_and_bg_separately", False)
    pol_orders_separate = config.get("background", "pol_orders_separate", [0])


    try:
        lambda_l2 = float(lambda_l2)
    except Exception as e:
        logger.warning(f"[WARNING] lambda_l2 is not a float: {lambda_l2} → {e}")
        lambda_l2 = 1e-3  # fallback
    
    # --- Determine box_sizes ---
    if fix_box:
        box_sizes = [fix_min_box]
    else:
        max_fwhm_extent = aper_sup * 2.3548  # twice major FWHM in pixels
        positions = np.column_stack([xcen, ycen])
        max_dist = np.max(pdist(positions)) if len(positions) > 1 else 0.0
        dynamic_min_box = int(np.ceil(max_dist + max_fwhm_extent + fix_min_box))
        dynamic_max_box = dynamic_min_box + fix_max_box
        box_sizes = list(range(dynamic_min_box + 1, dynamic_max_box + 2, 2))  # ensure odd
        
    best_result = None
    min_nmse = np.inf
    best_cutout = None
    best_header = None
    best_slice = None
    best_order = None
    best_box = None

    for box in box_sizes:
        half_box = box // 2 - 1
        xmin = max(0, int(np.min(xcen)) - half_box)
        xmax = min(nx, int(np.max(xcen)) + half_box + 1)
        ymin = max(0, int(np.min(ycen)) - half_box)
        ymax = min(ny, int(np.max(ycen)) + half_box + 1)

        cutout = np.array(image[ymin:ymax, xmin:xmax], dtype=np.float64)        
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
        
        mean_bg, median_bg, std_bg = sigma_clipped_stats(cutout, sigma=3.0, maxiters=10)
        cutout_rms = np.full_like(cutout, std_bg)

        mask_weights = ~SigmaClip(sigma=3.0, maxiters=10)(cutout).mask
        weights = None
        if weight_choice == "inverse_rms":
            weights = 1.0 / (cutout_rms + mean_bg)
        elif weight_choice == "snr":
            weights = cutout / (cutout_rms + mean_bg)
        elif weight_choice == "power_snr":
            weights = ((cutout / (cutout_rms + mean_bg)))**weight_power_snr
        elif weight_choice == "map":
            weights = cutout
        elif weight_choice == "mask":
            weights = mask_weights.astype(float)
            
            
        
        #--- Identify external sources inside box ---#
        external_sources = []
        for i in range(len(all_sources_xcen)):
            if i in group_indices:
                continue  # skip sources belonging to current group
            sx = all_sources_xcen[i]
            sy = all_sources_ycen[i]
            if xmin <= sx <= xmax and ymin <= sy <= ymax:
                external_sources.append((sx - xmin, sy - ymin))  # local cutout coords
    
  
        #=== Estimate separated background masking also external sources  ===#
        if fit_separately:
            xcen_cut_bg = np.copy(xcen_cut)
            ycen_cut_bg = np.copy(ycen_cut)

            if len(external_sources) > 0:
                ex_arr, ey_arr = np.array(external_sources).T
                xcen_cut_bg = np.concatenate([xcen_cut_bg, ex_arr])
                ycen_cut_bg = np.concatenate([ycen_cut_bg, ey_arr])
            
            cutout_bs, bg_poly = estimate_masked_background(
                cutout=cutout,
                cutout_header=cutout_header,
                xcen_cut=xcen_cut_bg,
                ycen_cut=ycen_cut_bg,
                aper_sup=aper_sup,
                max_fwhm_extent=max_fwhm_extent,
                orders=pol_orders_separate,
                suffix=suffix,
                count_source_blended_indexes=count_source_blended_indexes,
                config=config,
                logger=logger,
                logger_file_only=logger_file_only
            )
            cutout = cutout_bs
        else:
            bg_poly = np.zeros_like(cutout)
    

        #--- Create mask of external sources ---#
        mask = np.ones_like(cutout, dtype=bool)  # True = valid, False = masked
        masking_radius = max_fwhm_extent #config.get("background", "external_mask_radius", 1.5) * beam_pix  # or set a fixed value
            
        for ex, ey in external_sources:
            aperture = CircularAperture((ex, ey), r=masking_radius)
            mask_data = aperture.to_mask(method="center").to_image(cutout.shape)
            if mask_data is not None:
                mask[mask_data > 0] = False  # Mask external source region
            
        #--- Apply mask → set masked pixels to np.nan ---#
        cutout_masked = np.copy(cutout)
        cutout_masked[~mask] = np.nan 
                

        for order in orders:
            try:
                vary = config.get("fit_options", "vary", True)
                params = Parameters()

                # --- Add Gaussian components ---
                for i, (xc, yc) in enumerate(zip(xcen_cut, ycen_cut)):

                    # --- Local peak near (xc, yc) in cutout ---
                    prefix = f"g{i}_"
                    # params.add(f"{prefix}amplitude", value=np.max(cutout), min=0., max=1.1*np.max(cutout))
                    params.add(f"{prefix}amplitude", value=np.max(cutout), min=0., max=1.1*cutout[int(round(xc)),int(round(yc))])

                    params.add(f"{prefix}x0", value=xc, vary=False) #min=xc-0.05, max=xc+0.05)
                    params.add(f"{prefix}y0", value=yc, vary=False) #, min=yc-0.05, max=yc+0.05)
 
                    if vary == True:
                        params.add(f"{prefix}x0", value=xc, min=xc - 1, max=xc + 1)
                        params.add(f"{prefix}y0", value=yc, min=yc - 1, max=yc + 1)
                       
                    if vary == False:
                        params.add(f"{prefix}x0", value=xc, vary=False)
                        params.add(f"{prefix}y0", value=yc, vary=False) 

                    avg_aper = (aper_inf + aper_sup)/2. 
                    params.add(f"{prefix}sx", value=avg_aper*1.2, min=aper_inf, max=aper_sup)
                    params.add(f"{prefix}sy", value=avg_aper*0.8, min=aper_inf, max=aper_sup)
                    params.add(f"{prefix}theta", value=0.0, min=-np.pi/2, max=np.pi/2)


                # --- Add full 2D polynomial background (including cross terms) ---
                if not no_background:
                    for deg_x in range(order + 1):
                        for deg_y in range(order + 1):
                            pname = f"c{deg_x}_{deg_y}"
                            val = median_bg if (deg_x == 0 and deg_y == 0) else 0.0
                            params.add(pname, value=val, vary=True)

                def model_fn(params, x, y):
                    model = np.zeros_like(x, dtype=float)
                    for i in range(len(xcen_cut)):
                        prefix = f"g{i}_"
                        A = params[f"{prefix}amplitude"]
                        x0 = params[f"{prefix}x0"]
                        y0 = params[f"{prefix}y0"]
                        sx = params[f"{prefix}sx"]
                        sy = params[f"{prefix}sy"]
                        th = params[f"{prefix}theta"]
                        a = (np.cos(th)**2)/(2*sx**2) + (np.sin(th)**2)/(2*sy**2)
                        b = -np.sin(2*th)/(4*sx**2) + np.sin(2*th)/(4*sy**2)
                        c = (np.sin(th)**2)/(2*sx**2) + (np.cos(th)**2)/(2*sy**2)
                        model += A * np.exp(- (a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))

                    if not no_background:
                        for deg_x in range(order + 1):
                            for deg_y in range(order + 1):
                                pname = f"c{deg_x}_{deg_y}"
                                if pname in params:
                                    model += params[pname].value * (x ** deg_x) * (y ** deg_y)
                    # Final check
                    model = np.where(np.isfinite(model), model, 0.0)
                    return model
                

                def residual(params, x, y, data, weights=None):
                    model = model_fn(params, x, y)
                    resid = np.asarray(model - data, dtype=np.float64)  # Ensure float array
                
                    if weights is not None:
                        resid *= weights
                
                    # Ensure residual is a clean float64 array
                    resid = np.asarray(resid, dtype=np.float64).ravel()
                    
                    if use_l2 and not no_background:
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
                
                        
               # --- Call minimize with dynamic kwargs ONLY across good pixels (masked sources within each box) ---
                valid = ~np.isnan(cutout_masked)
                x_valid = xx.ravel()[valid.ravel()]
                y_valid = yy.ravel()[valid.ravel()]
                data_valid = cutout_masked.ravel()[valid.ravel()]
                weights_valid = weights.ravel()[valid.ravel()] if weights is not None else None

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
                    # Evaluate model on grid
                    model_eval = model_fn(result.params, xx, yy)
                
                    redchi = result.redchi

                    # Compute normalized mean squared error
                    numerator = np.sum((model_eval - cutout)**2)
                    denominator = np.sum(cutout**2) + 1e-12  # avoid division by zero
                    nmse = numerator / denominator
                    
                
                    logger_file_only.info(f"[SUCCESS] Fit (box={cutout.shape[1], cutout.shape[0]}, order={order}) → reduced chi² = {result.redchi:.5f}, NMSE = {nmse:.2e}")
                else:
                    logger_file_only.error(f"[FAILURE] Fit failed (box={cutout.shape[1], cutout.shape[0]}, order={order})")
                    
        
                if nmse < min_nmse:
                    best_result = result
                    best_order = order
                    best_nmse = nmse
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
            b = -np.sin(2*th)/(4*sx**2) + np.sin(2*th)/(4*sy**2)
            c = (np.sin(th)**2)/(2*sx**2) + (np.cos(th)**2)/(2*sy**2)
            gauss_vals += A * np.exp(- (a*(xx - x0)**2 + 2*b*(xx - x0)*(yy - y0) + c*(yy - y0)**2))

        bg_component = bg_vals - gauss_vals if not no_background else np.zeros_like(bg_vals)
        bg_mean = np.mean(bg_component) if not no_background else 0.0


        model_eval = model_fn(best_result.params, xx, yy)
        residual_map = best_cutout - model_eval
        

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
                hdul = fits.HDUList([hdu])
                
                # Write the FITS file
                hdul.writeto(filename, overwrite=True)
            
            save_fits(best_cutout, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "cutout", header=best_header)
            save_fits(model_eval, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "model", header=best_header)
            save_fits(residual_map, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}", "residual", header=best_header)

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
                label_name=f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}" if group_id is not None else "group",
                box_size=best_box,
                poly_order=best_order,
                nmse=best_nmse
            )
            

        return fit_status, best_result, model_fn, best_order, best_cutout, best_slice, best_header, bg_mean, best_box, best_nmse

    # Ensure return is always complete
    return 0, None, None, None, None, None, None, None