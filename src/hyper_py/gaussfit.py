import numpy as np
from lmfit import minimize, Parameters
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import CircularAperture
from hyper_py.visualization import plot_fit_summary
import matplotlib.pyplot as plt
import os



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
    ny, nx = image.shape

    if fix_box:
        box_sizes = [fix_min_box]
    else:
        max_fwhm_extent = aper_sup * 2.3548  # twice major FWHM in pixels
        dynamic_min_box = int(max_fwhm_extent) + fix_min_box
        dynamic_max_box = dynamic_min_box + fix_max_box
        box_sizes = list(range(dynamic_min_box + 1, dynamic_max_box + 2, 2))  # ensure odd

    best_result = None
    min_nmse = np.inf
    best_cutout = None
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
    
    
        #--- Create mask ---#
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
        


        # --- Background estimation on masked cutout (optional) ---
        if fit_separately:
            logger_file_only.info("[INFO] Estimating background separately on masked cutout...")
            mask_bg = np.ones_like(cutout, dtype=bool)
            r_mask = max_fwhm_extent  # already in pixels

            xcen_cut_bg = np.array([x0])
            ycen_cut_bg = np.array([y0])

            if len(external_sources) > 0:
                ex_arr, ey_arr = np.array(external_sources).T
                xcen_cut_bg = np.concatenate([xcen_cut_bg, ex_arr])
                ycen_cut_bg = np.concatenate([ycen_cut_bg, ey_arr])

            
            for xc, yc in zip(xcen_cut_bg, ycen_cut_bg):
                aperture = CircularAperture((xc, yc), r=r_mask)
                mask_obj = aperture.to_mask(method="center")
                mdata = mask_obj.to_image(cutout.shape)
                if mdata is not None:
                    mask_bg[mdata > 0] = False

            y_bg, x_bg = np.where(mask_bg)
            z_bg = cutout[y_bg, x_bg]
            yy, xx = np.indices(cutout.shape)
            
            # Select only valid (non-NaN) pixels
            valid = ~np.isnan(z_bg)
            x_bg_valid = x_bg[valid]
            y_bg_valid = y_bg[valid]
            z_bg_valid = z_bg[valid]


            poly_params = {}
            max_order = max(pol_orders_separate)
            for dx in range(max_order + 1):
                for dy in range(max_order + 1 - dx):
                    pname = f"c{dx}_{dy}"
                    # A = (x_bg ** dx) * (y_bg ** dy)
                    # denom = np.sum(A ** 2)
                    # poly_params[pname] = np.sum(z_bg * A) / denom if denom > 0 else 0.0
                    
                    A = np.ones_like(z_bg_valid)
                    A = (x_bg_valid ** dx) * (y_bg_valid ** dy)
                    denom = np.sum(A**2)
                    poly_params[pname] = np.sum(z_bg_valid * A) / denom if denom > 0 else 0.0

                    

            # Build polynomial model and subtract it
            bg_model = np.zeros_like(cutout)
            for pname, val in poly_params.items():
                dx, dy = map(int, pname[1:].split("_"))
                bg_model += val * (xx ** dx) * (yy ** dy)

            cutout -= bg_model
            logger_file_only.info("[INFO] Background subtracted from cutout.")


            # --- save separated background estimation in fits format --- #
            try:
                fits_bg_separate = config.get("fits_output", "fits_bg_separate", False)
                dir_comm =  config.get("paths", "dir_comm")
                fits_output_dir_bg_separate = dir_comm + config.get("fits_output", "fits_output_dir_bg_separate", "Fits/Bg_separate")  
            except:
                fits_bg_separate = False

            if fits_bg_separate:
                label_name = f"HYPER_MAP_{suffix}_ID_{source_id+1}"
                filename = f"{fits_output_dir_bg_separate}/{label_name}_bg_masked3D.fits"
            
                # Create a PrimaryHDU object and write the array into the FITS file
                hdu = fits.PrimaryHDU(bg_model)
                hdul = fits.HDUList([hdu])
                    
                # Write the FITS file
                hdul.writeto(filename, overwrite=True)


            # Optional 3D visualization
            try:
                visualize_bg = config.get("visualization", "visualize_bg_separate", False)                
                dir_comm =  config.get("paths", "dir_comm")
                output_dir = dir_comm + config.get("visualization", "output_dir_bg_separate")
            except:
                visualize_bg = False
                
            if visualize_bg:
                os.makedirs(output_dir, exist_ok=True)
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(xx, yy, bg_model, cmap="viridis", linewidth=0, antialiased=True)
                ax.set_xlabel("X (pix)", fontsize=8, fontweight="bold")
                ax.set_ylabel("Y (pix)", fontsize=8, fontweight="bold")
                ax.set_zlabel("Flux (Jy)", fontsize=8, fontweight="bold")
                ax.set_title("Initial Background (Isolated)", fontsize=10, fontweight="bold")
                
                label_str = f"HYPER_MAP_{suffix}_ID_{source_id+1}"
                outname = os.path.join(output_dir, f"{label_str}_bg_masked3D.png")
                
                plt.savefig(outname, dpi=300, bbox_inches="tight")
                plt.close()
                

        # --- Fit single 2D elliptical Gaussian (+ background) ---
        mean_bg, median_bg, std_bg = sigma_clipped_stats(cutout, sigma=3.0, maxiters=5)
        cutout_rms = np.full_like(cutout, std_bg)

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
                params.add("g_amplitude", value=np.max(cutout), min=0., max=1.2*np.max(cutout))
                
                if vary == True:
                    params.add("g_centerx", value=x0, min=x0 - 1, max=x0 + 1)
                    params.add("g_centery", value=y0, min=y0 - 1, max=y0 + 1)

                if vary == False:
                    params.add("g_centerx", value=x0, vary=False)
                    params.add("g_centery", value=y0, vary=False)

                params.add("g_sigmax", value=aper_inf, min=aper_inf, max=aper_sup)
                params.add("g_sigmay", value=aper_sup, min=aper_inf, max=aper_sup)
                params.add("g_theta", value=0.0) #, min=-np.pi/2, max=np.pi/2)

                if not no_background:
                    for dx in range(order + 1):
                        for dy in range(order + 1 - dx):
                            pname = f"c{dx}_{dy}"
                            val = median_bg if (dx == 0 and dy == 0) else 1.e-5
                            params.add(pname, value=val, vary=True)


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
                                model += params[pname] * (x ** dx) * (y ** dy)                                
                    return model


                def residual(p, x, y, data, weights=None):
                    model = model_fn(p, x, y)
                    resid = (model - data).ravel().astype(np.float64)
                    
                    if weights is not None:
                        resid *= weights

                    if use_l2 and not no_background:
                        penalty_values = [
                            float(params[p].value)
                            for p in params if p.startswith("c")
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
                    
        
                if nmse < min_nmse:
                    best_result = result
                    best_nmse = nmse
                    best_order = order
                    best_cutout = cutout
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
            def save_fits(array, output_dir, label_name, extension_name):
                 # Create the FITS filename based on the label and extension type
                filename = f"{output_dir}/{label_name}_{extension_name}.fits"
        
                # Create a PrimaryHDU object and write the array into the FITS file
                hdu = fits.PrimaryHDU(array)
                hdul = fits.HDUList([hdu])
                
                # Write the FITS file
                hdul.writeto(filename, overwrite=True)
            
            save_fits(best_cutout, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "cutout")
            save_fits(model_eval, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "model")
            save_fits(residual_map, fits_output_dir_fitting, f"HYPER_MAP_{suffix}_ID_{source_id+1}", "residual")


      
        
        return fit_status, best_result, model_fn, best_order, best_cutout, best_slice, bg_mean, best_nmse
    else:
        return 0, None, None, None, None, None, None, None