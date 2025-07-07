import numpy as np
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
from sklearn.linear_model import HuberRegressor, TheilSenRegressor

from astropy.wcs import WCS


def multigauss_background(minimize_method, image, header, xcen, ycen, nx, ny, all_sources_xcen, all_sources_ycen, aper_sup, max_fwhm_extent, 
                          box_sizes, pol_orders_separate, suffix, group_id, count_source_blended_indexes=None, config=None, logger=None, logger_file_only=None):
    """
    Estimate polynomial background in masked cutout, looping over box sizes and polynomial orders.
    """


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
        
    
    
    # - Initialize parameters - #
    best_params = {}
    best_order = None
    best_min = np.inf

    cutout_reference_mask = None




    for box in box_sizes:
        
        half_box = box // 2 -1
        xmin = max(0, int(np.mean(xcen)) - half_box)
        xmax = min(nx, int(np.mean(xcen)) + half_box + 1)
        ymin = max(0, int(np.mean(ycen)) - half_box)
        ymax = min(ny, int(np.mean(ycen)) + half_box + 1)

        cutout = np.array(image[ymin:ymax, xmin:xmax], dtype=np.float64)        
        if cutout.size == 0 or np.isnan(cutout).all():
            continue

        
        yy, xx = np.indices(cutout.shape)
        x0 = xcen - xmin
        y0 = ycen - ymin
        
        
        
        
        
        
        # ---Initialize mask: True = valid pixel for background fitting --- #
        mask_bg = np.ones_like(cutout, dtype=bool)
        
        all_sources_to_mask = []
        all_sources_to_mask.extend(zip(x0, y0))
        
        main_sources = []
        main_sources.extend(zip(x0, y0))
        
        external_sources = []
        
        #--- Identify external sources inside box and add to main source ---#
        mask_bg = np.ones_like(cutout, dtype=bool)
        
                             
                     
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
        
            # Mask pixels above 1-FWHM threshold (≈ 0.6065 × peak)
            threshold = g_fit.amplitude.value * np.exp(-0.5)
            mask_bg[model_vals > threshold] = False
        
        
        ### --- From now on, all photometry and background estimation is done on cutout_masked from external sources --- ###
        # --- Apply external sources mask → set masked pixels to np.nan --- #
        cutout_masked = np.copy(cutout)
        cutout_masked[~mask_bg] = np.nan

        

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
        
            # Mask pixels above 1-FWHM threshold (≈ 0.6065 × peak)
            threshold = g_fit.amplitude.value * np.exp(-0.5)
            mask_bg_all[model_vals > threshold] = False

        # --- Apply main sources mask → set masked pixels to np.nan --- #
        cutout_masked_all = np.copy(cutout_masked)
        cutout_masked_all[~mask_bg_all] = np.nan



        
        # - Estimate good pixels fpr background estimation only in cutout_masked_all - #
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
                half_ref_box = ref_box_size // 2 -1
                
                x_start = max(0, int((np.mean(x0))) - half_ref_box)
                x_end   = min(nx, int(np.mean(x0)) + half_ref_box +1)
                y_start = max(0, int((np.mean(y0))) - half_ref_box)
                y_end   = min(ny, int(np.mean(y0)) + half_ref_box +1)
                 
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

                
                # --- Crop current cutout to match reference size ---
                cutout_eval = cutout_masked_all[y_start:y_end, x_start:x_end]
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
                    best_bg_model = bg_model_full
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
        return cutout_masked, None, np.zeros_like(cutout), np.zeros_like(cutout), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}

    else:
        # Subtract background from the original cutout
        best_cutout -= best_bg_model
        best_cutout_masked -= best_bg_model
        
        logger_file_only.info(
            f"[INFO] Background subtracted using order {best_order} polynomial."
    )
 
    return best_cutout_masked, best_header, best_bg_model, best_mask_bg, best_x0, best_y0, best_xx, best_yy, best_xmin, best_xmax, best_ymin, best_ymax, best_box_sizes, best_order, best_params
    
 

    # # Final background subtraction
    # if best_bg_poly is None:
    #     logger_file_only.info(f"[WARNING] Background estimation failed.")
    #     cutout_bs = cutout.copy()
    #     bg_poly = np.zeros_like(cutout)
    # else:
    #     cutout_bs = cutout - best_bg_poly
    #     bg_poly = best_bg_poly
    #     logger_file_only.info(f"[INFO] Masked background polynomial subtracted using box size {box}.")





    # # === Optional 3D visualization ===
    # if config is not None:
    #     try:
    #         visualize_bg = config.get("visualization", "visualize_bg_separate", False)
    #     except:
    #         pass

    # # --- save separated background estimation in fits format --- #
    # try:
    #     fits_bg_separate = config.get("fits_output", "fits_bg_separate", False)
    #     dir_comm = config.get("paths", "dir_comm")
    #     fits_output_dir_bg_separate = dir_comm + config.get("fits_output", "fits_output_dir_bg_separate", "Fits/Bg_separate")  
    # except:
    #     fits_bg_separate = False

    # if fits_bg_separate:
    #     os.makedirs(fits_output_dir_bg_separate, exist_ok=True)
    #     label_name = f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}"
    #     filename = f"{fits_output_dir_bg_separate}/{label_name}_bg_masked3D.fits"
    #     convert_mjy = config.get("units", "convert_mJy")
    #     hdu = fits.PrimaryHDU(data=bg_poly, header=cutout_header)
    #     hdu.header['BUNIT'] = 'mJy/pixel' if convert_mjy else 'Jy/pixel'
    #     hdu.writeto(filename, overwrite=True)

    # # --- Visualize separated background estimation in png format --- #
    # if visualize_bg:
    #     logger_file_only.info("[INFO] Plotting 3D background model from masked map subtraction...")  
    #     dir_comm = config.get("paths", "dir_comm")
    #     output_dir_vis = dir_comm + config.get("visualization", "output_dir_bg_separate")
    #     os.makedirs(output_dir_vis, exist_ok=True)
    #     fig = plt.figure(figsize=(6, 5))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(xx_full, yy_full, bg_poly, cmap="viridis", linewidth=0, antialiased=True)
    #     ax.set_xlabel("X (pix)", fontsize=8, fontweight="bold")
    #     ax.set_ylabel("Y (pix)", fontsize=8, fontweight="bold")
    #     ax.set_zlabel("Flux (Jy)", fontsize=8, fontweight="bold")
    #     for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
    #         label.set_fontsize(8)
    #         label.set_fontweight("bold")
    #     ax.set_title("Initial Background Model from Masked Map", fontsize=10, fontweight="bold")
    #     plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.12)
    #     label_str = f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}" if count_source_blended_indexes is not None else "group"
    #     outname = os.path.join(output_dir_vis, f"{label_str}_bg_masked3D.png")
    #     plt.savefig(outname, dpi=300, bbox_inches="tight")
    #     plt.close()

    # return cutout_bs, bg_poly