import numpy as np
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.stats import SigmaClip

def estimate_masked_background(cutout, cutout_header, xcen_cut, ycen_cut, aper_sup, max_fwhm_extent, box_sizes, orders, suffix,
                                count_source_blended_indexes=None, config=None, logger=None, logger_file_only=None):
    """
    Estimate polynomial background in masked cutout, looping over box sizes and polynomial orders.
    """

    ny, nx = cutout.shape
    x_center = np.mean(xcen_cut)
    y_center = np.mean(ycen_cut)

    best_std = np.inf
    best_mean = np.inf
    best_bg_poly = None
    yy_full, xx_full = np.indices(cutout.shape)

    for box in box_sizes:
        half_box = box // 2
        x0_int, y0_int = int(round(x_center)), int(round(y_center))

        # Define crop region
        xmin = max(0, x0_int - half_box)
        xmax = min(nx, x0_int + half_box + 1)
        ymin = max(0, y0_int - half_box)
        ymax = min(ny, y0_int + half_box + 1)

        cut_local = cutout[ymin:ymax, xmin:xmax]
        yy, xx = np.indices(cut_local.shape)

        # Initialize valid mask
        mask_bg = np.ones_like(cut_local, dtype=bool)
        mask_bg[np.isnan(cut_local)] = False

        # Shift all centroids to local coordinates
        all_x = np.array(xcen_cut) - xmin
        all_y = np.array(ycen_cut) - ymin

        # Mask all sources
        for xc, yc in zip(all_x, all_y):
            aperture = CircularAperture((xc, yc), r=max_fwhm_extent)
            mask_obj = aperture.to_mask(method="center")
            mdata = mask_obj.to_image(cut_local.shape)
            if mdata is not None:
                mask_bg[mdata > 0] = False

        y_bg, x_bg = np.where(mask_bg)
        z_bg = cut_local[y_bg, x_bg]

        # Sigma clipping
        sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
        clipped = sigma_clip(z_bg)
        valid = ~clipped.mask
        x_valid = x_bg[valid]
        y_valid = y_bg[valid]
        z_valid = clipped.data[valid]

        if len(z_valid) < 10:
            continue

        # Try all polynomial orders
        for order in orders:
            terms = []
            param_names = []
            for dx in range(order + 1):
                for dy in range(order + 1 - dx):
                    terms.append((x_valid ** dx) * (y_valid ** dy))
                    param_names.append(f"c{dx}_{dy}")

            A = np.vstack(terms).T
            coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
            coeff_dict = dict(zip(param_names, coeffs))

            bg_model_local = np.zeros_like(cut_local)
            for pname, val in coeff_dict.items():
                dx, dy = map(int, pname[1:].split("_"))
                bg_model_local += val * (xx ** dx) * (yy ** dy)

            residual = cut_local - bg_model_local
            residual_clip = sigma_clip(residual)
            mean_res = np.nanmean(residual_clip)
            std_res = np.nanstd(residual_clip)

            # Save if best
            if std_res < best_std and abs(mean_res) < best_mean:
                best_std = std_res
                best_mean = abs(mean_res)
                best_bg_poly = np.zeros_like(cutout)
                best_bg_poly[ymin:ymax, xmin:xmax] = bg_model_local

    # Final background subtraction
    if best_bg_poly is None:
        logger_file_only.info(f"[WARNING] Background estimation failed.")
        cutout_bs = cutout.copy()
        bg_poly = np.zeros_like(cutout)
    else:
        cutout_bs = cutout - best_bg_poly
        bg_poly = best_bg_poly
        logger_file_only.info(f"[INFO] Masked background polynomial subtracted using box size {box}.")

    # === Optional 3D visualization ===
    if config is not None:
        try:
            visualize_bg = config.get("visualization", "visualize_bg_separate", False)
        except:
            pass

    # --- save separated background estimation in fits format --- #
    try:
        fits_bg_separate = config.get("fits_output", "fits_bg_separate", False)
        dir_comm = config.get("paths", "dir_comm")
        fits_output_dir_bg_separate = dir_comm + config.get("fits_output", "fits_output_dir_bg_separate", "Fits/Bg_separate")  
    except:
        fits_bg_separate = False

    if fits_bg_separate:
        os.makedirs(fits_output_dir_bg_separate, exist_ok=True)
        label_name = f"HYPER_MAP_{suffix}_ID_{count_source_blended_indexes[0]}_{count_source_blended_indexes[1]}"
        filename = f"{fits_output_dir_bg_separate}/{label_name}_bg_masked3D.fits"
        convert_mjy = config.get("units", "convert_mJy")
        hdu = fits.PrimaryHDU(data=bg_poly, header=cutout_header)
        hdu.header['BUNIT'] = 'mJy/pixel' if convert_mjy else 'Jy/pixel'
        hdu.writeto(filename, overwrite=True)

    # --- Visualize separated background estimation in png format --- #
    if visualize_bg:
        logger_file_only.info("[INFO] Plotting 3D background model from masked map subtraction...")  
        dir_comm = config.get("paths", "dir_comm")
        output_dir_vis = dir_comm + config.get("visualization", "output_dir_bg_separate")
        os.makedirs(output_dir_vis, exist_ok=True)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx_full, yy_full, bg_poly, cmap="viridis", linewidth=0, antialiased=True)
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