import numpy as np
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting

def masked_background_single_sources(
    cutout_masked,
    cutout,
    cutout_header,
    x0,
    y0,
    external_sources,
    max_fwhm_extent,
    all_box_sizes,
    pol_orders_separate,
    suffix,
    source_id,
    config,
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

    # Dimensions of the cutout
    ny, nx = cutout.shape

    # Initialize outputs
    best_bg_model = np.zeros_like(cutout)
    best_params = {}
    best_std = np.inf
    best_mean = np.inf
    best_order = None
    
    
    
    # --- Create Mask main source and external companions --- #
    # Initialize mask: True = valid pixel for background fitting
    mask_bg = np.ones_like(cutout_masked, dtype=bool)
    mask_bg[np.isnan(cutout_masked)] = False
    all_x = [x0]
    all_y = [y0]
    

    # Mask all sources using simple 2D Gaussian fitting
    cut_local = cutout_masked
    for xc, yc in zip(all_x, all_y):
        xc_int = int(round(xc))
        yc_int = int(round(yc))
    
        # Define small cutout around each source (e.g. 4*max_fwhm_extent)
        fit_size = round(max_fwhm_extent)*2  # half-size
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
            x_stddev=max_fwhm_extent/2.,
            y_stddev=max_fwhm_extent*2,
            theta=0.0,
            bounds={'x_stddev': (max_fwhm_extent/4., max_fwhm_extent*4), 'y_stddev': (max_fwhm_extent/4., max_fwhm_extent*4)}
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



    # ------------------ Determine list of box sizes to try ------------------
    
    # Dynamically define box size range based on box size limits defined for the main loop
    min_box = min(all_box_sizes)
    max_box = max(all_box_sizes)

    # Ensure odd box sizes
    if min_box % 2 == 0:
        min_box += 1
    if max_box % 2 == 0:
        max_box += 1
    box_sizes = list(range(min_box, max_box + 2, 2))

    # ------------------ Loop over box sizes ------------------
    for box in box_sizes:
        half_box = box // 2
        x0_int, y0_int = int(round(x0)), int(round(y0))

        # Determine cutout limits (crop to image boundaries)
        xmin = max(0, x0_int - half_box)
        xmax = min(nx, x0_int + half_box + 1)
        ymin = max(0, y0_int - half_box)
        ymax = min(ny, y0_int + half_box + 1)

        # Extract local cutout and corresponding masked version
        cut_masked = cutout_masked[ymin:ymax, xmin:xmax].copy()
        cut_raw = cutout[ymin:ymax, xmin:xmax]
        yy, xx = np.indices(cut_masked.shape)
        
        mask_bg = mask_bg[ymin:ymax, xmin:xmax].copy()

        # Extract valid pixel coordinates
        y_bg, x_bg = np.where(mask_bg)
        z_bg = cut_masked[y_bg, x_bg]

        # Sigma clip background pixel values
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        clipped = sigma_clip(z_bg)
        valid = ~clipped.mask
        x_valid = x_bg[valid]
        y_valid = y_bg[valid]
        z_valid = clipped.data[valid]

        if len(z_valid) < 10:
            continue  # Not enough pixels to fit


        # ------------------ Loop over polynomial orders ------------------
        for order in pol_orders_separate:
            # Build design matrix for 2D polynomial of given order
            terms = []
            param_names = []
            for dx in range(order + 1):
                for dy in range(order + 1 - dx):
                    terms.append((x_valid ** dx) * (y_valid ** dy))
                    param_names.append(f"c{dx}_{dy}")

            A = np.vstack(terms).T
            coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
            coeff_dict = dict(zip(param_names, coeffs))

            # Build background model over the cutout
            bg_model_local = np.zeros_like(cut_masked)
            for pname, val in coeff_dict.items():
                dx, dy = map(int, pname[1:].split("_"))
                bg_model_local += val * (xx ** dx) * (yy ** dy)

            # Compute residual map and statistics
            residual = cut_raw - bg_model_local
            residual_clip = sigma_clip(residual)
            mean_res = np.nanmean(residual_clip)
            std_res = np.nanstd(residual_clip)

            # Save best background model based on std and residual mean
            if std_res < best_std and abs(mean_res) < best_mean:
                best_bg_model = np.zeros_like(cutout)
                best_bg_model[ymin:ymax, xmin:xmax] = bg_model_local
                best_params = coeff_dict
                best_order = order
                best_std = std_res
                best_mean = abs(mean_res)


    # ------------------ Final background subtraction ------------------
    if best_order is None:
        # If no valid background was found, return unmodified cutout
        logger_file_only.warning("[WARNING] Background fit failed; returning original cutout.")
        return cutout, np.zeros_like(cutout), {}

    # Subtract background from the original cutout
    cutout -= best_bg_model
    logger_file_only.info(
        f"[INFO] Background subtracted using order {best_order} polynomial."
    )

    return cutout, best_bg_model, best_params