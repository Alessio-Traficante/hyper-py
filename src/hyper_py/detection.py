from email import header
import math

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from astropy.table import Table


def select_channel_map(map_struct):
    beam_dim_ref = map_struct["beam_dim"]
    pix_dim_ref = map_struct["pix_dim"]
    FWHM_pix = beam_dim_ref / pix_dim_ref
        
    return map_struct, FWHM_pix


def high_pass_filter(image, kernel_size_pix, FWHM_pix):

    if kernel_size_pix != 0 and kernel_size_pix % 2 == 0:
        kernel_size_pix -= 1    
    FWHM_int = math.floor(FWHM_pix)

    kernel_dim = kernel_size_pix if kernel_size_pix != 0 else FWHM_int**2

    ny, nx = image.shape
    kdim = min(kernel_dim, ny, nx)
    if kdim % 2 == 0:
        kdim -= 1

    # The original kernel (center = kdim²-1, rest = -1) is mathematically
    # equivalent to kdim² * (image - local_mean(image)).  uniform_filter
    # computes local_mean via separable 1-D box filters (prefix sums) in
    # O(N) regardless of kdim, versus O(N * kdim²) for direct convolution.
    # The overall sign / zero structure is identical; the kdim² scale factor
    # cancels in normalize_filtered_image (peak → 100).
    #
    # NaN handling: uniform_filter uses a sliding-window cumsum.  A single
    # NaN corrupts all subsequent pixels in its row/column (unlike convolve
    # which only spreads NaN within the kernel window).  Fix: fill NaN with
    # the image median before filtering, then zero those positions afterward.
    img_float = image.astype(float, copy=False)
    nan_mask = ~np.isfinite(img_float)
    if nan_mask.any():
        fill_val = float(np.nanmedian(img_float))
        img_work = np.where(nan_mask, fill_val, img_float)
    else:
        img_work = img_float
    filtered = img_float - uniform_filter(img_work, size=kdim, mode='nearest')
    filtered[nan_mask] = 0.0   # NaN pixels → 0 (clipped away anyway)
    filtered[filtered < 0] = 0.0
    return filtered


def normalize_filtered_image(filtered):
    # Step 1: Make a copy to avoid modifying input in-place
    filtered = np.array(filtered, copy=True)
    
    # Step 2: Set all values ≤ 0 to 0
    filtered[filtered <= 0] = 0.0

    # Step 3: Normalize to peak = 100 (only if peak > 0)
    peak = np.nanmax(filtered)
    normalized = (filtered / peak) * 100.0 if peak > 0 else filtered

    return normalized


# --- low values to get as many sources as possible in this first filter stage --- #
def estimate_rms(image, sigma_clip=3.0):
    values = image[image > 0]
    if len(values) == 0:
        return 0.0
    _, _, sigma = sigma_clipped_stats(values, sigma=sigma_clip, maxiters=10, mask_value=0.0)
    
    return sigma


def detect_peaks(filtered_image, threshold, fwhm_pix, roundlim=(-1.0, 1.0), sharplim=(-1.0, 2.0)):  
    
    finder = DAOStarFinder(
        threshold=threshold,
        fwhm=fwhm_pix,
        min_separation=0,
        roundlo=roundlim[0], roundhi=roundlim[1],
        sharplo=sharplim[0], sharphi=sharplim[1]
    )
    return finder(filtered_image)


def filter_peaks(peaks_table, fwhm_pix, image_shape, min_dist_pix, aper_sup):
    if peaks_table is None or len(peaks_table) == 0:
        return peaks_table

    if min_dist_pix is None:
        min_dist_pix = fwhm_pix

    ny, nx = image_shape
    margin = int(fwhm_pix)*aper_sup
    
    # Step 1: remove peaks too close to image border
    valid = (
        (peaks_table['xcentroid'] > margin) &
        (peaks_table['xcentroid'] < nx - margin) &
        (peaks_table['ycentroid'] > margin) &
        (peaks_table['ycentroid'] < ny - margin)
    )
    peaks = peaks_table[valid]

    if len(peaks) == 0:
        return peaks

    # Step 2: remove close neighbors (keep brightest)
    # KD-tree query_pairs finds all pairs with distance < min_dist_pix in
    # O(N log N + M) instead of the original O(N²) Python double-loop.
    # Pairs are returned as (i, j) with i < j in lexicographic order,
    # matching the original iteration order → identical output.
    coords = np.vstack([peaks['xcentroid'], peaks['ycentroid']]).T
    peak_vals = np.array(peaks['peak'])
    keep = np.ones(len(peaks), dtype=bool)

    if len(peaks) > 1:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(min_dist_pix, output_type='ndarray')
        # Sort pairs lexicographically to match original nested-loop order
        if len(pairs) > 0:
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs = pairs[order]
            for i, j in pairs:
                if not keep[i] or not keep[j]:
                    continue
                if peak_vals[i] >= peak_vals[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    
    return peaks[keep]


def snap_to_map_peak(peaks_table, real_map, search_radius_pix=1.0):
    """
    Refine each detected centroid to the sub-pixel flux-weighted maximum in
    the original map within ``search_radius_pix`` pixels of the DAOStarFinder
    position.

    ``search_radius_pix`` is specified directly in pixels (e.g. 1.0 = search
    within ±1 pixel).  This corrects the typical 1-pixel offset that
    DAOStarFinder introduces on the high-pass filtered image without being
    large enough to land on a neighbouring source's peak.

    Algorithm
    ---------
    1. Find the brightest integer pixel in the search window (real map).
    2. Compute a flux-weighted centroid in the 3x3 neighbourhood around
       that pixel using ABSOLUTE map coordinates (not cutout-relative).
       This is critical: any two sources that land on the same integer peak
       use the identical 3x3 patch and therefore get the identical sub-pixel
       position, which guarantees deduplication works correctly.

    Post-snap deduplication
    -----------------------
    Sources that snap to the same integer peak are grouped and only the
    brightest one (by real-map value at that pixel) is kept.  A second-pass
    KD-tree at 1.5 px then catches adjacent-pixel duplicates.

    Set ``snap_radius_pix: 0`` in the config to disable.
    """
    if peaks_table is None or len(peaks_table) == 0:
        return peaks_table

    h, w = real_map.shape
    R = max(1, int(np.ceil(search_radius_pix)))
    n = len(peaks_table)

    out = peaks_table.copy()
    orig_x = np.array([float(out['xcentroid'][k]) for k in range(n)], dtype=float)
    orig_y = np.array([float(out['ycentroid'][k]) for k in range(n)], dtype=float)

    # Step 1: find absolute integer peak for each source within its search window
    peak_xi = np.empty(n, dtype=int)
    peak_yi = np.empty(n, dtype=int)
    for k in range(n):
        x0 = int(round(orig_x[k]))
        y0 = int(round(orig_y[k]))
        x_lo = max(0, x0 - R);  x_hi = min(w, x0 + R + 1)
        y_lo = max(0, y0 - R);  y_hi = min(h, y0 + R + 1)
        cutout = real_map[y_lo:y_hi, x_lo:x_hi]
        if cutout.size == 0 or not np.any(np.isfinite(cutout)):
            peak_xi[k] = x0
            peak_yi[k] = y0
            continue
        flat_idx = np.nanargmax(cutout)
        yi_rel, xi_rel = np.unravel_index(flat_idx, cutout.shape)
        peak_xi[k] = x_lo + xi_rel
        peak_yi[k] = y_lo + yi_rel

    # Deduplication: sources sharing the same integer peak -> keep only the brightest
    peak_vals = np.array([
        real_map[int(np.clip(peak_yi[k], 0, h - 1)),
                 int(np.clip(peak_xi[k], 0, w - 1))]
        for k in range(n)
    ])
    keep = np.ones(n, dtype=bool)
    seen = {}  # (xi, yi) -> index of the kept source for that peak
    for k in range(n):
        pk = (int(peak_xi[k]), int(peak_yi[k]))
        if pk not in seen:
            seen[pk] = k
        else:
            prev = seen[pk]
            v_prev = peak_vals[prev] if np.isfinite(peak_vals[prev]) else -np.inf
            v_k   = peak_vals[k]    if np.isfinite(peak_vals[k])    else -np.inf
            if v_k > v_prev:
                keep[prev] = False
                seen[pk] = k
            else:
                keep[k] = False

    # Step 2: sub-pixel centroid in ABSOLUTE map coordinates for survivors.
    # Using absolute coords means sources that share a peak always get the
    # same 3x3 patch -> same centroid -> robust independence of search window.
    new_x = orig_x.copy()
    new_y = orig_y.copy()
    for k in range(n):
        if not keep[k]:
            continue
        xi, yi = int(peak_xi[k]), int(peak_yi[k])
        y_s = max(0, yi - 1);  y_e = min(h, yi + 2)
        x_s = max(0, xi - 1);  x_e = min(w, xi + 2)
        sub = real_map[y_s:y_e, x_s:x_e]
        sub_w = np.where(np.isfinite(sub) & (sub > 0), sub, 0.0)
        total_w = sub_w.sum()
        if total_w > 0:
            yy, xx = np.indices(sub.shape, dtype=float)
            new_x[k] = (sub_w * xx).sum() / total_w + x_s
            new_y[k] = (sub_w * yy).sum() / total_w + y_s
        else:
            new_x[k] = float(xi)
            new_y[k] = float(yi)

    survivors = np.where(keep)[0]
    for k in survivors:
        out['xcentroid'][k] = new_x[k]
        out['ycentroid'][k] = new_y[k]

    out = out[keep]

    # Second-pass dedup: sources that snapped to *adjacent* integer pixels (not
    # the same pixel) can still represent the same physical source.  Their
    # sub-pixel centroids end up within ~1.5 px of each other.  A 1.5 px
    # KD-tree threshold catches these without touching genuine distinct sources,
    # which are always >= FWHM (several pixels) apart on the real map.
    if len(out) > 1:
        sx = np.array([float(out['xcentroid'][k]) for k in range(len(out))], dtype=float)
        sy = np.array([float(out['ycentroid'][k]) for k in range(len(out))], dtype=float)
        sv = np.array([
            real_map[int(np.clip(round(sy[k]), 0, h - 1)),
                     int(np.clip(round(sx[k]), 0, w - 1))]
            for k in range(len(out))
        ])
        tree2 = cKDTree(np.column_stack([sx, sy]))
        pairs2 = tree2.query_pairs(1.5, output_type='ndarray')
        keep2 = np.ones(len(out), dtype=bool)
        if len(pairs2) > 0:
            order2 = np.lexsort((pairs2[:, 1], pairs2[:, 0]))
            for i, j in pairs2[order2]:
                if not keep2[i] or not keep2[j]:
                    continue
                v_i = sv[i] if np.isfinite(sv[i]) else -np.inf
                v_j = sv[j] if np.isfinite(sv[j]) else -np.inf
                if v_i >= v_j:
                    keep2[j] = False
                else:
                    keep2[i] = False
        out = out[keep2]

    return out


def compute_background_asymmetry(real_map, x, y, fwhm_pix, rms, n_angles=16, ring_factor=1.5,
                                  neighbor_positions=None, neighbor_mask_radius=None):
    """
    Background asymmetry score for a source at pixel (x, y).
    (See filter_by_shoulder docstring for full description.)
    """
    R = fwhm_pix * ring_factor
    h, w = real_map.shape
    angles = np.linspace(0, 2.0 * np.pi, n_angles, endpoint=False)

    # Vectorised ring sample positions
    xi_arr = np.round(x + R * np.cos(angles)).astype(int)
    yi_arr = np.round(y + R * np.sin(angles)).astype(int)

    # In-bounds mask
    in_bounds = (xi_arr >= 0) & (xi_arr < w) & (yi_arr >= 0) & (yi_arr < h)

    # Neighbour contamination mask (vectorised over angles × neighbours)
    if neighbor_positions is not None and neighbor_mask_radius is not None:
        nb = np.asarray(neighbor_positions, dtype=float)  # shape (M, 2)
        if nb.ndim == 2 and len(nb) > 0:
            nmr2 = neighbor_mask_radius ** 2
            dx = xi_arr[:, None] - nb[None, :, 0]   # (n_angles, M)
            dy = yi_arr[:, None] - nb[None, :, 1]
            contaminated = np.any(dx ** 2 + dy ** 2 < nmr2, axis=1)  # (n_angles,)
            in_bounds &= ~contaminated

    valid_idx = np.where(in_bounds)[0]
    if len(valid_idx) < max(4, n_angles // 4):
        return 0.0  # too few clean samples → can't judge, keep source

    ring_vals_raw = real_map[yi_arr[valid_idx], xi_arr[valid_idx]]
    ring_vals = ring_vals_raw[np.isfinite(ring_vals_raw)]

    if len(ring_vals) < max(4, n_angles // 4):
        return 0.0

    # Normalise by source peak; fall back to rms for very faint / non-positive peaks
    peak_flux = real_map[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))]
    norm = float(peak_flux) if np.isfinite(peak_flux) and peak_flux > 0 else float(rms)
    norm = max(norm, float(rms), 1e-30)

    ring_arr = np.sort(ring_vals)
    half = len(ring_arr) // 2
    score = (np.mean(ring_arr[half:]) - np.mean(ring_arr[:half])) / norm
    return float(score)


def filter_by_shoulder(peaks_table, real_map, fwhm_pix, rms_real,
                        max_asymmetry=None, n_angles=16, ring_factor=1.5,
                        neighbor_mask_fwhm=1.0):
    """
    Add ``SHOULDER_SCORE`` column to *peaks_table* and optionally remove shoulders.

    For each source, samples real_map on a ring at ``ring_factor * fwhm_pix``
    radius.  Ring samples within ``neighbor_mask_fwhm * fwhm_pix`` of another
    detected source are excluded (crowded-region protection).  The score is
    ``(mean_bright_half - mean_faint_half) / F_peak``; score ~ 0 = real source,
    score > 0.3 = likely shoulder.

    Parameters
    ----------
    max_asymmetry : float or None
        Remove sources with score > this value.  None = score only, no removal.
    neighbor_mask_fwhm : float
        Exclusion radius around each neighbour, in units of fwhm_pix.
    """
    if peaks_table is None or len(peaks_table) == 0:
        return peaks_table

    all_xy = np.array([[float(row['xcentroid']), float(row['ycentroid'])]
                       for row in peaks_table])
    nmr = neighbor_mask_fwhm * fwhm_pix

    # Pre-compute relevant neighbours for each source with a KD-tree.
    # A ring sample at ring_factor*FWHM from source k can only be
    # contaminated by a neighbour within (ring_factor + neighbor_mask_fwhm)*FWHM.
    max_relevant_dist = (ring_factor + neighbor_mask_fwhm) * fwhm_pix
    if len(all_xy) > 1:
        tree = cKDTree(all_xy)
        relevant_idx = tree.query_ball_point(all_xy, max_relevant_dist)
    else:
        relevant_idx = [[] for _ in range(len(all_xy))]

    scores = []
    for k, row in enumerate(peaks_table):
        nb_indices = [j for j in relevant_idx[k] if j != k]
        neighbors = all_xy[nb_indices] if nb_indices else None
        scores.append(
            compute_background_asymmetry(
                real_map,
                float(row['xcentroid']), float(row['ycentroid']),
                fwhm_pix, rms_real,
                n_angles=n_angles, ring_factor=ring_factor,
                neighbor_positions=neighbors,
                neighbor_mask_radius=nmr,
            )
        )

    out = peaks_table.copy()
    out['SHOULDER_SCORE'] = np.round(np.array(scores, dtype=float), 2)

    if max_asymmetry is not None:
        out = out[out['SHOULDER_SCORE'] <= float(max_asymmetry)]

    return out


# --- save only sources above a sigma-clipped rms estimation in the maps, or use a manual value ---
def filter_by_snr(peaks_table, real_map, rms_real, snr_threshold):
    if peaks_table is None or len(peaks_table) == 0:
        return peaks_table

    h, w = real_map.shape
    xs_raw = np.round(np.array(peaks_table['xcentroid'])).astype(int)
    ys_raw = np.round(np.array(peaks_table['ycentroid'])).astype(int)

    in_bounds = (xs_raw >= 0) & (xs_raw < w) & (ys_raw >= 0) & (ys_raw < h)
    xs = np.clip(xs_raw, 0, w - 1)
    ys = np.clip(ys_raw, 0, h - 1)

    peak_vals = real_map[ys, xs]
    snr = (peak_vals / rms_real) if rms_real > 0 else np.zeros(len(peaks_table))
    keep = in_bounds & (snr >= snr_threshold)

    return peaks_table[keep]


def detect_sources(map_struct_list, dist_limit_arcsec, real_map, rms_real, snr_threshold, roundlim, sharplim, config):
    map_struct, FWHM_pix = select_channel_map(map_struct_list)

    image = map_struct["map"]
    pix_dim_ref = map_struct["pix_dim"]
    beam_dim_ref = map_struct["beam_dim"]
    aper_sup=config.get("photometry", "aper_sup")
    aper_inf=config.get("photometry", "aper_inf")

    my_dist_limit_arcsec = beam_dim_ref if dist_limit_arcsec == 0 else dist_limit_arcsec
    dist_limit_pix = my_dist_limit_arcsec / pix_dim_ref


    # --- identify multiple peaks in filtered image and save good peaks with real snr threshold --- #
    kernel_size_pix=config.get("detection", "kernel_size_pix", 0)

    filtered = high_pass_filter(image, kernel_size_pix, FWHM_pix)
    norm_filtered = normalize_filtered_image(filtered)
        
    filtered_rms_detect = estimate_rms(norm_filtered)
    filtered_threshold = 2. * filtered_rms_detect
        
    peaks = detect_peaks(norm_filtered, filtered_threshold, FWHM_pix, roundlim=roundlim, sharplim=sharplim)
    if peaks is None or len(peaks) == 0:
        return Table({'xcentroid': [], 'ycentroid': [], 'peak': [], 'SHOULDER_SCORE': []})
    good_peaks = filter_peaks(peaks, FWHM_pix, image.shape, dist_limit_pix, aper_inf)
    snap_radius = config.get("detection", "snap_radius_pix", 1.0)
    if snap_radius and float(snap_radius) > 0:
        good_peaks = snap_to_map_peak(good_peaks, real_map,
                                      search_radius_pix=float(snap_radius))
    final_sources = filter_by_snr(good_peaks, real_map, rms_real, snr_threshold)

    # --- Shoulder / background-asymmetry filter ----------------------------
    # Always adds SHOULDER_SCORE column (useful for diagnostics / post-run
    # inspection).  Sources are removed only when shoulder_filter: true in
    # the config.  Set shoulder_filter: false (default) to get the complete
    # catalogue with scores but without any rejection.
    shoulder_filter    = config.get("detection", "shoulder_filter", False)
    shoulder_threshold = config.get("detection", "shoulder_threshold", 0.5)
    shoulder_ring      = config.get("detection", "shoulder_ring_factor", 1.5)
    shoulder_angles    = config.get("detection", "shoulder_n_angles", 16)
    shoulder_nb_mask   = config.get("detection", "shoulder_neighbor_mask", 1.0)

    final_sources = filter_by_shoulder(
        final_sources,
        real_map,
        FWHM_pix,
        rms_real,
        max_asymmetry=float(shoulder_threshold) if shoulder_filter else None,
        n_angles=int(shoulder_angles),
        ring_factor=float(shoulder_ring),
        neighbor_mask_fwhm=float(shoulder_nb_mask),
    )

    return final_sources
