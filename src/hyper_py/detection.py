from email import header
import math

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import convolve
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

    kernel = np.full((kdim, kdim), -1.0)
    kernel[kdim // 2, kdim // 2] = kdim**2 - 1.0
    filtered = convolve(image.astype(float), kernel, mode='nearest')
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

    # Step 2: remove close neighbors (keep brightest)
    coords = np.vstack([peaks['xcentroid'], peaks['ycentroid']]).T
    keep = np.ones(len(peaks), dtype=bool)

    for i in range(len(peaks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(peaks)):
            if not keep[j]:
                continue
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = np.hypot(dx, dy)
            if dist < min_dist_pix:
                if peaks[i]['peak'] >= peaks[j]['peak']:
                    keep[j] = False
                else:
                    keep[i] = False
                    
    return peaks[keep]


def compute_background_asymmetry(real_map, x, y, fwhm_pix, rms, n_angles=16, ring_factor=1.5,
                                  neighbor_positions=None, neighbor_mask_radius=None):
    """
    Background asymmetry score for a source at pixel (x, y).

    Samples ``real_map`` at ``n_angles`` equally-spaced points on a ring of
    radius ``ring_factor * fwhm_pix`` around the source.  Returns::

        score = (mean_bright_half - mean_faint_half) / F_peak

    where bright/faint halves come from sorting the ring samples and
    ``F_peak`` is the source peak flux at (x, y).  Falls back to ``rms``
    as denominator if the peak is not positive.

    At ring_factor = 1.5 a Gaussian source contributes < 0.2 % of its peak
    to the ring, so the score measures pure background asymmetry normalised
    by the source brightness — making it independent of the absolute flux
    scale of the map.

    Neighbor masking: if ``neighbor_positions`` is provided, ring sample
    points within ``neighbor_mask_radius`` pixels of any neighbouring source
    are skipped.  If too few clean samples remain after masking (crowded
    region), the function returns 0.0 (source is kept).

    score ~ 0.0–0.1  → symmetric background → likely real compact source
    score > 0.3–0.5  → one-sided emission   → likely shoulder artefact
    """
    R = fwhm_pix * ring_factor
    h, w = real_map.shape
    angles = np.linspace(0, 2.0 * np.pi, n_angles, endpoint=False)

    # Pre-compute squared neighbour mask radius to avoid sqrt in the inner loop
    nmr2 = neighbor_mask_radius ** 2 if neighbor_mask_radius is not None else None

    ring_vals = []
    for theta in angles:
        xi = int(round(x + R * np.cos(theta)))
        yi = int(round(y + R * np.sin(theta)))

        # Skip samples that land inside a neighbouring source's footprint
        if nmr2 is not None and neighbor_positions is not None:
            contaminated = any(
                (xi - nx) ** 2 + (yi - ny) ** 2 < nmr2
                for nx, ny in neighbor_positions
            )
            if contaminated:
                continue

        if 0 <= yi < h and 0 <= xi < w:
            v = real_map[yi, xi]
            if np.isfinite(v):
                ring_vals.append(v)

    # If too many samples were masked (crowded region), can't assess → keep source
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

    Parameters
    ----------
    max_asymmetry : float or None
        If not None, sources with ``SHOULDER_SCORE > max_asymmetry`` are removed.
        If None, scores are computed and stored but no sources are removed (full
        catalogue mode).
    neighbor_mask_fwhm : float
        Ring sample points within this many FWHM of another detected source are
        excluded before computing the asymmetry.  Sources in crowded regions
        where too many samples are masked automatically receive score 0 (kept).
        Default 1.0.

    Returns
    -------
    astropy.table.Table with ``SHOULDER_SCORE`` column appended.
    """
    # Build a list of all source positions for neighbour masking
    all_xy = np.array([[float(row['xcentroid']), float(row['ycentroid'])]
                       for row in peaks_table])
    nmr = neighbor_mask_fwhm * fwhm_pix  # neighbour mask radius in pixels

    scores = []
    for k, row in enumerate(peaks_table):
        xs = float(row['xcentroid'])
        ys = float(row['ycentroid'])
        # Neighbours = every other source (exclude self)
        neighbors = [(all_xy[j, 0], all_xy[j, 1])
                     for j in range(len(all_xy)) if j != k]
        scores.append(
            compute_background_asymmetry(
                real_map, xs, ys, fwhm_pix, rms_real,
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
    keep = []
    for row in peaks_table:
        x = int(round(row['xcentroid']))
        y = int(round(row['ycentroid']))
        if 0 <= y < real_map.shape[0] and 0 <= x < real_map.shape[1]:            
            peak_val = real_map[y, x] 
            snr = peak_val / rms_real if rms_real > 0 else 0
            keep.append(snr >= snr_threshold)
        else:
            keep.append(False)
            
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
    good_peaks = filter_peaks(peaks, FWHM_pix, image.shape, dist_limit_pix, aper_inf)
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
