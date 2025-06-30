import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import convolve
from astropy.table import Table
from astropy.wcs import WCS



def select_channel_map(map_struct):
    beam_dim_ref = map_struct["beam_dim"]
    pix_dim_ref = map_struct["pix_dim"]
    FWHM_pix = beam_dim_ref / pix_dim_ref
        
    return map_struct, FWHM_pix


def high_pass_filter(image, kernel_dim=9):
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
    peak = np.nanmax(filtered)
    return (filtered / peak) * 100.0 if peak > 0 else filtered


# --- low values to get as many sources as possible in this first filter stage --- #
def estimate_rms(image, sigma_clip=2.0):
    values = image[image > 0]
    if len(values) == 0:
        return 0.0
    _, _, sigma = sigma_clipped_stats(values, sigma=sigma_clip)
    return sigma


def detect_peaks(filtered_image, threshold, fwhm_pix, roundlim=(-1.0, 1.0), sharplim=(-1.0, 2.0)):
    finder = DAOStarFinder(
        threshold=threshold,
        fwhm=fwhm_pix,
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
    header = map_struct["header"]
    pix_dim_ref = map_struct["pix_dim"]
    beam_dim_ref = map_struct["beam_dim"]
    aper_sup=config.get("photometry", "aper_sup")

    my_dist_limit_arcsec = beam_dim_ref if dist_limit_arcsec == 0 else dist_limit_arcsec
    dist_limit_pix = my_dist_limit_arcsec / pix_dim_ref

    filtered = high_pass_filter(image)
    norm_filtered = normalize_filtered_image(filtered)
        
    rms_detect = estimate_rms(norm_filtered)
    threshold = snr_threshold * rms_detect
        
    peaks = detect_peaks(norm_filtered, threshold, FWHM_pix, roundlim=roundlim, sharplim=sharplim)
    good_peaks = filter_peaks(peaks, FWHM_pix, image.shape, dist_limit_pix, aper_sup)
    final_sources = filter_by_snr(good_peaks, real_map, rms_real, snr_threshold)

    return final_sources
