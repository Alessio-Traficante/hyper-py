from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


def read_and_prepare_map(filepath, beam, beam_area_arcsec2, beam_area_sr, convert_mjy=False):
    '''
    Load a FITS map and convert units as needed. 
    Always reduce WCS to 2D (RA, Dec) if higher dimensional axes are present.

    Parameters:
        filepath (str): Path to the FITS file.
        beam (float): Beam size in arcsec (FWHM).
        beam_area_arcsec2 (float): Beam area in arcsec^2.
        convert_jy_sr (bool): If True, convert from MJy/sr to Jy/pixel.
        convert_beam_jy (bool): If True, convert from Jy/beam to Jy/pixel.
        convert_mjy (bool): If True, convert Jy to mJy/pixel.

    Returns:
        dict: {
            'map': 2D image array,
            'header': cleaned FITS header,
            'pix_dim': pixel size in arcsec,
            'beam_dim': beam (arcsec),
            'band': extracted from header or placeholder
        }
    '''
    with fits.open(filepath) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header

    # --- Auto-squeeze down to 2D image ---
    while image_data.ndim > 2:
        image_data = image_data[0]


    # --- Get pixel scale from header ---
    pix_dim = abs(header.get('CDELT1', header.get('CD1_1', 1))) * 3600.0  # arcsec
    if pix_dim <= 0:
        raise ValueError("Invalid pixel scale in FITS header.")
        

    # --- Strip header down to 2D WCS only ---
    try:
        wcs_full = WCS(header)
        wcs_2d = wcs_full.celestial  # Only RA/DEC axes
        header = wcs_2d.to_header()
    except Exception as e:
        print(f"[WARNING] Could not clean WCS header: {e}")


    # --- Unit conversions ---
    bunit = header.get('BUNIT')
    if bunit == 'MJy /sr':
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        pix_area_sr = (pix_dim * arcsec_to_rad)**2
        image_data *= 1e6 * pix_area_sr  # MJy/sr to Jy/pixel
        
    if bunit == 'Jy/beam' or bunit == 'beam-1 Jy':    
        pix_area = pix_dim**2
        image_data /= (beam_area_arcsec2 / pix_area) # Jy/beam to Jy/pixel
        
    if convert_mjy:
        image_data *= 1e3  # Jy → mJy
 
    return {
        "map": image_data,
        "header": header,
        "pix_dim": pix_dim,
        "beam_dim": beam,
        "beam_area_arcsec2": beam_area_arcsec2,
        "beam_area_sr": beam_area_sr,
    }