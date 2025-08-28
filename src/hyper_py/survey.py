from astropy.io import fits
import numpy as np
from astropy import units as u

# Static beam info per survey_code  (with conversion micron -> GHz)

BEAM_INFO = {
    1: {'band': (70*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 5.2, 'area': None}, 
    2: {'band': (100*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 6.8, 'area': None},
    3: {'band': (160*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 12.0, 'area': None},
    4: {'band': (250*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 18.0, 'area': 423},
    5: {'band': (350*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 24.0, 'area': 751},
    6: {'band': (500*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 34.5, 'area': 1587},
    7: {'band': (450*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 8.0, 'area': None},
    8: {'band': (850*u.micron).to(u.GHz, equivalencies=u.spectral()), 'beam': 14.5, 'area': None}, 
    15: 'DYNAMIC',
}


arcsec2_to_sr = 2.35045e-11

def get_beam_info(survey_code, fits_file=None):
    beam_arcsec = []
    beam_area_arcsec2 = []
    beam_area_sr = []
    band_ref = []

    entry = BEAM_INFO.get(survey_code)

    if entry == 'DYNAMIC':
        if not fits_file:
            raise ValueError("FITS file required for ALMA Band 6 beam estimation.")
        with fits.open(fits_file) as hdul:
            hdr = hdul[0].header
            bmin = hdr.get('BMIN', 0) * 3600  # degrees → arcsec
            bmaj = hdr.get('BMAJ', 0) * 3600
            beam = np.sqrt(bmin * bmaj)
            
            beam_area_arcsec2 = 1.1331 * bmin * bmaj
            beam_area_sr = beam_area_arcsec2 * arcsec2_to_sr
            
            
            # area = 1.133 * bmin * bmaj *(3600 * np.pi / 180) ** 2 / SR_TO_ARCSEC2
    elif isinstance(entry, dict):
        with fits.open(fits_file) as hdul: hdr = hdul[0].header
        beam = np.mean(entry['beam']) if isinstance(entry['beam'], list) else entry['beam']
        beam_area_arcsec2 = entry['area']
        if beam_area_arcsec2 is None:
            beam_area_arcsec2 = 1.1331 * beam ** 2 
            beam_area_sr = beam_area_arcsec2 * arcsec2_to_sr
    else:
        raise ValueError(f"Survey code {survey_code} not recognized.")

    beam_arcsec = beam
    beam_area_arcsec2 = beam_area_arcsec2
    beam_area_sr = beam_area_sr
    
    
    # --- Search for frequency axis (CTYPEi == 'FREQ') --- #
    band_ref = None
    
    for i in range(1, 6):  # Loop over axis 1 to 5 in case of multi-dimensional FITS
        ctype = hdr.get(f'CTYPE{i}', '')
        if 'FREQ' in ctype.upper():
            crval = hdr.get(f'CRVAL{i}', 0.0)
            band_ref = crval * 1e-9  # Convert from Hz to GHz
            break
           
            
    return beam_arcsec, beam_area_arcsec2, beam_area_sr