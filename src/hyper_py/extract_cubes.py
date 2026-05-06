def extract_maps_from_cube(cube_names, dir_slices_out, dir_maps_in):
    """
    Extract 2D slices from 3D datacubes and return list of 2D FITS file paths.
    """
    from astropy.io import fits
    #import numpy as np
    import os

    extracted_maps = []
    # Ensure output directory exists
    os.makedirs(dir_slices_out, exist_ok=True)

    for cube_name in cube_names:
        cube_path = os.path.join(dir_maps_in, cube_name)

        with fits.open(cube_path, memmap=True) as hdul:
            data = hdul[0].data
            cube_header = hdul[0].header

            if data.ndim != 3:
                raise ValueError(f"{cube_name} is not a 3D datacube.")

            cards = [card for card in cube_header.cards if card.keyword != 'HISTORY']
            base_header = fits.Header(cards)
            base_header['NAXIS'] = 2

            keys_to_del = ['NAXIS3', 'CRVAL3', 'CDELT3', 'CTYPE3', 'CRPIX3', 'CUNIT3']
            for key in keys_to_del:
                if key in base_header:
                    del base_header[key]

            cube_base_name = os.path.splitext(cube_name)[0]


            for i in range(data.shape[0]):
                out_name = f"{cube_base_name}_slice_{i+1:03d}.fits"
                out_path = os.path.join(dir_slices_out, out_name)
                save_slice(i, data[i, :, :], base_header, out_path, out_name)
                extracted_maps.append(out_name)

            bmin = cube_header.get('BMIN', 0) * 3600  # degrees → arcsec
            bmaj = cube_header.get('BMAJ', 0) * 3600
            beam_area_arcsec2_datacubes = 1.1331 * bmin * bmaj

    return extracted_maps, cube_header, beam_area_arcsec2_datacubes

def save_slice(i, data_slice, base_header, out_path, out_name):

    from astropy.io import fits

    slice_header = base_header.copy()
    slice_header['CHAN_N'] = (i + 1, 'Original channel index')
    
    fits.writeto(out_path, data_slice, slice_header, overwrite=True)

    return out_name

