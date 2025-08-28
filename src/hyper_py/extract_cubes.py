def extract_maps_from_cube(cube_names, dir_slices_out, dir_maps_in):
    """
    Extract 2D slices from 3D datacubes and return list of 2D FITS file paths.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    import os


    extracted_maps = []

    for cube_name in cube_names:
        cube_path = os.path.join(dir_maps_in, cube_name)
        with fits.open(cube_path) as hdul:
            data = hdul[0].data
            cube_header = hdul[0].header
            wcs = WCS(cube_header)

            if data.ndim != 3:
                raise ValueError(f"{cube_name} is not a 3D datacube.")

            # Ensure output directory exists
            os.makedirs(dir_slices_out, exist_ok=True)

            for i in range(data.shape[0]):
                slice_data = data[i, :, :]
                slice_header = cube_header.copy()
                slice_header['CRPIX3'] = i + 1  # track the slice index
                slice_header['NAXIS'] = 2  # force 2D output
                for key in list(slice_header.keys()):
                    if key.startswith('NAXIS') and key != 'NAXIS1' and key != 'NAXIS2':
                        del slice_header[key]
                    if key.startswith('CRVAL3') or key.startswith('CDELT3') or key.startswith('CTYPE3'):
                        del slice_header[key]

                out_name = f"{os.path.splitext(cube_name)[0]}_slice_{i+1:03d}.fits"
                out_path = os.path.join(dir_slices_out, out_name)

                fits.writeto(out_path, slice_data, slice_header, overwrite=True)
                extracted_maps.append(out_name)

    return extracted_maps, cube_header