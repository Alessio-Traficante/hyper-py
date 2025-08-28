def create_background_cubes(background_slices, slice_cutout_header, cube_header, dir_slices_out, fix_min_box, convert_mjy, logger):
    
    import os
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    
    # 1. Determine common crop size
    all_shapes = [bg.shape for bg in background_slices]
    ny_list = [s[0] for s in all_shapes]
    nx_list = [s[1] for s in all_shapes]
    min_ny = min(ny_list)
    min_nx = min(nx_list)
    
    # 2. Find index of slice matching both min_ny and min_nx
    matching_index = None
    for i, (ny, nx) in enumerate(all_shapes):
        if ny == min_ny and nx == min_nx:
            matching_index = i
            break
    
    # 3. If no exact match, find best fit (one axis matches)
    if matching_index is None:
        for i, (ny, nx) in enumerate(all_shapes):
            if ny == min_ny or nx == min_nx:
                matching_index = i
                break
    
    # If still None (should not happen), fallback to first
    if matching_index is None:
        matching_index = 0
    
    # 4. Use that slice's header
    cropped_header = slice_cutout_header[matching_index].copy()
    
    # 5. Define crop with optional NaN padding
    def central_crop_or_pad(array, target_ny, target_nx):
        ny, nx = array.shape
        if ny == target_ny and nx == target_nx:
            return array
        else:
            cropped = np.full((target_ny, target_nx), np.nan, dtype=array.dtype)
            y0 = (ny - target_ny) // 2
            x0 = (nx - target_nx) // 2
            y1 = y0 + target_ny
            x1 = x0 + target_nx
            # Clip to valid range
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(ny, y1)
            x1 = min(nx, x1)
            sub = array[y0:y1, x0:x1]
    
            # Paste subarray into center of padded frame
            sy, sx = sub.shape
            start_y = (target_ny - sy) // 2
            start_x = (target_nx - sx) // 2
            cropped[start_y:start_y+sy, start_x:start_x+sx] = sub
            return cropped
    
    # 6. Centrally crop or pad all backgrounds to (min_ny, min_nx)
    cropped_bgs = [central_crop_or_pad(bg, min_ny, min_nx) for bg in background_slices]
    
    # 7. Stack into cube
    bg_cube = np.stack(cropped_bgs, axis=0)
    
    # 8. Adjust WCS header (preserve original logic)
    new_header = cube_header.copy()
        
    # 9. Update spatial WCS keywords (X and Y axes) from the cropped header
    spatial_keys = [
        'NAXIS1', 'NAXIS2',
        'CRPIX1', 'CRPIX2',
        'CRVAL1', 'CRVAL2',
        'CDELT1', 'CDELT2',
        'CTYPE1', 'CTYPE2',
        'CUNIT1', 'CUNIT2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
        'CROTA1', 'CROTA2'
    ]
    
    for key in spatial_keys:
        if key in cropped_header:
            new_header[key] = cropped_header[key]
    
    # 10. Update full shape to match the background cube
    new_header['NAXIS'] = 3
    new_header['NAXIS1'] = bg_cube.shape[2]  # X axis
    new_header['NAXIS2'] = bg_cube.shape[1]  # Y axis
    new_header['NAXIS3'] = bg_cube.shape[0]  # Z axis
    
    # 11. Ensure WCSAXES is at least 3
    new_header['WCSAXES'] = max(new_header.get('WCSAXES', 3), 3)
    
    # update units header 
    if convert_mjy:
        new_header['BUNIT'] = 'mJy'
    else:
        new_header['BUNIT'] = 'Jy'
    
    # Optional: clean inconsistent axis-specific keys (e.g., if 4D originally)
    for ax in [4, 5]:
        for prefix in ['CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
            key = f"{prefix}{ax}"
            if key in new_header:
                del new_header[key]

    output_cube_path = os.path.join(dir_slices_out, "background_cube_cut.fits")
    fits.PrimaryHDU(data=bg_cube, header=new_header).writeto(output_cube_path, overwrite=True)
    logger.info(f"📦 Background cube saved to: {output_cube_path}")

    # === Also create a full-size cube with padded background slices if cropped size is != original size (fix_min_box != 0) === #
    wcs_full = WCS(cube_header, naxis=2)
    xcen_all = []
    ycen_all = []
    
    for hdr in slice_cutout_header:
        ny, nx = cropped_bgs[0].shape
        x_c = nx / 2.0
        y_c = ny / 2.0
        wcs_cutout = WCS(hdr, naxis=2)
        skycoord = wcs_cutout.pixel_to_world(x_c, y_c)
        x_pix, y_pix = wcs_full.world_to_pixel(skycoord)
        xcen_all.append(x_pix)
        ycen_all.append(y_pix)

    if fix_min_box != 0:
        full_ny = cube_header['NAXIS2']
        full_nx = cube_header['NAXIS1']
        padded_bgs = []
        for i, cropped in enumerate(cropped_bgs):
            padded = np.full((full_ny, full_nx), np.nan, dtype=float)
            cy, cx = cropped.shape
            xcen_full = xcen_all[i]
            ycen_full = ycen_all[i]
            x0 = int(round(xcen_full - cx // 2))
            y0 = int(round(ycen_full - cy // 2))
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(x0 + cx, full_nx)
            y1 = min(y0 + cy, full_ny)
            sub = cropped[0:y1 - y0, 0:x1 - x0]
            padded[y0:y1, x0:x1] = sub
            padded_bgs.append(padded)
        bg_cube_full = np.stack(padded_bgs, axis=0)
        padded_header = cube_header.copy()
        padded_header['NAXIS1'] = full_nx
        padded_header['NAXIS2'] = full_ny
        padded_header['NAXIS3'] = bg_cube_full.shape[0]
        padded_header['WCSAXES'] = max(padded_header.get('WCSAXES', 3), 3)
        padded_header['BUNIT'] = 'mJy' if convert_mjy else 'Jy'
        for ax in [4, 5]:
            for prefix in ['CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
                key = f"{prefix}{ax}"
                if key in padded_header:
                    del padded_header[key]
        output_cube_full_path = os.path.join(dir_slices_out, "background_cube_fullsize.fits")
        fits.PrimaryHDU(data=bg_cube_full, header=padded_header).writeto(output_cube_full_path, overwrite=True)
        logger.info(f"📦 Full-size background cube saved to: {output_cube_full_path}")
