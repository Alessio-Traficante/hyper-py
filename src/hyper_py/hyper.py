import os
import sys
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import ascii, fits
from astropy.table import vstack
from astropy.wcs import WCS

from hyper_py.single_map import main as single_map
from hyper_py.config import HyperConfig
from hyper_py.logger import setup_logger
from .extract_cubes import extract_maps_from_cube

# Set multiprocessing start method
multiprocessing.set_start_method("spawn", force=True)

def run_hyper(cfg_path):
    # === Load config ===
    os.chdir(os.path.dirname(__file__))

    config_path = cfg_path if not None else "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = HyperConfig(config_path)

    # --- Initialize paths --- #
    # - common - #
    paths = cfg.get("paths")
    dir_root = paths["output"]["dir_root"]
    
    # # - input - #
    dir_maps = paths["input"]["dir_maps"]
    dir_slices_out = Path(dir_root, cfg.get("control")["dir_datacube_slices"])
    base_table_name = cfg.get("files", "file_table_base")
    map_names = cfg.get("files", "file_map_name")
    datacube = cfg.get("control", "datacube", False)
    fix_min_box = cfg.get("background", "fix_min_box", 3)     # minimum padding value (multiple of FWHM)
    convert_mjy=cfg.get("units", "convert_mJy")
    
    # If it's a path to a .txt file, read it #
    if isinstance(map_names, str) and map_names.endswith('.txt'):
        map_list_path = os.path.join(dir_maps, map_names)
        with open(map_list_path, 'r') as f:
            map_names = [line.strip() for line in f if line.strip()]
    # If it's a single string but not a .txt, wrap it in a list
    elif isinstance(map_names, str):
        map_names = [map_names]
        
    if datacube:
        map_names, cube_header = extract_maps_from_cube(map_names, dir_root, dir_maps)
        background_slices = []
        slice_cutout_header = []
    
    # - output - #
    output_dir = paths["output"]["dir_table_out"]
    
    # --- Set up logging for warnings --- #
    dir_log = paths["output"]["dir_log_out"]
    file_log = cfg.get("files", "file_log_name")
    log_path = os.path.join(dir_root, dir_log, file_log)
    
    # Ensure the log directory exists
    log_path_dir = os.path.join(dir_root, dir_log)
    os.makedirs(log_path_dir, exist_ok=True)

    logger, logger_file_only = setup_logger(log_path, logger_name="HyperLogger", overwrite=True)
    
    logger.info("******************* 🔥 Hyper starts !!! *******************")
    
    # --- Parallel control ---
    control_cfg = cfg.get("control", {})
    use_parallel = control_cfg.get("parallel_maps", False)
    n_cores = control_cfg.get("n_cores", os.cpu_count())

    # --- Main parallel or serial execution ---
    logger.info(f"🔄 Starting map analysis using {'multiprocessing' if use_parallel else 'serial'} mode")
    
    results = []

    if use_parallel:
        logger.info(f"📡 Running HYPER on {len(map_names)} maps using {n_cores} cores...")
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(single_map, name, cfg, dir_root): name
                for name in map_names
            }
            for future in as_completed(futures):
                map_name = futures[future]
                try:
                    suffix, bg_model, cutout_header, initial_header = future.result()
                    results.append(suffix)
                    if datacube:
                        background_slices.append(bg_model)
                        slice_cutout_header.append(cutout_header)
                    
                    logger.info(f"✅ Finished processing {map_name}")
                except Exception as e:
                    logger.error(f"❌ Error processing {map_name}: {e}")
    else:
        for map_name in map_names:
            logger.info(f"📡 Running HYPER on: {map_name}")
            suffix, bg_model, cutout_header, initial_header = single_map(map_name, cfg, dir_root, logger, logger_file_only)
            results.append(suffix)
            if datacube:
                background_slices.append(bg_model)
                slice_cutout_header.append(cutout_header)
                
                            
    # --- Collect all output tables --- #
    all_tables = []
    for suffix in results:
        try:
            suffix_clean = Path(suffix).stem  # remove ".fits"
            output_table_path = os.path.join(dir_root, output_dir, f"{base_table_name}_{suffix_clean}.txt")
            table = ascii.read(output_table_path, format="ipac")
            all_tables.append(table)
        except Exception as e:
            logger_file_only.error(f"[ERROR] Failed to load table for {suffix}: {e}")
    

    # === Merge and write combined tables ===
    final_table = vstack(all_tables)
    
    # Keep only the comments (headers) from the first table
    if hasattr(all_tables[0], 'meta') and 'comments' in all_tables[0].meta:
        final_table.meta['comments'] = all_tables[0].meta['comments']
    else:
        final_table.meta['comments'] = []
    
    # Output file paths
    ipac_path = os.path.join(dir_root, output_dir, f"{base_table_name}_ALL.txt")
    csv_path = os.path.join(dir_root, output_dir, f"{base_table_name}_ALL.csv")
    
    # Write outputs
    final_table.write(ipac_path, format='ipac', overwrite=True)
    final_table.write(csv_path, format='csv', overwrite=True)
    logger_file_only.info(f"\n✅ Final merged table saved to:\n- {ipac_path}\n- {csv_path}")
    
    # === Combine all bg_models into a datacube ===
    if datacube:
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

    
        output_cube_path = os.path.join(dir_root, dir_maps, "background_cube_cut.fits")
        fits.PrimaryHDU(data=bg_cube, header=new_header).writeto(output_cube_path, overwrite=True)
        logger.info(f"📦 Background cube saved to: {output_cube_path}")
    
    
    
        # === Also create a full-size cube with padded background slices if cropped size is != original size (fix_min_box != 0) === #
        # if fix_min_box != 0:
            # full_ny = cube_header['NAXIS2']
            # full_nx = cube_header['NAXIS1']
           
            # padded_bgs = []
            # for cropped in cropped_bgs:
            #     padded = np.full((full_ny, full_nx), np.nan, dtype=float)
            #     cy, cx = cropped.shape
                                
            #     y0 = (full_ny - cy) // 2
            #     x0 = (full_nx - cx) // 2
            #     padded[y0:y0+cy, x0:x0+cx] = cropped
            #     padded_bgs.append(padded)
           
            # # Stack into padded cube
            # bg_cube_full = np.stack(padded_bgs, axis=0)
           
            
            # # Adjust header: shift CRPIX relative to new_header (cropped)
            # padded_header = new_header.copy()
            # dx = (full_nx - new_header['NAXIS1']) / 2.0
            # dy = (full_ny - new_header['NAXIS2']) / 2.0
            # padded_header['CRPIX1'] += dx
            # padded_header['CRPIX2'] += dy
            
            # padded_header['NAXIS1'] = full_nx
            # padded_header['NAXIS2'] = full_ny
            # padded_header['NAXIS3'] = bg_cube_full.shape[0]
            
            # # update units header 
            # if convert_mjy:
            #     padded_header['BUNIT'] = 'mJy'
            # else:
            #     padded_header['BUNIT'] = 'Jy'
            
            # # Save full-size cube
            #  output_full_cube = os.path.join(dir_root, dir_maps, "background_cube_fullsize.fits")
            #  fits.PrimaryHDU(data=bg_cube_full, header=padded_header).writeto(output_full_cube, overwrite=True)
            #  logger.info(f"📦 Full-size padded background cube saved to: {output_full_cube}")
  
            
            # === Also create a full-size cube with background slices placed in original WCS positions === #
        # Get WCS of the original full cube (2D spatial only!)
        wcs_full = WCS(cube_header, naxis=2)
        
        xcen_all = []
        ycen_all = []
        
        for hdr in slice_cutout_header:
            # Build 2D WCS for the cropped cutout            
            ny, nx = cropped_bgs[0].shape

            
            # Compute pixel coordinates of the center of the cutout
            x_c = nx / 2.0
            y_c = ny / 2.0
        
            # Convert center pixel of cutout to world coordinates
            wcs_cutout = WCS(hdr, naxis=2)
            skycoord = wcs_cutout.pixel_to_world(x_c, y_c)
        
            # Convert world coordinates to pixel in full cube
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
        
                # Absolute position of the source in the full map
                xcen_full = xcen_all[i]
                ycen_full = ycen_all[i]
        
                # Compute top-left corner of where to place cropped cutout
                x0 = int(round(xcen_full - cx // 2))
                y0 = int(round(ycen_full - cy // 2))
        
                # Bounds check
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(x0 + cx, full_nx)
                y1 = min(y0 + cy, full_ny)
        
                # Matching region in cropped cutout
                sub = cropped[0:y1 - y0, 0:x1 - x0]
        
                # Insert cropped cutout in correct position
                padded[y0:y1, x0:x1] = sub
                padded_bgs.append(padded)
        
            # Stack into full cube
            bg_cube_full = np.stack(padded_bgs, axis=0)
        
            # Use original cube header (not from cropped version)
            padded_header = cube_header.copy()
            padded_header['NAXIS1'] = full_nx
            padded_header['NAXIS2'] = full_ny
            padded_header['NAXIS3'] = bg_cube_full.shape[0]
            padded_header['WCSAXES'] = max(padded_header.get('WCSAXES', 3), 3)
        
            # Update units
            padded_header['BUNIT'] = 'mJy' if convert_mjy else 'Jy'
        
            # Clean redundant axis-specific keys if present
            for ax in [4, 5]:
                for prefix in ['CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
                    key = f"{prefix}{ax}"
                    if key in padded_header:
                        del padded_header[key]
        
            # Save full cube
            output_cube_full_path = os.path.join(dir_root, dir_maps, "background_cube_fullsize.fits")
            fits.PrimaryHDU(data=bg_cube_full, header=padded_header).writeto(output_cube_full_path, overwrite=True)
            logger.info(f"📦 Full-size background cube saved to: {output_cube_full_path}")

    
    logger.info("****************** ✅ Hyper finished !!! ******************")