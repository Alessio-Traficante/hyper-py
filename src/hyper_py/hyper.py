import os
import sys
from pathlib import Path
# from parallel_utils import process_single_map

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from hyper_py.single_map import main as single_map
from hyper_py.config import HyperConfig

from hyper_py.logger import setup_logger

from concurrent.futures import ProcessPoolExecutor, as_completed

from astropy.io import ascii, fits
from astropy.table import vstack

import numpy as np
from extract_cubes import extract_maps_from_cube


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
    dir_comm = paths["dir_comm"]
    
    # # - input - #
    dir_maps = paths["input"]["dir_maps"]
    base_table_name = cfg.get("files", "file_table_base")
    map_names = cfg.get("files", "file_map_name")
    datacube = cfg.get("control", "datacube", False)

    
    # If it's a path to a .txt file, read it #
    if isinstance(map_names, str) and map_names.endswith('.txt'):
        map_list_path = os.path.join(dir_comm, dir_maps, map_names)
        with open(map_list_path, 'r') as f:
            map_names = [line.strip() for line in f if line.strip()]
    # If it's a single string but not a .txt, wrap it in a list
    elif isinstance(map_names, str):
        map_names = [map_names]
        
    if datacube:
        map_names = extract_maps_from_cube(map_names, dir_comm, dir_maps)
        background_slices = []
        slice_cutout_header = []
    
        
    # - output - #
    output_dir = paths["output"]["dir_table_out"]
    
    
    # --- Set up logging for warnings --- #
    dir_log = paths["output"]["dir_log_out"]
    file_log = cfg.get("files", "file_log_name")
    log_path = os.path.join(dir_comm, dir_log, file_log)
    
    # Ensure the log directory exists
    log_path_dir = os.path.join(dir_comm, dir_log)
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
                executor.submit(single_map, name, cfg, dir_comm): name
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
            suffix, bg_model, cutout_header, initial_header = single_map(map_name, cfg, dir_comm, logger, logger_file_only)
            results.append(suffix)
            if datacube:
                background_slices.append(bg_model)
                slice_cutout_header.append(cutout_header)
            
    
                            
    # --- Collect all output tables --- #
    all_tables = []
    for suffix in results:
        try:
            suffix_clean = Path(suffix).stem  # remove ".fits"
            output_table_path = os.path.join(dir_comm, output_dir, f"{base_table_name}_{suffix_clean}.txt")
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
    ipac_path = os.path.join(dir_comm, output_dir, f"{base_table_name}_ALL.txt")
    csv_path = os.path.join(dir_comm, output_dir, f"{base_table_name}_ALL.csv")
    
    # Write outputs
    final_table.write(ipac_path, format='ipac', overwrite=True)
    final_table.write(csv_path, format='csv', overwrite=True)
    logger_file_only.info(f"\n✅ Final merged table saved to:\n- {ipac_path}\n- {csv_path}")
    
    
    
    # === Combine all bg_models into a datacube ===
    if datacube:
        # 1. Determine common crop size
        min_ny = min(bg.shape[0] for bg in background_slices)
        min_nx = min(bg.shape[1] for bg in background_slices)
    
        def central_crop(array, ny, nx):
            y0 = (array.shape[0] - ny) // 2
            x0 = (array.shape[1] - nx) // 2
            return array[y0:y0+ny, x0:x0+nx]
    
        # 2. Centrally crop all backgrounds
        cropped_bgs = [central_crop(bg, min_ny, min_nx) for bg in background_slices]
    
        # 3. Stack into cube
        bg_cube = np.stack(cropped_bgs, axis=0)
    
        # 4. Adjust WCS header
        ref_header = initial_header.copy()
        
        # Compute x/y shifts for CRPIX update (from center-crop)
        old_ny, old_nx = background_slices[0].shape
        dy = (old_ny - min_ny) // 2
        dx = (old_nx - min_nx) // 2
        
        ref_header['CRPIX1'] -= dx
        ref_header['CRPIX2'] -= dy


        # ref_header['NAXIS3'] = bg_cube.shape[0]
        ref_header['CTYPE3'] = 'SLICE'
        ref_header['BUNIT'] = 'mJy/pixel'
    
         
        
        # Update the axes explicitly before creating the HDU
        ref_header['NAXIS'] = 3
        ref_header['NAXIS1'] = bg_cube.shape[2]  # nx
        ref_header['NAXIS2'] = bg_cube.shape[1]  # ny
        ref_header['NAXIS3'] = bg_cube.shape[0]  # nz

        # Clean structural keywords if they already exist elsewhere
        for key in ['SIMPLE', 'BITPIX', 'EXTEND']:
            if key in ref_header:
                del ref_header[key]

    
        output_cube_path = os.path.join(dir_comm, dir_maps, "combined_background_cube.fits")
        fits.PrimaryHDU(data=bg_cube, header=ref_header).writeto(output_cube_path, overwrite=True)
        logger.info(f"📦 Background cube saved to: {output_cube_path}")
    
    logger.info("****************** ✅ Hyper finished !!! ******************")
    
    
