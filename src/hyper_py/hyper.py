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
from hyper_py.create_background_slices import create_background_cubes
from .extract_cubes import extract_maps_from_cube

# Set multiprocessing start method
multiprocessing.set_start_method("spawn", force=True)

def start_hyper(cfg_path):
    # === Load config ===
    os.chdir(os.path.dirname(__file__))

    config_path = cfg_path if not None else "hyper_config.yaml"
    
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
        map_names, cube_header = extract_maps_from_cube(map_names, dir_slices_out, dir_maps)
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
        create_background_cubes(background_slices, slice_cutout_header, cube_header, dir_slices_out, fix_min_box, convert_mjy, logger)

    
    logger.info("****************** ✅ Hyper finished !!! ******************")