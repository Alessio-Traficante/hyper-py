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

from astropy.io import ascii
from astropy.table import vstack


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
    
    # If it's a path to a .txt file, read it #
    if isinstance(map_names, str) and map_names.endswith('.txt'):
        map_list_path = os.path.join(dir_comm, dir_maps, map_names)
        with open(map_list_path, 'r') as f:
            map_names = [line.strip() for line in f if line.strip()]
    # If it's a single string but not a .txt, wrap it in a list
    elif isinstance(map_names, str):
        map_names = [map_names]
        
    # - output - #
    output_dir = paths["output"]["dir_table_out"]
    
    
    # --- Set up logging for warnings --- #
    dir_log = paths["output"]["dir_log_out"]
    file_log = cfg.get("files", "file_log_name")
    log_path = os.path.join(dir_comm, dir_log, file_log)
    
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
                    table = future.result()
                    results.append((map_name))
                    logger.info(f"✅ Finished processing {map_name}")
                except Exception as e:
                    logger.error(f"❌ Error processing {map_name}: {e}")
    else:
        for map_name in map_names:
            logger.info(f"📡 Running HYPER on: {map_name}")
            single_map(map_name, cfg, dir_comm, logger, logger_file_only)
            results.append((map_name))
            
                            
    # --- Collect all output tables --- #
    all_tables = []
    for map_name in results:
        try:
            suffix = Path(map_name).stem
            output_table_path = os.path.join(dir_comm, output_dir, f"{base_table_name}_{suffix}.txt")
            table = ascii.read(output_table_path, format="ipac")
            all_tables.append(table)
        except Exception as e:
            logger_file_only.error(f"[ERROR] Failed to load table for {map_name}: {e}")
                    
    
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
    
    logger.info("****************** ✅ Hyper finished !!! ******************")
    
    
