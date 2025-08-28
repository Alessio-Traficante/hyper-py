import os
from pathlib import Path


def get_hyper_paths(cfg):
    """
    Prepares and returns all relevant I/O paths used in a HYPER run.
    """  
    # - Main paths - #
    paths = cfg.get("paths")
    dir_root = paths["output"]["dir_root"]

    # - Input - #
    dir_maps = paths["input"]["dir_maps"]
    file_map = 'temp'
    input_map_path = os.path.join(dir_maps, file_map)

    # - Output - #
    suffix = Path(file_map).stem

    # - Tables - #
    dir_table_out = paths["output"]["dir_table_out"]
    output_dir_path = os.path.join(dir_root, dir_table_out)
    base_name_with_suffix = f"{cfg.get('files', 'file_table_base')}_{suffix}"

    # - Region files - #
    dir_region_out = paths["output"]["dir_region_out"]
    region_base = os.path.join(dir_root, dir_region_out, f"{cfg.get('files', 'file_region_base')}_{suffix}")
    centroids_file = region_base + "_centroids.reg"
    ellipses_file = region_base + "_ellipses.reg"

    # - Log files - #
    dir_log = paths["output"]["dir_log_out"]
    file_log = cfg.get("files", "file_log_name")
    log_path = os.path.join(dir_root, dir_log, file_log)


    return {
        "dir_root": dir_root,
        "input_map_path": input_map_path,
        "suffix": suffix,
        "output_dir_path": output_dir_path,
        "base_table_name": cfg.get("files", "file_table_base"),
        "base_name_with_suffix": base_name_with_suffix,
        "region_base": region_base,
        "centroids_file": centroids_file,
        "ellipses_file": ellipses_file,
        "log_path": log_path
    }



def get_hyper_single_map_paths(cfg, map_name):
    """
    Prepares and returns all relevant I/O paths used in a HYPER run.
    """  
    
    if cfg is None:
        raise ValueError("cfg must be provided")

    # - Main paths - #
    paths = cfg.get("paths")
    
    if paths is None:
        raise ValueError("Missing 'paths' section in configuration.")
    
    dir_root = paths["output"]["dir_root"]
    if dir_root is None:
        raise ValueError("Missing 'dir_root' key in paths configuration.")

    # - Input - #
    dir_maps = paths["input"]["dir_maps"]
    file_map = map_name
    input_map_path = os.path.join(dir_maps, file_map)

    # - Output - #
    suffix = Path(file_map).stem


    # - Tables - #
    dir_table_out = paths["output"]["dir_table_out"]
    output_dir_path = os.path.join(dir_root, dir_table_out)
    # Ensure the output directory exists
    os.makedirs(output_dir_path, exist_ok=True)


    base_name_with_suffix = f"{cfg.get('files', 'file_table_base')}_{suffix}"


    # - Region files - #
    dir_region_out = paths["output"]["dir_region_out"]
    # Ensure the output directory exists
    output_dir_region_path = os.path.join(dir_root, dir_region_out)
    os.makedirs(output_dir_region_path, exist_ok=True)

    region_base = os.path.join(dir_root, dir_region_out, f"{cfg.get('files', 'file_region_base')}_{suffix}")
    centroids_file = region_base + "_centroids.reg"
    ellipses_file = region_base + "_ellipses.reg"


    # - Log files - #
    dir_log = paths["output"]["dir_log_out"]
    # Ensure the output directory exists
    output_dir_log_path = os.path.join(dir_root, dir_table_out)
    os.makedirs(output_dir_log_path, exist_ok=True)
    
    file_log = cfg.get("files", "file_log_name")
    log_path = os.path.join(dir_root, dir_log, file_log)


    return {
        "dir_root": dir_root,
        "input_map_path": input_map_path,
        "suffix": suffix,
        "output_dir_path": output_dir_path,
        "base_table_name": cfg.get("files", "file_table_base"),
        "base_name_with_suffix": base_name_with_suffix,
        "region_base": region_base,
        "centroids_file": centroids_file,
        "ellipses_file": ellipses_file,
        "log_path": log_path
    }