from __future__ import annotations
import sys
import os
import copy
from pathlib import Path
from typing import Dict, Any, List

try:
    import yaml
except Exception as ex:
    raise RuntimeError("PyYAML is required.") from ex


def find_package_config(pkg_dir: Path) -> Path:
    for name in ("config.yaml","config.yml","default_config.yaml","config_default.yaml","config_base.yaml"):
        p = pkg_dir / name
        if p.exists():
            return p
    for name in ("config.yaml","config.yml","default_config.yaml","config_default.yaml","config_base.yaml"):
        p = pkg_dir / "assets" / name 
        if p.exists():
            return p
    alts: List[Path] = list(pkg_dir.glob("*.yaml")) + list(pkg_dir.glob("*.yml"))
    if alts:
        return alts[0]
    raise FileNotFoundError(f"No configuration file found in '{pkg_dir}'.")


def read_list_file(list_path: Path) -> list[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    base = list_path.parent
    items: List[str] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = base / p
        items.append(str(p.resolve()))
    if not items:
        raise ValueError(f"No entries found in list file: {list_path}")
    return items


def ensure_dirs(paths: list[Path]) -> None:
    for d in paths:
        d.mkdir(parents=True, exist_ok=True)


def rewrite_paths_and_flags(cfg: Dict[str, Any], test_dir: Path, list_file: Path, maps: list[str], flags_true: bool, datacube: bool) -> Dict[str, Any]:
    if "paths" not in cfg or not isinstance(cfg["paths"], dict):
        raise ValueError("Config missing 'paths' section.")
    
    paths = cfg["paths"]
    path_maps = (test_dir / "maps").resolve()
    path_output = (test_dir / "output").resolve()

    folder_logs = "logs"
    folder_plots = "plots"
    folder_tables = "params"
    folder_regions = "regions"
    folder_fits = "fits"
    folder_fitting = "fitting"
    folder_deblended = "deblended"
    folder_bg_separate = "bg_separate"
    folder_maps_slices = "slices"

    ensure_dirs([path_output, path_output / folder_logs, path_output / folder_plots, path_output / folder_tables, path_output / folder_regions, path_output / folder_fits, 
                 path_output / folder_fits / folder_fitting, path_output / folder_fits / folder_deblended, path_output / folder_fits / folder_bg_separate, 
                 path_output / folder_plots / folder_fitting, path_output / folder_plots / folder_deblended, path_output / folder_plots / folder_bg_separate, 
                 path_output / folder_maps_slices])

    def set_if_present(d: Dict[str, Any], key: str, value: str) -> None:
        if key in d:
            d[key] = value

    pin: Dict[str, Any] = paths["input"]
    set_if_present(pin, "dir_maps", path_maps)

    pout: Dict[str, Any] = paths["output"]
    set_if_present(pout, "dir_root", path_output)
    set_if_present(pout, "dir_log_out", folder_logs)
    set_if_present(pout, "dir_table_out", folder_tables)
    set_if_present(pout, "dir_region_out", folder_regions)

    pfiles: Dict[str, Any] = cfg["files"]
    set_if_present(pfiles, "file_map_name", str(list_file))
    set_if_present(pfiles, "file_log_name", "hyper.log")

    pcube: Dict[str, Any] = cfg["control"]
    set_if_present(pcube, "datacube", bool(datacube))
    set_if_present(pcube, "dir_datacube_slices", str(Path(path_output, folder_maps_slices)))

    pfits: Dict[str, Any] = cfg["fits_output"]
    set_if_present(pfits, "fits_fitting", bool(flags_true))
    set_if_present(pfits, "fits_deblended", bool(flags_true))
    set_if_present(pfits, "fits_bg_separate", bool(flags_true))
    set_if_present(pfits, "fits_output_dir_fitting", str(Path(folder_fits, folder_fitting)))
    set_if_present(pfits, "fits_output_dir_deblended", str(Path(folder_fits, folder_deblended)))
    set_if_present(pfits, "fits_output_dir_bg_separate", str(Path(folder_fits, folder_bg_separate)))

    pplot: Dict[str, Any] = cfg["visualization"]
    set_if_present(pplot, "visualize_fitting", bool(flags_true))
    set_if_present(pplot, "visualize_deblended", bool(flags_true))
    set_if_present(pplot, "visualize_bg_separate", bool(flags_true))
    set_if_present(pplot, "output_dir_fitting", str((Path(folder_plots, folder_fitting))))
    set_if_present(pplot, "output_dir_deblended", str(Path(folder_plots, folder_deblended)))
    set_if_present(pplot, "output_dir_bg_separate", str(Path(folder_plots, folder_bg_separate)))

    return cfg


def paths_to_str(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [paths_to_str(i) for i in obj]
    else:
        return obj

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    test_dir = Path(__file__).resolve().parent
    maps_dir = test_dir / "maps"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        import hyper_py
        from hyper_py import start_hyper
    except Exception as ex:
        raise ImportError("Cannot import 'hyper_py'.") from ex

    pkg_dir = Path(hyper_py.__file__).resolve().parent
    base_cfg_path = find_package_config(pkg_dir)
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}

    list_2d = maps_dir / "test_2dmap.txt"
    maps_2d = read_list_file(list_2d)
    cfg_2d = rewrite_paths_and_flags(copy.deepcopy(base_cfg), test_dir, list_2d, maps_2d, flags_true=True, datacube=False)
    cfg_2d = paths_to_str(cfg_2d)
    cfg_2d_path = test_dir / "test_2dmap.config.yaml"
    cfg_2d_path.write_text(yaml.safe_dump(cfg_2d, sort_keys=False), encoding="utf-8")
    print(f"[hyper-py] Base config: {base_cfg_path}")
    print(f"[hyper-py] Test config: {cfg_2d_path}")
    print(f"[hyper-py] List: {list_2d}")
    for m in maps_2d:
        print(f"  - {m}")
    start_hyper(str(cfg_2d_path))

    list_cube = maps_dir / "test_datacube.txt"
    maps_cube = read_list_file(list_cube)
    cfg_cube = rewrite_paths_and_flags(copy.deepcopy(base_cfg), test_dir, list_cube, maps_cube, flags_true=False, datacube=True)
    cfg_cube = paths_to_str(cfg_cube)
    cfg_cube_path = test_dir / "test_datacube.config.yaml"
    cfg_cube_path.write_text(yaml.safe_dump(cfg_cube, sort_keys=False), encoding="utf-8")
    print(f"[hyper-py] Test config: {cfg_cube_path}")
    print(f"[hyper-py] List: {list_cube}")
    for m in maps_cube:
        print(f"  - {m}")
    start_hyper(str(cfg_cube_path))

    out_dir = test_dir / "output"
    produced = [p for p in out_dir.rglob("*") if p.is_file()]
    print(f"\n[hyper-py] Done. Produced {len(produced)} files under: {out_dir}")
    for p in sorted(produced):
        try:
            rel = p.relative_to(repo_root)
        except ValueError:
            rel = p
        print(f"  - {rel}")


if __name__ == "__main__":
    main()
