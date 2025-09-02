import sys
import os
import warnings
import shutil
import platform
from pathlib import Path

# Silence some known irrelevant warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
warnings.filterwarnings("ignore", message=".*Set OBSGEO-L to.*")
warnings.filterwarnings("ignore", message=".*Wrapping comment lines > 78 characters.*")
warnings.filterwarnings("ignore", message=".*more axes \\(4\\) than the image it is associated with \\(2\\).*")
warnings.filterwarnings("ignore", message=".*Set MJD-OBS to.*")

# Import the main entry point of the package
from hyper_py.hyper import start_hyper

# importlib.resources for packaged data (fallback to importlib_resources on older Python)
try:
    from importlib.resources import files as ir_files
except Exception:  # Python <3.9
    from importlib_resources import files as ir_files  # type: ignore

# Try ruamel.yaml for comment-preserving roundtrips
try:
    from ruamel.yaml import YAML  # type: ignore
    _HAS_RUAMEL = True
except Exception:
    _HAS_RUAMEL = False

_PKG = "hyper_py"  # top-level package name
_CONFIG_FILENAME = "hyper_config.yaml"  # custom config filename


def _user_config_path() -> Path:
    """Return the user-level config path depending on the operating system."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "HyperPy" / _CONFIG_FILENAME
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "hyper-py" / _CONFIG_FILENAME


def _load_default_template_path() -> Path:
    """
    Locate the packaged default config template.
    Fallback to local files when running directly from src (dev mode).
    """
    try:
        res = ir_files(_PKG).joinpath("assets/default_config.yaml")
        if res and os.path.isfile(str(res)):
            return Path(str(res))
    except Exception:
        pass

    # Fallbacks for development mode
    here = Path(__file__).parent
    for cand in (
        here / "assets" / "default_config.yaml",
        here / "config.yaml",
        here.parent / "config.yaml",
    ):
        if cand.is_file():
            return cand

    print("Error: default config template not found in package or src.")
    sys.exit(1)


def _ensure_parent(p: Path) -> None:
    """Ensure the parent directory exists."""
    p.parent.mkdir(parents=True, exist_ok=True)


def _copy_template_to(dest: Path) -> None:
    """Copy the default template into the destination config file."""
    src = _load_default_template_path()
    _ensure_parent(dest)
    with src.open("rb") as s, dest.open("wb") as d:
        shutil.copyfileobj(s, d)


def _resolve_config_path(argv: list[str]) -> Path:
    """Resolve the path of the configuration file from CLI args or default locations."""
    # 1) CLI argument
    if len(argv) > 1:
        return Path(argv[1]).expanduser().resolve()
    # 2) config file in the current working directory
    cwd_cfg = Path.cwd() / _CONFIG_FILENAME
    if cwd_cfg.is_file():
        return cwd_cfg
    # 3) user config path
    ucfg = _user_config_path()
    if ucfg.is_file():
        return ucfg
    # 4) if nothing found, suggest CWD/hyper_config.yaml as creation target
    return cwd_cfg


def _update_dir_root_preserving_comments(config_path: Path, new_dir_root: Path) -> None:
    """
    Update paths.output.dir_root preserving comments and formatting using ruamel.yaml.
    If ruamel.yaml is not available, do nothing (keep template intact and let the user edit).
    """
    if not _HAS_RUAMEL:
        print("Note: ruamel.yaml not installed; left config comments intact. "
              "Please edit 'paths.output.dir_root' manually in hyper_config.yaml.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f) or {}

    # Ensure nested mapping exists without destroying comments
    paths = data.get("paths")
    if paths is None:
        from ruamel.yaml.comments import CommentedMap  # type: ignore
        paths = CommentedMap()
        data["paths"] = paths

    output = paths.get("output")
    if output is None:
        from ruamel.yaml.comments import CommentedMap  # type: ignore
        output = CommentedMap()
        paths["output"] = output

    output["dir_root"] = str(new_dir_root / "output")

    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def main() -> None:
    """Main entry point for CLI execution."""
    cfg_path = _resolve_config_path(sys.argv)

    if not cfg_path.exists():
        # Create new hyper_config.yaml from packaged template
        _copy_template_to(cfg_path)
        # Update dir_root while preserving comments (if ruamel.yaml is available)
        _update_dir_root_preserving_comments(cfg_path, Path.cwd())
        print(f"⚠️  New {_CONFIG_FILENAME} created at: {cfg_path}")
        print("⚠️  Please edit it and run the command again.")
        sys.exit(0)

    # Run the real application
    start_hyper(str(cfg_path))


if __name__ == "__main__":
    main()
