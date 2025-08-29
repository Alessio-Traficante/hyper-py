import sys
import os
import shutil
import warnings
import yaml
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Block warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
warnings.filterwarnings("ignore", message=".*Set OBSGEO-L to.*")
warnings.filterwarnings("ignore", message=".*Wrapping comment lines > 78 characters.*")
warnings.filterwarnings("ignore", message=".*more axes \(4\) than the image it is associated with \(2\).*")
warnings.filterwarnings("ignore", message=".*Set MJD-OBS to.*")

from hyper_py.hyper import start_hyper

def update_dir_root(default_config, config_path, new_dir_root):
    """Create a new config.yaml in the specified directory with updated dir_root."""
    default_config = Path(default_config)
    config_path = Path(config_path)      
    cfg = yaml.safe_load(default_config.read_text(encoding="utf-8")) or {}
    cfg['paths']['output']['dir_root'] = str(Path(new_dir_root, "output"))
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_path):
        default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
        if not os.path.exists(default_config):
            print("Error: default config.yaml not found.")
            sys.exit(1)

        config_path = os.path.join(os.getcwd(), "config.yaml")

        update_dir_root(default_config, config_path, os.getcwd())
        print(f"⚠️  New config.yaml created in {config_path}")
        print("⚠️  Please edit the configuration file and set the correct parameters and paths before running again.")
        sys.exit(0)

    start_hyper(config_path)

if __name__ == "__main__":
    main()
