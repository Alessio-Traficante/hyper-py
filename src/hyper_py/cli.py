import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings

# Block warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
warnings.filterwarnings("ignore", message=".*Set OBSGEO-L to.*")
warnings.filterwarnings("ignore", message=".*Wrapping comment lines > 78 characters.*")
warnings.filterwarnings("ignore", message=".*more axes \(4\) than the image it is associated with \(2\).*")
warnings.filterwarnings("ignore", message=".*Set MJD-OBS to.*")

from hyper_py.hyper import run_hyper

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_hyper(config_path)

if __name__ == "__main__":
    main()
