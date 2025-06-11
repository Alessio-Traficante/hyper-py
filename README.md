# 💫 HYPER: Hybrid Photometry Photometry and Extraction Routine

**Author:** Alessio Traficante  
**Original reference:** Traficante et al. (2015), *MNRAS, 451, 3089*  

---

## Overview
HYPER is a flexible and modular Python-based pipeline for performing accurate source extraction and elliptical aperture photometry on astronomical maps. It is designed to reproduce and improve the performance of the original IDL-based HYPER algorithm introduced in Traficante et al. (2015).

The core objective of HYPER is to combine Gaussian fitting and polynomial background estimation to extract reliable fluxes for compact sources, especially in the presence of blending and spatially variable backgrounds.

---

## Philosophy
- Perform **aperture photometry** using source-dependent elliptical apertures derived from Gaussian fits
- Use a **polynomial background model**, estimated and optionally subtracted either jointly or separately from source fitting
- Handle both **isolated and blended sources**, using multi-Gaussian fitting for groups
- Offer high configurability through a YAML-based configuration file
- Provide robust visual diagnostics and clean output formats (e.g. IPAC tables, DS9 region files)

---

## Workflow Summary
1. **Input maps loading**
2. **Source detection** with configurable filters and DAOStarFinder
3. **Grouping** of nearby sources for joint fitting
4. **Background estimation** (optional, fixed or fitted)
5. **2D Gaussian fitting** with background polynomial (multi-source or isolated)
6. **Aperture photometry** using elliptical regions derived from fit parameters
7. **Output** generation: flux table, region files, diagnostics plots

---

## Parallel Processing

HYPER now supports **parallel execution** over multiple maps. If a list of FITS files is provided, HYPER will automatically:

- Launch one independent process per map (up to the number of available CPU cores)
- Run the full pipeline (detection, fitting, photometry) in parallel across different maps
- Maintain **individual log files** for each map
- Merge the final outputs (tables and diagnostics) into a single, combined summary

To enable parallelism, set the following parameters in your `config.yaml` file under the `control` section:

```yaml
control:
  parallel_maps: true      # Enable parallel execution across maps
  n_cores: 4               # Number of CPU cores to use
```

If `parallel_maps` is set to `false`, the pipeline will run in serial mode.

---

## 🚀 Prerequisites

Before using `hyper_py`, it is highly recommended to create a dedicated Python environment (using `conda`, `venv`, or another environment manager). This helps avoid dependency conflicts and ensures consistent behavior.

### Recommended Environment Setup

HYPER requires the following packages:

- Python >= 3.10
- `numpy`
- `astropy`
- `photutils`
- `lmfit` >= 1.3.3
- `matplotlib`

You can install them using `conda` or `pip`, depending on your setup.

### Option 1: Create a Conda Environment (Recommended)
```bash
conda create -n hyper_env python=3.10
conda activate hyper_env
conda install numpy astropy matplotlib
pip install photutils lmfit>=1.3.3
```

Or using an environment file:
```bash
conda env create -f environment.yml
conda activate Hyper
```

### Option 2: Install via `pip`
```bash
pip install numpy astropy matplotlib photutils "lmfit>=1.3.3"
```

---

## 🛠️ Installation

You can install and use `hyper_py` in two different ways, depending on your needs:

### Option 1: Use the Source Code (for development or integration)

1. Clone the repository:
```bash
git clone https://github.com/Alessio-Traficante/hyper-py.git
```

2. Set your `PYTHONPATH`:
```bash
cd hyper_py
export PYTHONPATH=$(pwd)/src
```

Or from within Python:
```python
import sys
sys.path.insert(0, "/absolute/path/to/hyper_py/src")
```

### Option 2: Install via `pip`
1. Build or download the `.whl` file from the `dist/` folder.
2. Install it:
```bash
pip install hyper_py-X.X.X-py3-none-any.whl
```

---

## 🎯 Usage

You can use `hyper_py` from either Python or the command line.

### From Python

```python
from hyper_py import run_hyper
run_hyper("path/to/config.yaml")
```

### From the Command Line

#### Using the source:
```bash
python -m hyper_py path/to/config.yaml
```

#### If installed via pip:
```bash
hyper_py path/to/config.yaml
```

---

## 🏗️ Configuration File (`config.yaml`)

HYPER is controlled entirely through a YAML configuration file. Below is a summary of its main sections:

### 📂 Paths
```yaml
paths:
  dir_comm: "/main/project/folder/"
  input:
    dir_maps: "Maps/"
  output:
    dir_table_out: "Params/"
    dir_region_out: "Region_files/"
    dir_log_out: "Logs/"
```

### 🧵 Control Flow
```yaml
control:
  parallel_maps: true
  n_cores: 4
  detection_only: false
  use_fixed_source_table: false
  fixed_source_table_path: "sources.txt"
```

### 🔍 Detection
```yaml
detection:
  sigma_thres: 5.0
  use_manual_rms: false
  rms_value: 1.e-6
  roundlim: [-2.0, 2.0]
  sharplim: [-1.0, 2.0]
  dist_limit_arcsec: 0
  fixed_peaks: false
  xcen_fix: [...]
  ycen_fix: [...]
```

### 📏 Photometry
```yaml
photometry:
  aper_inf: 1.0
  aper_sup: 2.0
  fixed_radius: true
  fwhm_1: [0.7]
  fwhm_2: [0.4]
  PA_val: [136.901]
```

### 🧮 Background
```yaml
background:
  no_background: true
  polynomial_orders: [0, 1]
  fix_min_box: 10
  fix_max_box: 20
  fit_gauss_and_bg_separately: true
  pol_orders_separate: [0, 1]
```

### 🧠 Fit Options
```yaml
fit_options:
  fit_method: "leastsq"
  max_nfev: 1000
  xtol: 1e-6
  ftol: 1e-6
  gtol: 1e-6
  calc_covar: false
  weights: "snr"
  power_snr: 5
  use_l2_regularization: true
  lambda_l2: 1e-4
  vary: false
```

### 💾 FITS Output
```yaml
fits_output:
  fits_fitting: true
  fits_deblended: false
  fits_bg_separate: true
  fits_output_dir_fitting: "Fits/Fitting/"
  fits_output_dir_deblended: "Fits/Deblended/"
  fits_output_dir_bg_separate: "Fits/Bg_separate/"
```

### 📊 Visualization
```yaml
visualization:
  visualize_fitting: false
  visualize_deblended: false
  visualize_bg_separate: false
  output_dir_fitting: "Plots/Fitting/"
  output_dir_deblended: "Plots/Deblended/"
  output_dir_bg_separate: "Plots/Bg_separate/"
```

---

## 📦 Code Modules

| File                | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `cli.py`            | Main launcher for multi-map analysis (parallel or serial)    |
| `hyper.py`          | Initialize Hyper runs                                        |
| `single_map.py`     | Core logic for running detection + photometry on one map     |
| `config.py`         | YAML parser with access interface                            |
| `detection.py`      | Source detection using high-pass filtering and DAOStarFinder |
| `fitting.py`        | Multi-Gaussian + background fitting engine                   |
| `gaussfit.py`       | Fitting routine for isolated Gaussian sources                |
| `photometry.py`     | Elliptical aperture photometry                               |
| `survey.py`         | Retrieves beam info and reference units                      |
| `bkg_single.py`     | Estimate background separately in isolated source runs       |
| `bkg_multigauss.py` | Estimate background separately in multi-Gaussian runs        |
| `map_io.py`         | FITS input and pre-processing (unit conversion)              |
| `data_output.py`    | Output table formatting and writing (IPAC, CSV)              |
| `paths_io.py`       | Handles file path construction for input/output files        |
| `logger.py`         | Custom logger supporting log file + screen separation        |
| `visualization.py`  | 2D/3D visual diagnostics of Gaussian/background fits         |
| `groups.py`         | Identifies source groups (blends vs. isolated)               |

---
