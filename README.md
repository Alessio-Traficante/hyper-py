# 💫 `Hyper-py`: Hybrid Photometry Photometry and Extraction Routine in Python

**Authors:** Alessio Traficante; Fabrizio De Angelis; Alice Nucara; Milena Benedettini
**Original reference:** Traficante et al. (2015), *MNRAS, 451, 3089*  

---

## Overview
`Hyper-py` is a flexible and modular Python-based pipeline for performing accurate source extraction and elliptical aperture photometry on astronomical maps. It is designed to reproduce and improve the performance of the original IDL-based HYPER algorithm introduced in Traficante et al. (2015).

The core objective of `Hyper-py` is to combine Gaussian fitting and polynomial background estimation to extract reliable fluxes for compact sources, especially in the presence of blending and spatially variable backgrounds.

---

## Philosophy
- Perform **aperture photometry** using source-dependent elliptical apertures derived from Gaussian fits
- Use a **polynomial background model**, estimated and optionally subtracted either jointly or separately from source fitting
- Handle both **isolated and blended sources**, using multi-Gaussian fitting for groups
- **Support 3D datacubes**: estimate polynomial backgrounds per spectral slice (with source masking) and optionally subtract them before fitting. 
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

Hyper-py now supports **parallel execution** over multiple maps or datacube slices. If a list of FITS files is provided, Hyper-py will automatically:

- Launch one independent process per map (up to the number of available CPU cores)
- Run the full pipeline (detection, fitting, photometry) in parallel across different maps
- Maintain **individual log files** for each map
- Merge the final outputs (tables and diagnostics) into a single, combined summary

To enable parallelism, set the following parameters in your `hyper_config.yaml` file under the `control` section:

```yaml
control:
  parallel_maps: true      # Enable parallel execution across maps
  n_cores: 4               # Number of CPU cores to use
```

If `parallel_maps` is set to `false`, the pipeline will run in serial mode.


## 🛠️ Installation
You can install and use `Hyper-py` in two different ways, depending on your needs:

### Option 1: Install via `pip` (for direct usage)
Install via PyPI:
```bash
pip install hyper-py-photometry
```

### Option 2: Use the Source Code (for development or integration)

If you want to modify, extend, or integrate `Hyper-py` in your own projects:

1. Clone the repository or download the source code.
```bash
git clone https://github.com/Alessio-Traficante/hyper-py.git
```

2. Make sure the `src/` directory is in your `PYTHONPATH`.
```bash
cd hyper_py
export PYTHONPATH=$(pwd)/src
```
   Or from within a Python script or interpreter:

```python
import sys
sys.path.insert(0, "/absolute/path/to/hyper_py/src")
```


## ✅ Requirements

Before using `Hyper-py`, make sure you have all the necessary Python dependencies installed. 

If you have installed `Hyper-py` via pip, all the requirements are automatically installed.
Otherwise, you can use the `requirements.txt` file, this will install the necessary packages using `pip`::
```bash
pip install -r requirements.txt
```


## 📄 Configuration File


`Hyper-py` requires a configuration file named **`hyper_config.yaml`** in order to run.  

>The first time you run `Hyper-py` a new hyper_config.yaml will be created automatically in the CWD, then you must setup all paths and paramenters.<br>
>If you already have a configuration file ready or you have moved the new configuration file to a different folder, provide the path as argument.

If no path is provided, the application will look for it in this order:  
1. Path passed as CLI argument  
2. `hyper_config.yaml` in the current working directory (CWD)  
3. User configuration directory  
    - Linux/macOS: `~/.config/hyper-py/hyper_config.yaml`  
    - Windows: `%APPDATA%\HyperPy\hyper_config.yaml`  
4. If not found, a new `hyper_config.yaml` will be created automatically in the CWD, copied from the package template (`assets/default_config.yaml`).  

> [!IMPORTANT]  
> <span style="font-weight:bold;">Before running the pipeline, you <span style="color:red; font-weight:bold;">must</span> edit **`hyper_config.yaml`** and set the correct parameters and paths.</span> 


### Configuration file lookup order

| Priority | Location                                   | Description                                                                 |
|----------|--------------------------------------------|-----------------------------------------------------------------------------|
| 1        | CLI argument                               | Path explicitly provided by the user, e.g. `hyper-py /path/to/hyper_config.yaml`. |
| 2        | Current Working Directory (CWD)            | Looks for `./hyper_config.yaml` in the folder where the command is executed. |
| 3        | User configuration directory               | - **Linux/macOS:** `~/.config/hyper-py/hyper_config.yaml`<br> - **Windows:** `%APPDATA%\HyperPy\hyper_config.yaml` |
| 4        | Auto-generated in CWD if none is found     | A new `hyper_config.yaml` is created, copied from the package template (`assets/default_config.yaml`). |

### 💡 Tips & Tricks

- **Use different configs**  
  You can maintain multiple configuration files (e.g., `hyper_config.dev.yaml` and `hyper_config.prod.yaml`) and choose which one to run. 
  <br>Eg. If you have installed via pip:
  ```bash
  hyper-py ./hyper_config.dev.yaml
  hyper-py ./hyper_config.prod.yaml
  ```


## 🚀 Usage

You can use `Hyper-py` either by importing and running it directly from Python, or via command line.

### 1. From Python

Import and run the `run_hyper` function, passing the path to your YAML configuration file.

```python
from hyper_py import run_hyper

run_hyper("path/to/hyper_config.yaml")
```
This is the recommended approach if you want to integrate `Hyper-py` into a larger Python application or workflow.

### 2. From Command Line Interface (CLI)

I) Using the source code:

You can execute the tool from the terminal:
```bash
python -m hyper_py path/to/hyper_config.yaml
```
This runs the main process using the configuration file specified.

II) If installed via pip:

Once the .whl package is installed (e.g., via pip install hyper_py-X.X.X-py3-none-any.whl), you can run it directly:
```bash
hyper_py path/to/hyper_config.yaml
```
OR
```bash
hyper-py path/to/hyper_config.yaml
```
OR
```bash
hyper path/to/hyper_config.yaml
```

## 💻 Using the Source Code in Visual Studio Code
To run or debug the source code using Visual Studio Code:
### 1. Open the project
- Open the project folder in VS Code.
- Make sure the Python extension is installed.
- Press Ctrl+Shift+P (or Cmd+Shift+P on macOS) and run Python: Select Interpreter.
- Choose the Hyper Conda environment (or another where the dependencies are installed).

### 2. Run and debug the code

To debug:
- Open src/hyper_py/hyper.py or run_hyper.py.
- Set breakpoints as needed.
- Press F5 or click the "Run and Debug" button in the sidebar.
- In the launch configuration, set the entry script to src/hyper_py/run_hyper.py.

Optional: You can add this to `.vscode/launch.json` for convenience:


```yaml
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger:Run Hyper",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/hyper_py/run_hyper.py",
      "console": "integratedTerminal",
      "args": ["path/to/hyper_config.yaml"],
    }
  ]
}
```
---
<br/><br/>


## ⚙️ Configuration File Reference (`hyper_config.yaml`)

The `hyper_config.yaml` file controls all aspects of the Hyper-py pipeline. Below is a detailed explanation of every entry, including its purpose, accepted values, default, and type.

### File Paths

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `paths.input.dir_maps`      | Directory containing input map files.                                         | `./maps`        | REQUIRED  |
| `paths.output.dir_root`     | Root directory for output data.                                               | `./output`      | REQUIRED  |
| `paths.output.dir_table_out`| Subdirectory of `dir_root` for photometry output tables.                      | `params`        | REQUIRED  |
| `paths.output.dir_region_out`| Subdirectory of `dir_root` for region files (output).                        | `regions`       | REQUIRED  |
| `paths.output.dir_log_out`  | Subdirectory of `dir_root` for log files.                                    | `logs`          | REQUIRED  |

### File Names

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `files.file_map_name`      | Input FITS map(s) list for analysis (in `dir_maps`).                        | `maps_list.txt` | REQUIRED  |
| `files.file_table_base`    | Base filename for photometry output tables (in `dir_table_out`).            | `params`        | REQUIRED  |
| `files.file_region_base`   | Base filename for output ellipse region files (in `dir_region_out`).        | `region_files`  | REQUIRED  |
| `files.file_log_name`      | Name of the global log file (in `dir_log_out`).                             | `hyper_py.log`  | REQUIRED  |

### Pipeline Control

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `control.parallel_maps`    | Enable parallel execution over multiple maps (`True`/`False`).                | `True`          | REQUIRED  |
| `control.n_cores`          | Number of CPU cores to use for multiprocessing.                              | `2`             | REQUIRED  |
| `control.detection_only`   | Only perform source detection without photometry (`True`/`False`).            | `False`         | REQUIRED  |
| `control.datacube`         | Select if the input map is a datacube (`True`/`False`).                       | `False`         | REQUIRED  |
| `control.dir_datacube_slices`| Subdirectory of `dir_root` for datacube slice FITS files.                   | `maps`          | OPTIONAL  |

### Units Conversion

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `units.convert_mJy`         | Convert fluxes to mJy in the final output (`True`/`False`).                  | `False` (Jy)    | REQUIRED  |

### Survey Settings

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `survey.survey_code`         | Numeric identifier for survey parameters (e.g., beam size).                 | `15 (params from map header)`            | REQUIRED  |

### Source Detection

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `detection.sigma_thres`      | Detection threshold in units of RMS (sigma).                                 | `4.0`           | REQUIRED  |
| `detection.use_manual_rms`   | Use manually provided RMS noise value (`True`/`False`).                      | `False`         | OPTIONAL  |
| `detection.rms_value`        | Manual RMS noise value (Jy), used if `use_manual_rms` is `True`.             | `1.e-6`         | OPTIONAL  |
| `detection.roundlim`         | Allowed source roundness range (min, max for DAOFIND).                       | `[-4.0, 4.0]`   | ADVANCED  |
| `detection.sharplim`         | Allowed source sharpness range (min, max for DAOFIND).                       | `[-2.0, 2.0]`   | ADVANCED  |
| `detection.use_fixed_source_table`| Use external IPAC table for peak/aperture (`True`/`False`).              | `False`         | OPTIONAL  |
| `detection.fixed_source_table_path` | Path to an external IPAC table with source information (in `dir_root`). The table must have **6 columns**:  
  - **ID**: Source identifier  
  - **xcen**: X coordinate (in map units, e.g. degrees or pixels)  
  - **ycen**: Y coordinate (in map units, e.g. degrees or pixels)  
  - **fwhm_1**: Major axis FWHM (arcsec)  
  - **fwhm_2**: Minor axis FWHM (arcsec)  
  - **PA**: Position angle (degrees, East of North)  
  The code will use only `xcen` and `ycen` if `detection.fixed_peaks = true`, only `fwhm_1`, `fwhm_2`, and `PA` if `photometry.fixed_radius = true`, or both sets of parameters if both options are enabled. | `source_table.txt` | OPTIONAL |
| `detection.fixed_peaks`      | Use fixed peaks instead of automatic (`True`/`False`).                       | `False`         | OPTIONAL  |
| `detection.xcen_fix`         | Fixed peak X coordinates (deg; used if `fixed_peaks` is `True`).              | `[1.0, 1.0]`    | OPTIONAL  |
| `detection.ycen_fix`         | Fixed peak Y coordinates (deg; used if `fixed_peaks` is `True`).              | `[1.0, 1.0]`    | OPTIONAL  |

### Photometry Settings

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `photometry.aper_inf`        | Minimum size factor for Gaussian FWHM (used as minimum radius for aperture photometry). This value multiplies the average beam FWHM to set the minimum allowed aperture size.              | `1.0`           | OPTIONAL  |
| `photometry.aper_sup` | Maximum size factor for Gaussian FWHM (used as maximum radius for aperture photometry). This value multiplies the average beam FWHM to set the maximum allowed aperture size for photometry. | `2.0` | OPTIONAL |
| `photometry.fixed_radius`    | Use fixed aperture radii (`True`/`False`).                                   | `False`         | OPTIONAL  |
| `photometry.fwhm_1`          | Fixed FWHM aperture radius major axis (arcsec; if `fixed_radius` is `True`). | `[0.0]`         | OPTIONAL  |
| `photometry.fwhm_2`          | Fixed FWHM aperture radius minor axis (arcsec; if `fixed_radius` is `True`). | `[0.0]`         | OPTIONAL  |
| `photometry.PA_val`          | Fixed aperture position angle (deg; if `fixed_radius` is `True`).            | `[0.0]`         | OPTIONAL  |

### Model Fit Settings

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `fit_options.fit_method`     | Optimization algorithm for Gaussian fitting.                              | `"least_squares"`| ADVANCED  |
| `fit_options.loss`           | Specifies the loss function used during Gaussian fitting optimization.  
  - `"linear"`: Standard least-squares loss (minimizes squared residuals; most common for well-behaved data).  
  - `"soft_l1"`: Soft L1 loss, less sensitive to outliers than linear; combines properties of L1 and L2 norms.  
  - `"huber"`: Huber loss, robust to outliers; behaves like linear for small residuals and like L1 for large residuals.  
  - `"cauchy"`: Cauchy loss, strongly suppresses the influence of outliers.  
  Choose a robust loss (e.g., `"huber"` or `"cauchy"`) if your data contains significant outliers or non-Gaussian noise. | `"linear"` | ADVANCED |
| `fit_options.f_scale`        | Relevant for `soft_l1`, `huber`, `cauchy` loss functions.                 | `0.1`           | ADVANCED  |
| `fit_options.max_nfev`       | Maximum number of function evaluations.                                   | `50000`         | ADVANCED  |
| `fit_options.xtol`           | Tolerance on parameter change for convergence.                            | `1e-8`          | ADVANCED  |
| `fit_options.ftol`           | Tolerance on cost function change for convergence.                        | `1e-8`          | ADVANCED  |
| `fit_options.gtol`           | Tolerance on gradient orthogonality.                                      | `1e-8`          | ADVANCED  |
| `fit_options.weights` | Specifies the weighting scheme used during Gaussian fitting.  
  - `"null"`: No weighting; all pixels are treated equally.  
  - `"inverse_rms"`: Weights are set as the inverse of the RMS noise, giving less weight to noisier pixels.  
  - `"snr"`: Weights are proportional to the signal-to-noise ratio (SNR) of each pixel.  
  - `"power_snr"`: Weights are proportional to the SNR raised to a user-defined power (`fit_options.power_snr`).  
  - `"map"`: Weights are set equal to the user-provided input map.  
  - `"mask"`: Weights are set to zero for masked pixels and one elsewhere, effectively ignoring masked regions.  
  Choose the scheme that best matches your data quality and analysis goals. | `"snr"` | OPTIONAL |
| `fit_options.power_snr`      | SNR exponent for weighting (if `weights` is `"power_snr"`).               | `5`             | OPTIONAL  |
| `fit_options.calc_covar`     | Estimate parameter covariance matrix (`True`/`False`).                    | `False`         | ADVANCED  |
| `fit_options.min_method` | Criterion used to select the best fit among multiple solutions:  
  - `"nmse"`: Normalized Mean Squared Error; selects the fit with the lowest mean squared residuals normalized by the data variance.  
  - `"redchi"`: Reduced Chi-Squared; selects the fit with the lowest reduced chi-squared statistic, accounting for the number of degrees of freedom.  
  - `"bic"`: Bayesian Information Criterion; selects the fit with the lowest BIC value, which penalizes model complexity to avoid overfitting.  
  Choose the method that best matches your scientific goals and data characteristics. | `"nmse"` | ADVANCED |
| `fit_options.verbose`        | Print full fit report (`True`/`False`).                                   | `False`         | ADVANCED  |
| `fit_options.use_l2_regularization`| Enable L2 regularization on background terms (`True`/`False`).        | `True`          | ADVANCED  |
| `fit_options.lambda_l2`      | Regularization strength.                                                  | `1e-4`          | ADVANCED  |
| `fit_options.vary`           | Allow source peak to vary during Gaussian fit (`True`/`False`).           | `False`         | ADVANCED  |
| `fit_options.bg_fitters`     | Background fitting methods to try (`least_squares`, `huber`, `theilsen`). | `['least_squares']`| ADVANCED  |
| `fit_options.huber_epsilons` | List of epsilon values for HuberRegressor.                                | `[1.1, 1.35, 1.7, 2.0]`| ADVANCED  |

### Background Estimation

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `background.fit_gauss_and_bg_separately`| Estimate Gaussian and background separately (`True`/`False`).            | `True`          | OPTIONAL  |
| `background.pol_orders_separate`     | Polynomial orders for separated background subtraction.                    | `[0, 1, 2]`     | OPTIONAL  |
| `background.fix_min_box` | Minimum box size for variable-size background fitting, expressed as a multiple of the source FWHM (half-size increment). **If set to `0`, the background is estimated over the entire map.** | `3` | OPTIONAL |
| `background.fix_max_box`             | Maximum box size (multiple of FWHMs) for background fitting.              | `5`             | OPTIONAL  |
| `background.fit_gauss_and_bg_together` | If `True`, the code fits Gaussian source components and the polynomial background **simultaneously** in a single optimization step. If `False`, background subtraction and Gaussian fitting are performed separately. Use `True` for joint modeling when the background and sources are strongly coupled. | `False` | REQUIRED |
| `background.polynomial_orders`       | Polynomial background orders for main fitting.                            | `[0]`           | OPTIONAL  |

### Fits Output Options

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `fits_output.fits_fitting`           | Save best fit model group FITS files (`True`/`False`).                  | `False`         | OPTIONAL  |
| `fits_output.fits_deblended`         | Save deblended per-source FITS files (`True`/`False`).                  | `False`         | OPTIONAL  |
| `fits_output.fits_bg_separate`       | Save best fit background separated model group FITS files (`True`/`False`).| `False`      | OPTIONAL  |
| `fits_output.fits_output_dir_fitting`| Subdirectory of `dir_root` for fitting FITS files.                      | `fits/fitting`  | OPTIONAL  |
| `fits_output.fits_output_dir_deblended`| Subdirectory of `dir_root` for deblended FITS files.                   | `fits/deblended`| OPTIONAL  |
| `fits_output.fits_output_dir_bg_separate`| Subdirectory of `dir_root` for background FITS files.                 | `fits/bg_separate`| OPTIONAL  |

### Visualization Options

| Entry                | Description                                                                 | Default         | Type      |
|----------------------|-----------------------------------------------------------------------------|-----------------|-----------|
| `visualization.visualize_fitting`      | Visualize final Gaussian+background fit (`True`/`False`).                | `False`         | OPTIONAL  |
| `visualization.visualize_deblended`    | Visualize per-source blended maps (`True`/`False`).                      | `False`         | OPTIONAL  |
| `visualization.visualize_bg_separate`  | Visualize background model from masked fit (`True`/`False`).              | `False`         | OPTIONAL  |
| `visualization.output_dir_fitting`     | Subdirectory of `dir_root` for fitting plots.                             | `plots/fitting` | OPTIONAL  |
| `visualization.output_dir_deblended`   | Subdirectory of `dir_root` for deblended plots.                           | `plots/deblended`| OPTIONAL  |
| `visualization.output_dir_bg_separate` | Subdirectory of `dir_root` for background plots.                          | `plots/bg_separate`| OPTIONAL  |

---

**Tip:**  
All entries can be customized in your `hyper_config.yaml`. If an entry is omitted, the default value will be used.



## 📦 Code Modules

| File                  | Description |
|-------------------------------|-------------|
| `run_hyper.py`                | Main launcher for multi-map analysis (parallel or serial)  
| `hyper.py`                    | Core logic for initializing the code run  
| `single_map.py`               | Core logic for running detection + photometry on one map  
| `config.py`                   | YAML parser with access interface  
| `logger.py`                   | Custom logger supporting log file + screen separation  
| `paths_io.py`                 | Handles file path construction for input/output files  
| `map_io.py`                   | FITS input and pre-processing (unit conversion)  
| `survey.py`                   | Retrieves beam info and reference units  
| `detection.py`                | Source detection using high-pass filtering and DAOStarFinder  
| `groups.py`                   | Identifies source groups (blends vs. isolated)  
| `bkg_single.py`               | Estimates and fits the background for single sources in maps or cubes
| `bck_multigauss.py`           | Estimates and fits the background for groups of sources using multi-Gaussian models
| `gaussfit.py`                 | Fitting routine for isolated Gaussian sources  
| `fitting.py`                  | Multi-Gaussian + background fitting engine  
| `photometry.py`               | Elliptical aperture photometry  
| `data_output.py`              | Output table formatting and writing (IPAC, CSV)  
| `visualization.py`            | 2D/3D visual diagnostics of Gaussian/background fits  
| `extract_cubes.py`            | Extracts 2D slices from 3D datacubes and saves them as FITS files. 
| `create_background_slices.py` | Creates and saves background slices from 3D datacubes for further analysis. 

---


## 🗺️ Minimal FITS Header Requirements

To ensure compatibility with Hyper-py, each input FITS file (2D map or 3D datacube) must include a minimal set of header keywords describing the coordinate system, pixel scale, units, and beam properties.

### Minimal Header for 2D Maps

| Keyword   | Description / Example Value                | Options / Notes                                 |
|-----------|-------------------------------------------|--------------------------------------------------|
| SIMPLE    | FITS standard compliance                  | `T` (required)                                   |
| BITPIX    | Data type                                 | `-64` (float64), `-32` (float32)                 |
| NAXIS     | Number of dimensions                      | `2`                                              |
| NAXIS1    | X axis length                             | Integer                                          |
| NAXIS2    | Y axis length                             | Integer                                          |
| CRPIX1    | Reference pixel X                         | Float                                            |
| CRPIX2    | Reference pixel Y                         | Float                                            |
| CDELT1    | Pixel scale X                             | Degrees/pixel (can also be `'CD1_1'`)            |
| CDELT2    | Pixel scale Y                             | Degrees/pixel (can also be `'CD2_1'`)            |
| CRVAL1    | Reference value X                         | RA (deg)                                         |
| CRVAL2    | Reference value Y                         | Dec (deg)                                        |
| CTYPE1    | Coordinate type X                         | `'RA---SIN'`, `'RA---TAN'`, `'GLON--CAR'`, etc.  |
| CTYPE2    | Coordinate type Y                         | `'DEC--SIN'`, `'DEC--TAN'`, `'GLAT--CAR'`, etc.  |
| CUNIT1    | Unit for X                                | `'deg'`, `'arcsec'`                              |
| CUNIT2    | Unit for Y                                | `'deg'`, `'arcsec'`                              |
| BUNIT     | Data unit                                 | `'Jy'`, `'Jy/beam'`, `'beam-1 Jy'`, `'MJy/sr'`   |
| BMAJ      | Beam major axis (deg)                     | Float                                            |
| BMIN      | Beam minor axis (deg)                     | Float                                            |
| BPA       | Beam position angle (deg)                 | Float                                            |
| OBJECT    | Map description                           | String                                           |

### Minimal Header for 3D Datacubes

| Keyword   | Description / Example Value                | Options / Notes                                 |
|-----------|-------------------------------------------|--------------------------------------------------|
| SIMPLE    | FITS standard compliance                  | `T` (required)                                   |
| BITPIX    | Data type                                 | `-32` (float32), `-64` (float64)                 |
| NAXIS     | Number of dimensions                      | `3`                                              |
| NAXIS1    | X axis length                             | Integer                                          |
| NAXIS2    | Y axis length                             | Integer                                          |
| NAXIS3    | Number of slices                          | Integer                                          |
| CRPIX1    | Reference pixel X                         | Float                                            |
| CRPIX2    | Reference pixel Y                         | Float                                            |
| CRPIX3    | Reference pixel Z (slice)                 | Float                                            |
| CDELT1    | Pixel scale X                             | Degrees/pixel (can also be `'CD1_1'`)            |
| CDELT2    | Pixel scale Y                             | Degrees/pixel (can also be `'CD2_1'`)            |
| CDELT3    | Channel width                             | Velocity or frequency units                      |
| CRVAL1    | Reference value X                         | RA (deg)                                         |
| CRVAL2    | Reference value Y                         | Dec (deg)                                        |
| CRVAL3    | Reference value Z (slice)                 | Velocity/frequency (e.g. `0.0`)                  |
| CTYPE1    | Coordinate type X                         | `'RA---SIN'`, `'RA---TAN'`, `'GLON--CAR'`, etc.  |
| CTYPE2    | Coordinate type Y                         | `'DEC--SIN'`, `'DEC--TAN'`, `'GLAT--CAR'`, etc.  |
| CTYPE3    | Coordinate type Z                         | `'VRAD'`, `'VELO-LSR'`, `'FREQ'`                 |
| CUNIT1    | Unit for X                                | `'deg'`, `'arcsec'`                              |
| CUNIT2    | Unit for Y                                | `'deg'`, `'arcsec'`                              |
| CUNIT3    | Unit for Z                                | `'km s-1'`, `'Hz'`                               |
| WCSAXES   | Number of WCS axes                        | `3`                                              |
| BUNIT     | Data unit                                 | `'Jy'`, `'Jy/beam'`, `'beam-1 Jy'`, `'MJy/sr'`   |
| BMAJ      | Beam major axis (deg)                     | Float                                            |
| BMIN      | Beam minor axis (deg)                     | Float                                            |
| BPA       | Beam position angle (deg)                 | Float                                            |
| OBJECT    | Cube description                          | String                                           |

### Notes & Options

- **Coordinate Systems:**  
  - Common values for `CTYPE1`/`CTYPE2` are `'RA---SIN'`, `'RA---TAN'`, `'DEC--SIN'`, `'DEC--TAN'`, `'GLON--CAR'`, `'GLAT--CAR'`.
  - For cubes, `CTYPE3` can be `'VRAD'` (velocity), `'VELO-LSR'`, or `'FREQ'` (frequency).
- **Units:**  
  - `CUNIT1`/`CUNIT2`: `'deg'` (degrees), `'arcsec'` (arcseconds)
  - `CUNIT3`: `'km s-1'` (velocity), `'Hz'` (frequency)
  - `BUNIT`: `'Jy'`, `'Jy/beam'`, `'beam-1 Jy'`, `'MJy/sr'` (must match your science case)
- **Beam Parameters:**  
  - `BMAJ`, `BMIN`: Beam size in degrees (convert from arcsec if needed: 1 arcsec = 1/3600 deg)
  - `BPA`: Beam position angle in degrees
- **Other:**  
  - Additional header keywords may be present, but the above are required for Hyper-py to interpret the map/cube correctly.

---

**Example: Minimal 2D Map Header**
```
SIMPLE  =                    T
BITPIX  =                  -64
NAXIS   =                    2
NAXIS1  =                  400
NAXIS2  =                  400
CRPIX1  =                200.0
CRPIX2  =                200.0
CDELT1  =  -3.000000000000E-03
CDELT2  =   3.000000000000E-03
CRVAL1  =                260.0
CRVAL2  =                 15.0
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CUNIT1  = 'deg     '
CUNIT2  = 'deg     '
BUNIT   = 'Jy      '
BMAJ    =              1.5E-05
BMIN    =              1.5E-05
BPA     =                  0.0
OBJECT  = '2D map for Hyper-py test'
END
```

**Example: Minimal Datacube Header**
```
SIMPLE  =                    T
BITPIX  =                  -32
NAXIS   =                    3
NAXIS1  =                  400
NAXIS2  =                  400
NAXIS3  =                    4
CRPIX1  =                200.0
CRPIX2  =                200.0
CRPIX3  =                    1
CDELT1  =  -2.500000000000E-03
CDELT2  =   2.500000000000E-03
CDELT3  =                  0.5
CRVAL1  =                260.0
CRVAL2  =                 15.0
CRVAL3  =                  0.0
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CTYPE3  = 'VRAD    '
CUNIT1  = 'deg     '
CUNIT2  = 'deg     '
CUNIT3  = 'km s-1  '
WCSAXES =                    3
BUNIT   = 'beam-1 Jy'
BMAJ    =              0.00015
BMIN    =              0.00015
BPA     =                  0.0
OBJECT  = 'Datacube for Hyper-py test'
END
```