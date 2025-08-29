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

To enable parallelism, set the following parameters in your `config.yaml` file under the `control` section:

```yaml
control:
  parallel_maps: true      # Enable parallel execution across maps
  n_cores: 4               # Number of CPU cores to use
```

If `parallel_maps` is set to `false`, the pipeline will run in serial mode.



## 🚀 Prerequisites

Before using `Hyper-py`, make sure you have all the necessary Python dependencies installed. The following core libraries are required:
	•	astropy
	•	photutils
	•	matplotlib
	•	lmfit
	
This will install the necessary packages using `pip`:

```bash
astropy photutils matplotlib lmfit
```


## 🛠️ Installation
You can install and use `Hyper-py` in two different ways, depending on your needs:

### Option 1: Use the Source Code (for development or integration)

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
### Option 2: Install via `pip` (for direct usage)
1. Build or download the .whl package (e.g., dist/hyper_py-X.X.X-py3-none-any.whl).
2. Install the wheel file using `pip`:
   
```bash
pip install hyper_py-X.X.X-py3-none-any.whl
```
Use the current file version in dist folder.



## 🎯 Usage

You can use `Hyper-py` either by importing and running it directly from Python, or via command line.
> [!IMPORTANT]  
>  `Hyper-py` needs a configuration file in order to run. If no configuration file path is provided, the default file located in the `src/` folder will be used.

### 1. From Python

Import and run the `cli` function, passing the path to your YAML configuration file.

```python
from hyper_py import run_hyper

run_hyper("path/to/config.yaml")
```
This is the recommended approach if you want to integrate `Hyper-py` into a larger Python application or workflow.

### 2. From Command Line Interface (CLI)

I) Using the source code:

You can execute the tool from the terminal:
```bash
python -m hyper_py path/to/config.yaml
```
This runs the main process using the configuration file specified.

II) If installed via pip:

Once the .whl package is installed (e.g., via pip install hyper_py-X.X.X-py3-none-any.whl), you can run it directly:
```bash
hyper_py path/to/config.yaml
```

## Using the Source Code in Visual Studio Code
To run or debug the source code using Visual Studio Code:
### 1. Open the project
- Open the project folder in VS Code.
- Make sure the Python extension is installed.
- Press Ctrl+Shift+P (or Cmd+Shift+P on macOS) and run Python: Select Interpreter.
- Choose the Hyper Conda environment (or another where the dependencies are installed).

### 2. Run and debug the code

To debug:
- Open src/hyper_py/hyper.py or cli.py.
- Set breakpoints as needed.
- Press F5 or click the "Run and Debug" button in the sidebar.
- In the launch configuration, set the entry script to src/hyper_py/cli.py.

Optional: You can add this to `.vscode/launch.json` for convenience:


```yaml
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger:Run Hyper",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/hyper_py/cli.py",
      "args": [],
      "console": "integratedTerminal",
      //"args": ["path/to/config.yaml"] // If you want to specify a different config file
    }
  ]
}
```
---
<br/><br/>

## 📦 Code Modules

| File                  | Description |
|-------------------------------|-------------|
| `cli.py`                      | Main launcher for multi-map analysis (parallel or serial)  
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
