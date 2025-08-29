#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of Reference catalog vs Hyper-py and Hyper-IDL per map, 
with direct IDL-vs-py scatter plots and combined percentage difference histograms.
@author: alessio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from astropy.table import Table
from astropy import coordinates as coords
from astropy import units as u

# ################## Parameters ##################
num_sources = 500
n_sigma = 4
tot_maps = 2
tolerance = 0.02 * u.arcsec
legend_properties = font_manager.FontProperties(weight='bold', size=12)

# Beam and pixel info for unit conversion (deg)
bmaj = 1.335631849037E-05
bmin = 1.069115267860E-05
cdelt = 3.611111108576E-06

beam_area_sr = 1.1331 * bmaj * bmin
pix_area_deg2 = cdelt**2
beam_area_pix = beam_area_sr / pix_area_deg2

# Paths
dir_in_ref = "/Users/alessio/Dropbox/Work/Python/Hyper/IDL_comparison/"
dir_in_py  = "/Users/alessio/Dropbox/Work/Python/Hyper/IDL_comparison/"
dir_in_idl = "/Users/alessio/Dropbox/Work/Python/Hyper/IDL_comparison/"
dir_out    = "/Users/alessio/Dropbox/Work/Python/Hyper/IDL_comparison/output/"

background = 'polynomial'
table_combined_stats = dir_out + 'combined_source_counts_comparison.txt'

# Summary table
summary_table = Table(
    names=['Catalog', 'Source_Type', 'Total', 'Matched', 'False', 'False_Percentage'],
    dtype=['i4', 'S10', 'i4', 'i4', 'i4', 'f4']
)

# ================== Helper Functions ==================
def match_catalogs(cat_coords, ref_coords):
    sep = ref_coords[:, None].separation(cat_coords[None, :])
    min_sep = sep.min(axis=1)
    min_idx = sep.argmin(axis=1)
    matched = min_sep < tolerance
    _, unique_idx = np.unique(min_idx[matched], return_index=True)
    filtered = np.zeros_like(matched, dtype=bool)
    filtered[np.where(matched)[0][unique_idx]] = True
    return filtered, min_idx

def scatter_comparison(ref_vals, cat_vals, flux_type, ylabel, label, color, marker, suffix):
    plt.figure(figsize=(8, 6))
    plt.scatter(ref_vals, cat_vals, color=color, marker=marker, label=label)
    minf = min(np.min(ref_vals), np.min(cat_vals))
    maxf = max(np.max(ref_vals), np.max(cat_vals))
    
    # plt.xlim(0, 1800)
    # plt.ylim(0, 1800)

    plt.plot([minf, maxf], [minf, maxf], 'k--', label='1:1 Line')
    plt.xlabel(f'{flux_type} ($\mu$Jy)', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.legend(prop=legend_properties)
    plt.title(f'{flux_type} Comparison: {label}', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', labelsize=12, width=2)
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontweight('bold')
    ax = plt.gca()
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)
    plt.savefig(f'{dir_out}Flux_Comparison_{suffix}.png')
    plt.close()

def plot_flux_difference_histogram(ref_list, py_list, idl_list, flux_type, filename):
    ref_all = np.concatenate(ref_list)
    py_all = np.concatenate(py_list)
    idl_all = np.concatenate(idl_list)

    # Percentage differences
    diff_py = 100. * (py_all - ref_all) / ref_all
    diff_idl = 100. * (idl_all - ref_all) / ref_all

    plt.figure(figsize=(8, 6))
    bins = np.linspace(
        min(diff_py.min(), diff_idl.min()),
        max(diff_py.max(), diff_idl.max()),
        50
    )
    
    plt.hist(diff_py, bins=bins, alpha=0.6, label='Hyper-py - Ref', color='blue', edgecolor='black')
    plt.hist(diff_idl, bins=bins, alpha=0.6, label='Hyper-IDL - Ref', color='green', edgecolor='black')

    for diff, label, color in [(diff_py, 'py', 'blue'), (diff_idl, 'idl', 'green')]:
        mean = np.mean(diff)
        median = np.median(diff)
        plt.axvline(mean, color=color, linestyle='--', linewidth=2, label=f'Mean {label}: {mean:.1f}%')
        plt.axvline(median, color=color, linestyle=':', linewidth=2, label=f'Median {label}: {median:.1f}%')

    plt.xlabel(f'{flux_type} Difference (% vs Reference)', fontsize=14, fontweight='bold')
    plt.ylabel('Counts', fontsize=14, fontweight='bold')
    plt.title(f'{flux_type} Difference: Hyper-py & Hyper-IDL vs Reference', fontsize=16, fontweight='bold')
    plt.legend(prop=legend_properties)
    plt.tick_params(axis='both', labelsize=12, width=2)
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontweight('bold')
    ax = plt.gca()
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)
    plt.savefig(f'{dir_out}{filename}')
    plt.close()

# Containers for histograms
py_all_hyper_py_id, py_all_hyper_idl_id, ref_all_peak, py_all_peak, idl_all_peak = [], [], [], [], []
ref_all_int, py_all_int, idl_all_int = [], [], []

# ================== Loop Over Maps ==================
for i in range(1, tot_maps + 1):
    ref = Table.read(f'{dir_in_ref}table_{num_sources}_Gaussians_{i}.txt', format='ipac')
    ref['RA'] = ref['RA'] * u.deg
    ref['DEC'] = ref['DEC'] * u.deg
    coords_ref = coords.SkyCoord(ra=ref['RA'], dec=ref['DEC'])

    hpy = Table.read(f'{dir_in_py}hyper_output_map_{num_sources}_Gaussians_{i}.txt', format='ipac')
    coords_py = coords.SkyCoord(ra=hpy['RA'], dec=hpy['DEC'])

    hidl = Table.read(f'{dir_in_idl}photometry_sources_1300_ellipses_1300_{background}_background_'+str(n_sigma)+f'sigma_ipac_{i}.txt', format='ipac')
    coords_idl = coords.SkyCoord(ra=hidl['ra'], dec=hidl['dec'])

    # Match each to reference
    m_py, idx_py = match_catalogs(coords_py, coords_ref)
    m_idl, idx_idl = match_catalogs(coords_idl, coords_ref)

    matched_py = hpy[idx_py[m_py]]
    ref_matched_py = ref[m_py]
    false_py = len(hpy) - len(matched_py)
    summary_table.add_row([i, 'Hyper-py', len(ref), len(matched_py), false_py, false_py / len(hpy) * 100.])

    matched_idl = hidl[idx_idl[m_idl]]
    ref_matched_idl = ref[m_idl]
    false_idl = len(hidl) - len(matched_idl)
    summary_table.add_row([i, 'Hyper-IDL', len(ref), len(matched_idl), false_idl, false_idl / len(hidl) * 100.])


    # Save IDs #
    hyper_py_id = matched_py['HYPER_ID']
    hyper_idl_id = matched_idl['HYPER_ID']

    # Peak Flux plots
    flux_peak_py_mjy = matched_py['FLUX_PEAK'] * beam_area_pix
    flux_peak_idl_mjy = matched_idl['FLUX_PEAK_JY']

    scatter_comparison(ref_matched_py['Flux_Peak'] * 1e6, flux_peak_py_mjy * 1e3, 'Flux Peak', 'Measured Flux Peak ($\mu$Jy)', 'Hyper-py', 'blue', 'o', f'Hyper_py_Peak_{i}')
    scatter_comparison(ref_matched_idl['Flux_Peak'] * 1e6, flux_peak_idl_mjy * 1e3, 'Flux Peak', 'Measured Flux Peak ($\mu$Jy)', 'Hyper-IDL', 'green', 's', f'Hyper_IDL_Peak_{i}')

    # Integrated Flux plots
    scatter_comparison(ref_matched_py['Flux_Integrated'] * 1e6, matched_py['FLUX'] * 1e3, 'Integrated Flux', 'Measured Integrated Flux ($\mu$Jy)', 'Hyper-py', 'blue', 'o', f'Hyper_py_Int_{i}')
    scatter_comparison(ref_matched_idl['Flux_Integrated'] * 1e6, matched_idl['FLUX'] * 1e3, 'Integrated Flux', 'Measured Integrated Flux ($\mu$Jy)', 'Hyper-IDL', 'green', 's', f'Hyper_IDL_Int_{i}')

    # Common matches across all three
    common = m_py & m_idl
    if np.any(common):
        ref_common = ref[common]
        py_common = hpy[idx_py[common]]
        idl_common = hidl[idx_idl[common]]
        
        
        id_hyper_py = py_common['HYPER_ID']
        id_hyper_idl = idl_common['HYPER_ID']
        
        py_all_hyper_py_id.append(id_hyper_py)
        py_all_hyper_idl_id.append(id_hyper_idl)


        peak_py = py_common['FLUX_PEAK'] * beam_area_pix *1.e3
        peak_idl = idl_common['FLUX_PEAK_JY'] *1.e3
        int_py = py_common['FLUX'] *1.e3
        int_idl = idl_common['FLUX'] *1.e3

        ref_all_peak.append(ref_common['Flux_Peak'] * 1e6)  # mJy
        py_all_peak.append(peak_py)
        idl_all_peak.append(peak_idl)

        ref_all_int.append(ref_common['Flux_Integrated'] * 1e6)  # mJy
        py_all_int.append(int_py)
        idl_all_int.append(int_idl)

        # Direct IDL vs py scatter
        scatter_comparison(peak_idl * 1e3, peak_py * 1e3, 'Flux Peak', 'Hyper-py Flux Peak ($\mu$Jy)', 'Hyper-IDL vs Hyper-py', 'purple', 'd', f'Hyper_IDL_vs_py_Peak_{i}')
        scatter_comparison(int_idl * 1e3, int_py * 1e3, 'Integrated Flux', 'Hyper-py Integrated Flux ($\mu$Jy)', 'Hyper-IDL vs Hyper-py', 'purple', 'd', f'Hyper_IDL_vs_py_Int_{i}')






from astropy.table import Table

# ================== Save Matched Source Table ==================

# Flatten all lists
all_ref_hyper_py_id = np.concatenate(py_all_hyper_py_id)
all_ref_hyper_idl_id = np.concatenate(py_all_hyper_idl_id)


all_ref_peak = np.concatenate(ref_all_peak)
all_py_peak = np.concatenate(py_all_peak)
all_idl_peak = np.concatenate(idl_all_peak)

all_ref_int = np.concatenate(ref_all_int)
all_py_int = np.concatenate(py_all_int)
all_idl_int = np.concatenate(idl_all_int)

# Percentage differences
diff_peak_py = 100. * (all_py_peak - all_ref_peak) / all_ref_peak
diff_peak_idl = 100. * (all_idl_peak - all_ref_peak) / all_ref_peak
diff_int_py = 100. * (all_py_int - all_ref_int) / all_ref_int
diff_int_idl = 100. * (all_idl_int - all_ref_int) / all_ref_int

# Build and write the output table
matched_table = Table(
    [
        np.arange(1, len(all_ref_peak) + 1),  # INDEX
        all_ref_hyper_py_id,
        all_ref_hyper_idl_id,
        all_ref_peak, all_py_peak, all_idl_peak,
        all_ref_int,  all_py_int,  all_idl_int,
        diff_peak_py, diff_peak_idl,
        diff_int_py,  diff_int_idl
    ],
    names=[
        'INDEX',
        'HYPER_ID_PY', 'HYPER_ID_IDL',
        'PEAK_REF_mJy', 'PEAK_PY_mJy', 'PEAK_IDL_mJy',
        'INT_REF_mJy',  'INT_PY_mJy',  'INT_IDL_mJy',
        'D_PEAK_PY_pct', 'D_PEAK_IDL_pct',
        'D_INT_PY_pct',  'D_INT_IDL_pct'
    ],
    dtype=[
        'int32', 'int32', 'int32',
        'float64', 'float64', 'float64',
        'float64', 'float64', 'float64',
        'float32', 'float32', 'float32', 'float32'
    ]
)

# Format columns
for col in [
    'PEAK_REF_mJy', 'PEAK_PY_mJy', 'PEAK_IDL_mJy',
    'INT_REF_mJy', 'INT_PY_mJy', 'INT_IDL_mJy',
    'D_PEAK_PY_pct', 'D_PEAK_IDL_pct',
    'D_INT_PY_pct', 'D_INT_IDL_pct'
]:
    matched_table[col].format = '6.1f'

# Save table
matched_table.meta['COMMENT'] = 'Flux comparison for sources matched in all three catalogs'
matched_table.write(dir_out + 'matched_flux_comparison_table.txt', format='ascii.ipac', overwrite=True)






# Histogram plots over all maps
plot_flux_difference_histogram(ref_all_peak, py_all_peak, idl_all_peak, 'Flux Peak', 'Flux_Diff_Histogram_Peak.png')
plot_flux_difference_histogram(ref_all_int, py_all_int, idl_all_int, 'Integrated Flux', 'Flux_Diff_Histogram_Int.png')

# Save summary stats
summary_table.write(table_combined_stats, format='ascii.ipac', overwrite=True)