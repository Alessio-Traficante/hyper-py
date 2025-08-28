import os
from astropy.table import Table


def write_tables(data_dict, output_dir, config, sigma_thres, real_rms, base_filename="hyper_output"):

    '''
    Write photometry results into formatted ECSV and IPAC tables,
    preserving numeric data types and precision.

    Parameters
    ----------
    data_dict : dict
        Dictionary of columns for the output table.
    output_dir : str
        Directory where the file will be written.
    base_filename : str
        Base name for the output file (without extension).
    '''

    if config.get('units', 'convert_mJy'):
        flux_units = 'mJy'
        flux_units_beam = 'mJy/beam'
    else:
        flux_units_beam = 'Jy/beam'
        flux_units = 'Jy/beam'
        
    units = {
        'MAP_ID': '', 'HYPER_ID': '', 'BAND': 'GHz',
        'RA': 'deg', 'DEC': 'deg', 'GLON': 'deg', 'GLAT': 'deg',
        'FLUX': flux_units, 'FLUX_ERR': flux_units,
        'FLUX_PEAK': flux_units_beam, 
        'RESIDUALS': flux_units,
        'FWHM_1': 'arcsec', 'FWHM_2': 'arcsec',
        'PA': 'deg', 'NMSE': '', 'CHI2_RED': '',
        'POLYN': '', 'STATUS': '', 'DEBLEND': '', 'CLUSTER': '',
    }

    descriptions = {
        'MAP_ID':    'Map identifier',
        'HYPER_ID':  'Source identifier',
        'FLUX_PEAK': 'Peak flux (' + flux_units_beam + ')',
        'FLUX':      'Integrated flux density (' + flux_units + ')',
        'FLUX_ERR':  'Flux density uncertainty (' + flux_units + ')',
        'RESIDUALS': 'Mean background level (' + flux_units + ')',
        'POLYN':     'Polynomial background order',
        'NMSE':      'Normalized Mean Squared Error of fit: scale-independent metric of model-data agreement',
        'CHI2_RED':  'Reduced chi-squared of Gaussian + background fit (valid only if residuals are Gaussian-distributed with constant variance)',
        'BIC':       'Bayesian Information Criterion: model selection metric that penalizes overfitting; lower BIC indicates better trade-off between goodness of fit and model complexity',
        'FWHM_1':    'First axis FWHM (arcsec)',
        'FWHM_2':    'Second axis FWHM (arcsec)',
        'PA':        'Position angle (deg East of North)',
        'STATUS':    'Fit status flag: 1 = fit succeeded, 0 = fit failed',
        'GLON':      'Galactic longitude (deg)',
        'GLAT':      'Galactic latitude (deg)',
        'RA':        'Right Ascension (deg, J2000)',
        'DEC':       'Declination (deg, J2000)',
        'DEBLEND':   'Deblending flag',
        'CLUSTER':   'Cluster flag',
    }
    

    format_dict = {
        'MAP_ID': 's', 'HYPER_ID': 'd', 'RA': '.5f', 'DEC': '.5f', 'GLON': '.5f', 'GLAT': '.5f',
        'FLUX': '.3f', 'FLUX_ERR': '.3f', 'FLUX_PEAK': '.5f',
        'FLUX_PEAK_JY': '.4f', 'RESIDUALS': '.5f',
        'FWHM_1': '.3f', 'FWHM_2': '.3f', 'PA': '.1f',
        'NMSE': '.3f', 'CHI2_RED': '.3f', 'BIC': '.2f', 'POLYN': 'd',
        'STATUS': 'd', 'DEBLEND': 'd', 'CLUSTER': 'd',
    }
    

    # Original numeric table
    table = Table(data_dict)    

    # Set units, descriptions, and numeric formats
    for col in table.colnames:
        if col in units and units[col]:
            table[col].unit = units[col]
        if col in descriptions:
            table[col].description = descriptions[col]
        if col in format_dict:
            table[col].format = format_dict[col]
            
            
    # Create your custom header lines explicitly
    if config.get('units', 'convert_mJy'): 
        rms_sentence = f"Estimated r.m.s. to identify sources: {real_rms:.5f} mJy"
    else:
        rms_sentence = f"Estimated r.m.s. to identify sources: {real_rms:.5f} Jy"    

    custom_header_lines = [
        " ****************** Hyper photometry ******************",
        f"Survey code: {config.get('survey', 'survey_code')}",
        f"Detection threshold: {sigma_thres} sigma",
        # f"Pixel size: {pix_dim:.3f} arcsec",
        # f"Background type: {'none' if background_type == 'none' else background_type}",
        f"Convert from MJy/sr: {config.get('units', 'convert_Jy')}",
        f"Convert from Jy/beam: {config.get('units', 'convert_beam_Jy')}",
        f"Convert to mJy: {config.get('units', 'convert_mJy')}",
        rms_sentence,
        " ******************************************************"
    ]

    # Add custom header to table metadata (ECSV)
    table.meta['comments'] = custom_header_lines + table.meta.get('comments', [])

    # Write ECSV table with custom header
    csv_output_path = os.path.join(output_dir, base_filename + ".csv")
    table.write(csv_output_path, format="ascii.ecsv", overwrite=True)

    # For IPAC, explicitly copy rows to preserve compatibility
    ipac_table = Table(names=table.colnames, dtype=[table[col].dtype for col in table.colnames])

    for row in table:
        ipac_table.add_row(row)

    # Copy formatting, units, and descriptions again explicitly to IPAC table
    for col in ipac_table.colnames:
        ipac_table[col].format = table[col].format
        ipac_table[col].unit = table[col].unit
        ipac_table[col].description = table[col].description

    # Add custom header lines explicitly to IPAC table comments
    ipac_table.meta['comments'] = custom_header_lines + [
        f"{col} = {ipac_table[col].description}" for col in ipac_table.colnames
    ]

    # Write IPAC table explicitly with custom headers
    ipac_output_path = os.path.join(output_dir, base_filename + ".txt")
    ipac_table.write(ipac_output_path, format="ipac", overwrite=True)
            