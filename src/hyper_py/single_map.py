import os
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS

from collections.abc import Iterable

from hyper_py.paths_io import get_hyper_single_map_paths
from hyper_py.survey import get_beam_info
from hyper_py.map_io import read_and_prepare_map
from hyper_py.detection import detect_sources
from hyper_py.data_output import write_tables
from hyper_py.groups import group_sources
from hyper_py.photometry import aperture_photometry_on_sources
from hyper_py.gaussfit import fit_isolated_gaussian
from hyper_py.fitting import fit_group_with_background
from hyper_py.visualization import plot_fit_summary
from hyper_py.logger import setup_logger


def main(map_name=None, cfg=None, dir_root=None, logger=None, logger_file_only=None):   
         
    paths_dict = get_hyper_single_map_paths(cfg, map_name)

    # - input/output paths - #
    datacube = cfg.get("control", "datacube", False)

    dir_root = cfg.get("paths", "output")["dir_root"]

    if datacube:
        input_map_path = Path(dir_root, cfg.get("control")["dir_datacube_slices"], map_name)
    else:
       input_map_path = paths_dict["input_map_path"]

    output_dir_path    = paths_dict["output_dir_path"]
    
    base_name_with_suffix = paths_dict["base_name_with_suffix"]
    centroids_file     = paths_dict["centroids_file"]
    ellipses_file      = paths_dict["ellipses_file"]
    suffix = paths_dict["suffix"]

    # - control - #
    detection_only = cfg.get("control", "detection_only", False) 
    fixed_radius = cfg.get("photometry", "fixed_radius", False)
    
    # - params - #
    survey_code = cfg.get("survey", "survey_code")
    
    # - visualization params - #
    try:
        visualize_deblended = cfg.get("visualization", "visualize_deblended", False)
        visualize_output_dir_deblended = os.path.join(dir_root, cfg.get("visualization", "output_dir_deblended", "images/deblended"))  
    except:
        visualize_deblended = False
        
   # - Fits save params - #     
    try:
        fits_deblended = cfg.get("fits_output", "fits_deblended", False)
        fits_output_dir_deblended = os.path.join(dir_root, cfg.get("fits_output", "fits_output_dir_deblended", "fits/deblended"))
    except:
        fits_deblended = False

        
    # - Setup log file specifically for this map - #
    log_new_map_file = f"Hyper_log_{suffix}.log"
    log_new_map_dir = cfg.get("paths")["output"]["dir_log_out"]
    log_path_each_map = os.path.join(dir_root, log_new_map_dir, log_new_map_file)
    
    # Give this map a unique logger name
    if logger is None:
        logger, logger_file_only = setup_logger(
            log_path=log_path_each_map,
            logger_name=f"HyperLogger_{suffix}", # unique name per map
            overwrite=True,
            process_name=os.path.basename(input_map_path)  # Use the map name as the process name
        )
    
    # initialize vectors and Table     
    radius_val_1 = []
    radius_val_2 = []
    PA_val = []
    sky_val = []
    poly_order_val = []
    nmse_val = []
    redchi_val = []
    bic_val = []
    updated_xcen = []
    updated_ycen = []
    flux_peak = []
    flux = []
    flux_err = []


    fit_statuts_val = []
    deblend_val = []
    cluster_val = []

    source_id_save = []  

    bg_model = None      
        
        
    # --- see if peaks position and aperture radius are fixed or not --- #
    def ensure_list(x, n):
        return x if isinstance(x, Iterable) and not isinstance(x, str) else [x] * n
    
    # --- Load fixed table if specified --- #
    use_fixed_table = cfg.get("detection", "use_fixed_source_table", False)
    fixed_radius = cfg.get("photometry", "fixed_radius", False)
    fixed_peaks  = cfg.get("detection", "fixed_peaks", False)
    
    if use_fixed_table:
        table_path = os.path.join(dir_root, cfg.get("detection", "fixed_source_table_path"))
        fixed_sources = ascii.read(table_path, format="ipac")
    
    
    # - read header and map - #
    beam_arcsec, beam_area_arcsec2, beam_area_sr = get_beam_info(survey_code, input_map_path)

    map_struct = read_and_prepare_map(
        filepath=input_map_path,
        beam=beam_arcsec,
        beam_area_arcsec2=beam_area_arcsec2,
        beam_area_sr = beam_area_sr,
        convert_mjy=cfg.get("units", "convert_mJy")
    )
    
    real_map = map_struct["map"]
    
    # # --- zero-mean for the input map --- #
    # map_zero_mean = real_map - np.nanmean(real_map)
    # real_map = map_zero_mean
    
    header = map_struct["header"]
    wcs = WCS(header)
    pix_dim = map_struct["pix_dim"]
    beam_dim = map_struct["beam_dim"]
    beam_area = map_struct["beam_area_arcsec2"]    
    

    # --- map rms used to define real sources in the map - accounting for non-zero background --- #
    use_maual_rms = cfg.get("detection", "use_manual_rms", False)
    if use_maual_rms == True:
        real_rms = cfg.get("detection", "rms_value", False)
    else:         
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        map_zero_mean_detect = real_map - np.nanmean(real_map)
        clipped = sigma_clip(map_zero_mean_detect)    
        real_rms = np.sqrt(np.nanmean(clipped**2))
        
        
    # --- run sources identification  --- #
    if fixed_peaks:
        if use_fixed_table:
            logger.info("[INFO] Using manually provided peak coordinates from IPAC table.")
            xcen = np.array(fixed_sources["RA"])   # these are assumed to be in world coordinates
            ycen = np.array(fixed_sources["DEC"])
            
            # Convert WCS coordinates to pixel positions
            xpix, ypix = wcs.wcs_world2pix(xcen, ycen, 0)    
            xcen = xpix
            ycen = ypix
            
        else:
            logger.info("[INFO] Using manually provided peak coordinates from config file.")
            xcen_fix = cfg.get("detection", "xcen_fix")
            ycen_fix = cfg.get("detection", "ycen_fix")
            
            # Convert WCS coordinates to pixel positions
            xpix, ypix = wcs.wcs_world2pix(xcen_fix, ycen_fix, 0)
            xcen = xpix
            ycen = ypix
        
        sources = xcen
        all_sources_xcen = xcen
        all_sources_ycen = ycen
     
    else:
        sources = detect_sources(
            map_struct_list=map_struct,
            dist_limit_arcsec=cfg.get("detection", "dist_limit_arcsec", 0),
            real_map=map_zero_mean_detect,
            rms_real=real_rms,
            snr_threshold=cfg.get("detection", "sigma_thres"),
            roundlim=cfg.get("detection", "roundlim", [-1.0, 1.0]),
            sharplim=cfg.get("detection", "sharplim", [-1.0, 2.0]),
            config=cfg
        )
        xcen = sources["xcentroid"]
        ycen = sources["ycentroid"]
        
        all_sources_xcen = xcen
        all_sources_ycen = ycen
        
    
    # -- if fixed_radius = True generate a xcen vector of aperture radii -- #
    if fixed_radius:        
        if use_fixed_table:
            fwhm_1_list = np.array(fixed_sources["FWHM_1"])
            fwhm_2_list = np.array(fixed_sources["FWHM_2"])
            PA_list     = np.array(fixed_sources["PA"])
        else:
            N = len(xcen)
            fwhm_1_list = ensure_list(cfg.get("photometry", "fwhm_1", 3.0), N)
            fwhm_2_list = ensure_list(cfg.get("photometry", "fwhm_2", 2.0), N)
            PA_list     = ensure_list(cfg.get("photometry", "PA_val", 0.0), N)    
    
        
    # --- organize sources in isolated or groups ---#
    start_group, common_group, deblend = group_sources(
        xcen=xcen,
        ycen=ycen,
        pix_dim=pix_dim,
        beam_dim=beam_dim,
        aper_sup=cfg.get("photometry", "aper_sup"),
    )
    
    tot_sources = len(sources)
    isolated = np.where(start_group == 0)[0]
    blended = np.where(start_group == 1)[0]

    logger.info(f"{tot_sources} sources above threshold.")
    logger.info(f"{len(isolated)} sources are isolated")
    logger.info(f"{len(blended)} sources are blended")

    
    if detection_only:
        logger.info("[INFO] Detection-only mode enabled. Skipping photometry and fitting.")
    
        # Convert to sky coordinates
        x_pixels = np.array(xcen, dtype=np.float64)
        y_pixels = np.array(ycen, dtype=np.float64)
        ra, dec = wcs.wcs_pix2world(x_pixels, y_pixels, 0)
        ra_save = np.where(ra < 0., ra + 360., ra)
        
        skycoords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        glon = skycoords.galactic.l.deg
        glat = skycoords.galactic.b.deg
    
        # Prepare zeroed output table
        N = len(xcen)
        data_dict = {
            "MAP_ID": [str(suffix)],
            "HYPER_ID": [0],
            "FLUX_PEAK": [0.0],
            "FLUX": [0.0],
            "FLUX_ERR": [0.0],
            "RESIDUALS": [0.0],
            "POLYN": [0],
            "NMSE": [0.0],
            "CHI2_RED": [0.0],
            "FWHM_1": [0.0],
            "FWHM_2": [0.0],
            "PA": [0.0],
            "STATUS": [0],
            "GLON": [0.0],
            "GLAT": [0.0],
            "RA": [0.0],
            "DEC": [0.0],
            "DEBLEND": [0],
            "CLUSTER": [0],
        }

        sigma_thres = cfg.get("detection", "sigma_thres")
        write_tables(data_dict, output_dir_path, cfg, sigma_thres, real_rms, base_filename=base_name_with_suffix)
    
        ######################## Write only the centroid region file ########################
        radecsys = (header.get("RADESYS") or header.get("RADECSYS") or wcs.wcs.radesys or "FK5").strip().upper()
        if "ICRS" in radecsys:
            ds9_coordsys = "icrs"
        elif "FK5" in radecsys:
            ds9_coordsys = "fk5"
        elif "GAL" in radecsys:
            ds9_coordsys = "galactic"
        else:
            ds9_coordsys = "fk5"
    
        with open(centroids_file, "w") as f:
            f.write("# Region file format: DS9 version 4.1\n")
            f.write("global color=cyan dashlist=8 3 width=1 font='helvetica 10 normal' select=1 "
                    "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0\n")
            f.write(f"{ds9_coordsys}\n")
            for xw, yw in zip(ra_save, dec):
                f.write(f"point({xw:.8f},{yw:.8f}) # point=cross\n")
    
        logger.info(f"Detection-only mode complete. Saved table and centroid region file for {N} sources.\n")
        return map_name, bg_model, header, header # ✅ Done!
    
  
    

    
######################## ISOLATED sources photometry ########################
    for idx_iso, i in enumerate(isolated):
                        
        logger_file_only.info(f"Photometry on isolated source {idx_iso + 1} of {len(isolated)}")
    
        fit_status, fit_result, model_fn, bg_order, cutout, (yslice, xslice), bg_mean, bg_model, cutout_header, final_nmse, final_redchi, final_bic = fit_isolated_gaussian(
            image=real_map,
            xcen=xcen[i],
            ycen=ycen[i],
            all_sources_xcen = all_sources_xcen,
            all_sources_ycen = all_sources_ycen,
            source_id=idx_iso,
            map_struct=map_struct,
            suffix=suffix,
            config=cfg,
            logger=logger,
            logger_file_only=logger_file_only
        )
        
        if fit_result is None:
            logger_file_only.error(f"Fit failed for isolated source {i}")
            continue
        
        # --- Extract fitted Gaussian parameters ---
        sig_x = fit_result.params["g_sigmax"].value
        sig_y = fit_result.params["g_sigmay"].value
        fwhm_x = 2.3548 * sig_x
        fwhm_y = 2.3548 * sig_y
        theta = np.rad2deg(fit_result.params["g_theta"].value) #+90.    # rotated for photometry #
        
        
        # --- radius fixed if decided in the config file ---
        if fixed_radius:     
            if len(fwhm_1_list) == 1:
                fwhm_x = fwhm_1_list[0] / pix_dim
                fwhm_y = fwhm_2_list[0] / pix_dim
                theta = PA_list[0]
            else:                
                fwhm_x = fwhm_1_list[i] / pix_dim
                fwhm_y = fwhm_2_list[i] / pix_dim
                theta = PA_list[i]
        
        # --- Evaluate full model on the cutout grid ---
        yy, xx = np.indices(cutout.shape)
 
        # --- Zero out the Gaussian component ---
        params_bg_only = fit_result.params.copy()
        for name in params_bg_only:
            if name.startswith("g_"):
                params_bg_only[name].set(value=0.0)

        model_bg_only = model_fn(params_bg_only, xx, yy)
        
        # --- Final cleaned map for aperture photometry ---
        source_only_map = cutout - model_bg_only 
                
        # --- Photometry: use centroid within cutout ---
        phot_single = aperture_photometry_on_sources(
            image=source_only_map,
            xcen=[fit_result.params["g_centerx"].value],  # relative coordinates inside cutout
            ycen=[fit_result.params["g_centery"].value],
            config=cfg,
            radius_val_1=[fwhm_x],
            radius_val_2=[fwhm_y],
            PA_val=[theta]
        )

        # --- populate the Table --- #
 
        # - flux peak in mJy/beam -#
        xc_rel = int(round(fit_result.params["g_centerx"].value)) 
        yc_rel = int(round(fit_result.params["g_centery"].value))
        flux_peak_mjy_pix = source_only_map[yc_rel, xc_rel]         # in mJy/pixel           
        beam_area_pix = beam_area / (pix_dim**2)               # beam area in pixel²
        flux_peak_mjy_beam = flux_peak_mjy_pix / beam_area_pix # → mJy/beam
        flux_peak.append(flux_peak_mjy_beam) 


        flux.append(phot_single["flux"][0])
        flux_err.append(phot_single["error"][0])
        
        radius_val_1.append(fwhm_x * pix_dim)          # save value in arcsec
        radius_val_2.append(fwhm_y * pix_dim)          # save value in arcsec
        PA_val.append(theta)
        updated_xcen.append(fit_result.params["g_centerx"].value + xslice.start)
        updated_ycen.append(fit_result.params["g_centery"].value + yslice.start)
        sky_val.append(bg_mean)
        poly_order_val.append(bg_order)
        nmse_val.append(final_nmse)
        redchi_val.append(final_redchi)
        bic_val.append(final_bic)
        
        fit_statuts_val.append(fit_status)
        deblend_val.append(0)       # not deblended
        cluster_val.append(1)       # only one source
        
        source_id_save.append(i+1)    #source_id to save in params files
            
    tot_fitted_isolated = len(updated_xcen)    
    logger.info(f"✓ Fitted {tot_fitted_isolated} isolated sources with Gaussian + background")

    


######################## BLENDED sources photometry ########################
    seen_groups = set()
    group_counter = 0
    total_groups = len(set([tuple(sorted(common_group[i][common_group[i] >= 0])) for i in blended]))
    
    count_blended_sources = 0
    for i in blended:
        group_indices = common_group[i]
        group_indices = group_indices[group_indices >= 0]
        group_key = tuple(sorted(group_indices))
                
        if group_key in seen_groups:
            continue  # already processed this group
    
        seen_groups.add(group_key)
        group_counter += 1
        
        #- counts indexes for plots -#
        count_source_blended_indexes = (tot_fitted_isolated + count_blended_sources +1, tot_fitted_isolated + count_blended_sources +len(group_indices))
    
        logger_file_only.info(f"Photometry on source group {group_counter} of {total_groups} ({len(group_indices)} sources)")    

        group_x = xcen[group_indices]
        group_y = ycen[group_indices]        
    
        fit_status, fit_result, model_fn, bg_order, cutout, cutout_masked_full, cutout_slice, cutout_header, bg_mean, bg_model, box_size, final_nmse, final_redchi, final_bic = fit_group_with_background(
            image=real_map,
            xcen=group_x,
            ycen=group_y,
            all_sources_xcen = all_sources_xcen,
            all_sources_ycen = all_sources_ycen,
            group_indices = group_indices,
            map_struct=map_struct,
            config=cfg,
            suffix=suffix,
            logger=logger,
            logger_file_only=logger_file_only,
            group_id=group_indices,
            count_source_blended_indexes = count_source_blended_indexes
        )
        
        
        # --- radius fixed if decided in the config file ---
        if fixed_radius == True:
            if len(fwhm_1_list) == 1:
                fwhm_x_group = np.full(len(group_indices), fwhm_1_list[0] / pix_dim)
                fwhm_y_group = np.full(len(group_indices), fwhm_2_list[0] / pix_dim)
                theta_group  = np.full(len(group_indices), PA_list[0])
            else:                
                fwhm_x_group = fwhm_1_list[group_indices] / pix_dim
                fwhm_y_group = fwhm_2_list[group_indices] / pix_dim
                theta_group = PA_list[group_indices]

        
        if fit_result is None:
            logger.error(f"Group fit failed for sources {group_key}")
            continue        
        
        # Unpack slices
        yslice, xslice = cutout_slice
        x0_global = xslice.start
        y0_global = yslice.start
        
        # Pre-evaluate full model on the same grid
        xx, yy = np.meshgrid(np.arange(cutout.shape[1]), np.arange(cutout.shape[0]))
        
        for j, idx in enumerate(group_indices):        
            # --- Create residual map where only source j is preserved ---
            params_sub = fit_result.params.copy()
            
            # Zero the current source j only
            for name in params_sub:
                if name.startswith(f"g{j}_"):
                    params_sub[name].set(value=0.0)
                
            model_without_j = model_fn(params_sub, xx, yy)
            source_only_map = cutout - model_without_j  #  subtract background + companions
                                
            # --- Extract Gaussian parameters for aperture ---
            sig_x = fit_result.params[f"g{j}_sx"].value
            sig_y = fit_result.params[f"g{j}_sy"].value
            fwhm_x = 2.3548 * sig_x  # FWHM in pixels
            fwhm_y = 2.3548 * sig_y
            theta = np.rad2deg(fit_result.params[f"g{j}_theta"].value)
            
            # --- radius fixed if decided in the config file ---
            if fixed_radius == True:
                fwhm_x = fwhm_x_group[j] 
                fwhm_y = fwhm_y_group[j]
                theta = theta_group[j]


            
            # --- Perform aperture photometry on residual image relative to cutout --- #
            phot_res = aperture_photometry_on_sources(
                image=source_only_map,
                xcen=[fit_result.params[f"g{j}_x0"].value], 
                ycen=[fit_result.params[f"g{j}_y0"].value],
                config=cfg,
                radius_val_1=[fwhm_x],
                radius_val_2=[fwhm_y],
                PA_val=[theta]
            )
            
            
            # -- Save the cutout, model, and residual maps for deblended sources as fits files -- #
            if fits_deblended:
                def save_fits(array, output_dir, label_name, extension_name, header=None):
                    
                    # Ensure the output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                     # Create the FITS filename based on the label and extension type
                    filename = f"{output_dir}/{label_name}_{extension_name}.fits"
            
                    # Create a PrimaryHDU object and write the array into the FITS file
                    hdu = fits.PrimaryHDU(data=array, header=header)
                    
                    convert_mjy=cfg.get("units", "convert_mJy")
                    if convert_mjy:
                        hdu.header['BUNIT'] = 'mJy/pixel'
                    else: hdu.header['BUNIT'] = 'Jy/pixel'    

                    hdul = fits.HDUList([hdu])
                    # Write the FITS file
                    hdul.writeto(filename, overwrite=True)
                
                save_fits(cutout, fits_output_dir_deblended, f"HYPER_MAP_{suffix}_ID_{tot_fitted_isolated + count_blended_sources +1 +j}_single_source", "cutout", header=cutout_header)
                save_fits(cutout_masked_full, fits_output_dir_deblended, f"HYPER_MAP_{suffix}_ID_{tot_fitted_isolated + count_blended_sources +1 +j}_single_source", "cutout_masked_full", header=cutout_header)
                save_fits(model_without_j, fits_output_dir_deblended, f"HYPER_MAP_{suffix}_ID_{tot_fitted_isolated + count_blended_sources +1 +j}_single_source", "model", header=cutout_header)
                save_fits(source_only_map, fits_output_dir_deblended, f"HYPER_MAP_{suffix}_ID_{tot_fitted_isolated + count_blended_sources +1 +j}_single_source", "residual", header=cutout_header)

            
            # --- visualize plots of the cutout, model, and residual maps for deblended sources an png files --- #
            if visualize_deblended:     
                plot_fit_summary(
                    cutout=cutout,                          # original cutout background subtracted
                    cutout_masked_full=cutout_masked_full,  # original cutout masked
                    model=model_without_j,               # model of only source j
                    residual=source_only_map,         # what you're analyzing
                    output_dir=visualize_output_dir_deblended,    # or configurable
                    label_name=f"HYPER_MAP_{suffix}_ID_{tot_fitted_isolated + count_blended_sources +1 +j}_single_source",  # unique label per source
                    box_size=box_size,
                    poly_order=bg_order,
                    nmse=final_nmse
                    )

            # Save results in Table 
            xc_rel = int(round(fit_result.params[f"g{j}_x0"].value)) 
            yc_rel = int(round(fit_result.params[f"g{j}_y0"].value))
            flux_peak_mjy_pix = source_only_map[yc_rel, xc_rel]         # in mJy/pixel       
            beam_area_pix = beam_area / (pix_dim**2)               # beam area in pixel²
            flux_peak_mjy_beam = flux_peak_mjy_pix / beam_area_pix # → mJy/beam
            flux_peak.append(flux_peak_mjy_beam) 
                                           
            flux.append(phot_res["flux"][0])
            flux_err.append(phot_res["error"][0])

            radius_val_1.append(fwhm_x * pix_dim)        # save value in arcsec
            radius_val_2.append(fwhm_y * pix_dim)        # save value in arcsec
            PA_val.append(theta)
            updated_xcen.append(fit_result.params[f"g{j}_x0"].value + x0_global)
            updated_ycen.append(fit_result.params[f"g{j}_y0"].value + y0_global)

            sky_val.append(bg_mean)
            poly_order_val.append(bg_order)
            nmse_val.append(final_nmse)
            redchi_val.append(final_redchi)
            bic_val.append(final_bic)
            
            fit_statuts_val.append(fit_status)
            deblend_val.append(1)                   # multi-Gaussian fit
            cluster_val.append(len(group_indices))  # number of sources in the group
            
            
        count_blended_sources = count_blended_sources + len(group_indices)        
            

    cluster_array = np.array(cluster_val)
    tot_blended_sources = np.sum(cluster_array >= 2)
    logger.info(f"✓ Fitted {tot_blended_sources} blended sources with Multiple Gaussians + background")



######################## Parameters for Table and region file ########################
            
    # Assuming you have your WCS object (usually from your FITS header)
    header = map_struct["header"]
    # Convert pixel coordinates to sky coordinates (RA, Dec)
    x_pixels = np.array(updated_xcen, dtype=np.float64)
    y_pixels = np.array(updated_ycen, dtype=np.float64)
    
    # Initialize WCS from header
    wcs = WCS(header)
    ra, dec = wcs.wcs_pix2world(x_pixels, y_pixels, 0)
    ra_save = np.where(ra < 0., ra + 360., ra)
        
    skycoords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    glon = skycoords.galactic.l.deg
    glat = skycoords.galactic.b.deg


######################## Write Table after photometry ########################

    if len(updated_xcen) == 0:
        data_dict = {
            "MAP_ID": [str(suffix)],
            "HYPER_ID": [0],
            "FLUX_PEAK": [0.0],
            "FLUX": [0.0],
            "FLUX_ERR": [0.0],
            "RESIDUALS": [0.0],
            "POLYN": [0],
            "NMSE": [0.0],
            "CHI2_RED": [0.0],
            "FWHM_1": [0.0],
            "FWHM_2": [0.0],
            "PA": [0.0],
            "STATUS": [0],
            "GLON": [0.0],
            "GLAT": [0.0],
            "RA": [0.0],
            "DEC": [0.0],
            "DEBLEND": [0],
            "CLUSTER": [0],
        }
    else:
        data_dict = {
            "MAP_ID": [str(suffix)] * len(updated_xcen),
            "HYPER_ID": list(range(1, len(updated_xcen) + 1)),
            "FLUX_PEAK": list(flux_peak),
            "FLUX": list(flux),
            "FLUX_ERR": list(flux_err),
            "RESIDUALS": list(sky_val),
            "POLYN": list(poly_order_val),
            "NMSE": list(nmse_val),
            "CHI2_RED": list(redchi_val),
            "BIC": list(bic_val),
            "FWHM_1": list(radius_val_1),
            "FWHM_2": list(radius_val_2),
            "PA": list(PA_val),
            "STATUS": list(fit_statuts_val),
            "GLON": list(glon),
            "GLAT": list(glat),
            "RA": list(ra_save),
            "DEC": list(dec),
            "DEBLEND": list(deblend_val),
            "CLUSTER": list(cluster_val),
        }

    
    
    # -- Print the output directory and file path --#
    sigma_thres=cfg.get("detection", "sigma_thres")
    write_tables(data_dict, output_dir_path, cfg, sigma_thres, real_rms, base_filename=base_name_with_suffix)
    logger_file_only.info(f"✅ eCSV and IPAC tables written to: {output_dir_path}")




    ######################## Write Region file ########################
    # Convert PA to DS9 convention
    # Initialize WCS from header
    theta_DS9 = [(pa + 90) % 180 for pa in PA_val]
    
    
    # --- Extract coordinate system ---
     # Extract WCS and coordinate system
    wcs = WCS(header)
    ra = ra_save

    # Map RADESYS to DS9 coordinate system name
    radecsys = (header.get("RADESYS") or header.get("RADECSYS") or wcs.wcs.radesys or "FK5").strip().upper()
    if "ICRS" in radecsys:
        ds9_coordsys = "icrs"
    elif "FK5" in radecsys:
        ds9_coordsys = "fk5"
    elif "GAL" in radecsys:
        ds9_coordsys = "galactic"
    else:
        ds9_coordsys = "fk5"  # Fallback
        
        
    # --- Write centroids only file ---
    with open(centroids_file, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=cyan dashlist=8 3 width=1 font='helvetica 10 normal' select=1 "
                "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0\n")
        f.write(f"{ds9_coordsys}\n")
        for xw, yw in zip(ra, dec):
            f.write(f"point({xw:.8f},{yw:.8f}) # point=cross\n")
    
    # --- Write ellipses file, with ellipses and centroids ---
    with open(ellipses_file, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=magenta dashlist=8 3 width=1 font='helvetica 10 normal' select=1 "
                "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0\n")
        f.write(f"{ds9_coordsys}\n")
        for i, (xw, yw, a, b, angle) in enumerate(zip(ra, dec, radius_val_1, radius_val_2, theta_DS9), 1):
            f.write(f"ellipse({xw:.8f},{yw:.8f},{a:.4f}\",{b:.4f}\",{angle:.3f})\n")
            f.write(f"point({xw:.8f},{yw:.8f}) # point=cross text={{ID {i}}}\n")
        
        
    if bg_model is not None:    
        return map_name, bg_model, cutout_header, header
    else:
        valid_real_map_nobg = ~np.isnan(real_map)        
        mean_valid_real_map_nobg, median_valid_real_map_nobg, std_valid_real_map_nobg = sigma_clipped_stats(real_map[valid_real_map_nobg], sigma=3.0, maxiters=5)
        real_map_nobg = np.full_like(real_map, median_valid_real_map_nobg)
        real_map_nobg[np.isnan(real_map)] = np.nan        
        bg_model = real_map_nobg
        return map_name, bg_model, header, header
    
#################################### MAIN CALL ####################################
if __name__ == "__main__":
    main()
