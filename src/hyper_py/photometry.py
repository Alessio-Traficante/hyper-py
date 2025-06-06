
import numpy as np
from photutils.aperture import EllipticalAperture, aperture_photometry
from astropy.table import Table

def aperture_photometry_on_sources(image, xcen, ycen, config,
                                   radius_val_1=None, radius_val_2=None, PA_val=None):
    '''
    Perform elliptical aperture photometry on given source positions.

    Parameters:
        image : 2D array
        xcen, ycen : 1D arrays of centroid positions (in pixels)
        config : HyperConfig instance
        radius_val_1, radius_val_2, PA_val : arrays of ellipse parameters in pixels
            If None, use values from config.

    Returns:
        Table with flux and uncertainty
    '''


    if len(radius_val_1) == 1:
        radius_val_1 = radius_val_1 * len(xcen)
    if len(radius_val_2) == 1:
        radius_val_2 = radius_val_2 * len(xcen)
    if len(PA_val) == 1:
        PA_val = PA_val * len(xcen)

    fluxes = []
    errors = []

    for i in range(len(xcen)):   
        a = radius_val_1[i]
        b = radius_val_2[i]
        theta = np.deg2rad(PA_val[i] + 90.)  # rotated only for photometry (x and y axes inverted here)
        
        position = (xcen[i], ycen[i])
        aperture = EllipticalAperture(position, a, b, theta=theta)
        
        # print(aperture)    
        phot_table = aperture_photometry(image, aperture, method='exact')
        flux = phot_table['aperture_sum']
        
        # print(phot_table)
        
        # --- Perform area-weighted photometry on an image within an elliptical aperture --- #
        # flux = area_weighted_photometry(image, xcen[i], ycen[i], a, b, theta)
         
        
        # Estimate noise inside aperture
        mask = aperture.to_mask(method="exact")
        data_cutout = mask.cutout(image)
        if data_cutout is not None:
            aperture_data = data_cutout * mask.data
            noise = np.nanstd(aperture_data)
            error = noise * np.sqrt(np.sum(mask.data**2))
        else:
            error = 0.0

        fluxes.append(flux)
        errors.append(error)

    return Table(data={"x": xcen, "y": ycen, "flux": fluxes, "error": errors})




# --- Function to calculate area-weighted flux inside an elliptical aperture --- #
def area_weighted_photometry(image, xcen, ycen, a, b, theta, method="exact"):
    """
    Perform area-weighted photometry on an image within an elliptical aperture.

    Parameters:
        image: 2D numpy array of image data
        xcen, ycen: coordinates of the center of the aperture
        a, b: semi-major and semi-minor axes of the elliptical aperture
        theta: position angle of the ellipse
        method: method used for photometry (not used here but can be extended)

    Returns:
        total_flux: total flux inside the elliptical aperture
    """
    total_flux = 0
    
    # Get the shape of the image
    ysize, xsize = image.shape

    # Create a meshgrid for pixel coordinates (Note the order: y-axis, then x-axis)
    y, x = np.indices((ysize, xsize))  # y -> rows, x -> columns


    # Transform the pixel coordinates to match the elliptical aperture
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))

    # Rotate the coordinates according to the position angle
    x_rot = (x - xcen) * cos_theta + (y - ycen) * sin_theta
    y_rot = -(x - xcen) * sin_theta + (y - ycen) * cos_theta

    # Check if each pixel lies within the elliptical aperture
    inside_aperture = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1
    

    # Now calculate the weighted flux (sum of fluxes for each pixel inside the aperture)
    for i in range(ysize):
        for j in range(xsize):
            if inside_aperture[i, j]:
                # For pixels inside the ellipse, add their full flux
                total_flux += image[i, j]
                # print(image[i, j])
    return total_flux
    

