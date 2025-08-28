import os

import matplotlib.pyplot as plt
import numpy as np


def plot_fit_summary(cutout, cutout_masked_full, model, residual, output_dir, label_name="fit", dpi=300,
                     box_size=None, poly_order=None, nmse=None):
    """
    Save 2D and 3D plots of cutout, model, and residual.

    Parameters
    ----------
    cutout : 2D array
        Original data image (cutout).
    model : 2D array
        Fitted Gaussian + background model.
    residual : 2D array
        Difference between data and model.
    output_dir : str
        Directory to save PNG files.
    label : str
        Filename prefix for saving.
    dpi : int
        Resolution for PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shared color scale based on cutout range, ignoring NaNs
    vmin_shared = np.nanmin(cutout)
    vmax_shared = np.nanmax(cutout)
    
    # Individual residual rescaled range, ignoring NaNs
    vmin_resid = np.nanmin(residual)
    vmax_resid = np.nanmax(residual)

    data_list = [cutout, cutout_masked_full, model, residual, residual]
    file_tags = ["cutout", "cutout masked full", "model", "residual", "residual_rescaled"]

    if box_size is not None:
        titles = [
            f"Map Cutout Back. subt.  • Box = {box_size} px   ",
            f"Original Map Cutout masked  • Box = {box_size} px   ",
            f"Model (Gauss. + Backgr.)  • Box = {box_size} px   ",
            f"Residual (Data − Model)  • Box = {box_size} px   ",
            f"Residual (Rescaled)  • Box = {box_size} px   "
        ]
    else:
        titles = [
            "Map Cutout Back. subt  • Box = NaN ",
            "Original Map Cutout masked  • Box = NaN   ",
            "Model (Gauss. + Backgr.)  • Box = NaN ",
            "Residual (Data − Model)  • Box = NaN ",
            "Residual (Rescaled)  • Box = NaN "
        ]

    # Color limits: first three plots share cutout scale, last uses residual scale
    vmins = [vmin_shared, vmin_shared, vmin_shared, vmin_resid]
    vmaxs = [vmax_shared, vmax_shared, vmax_shared, vmax_resid]


    # ------------------------ 2D plots --------------------------- #
    for arr, title, tag, vmin, vmax in zip(data_list, titles, file_tags, vmins, vmaxs):
        fig, ax = plt.subplots(figsize=(5, 4))
        
        im = ax.imshow(arr, origin='lower', cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Flux (Jy)", fontweight="bold", fontsize=9)
    
        # Bold colorbar tick labels
        cb.ax.tick_params(labelsize=8)
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")
    
        # Bold axis labels
        ax.set_xlabel("X (pix)", fontweight="bold", fontsize=9)
        ax.set_ylabel("Y (pix)", fontweight="bold", fontsize=9)
    
        # Bold tick labels
        ax.tick_params(axis='both', labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
    
        # Format title with fit info, if provided
        if box_size is not None and poly_order is not None and nmse is not None:
            subtitle = f"• Polynomial Order = {poly_order}   • NMSE = {nmse:.3f}"
            full_title = f"{title}\n{subtitle}"
        else:
            full_title = title
    
        ax.set_title(full_title, fontsize=10, fontweight="bold", linespacing=1.4)
    
        # Save with tight bounding box
        outname = os.path.join(output_dir, f"{label_name}_{tag}_2D.png")
        fig.savefig(outname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    

    # ------------------------ 3D surface plots --------------------------- #
    ny, nx = cutout.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    for arr, title, tag, vmin, vmax in zip(data_list, titles, file_tags, vmins, vmaxs):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot 3D surface
        surf = ax.plot_surface(X, Y, arr, cmap='viridis',
                               vmin=vmin, vmax=vmax,
                               linewidth=0, antialiased=True)
    
        # Z limits scaled to color range
        ax.set_zlim(vmin, vmax)
    
        # Set title with fit info
        if poly_order is not None and nmse is not None:
            subtitle = f"• Polynomial Order = {poly_order}   • NMSE = {nmse:.3f}"
            full_title = f"{title}\n{subtitle}"
        else:
            full_title = title
    
        ax.set_title(full_title, fontsize=9, fontweight="bold", pad=-40)
    
        # Axis labels
        ax.set_xlabel("X (pix)", fontweight="bold", fontsize=9, labelpad=2)
        ax.set_ylabel("Y (pix)", fontweight="bold", fontsize=9, labelpad=2)
        ax.set_zlabel("Flux (Jy)", fontweight="bold", fontsize=9, labelpad=2)
    
        # Bold tick labels
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(8)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(8)
        for label in ax.get_zticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(8)
    
        # Adjust margins to make room
        fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    
        # Save using tight layout with padding
        outname = os.path.join(output_dir, f"{label_name}_{tag}_3D.png")
        fig.savefig(outname, dpi=dpi, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)