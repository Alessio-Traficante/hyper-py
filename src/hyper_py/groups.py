import numpy as np

def group_sources(xcen, ycen, pix_dim, beam_dim, aper_sup):
    '''
    Groups sources based on proximity within the beam scale, ensuring no duplicate groups and transitive merging.

    Parameters:
        xcen, ycen: arrays of source positions in pixels
        pix_dim: pixel scale (arcsec)
        beam_dim: beam size (arcsec)
        aper_sup: aperture scaling factor

    Returns:
        start_group: 1 for blended sources, 0 otherwise
        common_group: 2D array of group membership (list of sources per group)
        deblend: number of neighbors (for Gaussian deblending)
    '''
    n = len(xcen)
    xcen = np.array(xcen)
    ycen = np.array(ycen)

    max_dist = beam_dim * aper_sup * 2.0
    max_dist_pix = max_dist / pix_dim
        
    start_group = np.zeros(n, dtype=int)
    common_group = -1 * np.ones((n, n), dtype=int)  # Initialize common_group as a 2D array
    deblend = np.zeros(n, dtype=int)
    
    # Each source is initially its own group
    group_assignment = np.arange(n, dtype=int)
    
    def find(group_id):
        if group_assignment[group_id] != group_id:
            group_assignment[group_id] = find(group_assignment[group_id])  # Path compression
        return group_assignment[group_id]
    
    def union(group1, group2):
        root1 = find(group1)
        root2 = find(group2)
        if root1 != root2:
            group_assignment[root2] = root1  # Merge the groups

    # First pass: union sources that are within max_dist_pix of each other
    for i in range(n):
        dx = xcen[i] - xcen
        dy = ycen[i] - ycen
        dist = np.sqrt(dx**2 + dy**2)
        same_group = np.where(dist < max_dist_pix)[0]

        for j in same_group:
            if find(i) != find(j):
                union(i, j)

    # Essential fix: flatten all group pointers after union phase
    for i in range(n):
        group_assignment[i] = find(i)

    # Second pass: assign group info for each source (same as your original)
    for i in range(n):
        group_members = np.where(group_assignment == group_assignment[i])[0]
        common_group[i, :len(group_members)] = group_members
        deblend[i] = len(group_members) - 1
        if len(group_members) > 1:
            start_group[i] = 1

    return start_group, common_group, deblend