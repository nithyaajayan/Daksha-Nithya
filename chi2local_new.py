from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import numpy as np

def chi2localisation_new(counts_on_17_panels, NSIDE, panels, noise):
    
    panel_orient_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
                         [90, 0], [90, 45], [90, 90], [90, 135], [90, 180], 
                         [90, 225], [90, 270], [90, 315], 
                         [180, 45], [180, 135], [180, 225], [180, 315]]

    all_sources = counts_on_17_panels - noise
    all_sources[all_sources < 0] = np.nan

    top_indices = np.argsort(-all_sources)[:panels]
    top_sources = all_sources[top_indices]
    top_panels = np.array([panel_orient_list[i] for i in top_indices])

    NPIX = hp.nside2npix(NSIDE)
    theta_sky, phi_sky = hp.pix2ang(NSIDE, np.arange(NPIX))
    ra_sky = np.degrees(phi_sky)
    dec_sky = 90 - np.degrees(theta_sky)

    sky_coords = SkyCoord(ra=ra_sky * u.deg, dec=dec_sky * u.deg)
    panel_coords = SkyCoord(ra=top_panels[:, 1]*u.deg, dec=(90 - top_panels[:, 0]) * u.deg)

    gamma_matrix = panel_coords[:,None].separation(sky_coords).deg
    
    cos_gamma_matrix = np.cos(np.radians(gamma_matrix))
    cos_gamma_matrix[cos_gamma_matrix < 0] = np.nan  

    sum_cos_gamma = np.nansum(cos_gamma_matrix, axis=0)
    total_counts = np.nansum(top_sources)

    with np.errstate(divide='ignore', invalid='ignore'):
        F_estimates = total_counts / sum_cos_gamma  #(NPIX,)

    expected_all = F_estimates * cos_gamma_matrix

    chi_squared_all = np.nansum(((top_sources[:, None] - expected_all) ** 2) / top_sources[:, None], axis=0)

    min_index = np.nanargmin(chi_squared_all)
    theta_min_r, phi_min_r = hp.pix2ang(NSIDE, min_index)

    theta_min = np.degrees(theta_min_r)
    phi_min = np.degrees(phi_min_r)

    dec_min = 90 - theta_min
    ra_min = phi_min

    F_at_transient = F_estimates[min_index]

    return float(ra_min), float(dec_min), float(F_at_transient)
