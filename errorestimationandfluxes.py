# %%
import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

observed_data = pd.read_csv("face_counts_new.csv", header=None, sep=' ')
noise = 3000
sources_data_all = observed_data - noise
sources_data_all[sources_data_all < 0] = np.nan

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
panel_orientation_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270],
    [90, 0], [90, 45], [90, 90], [90, 135], [90, 180],
    [90, 225], [90, 270], [90, 315], [180, 45], [180, 135],
    [180, 225], [180, 315]]

theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
sky_coords = SkyCoord(ra=np.degrees(phi)*u.deg, dec=(90 - np.degrees(theta))*u.deg)

result_list = []

for obs_index in range(10): 
    source_panel_reading = sources_data_all.iloc[obs_index, :].to_numpy()
    n = 8
    top_indices = np.argsort(-source_panel_reading)[:n]
    source_panel = source_panel_reading[top_indices]
    panel_orient = np.array([panel_orientation_list[i] for i in top_indices])

    panel_coords = SkyCoord(ra=panel_orient[:, 1] * u.deg,dec=(90 - panel_orient[:, 0]) * u.deg)

    gamma_matrix = sky_coords.separation(panel_coords[:, None]).deg
    cos_gamma_matrix = np.cos(np.radians(gamma_matrix))
    cos_gamma_matrix[cos_gamma_matrix < 0] = np.nan

    # Estimate F
    sum_cos_gamma = np.sum(cos_gamma_matrix, axis=0)
    total_counts = np.nansum(source_panel)
    with np.errstate(divide='ignore', invalid='ignore'):
        F_estimate = total_counts / sum_cos_gamma
    avg_F = np.nanmean(F_estimate)

    #chi2 map
    chi_squared_all = np.zeros((NPIX))
    for i in range(n):
        theta_panel, phi_panel = panel_orient[i]
        obs = source_panel[i]
        panel_coord = SkyCoord(ra=phi_panel * u.deg, dec=(90 - theta_panel) * u.deg)
        
        gamma = sky_coords.separation(panel_coord).deg
        cos_gamma = np.cos(np.radians(gamma))
        expected = avg_F * cos_gamma
        chi_squared_all += ((obs - expected) ** 2) / obs

    #min location
    min_index = np.argmin(chi_squared_all)
    theta_min, phi_min = hp.pix2ang(NSIDE, min_index)
    ra_min = np.degrees(phi_min)
    dec_min = 90 - np.degrees(theta_min)

    sky_target = SkyCoord(ra=ra_min * u.deg, dec=dec_min * u.deg)
    theta_target = np.radians(90 - sky_target.dec.deg)
    phi_target = np.radians(sky_target.ra.deg)
    pix_target = hp.ang2pix(NSIDE, theta_target, phi_target)
    F_at_transient = F_estimate[pix_target]

    #error
    delta_chi = chi_squared_all - np.min(chi_squared_all)
    ra_deg = np.degrees(phi)
    dec_deg = 90 - np.degrees(theta)
    points = np.vstack((ra_deg, dec_deg)).T
    values = delta_chi

    def walk_line(ra0, dec0, d_ra=0.1, d_dec=0.1, axis='ra'):
        errors = []
        for direction in [1, -1]:
            for i in range(1, 200):  # max 20 degrees
                ra = ra0 + direction * d_ra * i if axis == 'ra' else ra0
                dec = dec0 + direction * d_dec * i if axis == 'dec' else dec0
                if not (-90 <= dec <= 90): break
                ra = ra % 360
                test_point = np.array([[ra, dec]])
                interpolated_val = griddata(points, values, test_point, method='linear')[0]
                if interpolated_val is None or interpolated_val >= 2.3:
                    center = SkyCoord(ra0*u.deg, dec0*u.deg)
                    edge = SkyCoord(ra*u.deg, dec*u.deg)
                    sep = center.separation(edge).deg
                    errors.append(sep)
                    break
        return np.mean(errors) if errors else np.nan

    ra_error = walk_line(ra_min, dec_min, axis='ra')
    dec_error = walk_line(ra_min, dec_min, axis='dec')

    result_list.append({
        'Row': obs_index + 1,
        'RA (deg)': ra_min,
        'Dec (deg)': dec_min,
        'RA error (deg)': ra_error,
        'DEC error (deg)': dec_error,
        'Estimated Flux': F_at_transient
    })

df_all_fluxes = pd.DataFrame(result_list)
print(df_all_fluxes)


# %%



