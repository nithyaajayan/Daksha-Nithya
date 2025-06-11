import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import healpy as hp
import ligo.skymap.plot
from astropy.coordinates import SkyCoord
import astropy.units as u

observed_data = pd.read_csv("face_counts_new.csv",header=None, sep=' ')
noise = 3000

sources_data_all = observed_data - noise
sources_data_all[sources_data_all<0] = np.nan

source_panel_reading1 = sources_data_all.iloc[0,:].to_numpy()

n=5

top_indices = np.argsort(-source_panel_reading1)[:n]

source_panel=source_panel_reading1[top_indices]

panel_orientation_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
                          [90, 0], [90, 45], [90, 90], [90, 135], [90, 180], 
                          [90, 225], [90, 270], [90, 315], [180, 45], [180, 135], 
                          [180, 225], [180, 315]]

panel_orient = np.array([panel_orientation_list[i] for i in top_indices])

#sky coords

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)

theta,phi = hp.pix2ang(NSIDE, np.arange(NPIX))

dec = 90 - np.degrees(theta)
ra = np.degrees(phi)

sky_coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

panel_coords = SkyCoord(ra=panel_orient[:, 1] * u.deg, dec=(90 - panel_orient[:, 0]) * u.deg)

#shape: (n, NPIX), 5,12288
gamma_matrix = sky_coords.separation(panel_coords[:, None]).deg

cos_gamma_matrix = np.cos(np.radians(gamma_matrix))
cos_gamma_matrix[cos_gamma_matrix < 0] = np.nan

# Estimate F
sum_cos_gamma = np.sum(cos_gamma_matrix, axis=0)
total_counts = np.nansum(source_panel)  
with np.errstate(divide='ignore', invalid='ignore'):
    F_estimate = total_counts / sum_cos_gamma

avg_F = np.nanmean(F_estimate)

# Initialize final chi-squared map
chi_squared_all = np.zeros((NPIX))


for i in range(n):
    theta_panel, phi_panel = panel_orient[i]
    obs = source_panel[i]

    panel_dec = 90 - theta_panel
    panel_ra = phi_panel

    panel_coord = SkyCoord(ra=panel_ra * u.deg, dec=panel_dec * u.deg)
    gamma = sky_coords.separation(panel_coord).deg
    cos_gamma = np.cos(np.radians(gamma))
    #cos_gamma[cos_gamma < 0] = np.nan

    expected = avg_F * cos_gamma

    chi_squared_all += ((obs - expected) ** 2) / obs
    

min_index = np.argmin(chi_squared_all)
theta_min, phi_min = hp.pix2ang(NSIDE, min_index)
min_chi_square = chi_squared_all[min_index]
delta_chi_sq = chi_squared_all - min_chi_square

dec_min = 90 - np.degrees(theta_min)
ra_min = np.degrees(phi_min)

plt.figure(figsize=(12, 12))
ax = plt.axes(projection="astro hours mollweide")
cmap = matplotlib.colormaps.get_cmap('cylon') 
im = ax.imshow_hpx(delta_chi_sq, cmap=cmap)
ax.contour_hpx(delta_chi_sq, levels=[2.3, 4.61, 9.21 ], colors=["k", "r", "b"], linewidths=1)