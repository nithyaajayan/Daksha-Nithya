import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

observations = pd.read_csv('facecounts.csv',header=None, sep=' ')
observed_normal = 2780
noise = 2500

source_counts = observations - noise
expected_normal = observed_normal - noise

panel_sources = [source_counts.iloc[:,i].to_numpy() for i in range(source_counts.shape[1])]

panel_orientation_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270],[90, 0], [90, 45], [90, 90], [90, 135], [90, 180], [90, 225], [90, 270], [90, 315]]
panel_orients = np.array(panel_orientation_list)

#Sky Coordinates

NSIDE = 32
NPIX = hp.nside2npix(NSIDE) #12288 grid points

theta,phi = hp.pix2ang(NSIDE, np.arange(NPIX))

dec = 90 - np.degrees(theta)
ra = np.degrees(phi)

sky_coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

#Panel Coordinated and Chi Square computation

chi_squared_list = []

for i,(theta,phi) in enumerate(panel_orients):
    panel_dec = 90-theta
    panel_ra = phi

    panel_coords = SkyCoord(ra=panel_ra*u.deg, dec=panel_dec*u.deg)

    gamma = sky_coords.separation(panel_coords).deg
    cos_gamma = np.cos(np.radians(gamma))
    cos_gamma[cos_gamma<0] = 0

    expected_counts = expected_normal * cos_gamma
    observed_counts = panel_sources[i]

    chi_sq = np.zeros_like(expected_counts)

    for obs in observed_counts:
        with np.errstate(divide='ignore',invalid='ignore'):
            chi_sq += np.where(expected_counts>0, ((obs-expected_counts)**2)/expected_counts,np.inf)
    
    chi_squared_list.append(chi_sq)

chi_squared_stack = np.stack(chi_squared_list,axis=0)
chi_squared_final = np.min(chi_squared_stack, axis=0)

#Minimum value of chi square and its coords

min_index = np.argmin(chi_squared_final)
theta_min,phi_min = hp.pix2ang(NSIDE,min_index)

theta_deg = np.degrees(theta_min)
phi_deg = np.degrees(phi_min)

hp.mollview(chi_squared_final, title='Chi-Squared Sky Map',unit=r'chi_square',cmap='YlOrRd',norm='hist')
hp.projplot(theta_min, phi_min, 'ko', markersize=5)
hp.projtext(theta_min, phi_min - 0.1, f"({int(theta_deg)}°, {int(phi_deg)}°)", color='black', fontsize=10)
hp.graticule()
plt.show()