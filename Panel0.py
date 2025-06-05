import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

#recorded values without background (= -2500)
recorded = [1349, 2470,  781,  635, 1840,  461,  486, 3052,  505, 2647] 
expected_normal = 280

#panel orientation
panel_theta = 0
panel_phi = 0

#in celestial coordinates
panel_dec = 90-panel_theta
panel_ra = panel_phi

panel_coord = SkyCoord(ra=panel_ra*u.deg, dec=panel_dec*u.deg)

# Sky grid
theta_vals = np.linspace(0, 90, 50)   # elevation
phi_vals = np.linspace(0, 90, 50)     # azimuth

theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

#in celestial coordinates
dec_grid = 90 - theta_grid
ra_grid = phi_grid

sky_coords = SkyCoord(ra=ra_grid*u.deg, dec=dec_grid*u.deg)

gamma_angle = sky_coords.separation(panel_coord).deg
cos_gamma = np.cos(np.radians(gamma_angle))

expected_counts = expected_normal * cos_gamma

chi_squared = np.zeros_like(expected_counts) #create 50x50 zero matrix
for obs in recorded:
    chi_squared += ((obs-expected_counts)**2)/(expected_counts)

#---------------------------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')

lon = np.linspace(0, np.pi/2,50)
lat = np.linspace(0, np.pi/2,50)

Lon,Lat = np.meshgrid(lon,lat)

#im = ax.pcolormesh(Lon,Lat,chi_squared, cmap=plt.cm.jet)
#plt.show()

c = ax.pcolormesh(Lon, Lat, chi_squared, cmap='jet', shading='auto')
plt.colorbar(c, ax=ax, orientation='horizontal', label='Chi-squared')

ax.set_title("Chi-Squared Map")
ax.grid(True)
plt.tight_layout()
plt.show()
