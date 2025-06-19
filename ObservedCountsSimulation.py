import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def simulate_detector_counts(ra,dec,normalflux,noise):
    
    panel_orientation_list = np.array([[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
    [90, 0], [90, 45], [90, 90], [90, 135], 
    [90, 180], [90, 225], [90, 270], [90, 315], 
    [180, 45], [180, 135], [180, 225], [180, 315]])
    
    face_coords = SkyCoord(ra=panel_orientation_list[:, 1], dec=90 - panel_orientation_list[:, 0], unit='deg')

    GRB_coords = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)

    gamma = face_coords.separation(GRB_coords)
    cos_gamma = np.cos(np.radians(gamma))
    cos_gamma[cos_gamma<0]=0

    expected_counts = noise + normalflux*cos_gamma

    observed_counts = np.random.poisson(expected_counts)

    return observed_counts



