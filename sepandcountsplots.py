# %%
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy_healpix as ah
import ligo.skymap.moc
import astropy.constants as const
from astropy import table
from scipy.special import gammaln
import matplotlib.pyplot as plt

# def expected_counts(x, Nm_inj, theta_inj, phi_inj, B_inj, detector_conf, burst_duration):
def expected_counts(x: np.ndarray, Nm_inj: float, theta_inj: float, phi_inj: float, B_inj: float, detector_conf: np.ndarray, burst_duration: float)->np.ndarray:
    """
    Given the source parameters and detector configuration, calculates the expected counts on each surface
    
    Parameters
    ----------
    x: array
        Array containing indices of the detector faces
    Nm_inj: float
        Number of photons emitted by the source (at normal incidence) at one face
    theta_inj: float
        Declination of the source
    phi_inj: float
        Right ascension of the source
    B_inj: float
        Background counts per second per surface
    detector_conf: array
        Array containing the detector configuration (theta, phi) for each face
    burst_duration: float
        Duration of the burst in seconds

    Returns
    -------
    n_expected: array
        Array containing the expected counts on each surface
    """

    x_src=np.sin(theta_inj*u.deg)*np.cos(phi_inj*u.deg)
    y_src=np.sin(theta_inj*u.deg)*np.sin(phi_inj*u.deg)
    z_src=np.cos(theta_inj*u.deg)
    V_source=np.array([x_src,y_src,z_src])

    x_surf=np.sin(detector_conf[:,0]*u.deg)*np.cos(detector_conf[:,1]*u.deg)
    y_surf=np.sin(detector_conf[:,0]*u.deg)*np.sin(detector_conf[:,1]*u.deg)
    z_surf=np.cos(detector_conf[:,0]*u.deg)
    V_surface=np.array([x_surf,y_surf,z_surf])

    n_expected=Nm_inj*np.dot(V_source.T,V_surface)
    n_expected[n_expected<0]=0
    n_expected+=B_inj*burst_duration
    return n_expected

def observed_counts(n_expected):

    n_observed= np.random.poisson(n_expected)
    return n_observed

panel_orientation_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
                          [90, 0], [90, 45], [90, 90], [90, 135], [90, 180], 
                          [90, 225], [90, 270], [90, 315], [180, 45], [180, 135], 
                          [180, 225], [180, 315]]

panel_orient = np.array(panel_orientation_list)

x = np.array(range(17))
detector_conf = panel_orient
burst_duration = 1
B_inj = 3000
Nm_inj = 2780

dataset = pd.read_csv('projection_result_fixed_NEW.csv',skiprows=1,header=None,sep=' ')

ra_dataset = dataset[18].to_numpy()
dec_dataset = dataset[19].to_numpy()
#print(ra_dataset,dec_dataset)

obs_counts_set = dataset.iloc[:,1:18].to_numpy()
#print(obs_counts_set.shape)

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("GRB_analysis_plots.pdf") as pdf:
    for i in range(10):
        exp_counts = expected_counts(x, Nm_inj, dec_dataset[i], ra_dataset[i], B_inj, detector_conf, burst_duration)
        est_obscounts = observed_counts(exp_counts)

        panel_coords = SkyCoord(ra=panel_orient[:, 1] * u.deg,dec=(90 - panel_orient[:, 0]) * u.deg)
        GRB_coords = SkyCoord(ra = ra_dataset[i]*u.deg,dec=dec_dataset[i]*u.deg)

        gamma = GRB_coords.separation(panel_coords)
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top plot: Gamma (angular separation)
        axs[0].plot(gamma, marker='o', color='teal')
        axs[0].set_ylabel("Separation (deg)")
        axs[0].set_title(f"Source {i+1}")
        axs[0].grid(True)

        # Bottom plot: Simulated vs Actual observed counts
        axs[1].plot(est_obscounts, label='Simulated from Given Coords', marker='x')
        axs[1].plot(obs_counts_set[i], label='Given Observed Counts', marker='s')
        axs[1].set_xlabel("Phase") 
        axs[1].set_ylabel("Counts")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)




