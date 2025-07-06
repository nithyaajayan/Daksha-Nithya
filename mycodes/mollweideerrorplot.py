import healpy as hp
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import healpy as hp
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)

data=np.load("data/test/locmul_ndir_00100_ninj_1000_flu_1e-06_alpha_-1.00_beta_-1.50_Ep_1000.00_NSIDE_128_faces_05.npz",allow_pickle=True)
phi_rad = np.radians(data['true_ra'])
theta_rad =np.radians(90 - data['true_dec'])
results = data['results']
pix_id = hp.ang2pix(NSIDE,theta_rad,phi_rad)

mu_phi_chi2 = defaultdict(list)
sigma_phi_chi2 = defaultdict(list)
mu_phi_vec = defaultdict(list)
sigma_phi_vec = defaultdict(list)

def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def plotdistribution(data, param):

    if (param == "phi"):
        bins = np.arange(-5.5, 5.6, 0.5)
        bins_cen = (bins[:-1] + bins[1:]) / 2
    if (param=="theta"):
        bins = np.arange(-5.5, 5.6, 0.5)
        bins_cen = (bins[:-1] + bins[1:]) / 2
    if (param=="fluence"):
        bins = np.arange(-150.5, 150.6, 5)
        bins_cen = (bins[:-1] + bins[1:]) / 2

    data = data[~np.isnan(data)]

    bins_cen += np.median(data)
    bins += np.median(data)

    counts, _ = np.histogram(data, bins=bins)

    total_counts = simpson(counts, bins_cen)
    norm_counts = counts / total_counts

    errors = np.sqrt(counts) / total_counts

    p0 = [np.mean(data), np.std(data)]

    popt, _ = curve_fit(gaussian, bins_cen, norm_counts, p0=p0, sigma=errors, absolute_sigma=True)

    return popt[0],popt[1]


for i in range(len(results)):
    pixel = pix_id[i]
    injected = results[i,:,:]
    true_phi = np.degrees(phi_rad[i])

    isnan=0
    for j in range(6):
        if (~np.isnan(injected[:,j])).sum()/injected[:,j].size > 0.5:
            isnan+=1
        
    if isnan!=6:
        continue

    diff_phi_chi2 = injected[:,0] - true_phi
    diff_phi_vec = injected[:,3] - true_phi

    mu_c, sig_c = plotdistribution(diff_phi_chi2,param="phi")
    mu_v,sig_v = plotdistribution(diff_phi_vec,param="phi")

    mu_phi_chi2[pixel].append(mu_c)
    sigma_phi_chi2[pixel].append(sig_c)
    mu_phi_vec[pixel].append(mu_v)
    sigma_phi_vec[pixel].append(sig_v)


mu_map_chi2 = np.full(NPIX, np.nan)
sigma_map_chi2 = np.full(NPIX, np.nan)
mu_map_vec = np.full(NPIX,np.nan)
sigma_map_vec = np.full(NPIX,np.nan)

for pix in mu_phi_chi2:
    mu_map_chi2[pix] = np.abs(np.mean(mu_phi_chi2[pix]))
    sigma_map_chi2[pix] = np.abs(np.mean(sigma_phi_chi2[pix]))
    mu_map_vec[pix] = np.abs(np.mean(mu_phi_vec[pix]))
    sigma_map_vec[pix] = np.abs(np.mean(sigma_phi_vec[pix]))

hp.mollview(mu_map_chi2, title="$\mu$ $\Delta$ $\phi$ Chi2", unit="deg")
hp.mollview(sigma_map_chi2, title="$\sigma$ $\Delta$ $\phi$ Chi2", unit="deg")
hp.mollview(mu_map_vec, title="$\mu$ $\Delta$ $\phi$ Vector", unit="deg")
hp.mollview(sigma_map_vec, title="$\sigma$ $\Delta$ $\phi$ Vector", unit="deg")




