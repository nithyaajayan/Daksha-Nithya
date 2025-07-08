import healpy as hp
import numpy as np
from collections import defaultdict

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)

data=np.load("data/flu_1e-6/locmul_ndir_10000_ninj_1000_flu_1e-06_alpha_-1.08_beta_-2.14_Ep_196.00_NSIDE_128_faces_08.npz",allow_pickle=True)
phi_rad = np.radians(data['true_ra'])
theta_rad =np.radians(90 - data['true_dec'])
results = data['results']
pix_id = hp.ang2pix(NSIDE,theta_rad,phi_rad)

mu_phi_chi2 = defaultdict(list)
sigma_phi_chi2 = defaultdict(list)
mu_phi_vec = defaultdict(list)
sigma_phi_vec = defaultdict(list)
mu_theta_chi2 = defaultdict(list)
sigma_theta_chi2 = defaultdict(list)
mu_theta_vec = defaultdict(list)
sigma_theta_vec = defaultdict(list)

error = defaultdict(list)

from scipy.optimize import curve_fit
from scipy.integrate import simpson

def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def plotdistribution(data, param):

    if (param == "phi"):
        bins = np.arange(-5.5, 5.6, 0.5)
        bins_cen = (bins[:-1] + bins[1:]) / 2
    if (param=="theta"):
        bins = np.arange(-5.5, 5.6, 0.5)
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
    injected = results[i,:,:]       #1000 x 6
    true_phi = np.degrees(phi_rad[i])
    true_theta = np.degrees(theta_rad[i])

    isnan=0
    for j in range(6):
        if (~np.isnan(injected[:,j])).sum()/injected[:,j].size > 0.5:
            isnan+=1
        
    if isnan<=3:
        continue

    mu_c_ph, sig_c_ph = plotdistribution(injected[:,0],param="phi")
    mu_v_ph,sig_v_ph = plotdistribution(injected[:,3],param="phi")
    mu_c_th, sig_c_th = plotdistribution(injected[:,1],param="theta")
    mu_v_th,sig_v_th = plotdistribution(injected[:,4],param="theta")

    diff_phi_chi2 = mu_c_ph - true_phi
    diff_phi_vec = mu_v_ph - true_phi
    diff_theta_chi2 = mu_c_th - true_theta
    diff_theta_vec = mu_v_th - true_theta


    mu_phi_chi2[pixel].append(diff_phi_chi2)
    sigma_phi_chi2[pixel].append(sig_c_ph)
    mu_phi_vec[pixel].append(diff_phi_vec)
    sigma_phi_vec[pixel].append(sig_v_ph)
    mu_theta_chi2[pixel].append(diff_theta_chi2)
    sigma_theta_chi2[pixel].append(sig_c_th)
    mu_theta_vec[pixel].append(diff_theta_vec)
    sigma_theta_vec[pixel].append(sig_v_th)


mu_map_chi2_ph = np.full(NPIX, np.nan)
sigma_map_chi2_ph = np.full(NPIX, np.nan)
mu_map_vec_ph = np.full(NPIX,np.nan)
sigma_map_vec_ph = np.full(NPIX,np.nan)
mu_map_chi2_th = np.full(NPIX, np.nan)
sigma_map_chi2_th = np.full(NPIX, np.nan)
mu_map_vec_th = np.full(NPIX,np.nan)
sigma_map_vec_th = np.full(NPIX,np.nan)

for pix in mu_phi_chi2:

    mu_map_chi2_ph[pix] = np.abs(np.mean(mu_phi_chi2[pix]))
    sigma_map_chi2_ph[pix] = np.abs(np.mean(sigma_phi_chi2[pix]))
    mu_map_vec_ph[pix] = np.abs(np.mean(mu_phi_vec[pix]))
    sigma_map_vec_ph[pix] = np.abs(np.mean(sigma_phi_vec[pix]))
    mu_map_chi2_th[pix] = np.abs(np.mean(mu_theta_chi2[pix]))
    sigma_map_chi2_th[pix] = np.abs(np.mean(sigma_theta_chi2[pix]))
    mu_map_vec_th[pix] = np.abs(np.mean(mu_theta_vec[pix]))
    sigma_map_vec_th[pix] = np.abs(np.mean(sigma_theta_vec[pix]))

import healpy as hp

hp.mollview(mu_map_chi2_ph, title="$\mu$ $\Delta$ $\phi$ Chi2", unit="deg")
hp.mollview(sigma_map_chi2_ph, title="$\sigma$ $\Delta$ $\phi$ Chi2", unit="deg")
hp.mollview(mu_map_vec_ph, title="$\mu$ $\Delta$ $\phi$ Vector", unit="deg")
hp.mollview(sigma_map_vec_ph, title="$\sigma$ $\Delta$ $\phi$ Vector", unit="deg")
hp.mollview(mu_map_chi2_th, title="$\mu$ $\Delta$ $\\theta$ Chi2", unit="deg")
hp.mollview(sigma_map_chi2_th, title="$\sigma$ $\Delta$ $\\theta$ Chi2", unit="deg")
hp.mollview(mu_map_vec_th, title="$\mu$ $\Delta$ $\\theta$ Vector", unit="deg")
#hp.mollview(sigma_map_vec_th, title="$\sigma$ $\Delta$ $\\theta$ Vector", unit="deg")

