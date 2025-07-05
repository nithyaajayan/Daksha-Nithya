import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson

def isotropicpoints(n):
    GRB_ra_list =[]
    GRB_dec_list=[]

    while len(GRB_ra_list)<n:
        x=np.random.uniform(0,1,n)
        theta = np.arccos(x)
        phi = np.random.uniform(0,2*np.pi,n)

        ra= np.degrees(phi)
        dec=90-np.degrees(theta)

        mask = ((ra<135)|(ra>315))&((dec>=0)&(dec<=90))

        GRB_ra_list.extend(ra[mask])
        GRB_dec_list.extend(dec[mask])

    GRB_ra_array = np.array(GRB_ra_list[:n])
    GRB_dec_array = np.array(GRB_dec_list[:n])

    return GRB_ra_array, GRB_dec_array



def observedcounts(GRB_RA,GRB_Dec,normalflux,noise):

    panel_orient_list = np.array([[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
    [90, 0], [90, 45], [90, 90], [90, 135], 
    [90, 180], [90, 225], [90, 270], [90, 315], 
    [180, 45], [180, 135], [180, 225], [180, 315]])

    GRB_coords = SkyCoord(ra=GRB_RA*u.deg, dec=GRB_Dec*u.deg)

    panel_coords = SkyCoord(ra=panel_orient_list[:,1] ,dec=90-panel_orient_list[:,0], unit='deg')

    gamma = GRB_coords.separation(panel_coords)
    cos_gamma=np.cos(np.radians(gamma))
    cos_gamma[cos_gamma<0]=0

    expected = noise + cos_gamma*normalflux
    observed = np.random.poisson(expected)

    return observed

def chi2localisation(counts_on_17_panels,NSIDE,panels,noise):

    panel_orient_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
    [90, 0], [90, 45], [90, 90], [90, 135], [90, 180], [90, 225], [90, 270], [90, 315], 
    [180, 45], [180, 135], [180, 225], [180, 315]]
    n=5
    
    all_sources = counts_on_17_panels - noise
    all_sources[all_sources<0]=0

    top_indices = np.argsort(-all_sources)[:panels]
    top_sources = all_sources[top_indices]
    top_panels = np.array([panel_orient_list[i] for i in top_indices])

    NPIX=hp.nside2npix(NSIDE)
    theta_sky,phi_sky = hp.pix2ang(NSIDE, np.arange(NPIX))
    ra_sky = np.degrees(phi_sky)
    dec_sky=90-np.degrees(theta_sky)

    sky_coords = SkyCoord(ra=ra_sky*u.deg,dec=dec_sky*u.deg)
    panel_coords = SkyCoord(ra=top_panels[:,1]*u.deg,dec=(90-top_panels[:,0])*u.deg)

    gamma_matrix = sky_coords.separation(panel_coords[:, None]).deg

    cos_gamma_matrix = np.cos(np.radians(gamma_matrix))
    cos_gamma_matrix[cos_gamma_matrix < 0] = np.nan

    # Estimate F
    sum_cos_gamma = np.sum(cos_gamma_matrix, axis=0)
    total_counts = np.nansum(top_sources)  
    with np.errstate(divide='ignore', invalid='ignore'):
        F_estimate = total_counts / sum_cos_gamma

    avg_F = np.nanmean(F_estimate)

    # Initialize final chi-squared map
    chi_squared_all = np.zeros((NPIX))


    for i in range(n):
        theta_panel, phi_panel = top_panels[i]
        obs = top_sources[i]

        panel_dec = 90 - theta_panel
        panel_ra = phi_panel

        panel_coord = SkyCoord(ra=panel_ra * u.deg, dec=panel_dec * u.deg)
        gamma = sky_coords.separation(panel_coord).deg
        cos_gamma = np.cos(np.radians(gamma))
        #cos_gamma[cos_gamma < 0] = np.nan

        expected = avg_F * cos_gamma

        chi_squared_all += ((obs - expected) ** 2) / obs
    
    min_index = np.argmin(chi_squared_all)
    theta_min_r, phi_min_r = hp.pix2ang(NSIDE, min_index)

    theta_min = np.degrees(theta_min_r)
    phi_min = np.degrees(phi_min_r)

    dec_min = 90 - theta_min
    ra_min = phi_min
    
    sky_target = SkyCoord(ra=ra_min * u.deg, dec=dec_min * u.deg)
    theta_target = np.radians(90 - sky_target.dec.deg)
    phi_target = np.radians(sky_target.ra.deg)
    pix_target = hp.ang2pix(NSIDE, theta_target, phi_target)
    F_at_transient = F_estimate[pix_target]


    return float(phi_min),float(theta_min),float(F_at_transient)



def vectorlocalisation(counts_from_17_panels,noise):

    n=3
    all_sources = counts_from_17_panels - noise
    top_indices = np.argsort(-all_sources)[:n]
    top_sources = all_sources[top_indices]

    panel_orientation_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
    [90, 0], [90, 45], [90, 90], [90, 135], [90, 180], [90, 225], [90, 270], [90, 315], 
    [180, 45], [180, 135], [180, 225], [180, 315]]

    panel_orient = [(np.radians(t),np.radians(p)) for t,p in panel_orientation_list]
    
    sources_matrix = []
    angles_matrix=[]

    for i, counts in zip(top_indices,top_sources):
        theta,phi=panel_orient[i]

        x=np.sin(theta)*np.cos(phi)
        y=np.sin(theta)*np.sin(phi)
        z=np.cos(theta)

        angles_matrix.append([x,y,z])
        sources_matrix.append(counts)
    
    sources_matrix=np.array(sources_matrix)
    angles_matrix=np.array(angles_matrix)

    angles_matrix_det = np.linalg.det(angles_matrix)
    
    if angles_matrix_det<1e-3:
        return None

    r_vec = np.zeros(3)
    for i in range(3):
        angles_matrix_copy = angles_matrix.copy()
        angles_matrix_copy[:,i] = sources_matrix
        r_vec[i] = np.linalg.det(angles_matrix_copy) / angles_matrix_det
    
    
    r_unit = r_vec / np.linalg.norm(r_vec)

    theta_r,phi_r = hp.vec2ang(r_unit)

    phi_min = np.degrees(phi_r.item())
    theta_min = np.degrees(theta_r.item())

    return phi_min, theta_min, np.linalg.norm(r_vec)


def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def plotdistribution(data, param,true_val,method):
    fig, ax = plt.subplots(figsize=(8, 4))

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

    ax.errorbar(bins_cen, norm_counts, yerr=errors, color='skyblue', label='Data',ds='steps-mid')

    p0 = [np.median(data), np.std(data)]

    popt, pcov = curve_fit(gaussian, bins_cen, norm_counts, p0=p0, sigma=errors, absolute_sigma=True)

    ax.axvline(popt[0], color='red', linestyle='--')
    ax.axvline(true_val, color='black', linestyle='-', label='True Value')

    y_vals = gaussian(bins_cen, *popt)
    ax.plot(bins_cen, y_vals, 'r--')

    title = f"Injected {method} {param}: {true_val:.4f} $\mu$={popt[0]:.2f}, $\sigma$={popt[1]:.2f}"
    ax.set_title(title)
    ax.set_xlabel(f"{param}")
    ax.set_ylabel("Normalised Counts")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig


def bandcounts(E, A, alpha, beta, Ep):

    E0= Ep/(alpha+2)
        
    return np.where(
        E < (alpha-beta)*E0, A * (E / 100)**alpha * np.exp(-E / E0),
        A * (((alpha-beta)*E0 / 100)**(alpha - beta)) * np.exp(beta - alpha) * (E / 100)**beta)