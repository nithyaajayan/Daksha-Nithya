# %% [markdown]
# Points to be Simulated 

# %%
import numpy as np

def uniform_points(n):
    x = np.random.uniform(0,1,n)
    theta = np.arccos(x)
    phi = np.random.uniform(0, 2 * np.pi, n)

    GRB_ra = np.degrees(phi)
    GRB_dec = 90-np.degrees(theta)

    return GRB_ra,GRB_dec
    

# %% [markdown]
# Simulate Injections

# %%
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def simulate_observed_counts(GRB_RA,GRB_Dec,normalflux,noise):
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

# %% [markdown]
# Chi2 Localisation

# %%
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord

def Chi2_localisation(counts_on_17_panels,noise):
    panel_orient_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], 
    [90, 0], [90, 45], [90, 90], [90, 135], [90, 180], [90, 225], [90, 270], [90, 315], 
    [180, 45], [180, 135], [180, 225], [180, 315]]

    all_sources = counts_on_17_panels - noise
    all_sources[all_sources<0]=0

    n=5
    top_indices = np.argsort(-all_sources)[:n]
    top_sources = all_sources[top_indices]
    top_panels = np.array([panel_orient_list[i] for i in top_indices])

    NSIDE=128
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
    theta_min, phi_min = hp.pix2ang(NSIDE, min_index)
    dec_min = 90 - np.degrees(theta_min)
    ra_min = np.degrees(phi_min)
    
    sky_target = SkyCoord(ra=ra_min * u.deg, dec=dec_min * u.deg)
    theta_target = np.radians(90 - sky_target.dec.deg)
    phi_target = np.radians(sky_target.ra.deg)
    pix_target = hp.ang2pix(NSIDE, theta_target, phi_target)
    F_at_transient = F_estimate[pix_target]


    return float(ra_min),float(dec_min),float(F_at_transient)

# %% [markdown]
# Vector Localisation

# %%
import numpy as np
import healpy as hp

def Vector_localisation(counts_from_17_panels,noise):
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
    
    if angles_matrix_det<=0:
        return None

    r_vec = np.zeros(3)
    for i in range(3):
        angles_matrix_copy = angles_matrix.copy()
        angles_matrix_copy[:,i] = sources_matrix
        r_vec[i] = np.linalg.det(angles_matrix_copy) / angles_matrix_det
    
    
    r_unit = r_vec / np.linalg.norm(r_vec)

    theta_r,phi_r = hp.vec2ang(r_unit)

    ra = np.degrees(phi_r.item())
    dec= 90 - np.degrees(theta_r.item())

    return ra, dec, np.linalg.norm(r_vec)

# %% [markdown]
# Plot Distribution Function

# %%
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_distribution(data1, data2, label1, label2, true_val, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 4))

    bins =100
    ax.hist(data1, bins=bins, histtype='step', linewidth=2, label=label1, color='skyblue', density=True)
    ax.hist(data2, bins=bins, histtype='step', linewidth=2, label=label2, color='red', density=True)

    # Mean
    ax.axvline(np.mean(data1), color='skyblue', linestyle='--', label=f'{label1} mean={np.mean(data1):.2f}')
    ax.axvline(np.mean(data2), color='red', linestyle='--', label=f'{label2} mean={np.mean(data2):.2f}')
    ax.axvline(true_val, color='k', linestyle='-', label='True Value')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig


# %% [markdown]
# Database

# %%
import pandas as pd
import numpy as np

n=300
noise=3000
flux=2780
database=[]
chi2_result=[]
vector_result=[]
true_GRB_ra, true_GRB_dec = [],[]

GRB_ra_array,GRB_dec_array = uniform_points(n)

for GRB_ra,GRB_dec in zip(GRB_ra_array,GRB_dec_array):
    counts = simulate_observed_counts(GRB_ra,GRB_dec,flux,noise)

    vec_result = Vector_localisation(counts,noise)
    
    if vec_result is None:
        continue

    chira,chidec,chiflux = Chi2_localisation(counts,noise)
    chi2_result.append((chira,chidec,chiflux))

    vecra,vecdec,vecflux = Vector_localisation(counts,noise)
    vector_result.append((vecra,vecdec,vecflux))

    database.append({'Counts': counts,'GRB_ra': GRB_ra,'GRB_dec': GRB_dec})

    true_GRB_ra.append(GRB_ra)
    true_GRB_dec.append(GRB_dec)

df=pd.DataFrame(database)
df.to_csv('100uniformcounts.csv')

chi2_ra,chi2_dec,chi2_flux = np.array(chi2_result).T
vec_ra,vec_dec,vec_flux = np.array(vector_result).T

#errors
chi2ra_error = chi2_ra- np.array(true_GRB_ra)
chi2dec_error = chi2_dec - np.array(true_GRB_dec)
chi2flux_error = chi2_flux - flux

vecra_error = vec_ra-np.array(true_GRB_ra)
vecdec_error = vec_dec - np.array(true_GRB_dec)
vecflux_error = vec_flux - flux 


# %%
plot_distribution(chi2ra_error,vecra_error,'Chi2','Vector',0,'RA error','difference')
plot_distribution(chi2dec_error,vecdec_error,'Chi2','Vector',0,'Dec error','difference')
#plot_distribution(chi2flux_error,vecflux_error,'Chi2','Vector',0,'Flux error','difference')





