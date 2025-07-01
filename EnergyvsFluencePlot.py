import pandas as pd
import numpy as np

data = pd.read_csv("Daksha-Nithya/mycodes/czt.dat", comment='#',
                   sep='\s+', header=None,     
                   names=["Energy_MeV", "Coherent", "Incoherent", "Photoelectric",
                          "PairProd_Nuclear", "PairProd_Electron", "Total_with_Coherent", "Total_wo_Coherent"])

mass_thickness_coeff = data["Total_with_Coherent"].to_numpy()
density=5.78
thickness = 1
linear_att = mass_thickness_coeff*density*thickness

eff = 1 - np.exp((-1)*linear_att)

def bandcounts(E, A, alpha, beta, Ep):

    E0= Ep/(alpha+2)
        
    return np.where(
        E < (alpha-beta)*E0, A * (E / 100)**alpha * np.exp(-E / E0),
        A * (((alpha-beta)*E0 / 100)**(alpha - beta)) * np.exp(beta - alpha) * (E / 100)**beta)

from scipy.integrate import simpson
import astropy.units as u

alpha=-1
beta=-1.5
delT=1
Ep=1e3
area = 304
targetfluence=1e-6

E_MeV = data["Energy_MeV"].to_numpy()
E_grid = E_MeV*1000

integrand = E_grid*u.keV.to(u.erg) * bandcounts(E_grid,1,alpha,beta,Ep)  #erg/cm2/sec/keV

fluence_temp = simpson(integrand, E_grid) *delT #erg/cm2

A_scaling = targetfluence / fluence_temp
band_counts_new = eff*bandcounts(E_grid,A_scaling,alpha,beta,Ep)

#photons = simpson(band_counts_new,E_grid) *area *delT

import matplotlib.pyplot as plt

plt.plot(E_grid,band_counts_new)
plt.xscale('log')
plt.yscale('log')
plt.title('Energy vs Photon Fluence')
plt.xlabel('Energies (keV)')
plt.ylabel('Photon Fluence')
plt.grid(True, which='both', linestyle='--', alpha=0.6)