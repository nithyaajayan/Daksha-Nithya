import sys
import os
sys.path.append(os.path.abspath(".."))

from localisation.locfunction import *
import numpy as np

n = 100
noise = 3000
flux = 2780
NSIDE = 128
panels = 5

injection_dict = {}
error_dict = {}  
results_dict = {}

GRB_ra_array, GRB_dec_array = isotropicpoints(n)

for GRB_ra, GRB_dec in zip(GRB_ra_array, GRB_dec_array):
    injections = []
    errors_list = []
    results_list = []

    for _ in range(1000):
        counts = observedcounts(GRB_ra, GRB_dec, flux, noise)
        injections.append(counts)

        vec_result = vectorlocalisation(counts, noise)
        if vec_result is None:
            continue

        chira, chidec, chiflux = chi2localisation(counts, NSIDE, panels, noise)
        vecra, vecdec, vecflux = vectorlocalisation(counts, noise)

        errors_list.append([
            chira - GRB_ra,
            chidec - GRB_dec,
            chiflux - flux,
            vecra - GRB_ra,
            vecdec - GRB_dec,
            vecflux - flux
        ])

        results_list.append([
            GRB_ra, GRB_dec,
            chira, chidec, chiflux,
            vecra, vecdec, vecflux
        ])

    key = f"{GRB_ra:.5f}_{GRB_dec:.5f}"
    injection_dict[key] = np.array(injections)
    error_dict[key] = np.array(errors_list)
    results_dict[key] = np.array(results_list)

np.savez("grouped_injections_by_source.npz", **injection_dict)
np.savez("sourcewise_errors.npz", **error_dict)
np.savez("sourcewise_results.npz", **results_dict)
