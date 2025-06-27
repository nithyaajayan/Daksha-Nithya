# plot.py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from localisation.locfunction import plotdistribution

def main(error_file):
    errors = np.load(error_file)
    
    all_errors = []
    for key in errors.files:
        all_errors.append(errors[key])
    
    all_errors = np.concatenate(all_errors, axis=0)

    chi2raerror = all_errors[:,0]
    chi2decerror = all_errors[:,1]
    chi2fluxerror = all_errors[:,2]
    vecraerror = all_errors[:,3]
    vecdecerror = all_errors[:,4]
    vecfluxerror = all_errors[:,5]

    bins = np.sqrt(all_errors.shape)
    
    plotdistribution(chi2raerror,vecraerror,'Chi2 RA Error',"Vector RA Error",0,"RA Error Distribution", "Error",bins)
    plotdistribution(chi2decerror,vecdecerror,'Chi2 Dec Error',"Vector Dec Error",0,"Dec Error Distribution", "Error",bins)
    plotdistribution(chi2fluxerror,vecfluxerror,'Chi2 Flux Error',"Vector Flux Error",0,"Flux Error Distribution", "Error",bins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_file", type=str, default="localisation_errors.npz")
    args = parser.parse_args()
    main(args.error_file)
