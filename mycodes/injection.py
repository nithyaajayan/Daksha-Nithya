import numpy as np
import argparse
import sys
import astropy.units as u
from scipy.integrate import simpson
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from localisation.locfunction import isotropicpoints, observedcounts,bandcounts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources",type=int,required=True,help="Number of isotropic injections")
    parser.add_argument("--injections",type=int,required=True,help="Number of injections")
    parser.add_argument("--bkg",type=float,default=3000,help="Background noise")
    parser.add_argument("--inputfluence", type=float,required=True,help="Input Fluence")
    parser.add_argument("--alpha", type=float,required=True,help="Alpha")
    parser.add_argument("--beta", type=float,required=True,help="Beta")
    parser.add_argument("--Ep", type=float,required=True,help="Energy peak")
    parser.add_argument("--delT", type=float,default=1,help="Burst Duration")
    args = parser.parse_args()

    area = 304

    E_grid = np.logspace(np.log10(20), np.log10(200), 101)
    integrand = E_grid*u.keV.to(u.erg) * bandcounts(E_grid,1,args.alpha,args.beta,args.Ep) #erg/cm2/sec/keV

    fluence_temp = simpson(integrand, E_grid) *args.delT #erg/cm2

    A_scaling = args.inputfluence / fluence_temp
    band_counts_new = bandcounts(E_grid,A_scaling,args.alpha,args.beta,args.Ep)

    photons = simpson(band_counts_new,E_grid) *area *args.delT
    
    GRB_ra_array,GRB_dec_array = isotropicpoints(args.sources)
    injection_array=np.zeros((args.sources,args.injections,17))

    for i,(GRB_ra,GRB_dec) in enumerate(zip(GRB_ra_array,GRB_dec_array)):
        for j in range(args.injections):
            counts = observedcounts(GRB_ra,GRB_dec,photons,args.bkg)
            injection_array[i,j,:]=counts

    filename = (
        f"sim_ndir_{args.sources:05d}"
        f"_ninj_{args.injections:04d}"
        f"_flu_{args.inputfluence:.0e}"
        f"_alpha_{args.alpha:+.2f}"
        f"_beta_{args.beta:+.2f}"
        f"_Ep_{args.Ep:06.2f}.npz")

    np.savez(filename, ra=GRB_ra_array,dec=GRB_dec_array,counts=injection_array,
             metadata = dict(sources=args.sources,
                             injections=args.injections,
                             bkg=args.bkg,
                             alpha=args.alpha,
                             beta=args.beta,
                             Ep=args.Ep,
                             delT=args.delT,
                             area=area,
                             input_fluence=args.inputfluence,
                             photons=photons))

