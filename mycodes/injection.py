import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from localisation.locfunction import isotropicpoints, observedcounts


def main(sources,injections,noise,flux,outfile):
    GRB_ra_array,GRB_dec_array = isotropicpoints(sources)
    injection_dict={}

    for GRB_ra,GRB_dec in zip(GRB_ra_array,GRB_dec_array):
        injection_list =[]
        for _ in range(injections):
            counts = observedcounts(GRB_ra,GRB_dec,flux,noise)
            injection_list.append(counts)
        key = f"{GRB_ra:.5F}_{GRB_dec:.5f}"
        injection_dict[key] = np.array(injection_list)

    np.savez(outfile, **injection_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources",type=int,default=100,help="Number of isotropic injections")
    parser.add_argument("--injections",type=int,default=1000,help="Number of injections")
    parser.add_argument("--noise",type=float,default=3000,help="Background noise")
    parser.add_argument("--flux", type=float,default=2780,help="Normal flux")
    parser.add_argument("--outfile",type=str,default="injectiondataset.npz")
    args = parser.parse_args()
    main(args.sources,args.injections,args.noise,args.flux,args.outfile)



