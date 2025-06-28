import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from localisation.locfunction import isotropicpoints, observedcounts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources",type=int,required=True,help="Number of isotropic injections")
    parser.add_argument("--injections",type=int,required=True,help="Number of injections")
    parser.add_argument("--noise",type=float,default=3000,help="Background noise")
    parser.add_argument("--flux", type=float,required=True,help="Normal flux")
    parser.add_argument("--outfile",type=str,default="injectiondataset.npz")
    args = parser.parse_args()
    
    GRB_ra_array,GRB_dec_array = isotropicpoints(args.sources)
    injection_array=np.zeros((args.sources,args.injections,17))

    for i,(GRB_ra,GRB_dec) in enumerate(zip(GRB_ra_array,GRB_dec_array)):
        for j in range(args.injections):
            counts = observedcounts(GRB_ra,GRB_dec,args.flux,args.noise)
            injection_array[i,j,:]=counts

    np.savez(args.outfile, ra=GRB_ra_array,dec=GRB_dec_array,counts=injection_array)
