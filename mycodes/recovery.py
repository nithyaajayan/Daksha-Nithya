import numpy as np
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from localisation.locfunction import chi2localisation,vectorlocalisation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--injectionfile",type=str,required=True)
    parser.add_argument("--noise",type=int,default=3000)
    parser.add_argument("--NSIDE",type=int,default=128)
    parser.add_argument("--panels",type=int,default=5)
    parser.add_argument("--flux",type=float,required=True)
    parser.add_argument("--result_file",type=str,required=True)
    args = parser.parse_args()

    file =np.load(args.injectionfile)
    data = file['counts']
    true_GRB_ra = file['ra']
    true_GRB_dec=file['dec']

    num_sources, num_injections, _ = data.shape
    result_array=np.zeros((num_sources, num_injections, 6))

    for i in range(num_sources):
        for j in range(num_injections):
            counts=data[i,j,:]
            vec_result = vectorlocalisation(counts,args.noise)
            if vec_result is None:
                continue

            chi2_result = chi2localisation(counts,args.NSIDE,args.panels,args.noise)
                
            result_array[i,j,0:3]=chi2_result
            result_array[i,j,3:6]=vec_result

    np.savez(args.result_file, true_ra=true_GRB_ra,true_dec=true_GRB_dec,results=result_array)    
