import numpy as np
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from localisation.locfunction import chi2localisation,vectorlocalisation

def main(injectionfile,noise,NSIDE,panels,flux,error_file,result_file):
    data=np.load(injectionfile)
    error_dict={}
    result_dict={}

    for key in data.files:
        GRB_ra,GRB_dec = map(float,key.split('_'))
        injections = data[key]
        error_list =[]
        result_list =[]

        for counts in injections:
            vec_result = vectorlocalisation(counts,noise)
            if vec_result is None:
                continue

            chira,chidec,chiflux = chi2localisation(counts,NSIDE,panels,noise)
            vecra,vecdec,vecflux = vec_result

            error_list.append([
                chira - GRB_ra,
                chidec - GRB_dec,
                chiflux - flux,
                vecra - GRB_ra,
                vecdec - GRB_dec,
                vecflux - flux
            ])

            result_list.append([
                GRB_ra,GRB_dec,
                chira,chidec, chiflux,
                vecra,vecdec, vecflux
            ])

        error_dict[key] = np.array(error_list)
        result_dict[key] = np.array(result_list)
    
    np.savez(error_file,**error_dict)
    np.savez(result_file,**result_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--injectionfile",type=str,default="injectiondataset.npz")
    parser.add_argument("--noise",type=int,default=3000)
    parser.add_argument("--NSIDE",type=int,default=128)
    parser.add_argument("--panels",type=int,default=5)
    parser.add_argument("--flux",type=float,default=2780)
    parser.add_argument("--error_file",type=str,default="localisation_errors.npz")
    parser.add_argument("--result_file",type=str,default="localisation_results.npz")
    args = parser.parse_args()
    main(args.injectionfile,args.noise,args.NSIDE,args.panels,args.flux,args.error_file,args.result_file)
    
