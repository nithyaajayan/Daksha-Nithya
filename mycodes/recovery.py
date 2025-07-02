import numpy as np
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from localisation.locfunction import chi2localisation,vectorlocalisation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--injectionfile",type=str,required=True)
    parser.add_argument("--noise",type=int,default=3000)
    parser.add_argument("--NSIDE",type=int,default=128)
    parser.add_argument("--faces",type=int,default=5)
    args = parser.parse_args()

    file = np.load(args.injectionfile, allow_pickle=True)
    metadata = file['metadata'].item()
    data = file['counts']
    true_GRB_ra = file['ra']
    true_GRB_dec=file['dec']

    num_sources, num_injections, _ = data.shape
    result_array=np.zeros((num_sources, num_injections, 6))

    starttime=time.time()

    for i in range(num_sources):
        for j in range(num_injections):
            counts=data[i,j,:]
            vec_result = vectorlocalisation(counts,args.noise)
            chi2_result = chi2localisation(counts,args.NSIDE,args.faces,args.noise)
                
            result_array[i,j,0:3]=chi2_result
            result_array[i,j,3:6]=vec_result
    
    endtime=time.time()
    elapsed_time = endtime-starttime

    filename = (
    f"locsin_ndir_{metadata['sources']:05d}"
    f"_ninj_{metadata['injections']:04d}"
    f"_flu_{metadata['input_fluence']:.0e}"
    f"_alpha_{metadata['alpha']:+.2f}"
    f"_beta_{metadata['beta']:+.2f}"
    f"_Ep_{metadata['Ep']:06.2f}"
    f"_NSIDE_{args.NSIDE}"
    f"_faces_{args.faces:02d}.npz"
    )
    print(f"Finished localisation. Total processing time: {elapsed_time} seconds.")

    np.savez(filename,
                true_ra=true_GRB_ra,
                true_dec=true_GRB_dec,
                results=result_array,
                processing_time_sec=elapsed_time,
                metadata=dict(
                    sources=metadata['sources'],
                    injections=metadata['injections'],
                    input_fluence=metadata['input_fluence'],
                    alpha=metadata['alpha'],
                    beta=metadata['beta'],
                    Ep=metadata['Ep'],
                    noise=args.noise,
                    NSIDE=args.NSIDE,
                    faces=args.faces))   
