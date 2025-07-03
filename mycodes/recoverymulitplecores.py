import numpy as np
import argparse
import sys
import time 
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from localisation.locfunction import chi2localisation, vectorlocalisation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--injectionfile", type=str, required=True)
    parser.add_argument("--noise", type=int, default=3000)
    parser.add_argument("--NSIDE", type=int, default=128)
    parser.add_argument("--faces", type=int, default=5)
    parser.add_argument("--ncores", type=int, default=10)
    args = parser.parse_args()

    file = np.load(args.injectionfile, allow_pickle=True)
    metadata = file['metadata'].item()

    data = file['counts']
    true_GRB_ra = file['ra']
    true_GRB_dec = file['dec']

    input_fluence = metadata['input_fluence']
    fluence_folder = f"flu_{input_fluence:.0e}"
    base_path = Path.cwd() / 'data' / fluence_folder

    base_path.mkdir(parents=True,exist_ok=True)

    num_sources, num_injections, _ = data.shape
    result_array = np.zeros((num_sources, num_injections, 6))

    #everything needed for each injection
    index_pairs = list(np.ndindex(num_sources, num_injections))
    tasks = [(i, j, data[i, j, :], args.NSIDE, args.faces, args.noise) for i, j in index_pairs]

    starttime = time.time()

    def wrapper(args_tuple):
        i, j, counts, NSIDE, faces, noise = args_tuple
        vec_result = vectorlocalisation(counts, noise)
        chi2_result = chi2localisation(counts, NSIDE, faces, noise)
        return (i, j, chi2_result, vec_result)

    with Pool(args.ncores) as pool:
        results = pool.map(wrapper, tasks)

    for i, j, chi2_result, vec_result in results:
        result_array[i, j, 0:3] = chi2_result
        result_array[i, j, 3:6] = vec_result

    endtime=time.time()
    elapsed_time = endtime-starttime

    filename = (
        f"locmul_ndir_{metadata['sources']:05d}"
        f"_ninj_{metadata['injections']:04d}"
        f"_flu_{metadata['input_fluence']:.0e}"
        f"_alpha_{metadata['alpha']:+.2f}"
        f"_beta_{metadata['beta']:+.2f}"
        f"_Ep_{metadata['Ep']:06.2f}"
        f"_NSIDE_{args.NSIDE}"
        f"_faces_{args.faces:02d}.npz"
    )
    print(f"Finished localisation. Total processing time: {elapsed_time} seconds.")

    np.savez(base_path/filename,
             true_ra=true_GRB_ra,
             true_dec=true_GRB_dec,
             results=result_array,
             processing_time_sec=elapsed_time,
             metadata=dict(
                 sources=metadata['sources'],
                 injections=metadata['injections'],
                 alpha=metadata['alpha'],
                 beta=metadata['beta'],
                 Ep=metadata['Ep'],
                 input_fluence=input_fluence,
                 photons=metadata['photons'],
                 noise=args.noise,
                 NSIDE=args.NSIDE,
                 faces=args.faces,
                 cores=args.ncores))
