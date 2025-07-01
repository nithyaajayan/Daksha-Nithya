import numpy as np
import argparse
import sys, os
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

    num_sources, num_injections, _ = data.shape
    result_array = np.zeros((num_sources, num_injections, 6))

    #everything needed for each injection
    index_pairs = list(np.ndindex(num_sources, num_injections))
    tasks = [(i, j, data[i, j, :], args.NSIDE, args.faces, args.noise) for i, j in index_pairs]


    def wrapper(args_tuple):
        i, j, counts, NSIDE, faces, noise = args_tuple
        vec_result = vectorlocalisation(counts, noise)
        chi2_result = chi2localisation(counts, NSIDE, faces, noise)
        return (i, j, chi2_result, vec_result)

    with Pool(args.processes) as pool:
        results = pool.starmap(wrapper, tasks)

    for i, j, chi2_result, vec_result in results:
        result_array[i, j, 0:3] = chi2_result
        result_array[i, j, 3:6] = vec_result

    filename = (
        f"loc_ndir_{metadata['sources']:05d}"
        f"_ninj_{metadata['injections']:04d}"
        f"_flu_{metadata['target_fluence']:.0e}"
        f"_alpha_{metadata['alpha']:+.2f}"
        f"_beta_{metadata['beta']:+.2f}"
        f"_Ep_{metadata['Ep']:06.2f}"
        f"_NSIDE_{args.NSIDE}"
        f"_faces_{args.faces:02d}.npz"
    )

    np.savez(filename,
             true_ra=true_GRB_ra,
             true_dec=true_GRB_dec,
             results=result_array,
             metadata=dict(
                 sources=metadata['sources'],
                 injections=metadata['injections'],
                 target_fluence=metadata['target_fluence'],
                 alpha=metadata['alpha'],
                 beta=metadata['beta'],
                 Ep=metadata['Ep'],
                 noise=args.noise,
                 NSIDE=args.NSIDE,
                 faces=args.faces,
                 cores=args.processes))
