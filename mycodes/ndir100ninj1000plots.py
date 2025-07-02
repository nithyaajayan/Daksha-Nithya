import numpy as np

data=np.load("Daksha-Nithya/mycodes/locmul_ndir_00100_ninj_1000_flu_1e-06_alpha_-1.00_beta_-1.50_Ep_1000.00_NSIDE_128_faces_05.npz",allow_pickle=True)
true_phi_array = data['true_ra']
true_theta_array =90- data['true_dec']
results = data['results']

injdata=np.load("Daksha-Nithya/mycodes/sim_ndir_00100_ninj_1000_flu_1e-06_alpha_-1.00_beta_-1.50_Ep_1000.00.npz",allow_pickle=True)
meta=injdata['metadata'].item()
true_fluence=meta["photons"].item()

from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path("Daksha-Nithya").resolve()))

from localisation.locfunctionnew import plotdistribution

with PdfPages("sourcewise_differences.pdf") as pdf:
    for source in range(5):
        injected = results[source, :, :]
        true_phi = true_phi_array[source]
        true_theta = true_theta_array[source]

        diffs = {
            "Theta Chi2 Error": true_theta - injected[:, 1],
            "Phi Chi2 Error": true_phi - injected[:, 0],
            "Fluence Chi2 Error": true_fluence - injected[:, 2],
            "Theta Vector Error": true_theta - injected[:, 4],
            "Phi Vector Error": true_phi - injected[:, 3],
            "Fluence Vector Error": true_fluence - injected[:, 5],
        }

        for title, diff in diffs.items():
            if not np.all(np.isnan(diff)):
                fig = plotdistribution(
                    data=diff[~np.isnan(diff)],
                    title=f"Source {source+1} — {title}",
                    xlabel=title.split(':')[0],
                    ylabel="Normalized Count",
                    bins=50
                )
                pdf.savefig(fig)
                plt.close(fig)
