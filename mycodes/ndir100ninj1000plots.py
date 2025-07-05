import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path("Daksha-Nithya").resolve()))

from localisation.locfunctionnew import plotdistribution

data=np.load("data/test/locmul_ndir_00100_ninj_1000_flu_1e-06_alpha_-1.00_beta_-1.50_Ep_1000.00_NSIDE_128_faces_05.npz",allow_pickle=True)
true_phi_array = data['true_ra']
true_theta_array =90- data['true_dec']
results = data['results']

injdata=np.load("data/test/sim_ndir_00100_ninj_1000_flu_1e-06_alpha_-1.00_beta_-1.50_Ep_1000.00.npz",allow_pickle=True)
meta=injdata['metadata'].item()
true_fluence=meta["photons"].item()


with PdfPages("sourcewise_errorplots.pdf") as pdf:
    for source in range(2):
        injected = results[source, :, :]
        true_phi = true_phi_array[source]
        true_theta = true_theta_array[source]

        isnan=0
        for i in range(6):
            if (~np.isnan(injected[:,i])).sum()/injected[:,i].size > 0.5:
                isnan+=1
            
        if isnan!=6:
            continue

        for i in range(6):
            if (i%3== 0):
                param="phi"
                true_val= true_phi
            if (i%3==1):
                param="theta"
                true_val=true_theta
            if (i%3==2):
                param="fluence"
                true_val=true_fluence

            if i<3:
                method='chi'
            else:
                method='vector'

            data = injected[:,i]
            fig = plotdistribution(data=data,param=param,true_val=true_val,method=method)
            pdf.savefig(fig)
            plt.close(fig)