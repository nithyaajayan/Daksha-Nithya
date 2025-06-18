import pandas as pd
import numpy as np
import healpy as hp

panel_orient_list = [[0, 0], [45, 0], [45, 90], [45, 180], [45, 270], [90, 0], 
                     [90, 45], [90, 90], [90, 135], [90, 180], [90, 225], 
                     [90, 270], [90, 315], [180, 45], [180, 135], [180, 225], [180, 315]]

panel_orient = [(np.radians(t),np.radians(p)) for t,p in panel_orient_list]

dataset = pd.read_csv("face_counts_new.csv",header=None,sep=' ')
noise = 3000
sources_all = dataset-noise
sources_all[sources_all<0] = np.nan
n=7
result=[]

for row in range(10):
    sources = sources_all.iloc[row,:].to_numpy()
    top_indices = np.argsort(-sources)[:n]
    top_counts = sources[top_indices]
    
    sources_matrix = []
    angles_matrix=[]

    for i,counts in zip(top_indices,top_counts):
        theta,phi = panel_orient[i]

        x= np.sin(theta) * np.cos(phi)
        y= np.sin(theta) * np.sin(phi)
        z= np.cos(theta)

        angles_matrix.append([x,y,z])
        sources_matrix.append(counts)
    
    sources_matrix=np.array(sources_matrix)
    angles_matrix=np.array(angles_matrix)

    r_vec, _, _, _ = np.linalg.lstsq(angles_matrix, sources_matrix, rcond=None)
    r_unit = r_vec / np.linalg.norm(r_vec)

    theta_r, phi_r = hp.vec2ang(r_unit)

    ra = np.degrees(phi_r.item())
    dec = 90 - np.degrees(theta_r.item())

    result.append({
        'Row': row + 1,
        'RA (deg)': ra,
        'Dec (deg)': dec,
        'Norm': np.linalg.norm(r_vec)
    })

result_df = pd.DataFrame(result)
result_df.set_index('Row',inplace=True)
print(result_df)



