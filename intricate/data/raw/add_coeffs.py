import numpy as np
import pandas as pd
from os import path

rr_t = ["RRab", "RRc"]
local = path.split(path.abspath(__file__))[0]
base = pd.read_csv(path.join(local, "new_N_templates.csv"))
base["source_id"] = base["source_id"].map(lambda x: int(x.split("-")[-1]))
bands = ["g", "r", "i", "z"]
#source_id,N_g,N_r,N_i,N_z
# print(base["source_id"])
# exit()
# ["ID", "Band", "Period", "Coefficients"]

for x in rr_t:
    crt = np.load(path.join(local, f"fourier_{x}_DECam.npz"), allow_pickle=True)
    all_data = [crt[i] for i in ["ID", "Band", "Period", "Coefficients"]]
    new_coeffs = np.copy(all_data[3])

    for u in range(len(all_data[0])):
        my_band = all_data[1][u]
        if my_band != 0:#(my_band == 3) & (all_data[0][u] == 10891):
            # print(all_data[3][u])
            idx = base.index[all_data[0][u] == base["source_id"]].to_numpy()[0]
            limit = base.at[idx, f"N_{bands[my_band]}"]
            # print(limit)
            # print(all_data[3][u])
            crt_size = len(all_data[3][u])
            mid_point = crt_size//2
            size_diff = mid_point - limit
            upper_limit = crt_size if size_diff == 0 else -size_diff
            new_coeffs[u] = np.concatenate([all_data[3][u][:1+limit],all_data[3][u][mid_point+1:upper_limit]])
            # print(all_data[3][u])
            # print(new_coeffs[u])
            # exit()
    print(len(new_coeffs), [new_coeffs[x] == all_data[3][x] for x in range(len(new_coeffs))])
    np.savez(path.join(local, f"fourier_{x}_DECam_new.npz"), ID = all_data[0], Band  = all_data[1], Period  = all_data[2], Coefficients  = new_coeffs)
