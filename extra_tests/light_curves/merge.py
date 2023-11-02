import pandas as pd
from os import path
import numpy as np
import os

local = path.split(path.abspath(__file__))[0]

all_available = np.array([f[:-4] for f in os.listdir(path.join(local, "RRab")) if f.endswith(".csv")] + [f[:-4] for f in os.listdir(path.join(local, "RRc")) if f.endswith(".csv")])

df_n = pd.read_csv(path.join(local, "N_templates.csv"))
n_id = df_n.source_id.values

df_o = pd.read_csv(path.join(local, "OGLE_RR.csv"), usecols = ["ID","Type","P_1"])
o_id = df_o.ID.values
o_types = df_o.Type.values
o_p = df_o.P_1.values

all_ab = np.full_like(o_id, False)
all_c = np.full_like(o_id, False)
Ns = [[np.ones_like(o_id), np.ones_like(o_id)] for _ in range(4)] # per band, per RR type [ab, c]

bands = {0 : "g",
         1 : "r",
         2 : "i",
         3 : "z"}

for i in range(len(n_id)):
    if n_id[i] in all_available:
        mask = n_id[i] == o_id
        my_type = o_types[mask]
        if my_type == "RRab":
            all_ab[mask] = True
            for b in range(4): 
                Ns[b][0][mask] = df_n.at[i, f"N_{bands[b]}"]
        elif my_type == "RRc":
            all_c[mask] = True
            for b in range(4): 
                Ns[b][1][mask] = df_n.at[i, f"N_{bands[b]}"]
        else:
            print(f"{n_id[i]} is RRd!")


all_ab = all_ab.astype(bool)
all_c = all_c.astype(bool)

ab_new = pd.DataFrame(data = {"ID" : o_id[all_ab],
                               "P" : o_p[all_ab]})
c_new = pd.DataFrame(data = {"ID" : o_id[all_c],
                              "P" : o_p[all_c]})

for b in range(4):
    ab_new[f"n_{bands[b]}"] = Ns[b][0][all_ab]
    c_new[f"n_{bands[b]}"] = Ns[b][1][all_c]

ab_new.to_csv(path.join(local, "ab_info.csv"), index = False)
c_new.to_csv(path.join(local, "c_info.csv"), index = False)