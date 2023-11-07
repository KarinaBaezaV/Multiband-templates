import os
from os import path
import pandas as pd

local = path.split(path.abspath(__file__))[0]
band_I = [x for x in os.listdir(path.join(local, "Band_I")) if x.endswith(".dat")]
band_V = [x for x in os.listdir(path.join(local, "Band_V")) if x.endswith(".dat")]

done = []

for f in band_I:
    out = pd.read_csv(path.join(local, "Band_I", f))
    if f in band_V:
        out2 = pd.read_csv(path.join(local, "Band_V", f))
        out = pd.concat([out, out2])
    out.to_csv(path.join(local, f), index=False)

