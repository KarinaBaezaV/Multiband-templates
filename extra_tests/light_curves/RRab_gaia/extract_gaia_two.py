import os
import pandas as pd
from os import path

local = path.split(path.abspath(__file__))[0]
files = [f for f in os.listdir(path.join(local, "raw")) if f.endswith('.dat')]

matches = pd.read_csv(path.join(local, "..", 'Gaia_RR.csv'))
noms = (matches['ID']).values
gaid = (matches['Source']).values


for i in files: 
    data = pd.read_csv(path.join(local, "raw", i))
    data['time'] = data['time'].apply(lambda x: x + 2455197.5 - 2450000)
    identifier = i[:-4].split()[-1]
    # print(i)
    select = int(identifier) == gaid
    df = pd.DataFrame()
    df['HJD'] = data['time'].values
    df['MAG'] = data['mag'].values
    df['ERR'] = 1.086/data['flux_over_error'].values
    df["Band"] = data.band.values
    df.to_csv(path.join(local, f"{noms[select][0]}.dat"), index = False)


