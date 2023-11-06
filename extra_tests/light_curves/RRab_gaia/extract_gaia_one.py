import os
from os import path
from astropy.io.votable import parse_single_table
import pandas as pd

local = path.split(path.abspath(__file__))[0]
files = [f for f in os.listdir(local) if f.endswith('.xml')]


for i in files: 
    votable = parse_single_table(path.join(local, i))
    data = votable.array
    #print(data['band'])
    #exit()
    #print(votable.fields)
    #exit()
    df = pd.DataFrame()
    #df['source_id'] = data['source_id']
    df['transit_id'] = data['transit_id']
    df['band'] = data['band']
    df['time'] = data['time']
    df['mag'] = data['mag']
    df['flux'] = data['flux']
    df['flux_error'] = data['flux_error']
    df['flux_over_error'] = data['flux_over_error']
    df['rejected_by_photometry'] = data['rejected_by_photometry']
    df['rejected_by_variability'] = data['rejected_by_variability']
    df['other_flags'] = data['other_flags']
    df['solution_id'] = data['solution_id']
    df['band'] = df['band']
    idd = data['source_id'][0]
    df.to_csv(path.join(local, "raw", f"{i[:-4]}.dat"), index = False)


