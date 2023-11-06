from astroquery.gaia import Gaia
import pandas as pd
from os import path

local = path.split(path.abspath(__file__))[0]
src = pd.read_csv(path.join(local,"..",'Gaia_RR.csv'))

retrieval_type = 'EPOCH_PHOTOMETRY'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'


datalink = Gaia.load_data(ids=src['Source'].values, data_release = data_release, retrieval_type=retrieval_type, data_structure = data_structure, verbose = False, output_file = path.join(local, "All_results.xml"))
dl_keys  = [inp for inp in datalink.keys()]
dl_keys.sort()

print(f'The following Datalink products have been downloaded:')
for dl_key in dl_keys:
    print(f' * {dl_key}')
