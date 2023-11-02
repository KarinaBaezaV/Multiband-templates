import numpy as np
import pandas as pd
from os import path
from sklearn.linear_model import HuberRegressor as HR
from sklearn.linear_model import LinearRegression as LR
from matplotlib import pyplot as plt

local = path.split(path.abspath(__file__))[0]

def to_phase(times, period):
    ratio = times/period
    return ratio - np.floor(ratio)

def fourier(period,times,mags,n, regressor, weights):
    temp = []
    ph1 = to_phase(times, period)
    temp.append(np.ones_like(ph1))
    
    for i in range(1,n+1):
        arg = 2*i*np.pi*ph1
        temp.append(np.sin(arg))
        temp.append(np.cos(arg))

    X_tr = np.column_stack(temp)
    Y_tr = mags

    model = regressor(fit_intercept = False, max_iter=1000000)
    model.fit(X_tr,Y_tr, weights)

    coeff = model.fit(X_tr,Y_tr)
    coeff = coeff.coef_ 

    return coeff

def eval_fourier(period, times, coefs):
    n = len(coefs)//2
    output = np.ones_like(times)*coefs[0]

    for x in range(1, n+1):
        arg = 2*np.pi*x*times/period
        to_sum = coefs[2*x - 1]*np.sin(arg) + coefs[2*x]*np.cos(arg)
        output += to_sum
    
    return output

# Uncertainty on x when folding?

df_ab = pd.read_csv(path.join(local, "light_curves", "ab_info.csv"))
df_c = pd.read_csv(path.join(local, "light_curves", "c_info.csv"))

# max N in ab: 15
# max N in c: 8

process_ab = True
process_c = False

bands = {0 : "g",
         1 : "r",
         2 : "i",
         3 : "z"}

if process_ab:
    ids = df_ab.ID.values
    periods = df_ab.P.values
    Ns = [df_ab[f"n_{x}"].values for x in "griz"]
    df_out = pd.DataFrame()
    # out_id = []
    # out_period = []
    C = [[] for x in range(4)]
    As = [[] for x in range(60)]
    Bs = [[] for x in range(60)]
    
    for i in range(len(ids)):
        base = 0
        crt_p = periods[i]
        crt_id = ids[i]
        crt_curve = pd.read_csv(path.join(local, "light_curves", "RRab", f"{crt_id}.csv"), usecols = ["Mag", "HJD", "Err", "band"])
        
        for b in range(4):
            crt_n = Ns[b][i]
            crt_band = bands[b]
            temp_curve = crt_curve[crt_curve.band == crt_band]
            crt_times = temp_curve.HJD.values
            crt_mags = temp_curve.Mag.values
            # crt_phase = to_phase(crt_times, crt_p)
            # sorter = np.argsort(crt_phase)
            f_coefs = fourier(crt_p, crt_times, crt_mags, crt_n, HR, 1/temp_curve.Err.values)
            
            # x = np.arange(0, 1.01, 0.01)
            # y = eval_fourier(crt_p, crt_p * x, f_coefs)


            # plt.scatter(crt_phase[sorter], crt_mags[sorter], c = 'k', s = 5)
            # plt.plot(x, y, c = 'r')
            # plt.gca().invert_yaxis()
            # plt.show()
            # out_id.append(crt_id)
            # out_period.append(crt_p)
            C[b].append(f_coefs[0])
            for k in range(1,16):
                if k <= crt_n:
                    # print(base + k - 1)
                    # print(crt_band)
                    As[base + k - 1].append(f_coefs[2*k - 1])
                    Bs[base + k - 1].append(f_coefs[2*k])
                else:
                    As[base + k - 1].append(np.nan)
                    Bs[base + k - 1].append(np.nan)
            base += 15
    df_out["ID"] = ids
    df_out["P"] = periods
    for b in range(4):
        this_band = bands[b]
        df_out[f"C{this_band}"] = C[b]
        base = b*15
    
        for m in range(base, base + 15):
            df_out[f"A{this_band}_{m+1-base}"] = As[m]
            df_out[f"B{this_band}_{m+1-base}"] = Bs[m]
    df_out.to_csv(path.join(local, "new_fits_ab.csv"), index = False)


