import numpy as np
import pandas as pd
from os import path
# from statsmodels.regression.linear_model import WLS
import statsmodels.api as sm
from matplotlib import pyplot as plt

local = path.split(path.abspath(__file__))[0]

def to_phase(times, period):
    ratio = times/period
    return ratio - np.floor(ratio)

def fourier(period,times,mags,n, regressor):
    temp = []
    ph1 = to_phase(times, period)
    temp.append(np.ones_like(ph1))
    
    for i in range(1,n+1):
        arg = 2*i*np.pi*ph1
        temp.append(np.sin(arg))
        temp.append(np.cos(arg))

    X_tr = np.column_stack(temp)
    Y_tr = mags

    model = regressor(Y_tr, X_tr)

    results = model.fit()
    coeff = results.params
    errs = np.sqrt(np.diag(results.cov_params()))
    # print(results.summary())
    # exit()
    
    return coeff, errs

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
prev_data = pd.read_csv(path.join(local, "sm_new_fits_ab.csv"))
r_phases = prev_data.Ph_r_max.values
prev_ids = prev_data.ID.values

# max N in ab: 15
# max N in c: 8

process_ab = True
process_c = False

bands = {0 : "g",
         1 : "r",
         2 : "i",
         3 : "z"}
col_dict = {}

if process_ab:
    ids = df_ab.ID.values
    periods = df_ab.P.values
    Ns = [df_ab[f"n_{x}"].values for x in "griz"]
    
    # out_id = []
    # out_period = []
    C = [[] for x in range(4)]
    C_e = [[] for x in range(4)]
    As = [[] for x in range(60)]
    As_e = [[] for x in range(60)]
    Bs = [[] for x in range(60)]
    Bs_e = [[] for x in range(60)]
    Ph_max = [[] for x in range(4)]
    Ph_max_e = [[] for x in range(4)]
    Ph_min = [[] for x in range(4)]
    Ph_min_e = [[] for x in range(4)]

    x = np.arange(0, 1.001, 0.001)
    for i in range(len(ids)):
        base = 0
        crt_p = periods[i]
        crt_id = ids[i]
        crt_ph_max = r_phases[prev_ids == crt_id][0]

        crt_curve = pd.read_csv(path.join(local, "light_curves", "RRab", f"{crt_id}.csv"), usecols = ["Mag", "HJD", "Err", "band"])
        
        for b in range(4):
            crt_n = Ns[b][i]
            crt_band = bands[b]
            temp_curve = crt_curve[crt_curve.band == crt_band]
            crt_times = temp_curve.HJD.values - (crt_ph_max + 0.5)*crt_p
            crt_mags = temp_curve.Mag.values
            crt_phase = to_phase(crt_times, crt_p)
            # sorter = np.argsort(crt_phase)
            f_coefs, f_errs = fourier(crt_p, crt_times, crt_mags, crt_n, sm.RLM)
            max_data = np.argmin(crt_mags)
            min_data = np.argmax(crt_mags)
            
            y = eval_fourier(crt_p, crt_p * x, f_coefs)
            max_y = np.argmin(y) # remember, minimum magnitude is maximum brightness
            min_y = np.argmax(y) # and maximum magnitude is minimum brightness
            # Estimate phase of maximum/minimum as mean between fit and data. Uncertainty is the mean difference

            fit_phase_max = x[max_y]
            data_phase_max = crt_phase[max_data]
            mean_phase_max = (fit_phase_max + data_phase_max)/2.
            half_diff_phase_max = np.abs(fit_phase_max - data_phase_max)/2.

            fit_phase_min = x[min_y]
            data_phase_min = crt_phase[min_data]
            mean_phase_min = (fit_phase_min + data_phase_min)/2.
            half_diff_phase_min = np.abs(fit_phase_min - data_phase_min)/2.

            Ph_max[b].append(mean_phase_max)
            Ph_max_e[b].append(half_diff_phase_max)

            Ph_min[b].append(mean_phase_min)
            Ph_min_e[b].append(half_diff_phase_min)


            # plt.scatter(crt_phase[sorter], crt_mags[sorter], c = 'k', s = 5)
            # plt.plot(x, y, c = 'r')
            # plt.gca().invert_yaxis()
            # plt.show()
            # out_id.append(crt_id)
            # out_period.append(crt_p)
            C[b].append(f_coefs[0])
            C_e[b].append(f_errs[0])
            for k in range(1,16):
                if k <= crt_n:
                    # print(base + k - 1)
                    # print(crt_band)
                    As[base + k - 1].append(f_coefs[2*k - 1])
                    Bs[base + k - 1].append(f_coefs[2*k])
                    As_e[base + k - 1].append(f_errs[2*k - 1])
                    Bs_e[base + k - 1].append(f_errs[2*k])
                else:
                    As[base + k - 1].append(np.nan)
                    Bs[base + k - 1].append(np.nan)
                    As_e[base + k - 1].append(np.nan)
                    Bs_e[base + k - 1].append(np.nan)

            base += 15
    col_dict["ID"] = ids
    col_dict["P"] = periods

    for s in range(4):
        this_band = bands[s]
        col_dict[f"Ph_{this_band}_max"] = Ph_max[s]
        col_dict[f"Ph_{this_band}_max_e"] = Ph_max_e[s]
        col_dict[f"Ph_{this_band}_min"] = Ph_min[s]
        col_dict[f"Ph_{this_band}_min_e"] = Ph_min_e[s]

    for b in range(4):
        this_band = bands[b]
        col_dict[f"C{this_band}"] = C[b]
        col_dict[f"C{this_band}_e"] = C_e[b]
        base = b*15
    
        for m in range(base, base + 15):
            col_dict[f"A{this_band}_{m+1-base}"] = As[m]
            col_dict[f"A{this_band}_{m+1-base}_e"] = As_e[m]
            col_dict[f"B{this_band}_{m+1-base}"] = Bs[m]
            col_dict[f"B{this_band}_{m+1-base}_e"] = Bs_e[m]
    
    df_out = pd.DataFrame(data=col_dict)
    df_out.to_csv(path.join(local, "sm_new_fits_ab_phase.csv"), index = False)


