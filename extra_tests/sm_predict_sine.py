import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def to_phase(times, period):
    ratio = times/period
    return ratio - np.floor(ratio)

def eval_fourier(period, times, coefs):
    n = len(coefs)//2
    output = np.ones_like(times)*coefs[0]
    for x in range(1, n+1):
        arg = 2*np.pi*x*times/period
        to_sum = coefs[2*x - 1]*np.sin(arg) + coefs[2*x]*np.cos(arg)
        output += to_sum
    
    return output

local = path.split(path.abspath(__file__))[0]
fit_sin_pre = lambda x,a,b: a*np.sin((x-b)*2*np.pi)
fit_cos_pre = lambda x,a,b: a*np.cos((x-b)*2*np.pi)
base = pd.read_csv(path.join(local, "sm_new_fits_ab.csv"))
ids = base.ID.values

argX = base.Ph_r_max.values
custom_val_sin = [1.34,1.4,1.65,1.8]
displacement = [0.4, 0.4, 0.45, 0.52]
sin_coef = []
cos_coef = []
plot_fits = False
# It seems that there is an additional period dependence ...

for k in range(1,5):
    fit_sin = lambda x,a: fit_sin_pre(x,a,displacement[k-1])
    fit_cos = lambda x,a: fit_cos_pre(x,a,displacement[k-1])
    argY1 = base[f"Ag_{k}"].values
    argY2 = base[f"Bg_{k}"].values
    # argY1_e = base[f"Ag_{k}_e"].values
    not_nan = ~np.isnan(argY1)
    modX = to_phase(argX, 1/k)
    cofs1, cov1 = curve_fit(fit_sin, modX[not_nan], argY1[not_nan])
    cofs2, cov2 = curve_fit(fit_cos, modX[not_nan], argY2[not_nan])
    fit_x = np.arange(0, 1 + 0.01, 0.01)

    cofs1[0] = cofs1[0]*custom_val_sin[k-1]
    cofs2[0] = cofs2[0]*custom_val_sin[k-1]

    fit_y1 = fit_sin(fit_x, *cofs1)
    fit_y2 = fit_cos(fit_x, *cofs2)

    if plot_fits:
        print(f"Sin: {cofs1}")
        plt.scatter(modX, argY1, c = "k", s = 5)
        plt.plot(fit_x, fit_y1)
        plt.title(f"Sin {k}")
        plt.show()

        print(f"Cos: {cofs2}")
        plt.scatter(modX, argY2, c = "k", s = 5)
        plt.plot(fit_x, fit_y2)
        plt.title(f"Cos {k}")
        plt.show()

    sin_coef.append(cofs1[0])
    cos_coef.append(cofs1[0])

# ords = [k for k in range(1,5)]
# coef_name = "abc"
# for k in range(3):
#     plt.scatter(ords, sin_coef[k], label = coef_name[k])
# plt.legend()
# plt.show()

# Test 12437, index 108

choice = 108

personal_info, coeffs = np.split(base.iloc[[choice]].to_numpy().flatten(), [18])
crt_ph_max = personal_info[6]

plot_fits2 = False
if plot_fits2:

    for k in range(1,5):
        argY1 = base[f"Ag_{k}"].values
        argY2 = base[f"Bg_{k}"].values
        crt_a = argY1[choice]
        crt_b = argY2[choice]
        crt_x = to_phase(crt_ph_max, 1/k)

        not_nan = ~np.isnan(argY1)
        modX = to_phase(argX, 1/k)


        plt.scatter(modX, argY1, c = "k", s = 5)
        plt.scatter(crt_x, crt_a, c = "r", s = 5)
        plt.title(f"Sin {k}")
        plt.show()

        plt.scatter(modX, argY2, c = "k", s = 5)
        plt.scatter(crt_x, crt_b, c = "r", s = 5)
        plt.title(f"Cos {k}")
        plt.show()



exit()

coeffs = coeffs.astype(float)

temp_splits = np.argwhere(np.isnan(coeffs)).flatten()
splits = [temp_splits[0]]

for s in range(1, len(temp_splits)):
    if temp_splits[s-1] != (temp_splits[s]-1):
        splits.append(temp_splits[s])

pre_coeffs = [x[~np.isnan(x)] for x in np.split(coeffs, splits)]
c_errs = [x[1::2] for x in pre_coeffs]
f_coeffs = [x[::2] for x in pre_coeffs]

data = pd.read_csv(path.join(local, "light_curves", "RRab", f"{personal_info[0]}.csv"))
period = personal_info[1]
times = data.HJD.values
mags = data.Mag.values
bands = data.band.values

b_info = {"g" : [0, 4769.90],
          "r" : [1, 6370.44],
          "i" : [2, 7774.30],
          "z" : [3, 9154.88]}


f_x = np.arange(0,1.01,0.01)


for b in "g": # Only testing g...
    my_info = b_info[b]
    b_mask = b == bands
    x_data = to_phase(times[b_mask], period)
    y_data = mags[b_mask]
    med_mag = np.median(y_data)
    test_coefs = [med_mag]
    for k in range(4):
        my_ph = to_phase(crt_ph_max, 1/(k+1))
        test_coefs.append(fit_sin_pre(my_ph,sin_coef[k], displacement[k]))
        test_coefs.append(fit_sin_pre(my_ph,cos_coef[k], displacement[k]))
    test_coefs = np.array(test_coefs)

    f_y = eval_fourier(period, f_x*period,f_coeffs[my_info[0]])
    test_y = eval_fourier(period, f_x*period, test_coefs)
    print(f_coeffs[my_info[0]])
    print(test_coefs)
    plt.title(f"{personal_info[0]}: {b}-band light-curve as a function max-phase in r")
    plt.scatter(x_data, y_data, c = "k", s = 5, label = "Data")
    plt.plot(f_x, f_y, label = "Fourier fit")
    # plt.plot(f_x, test_y, label = "Phase prediction")
    # plt.plot(f_x, y_slope, label = "Prediction")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()
    plt.close()