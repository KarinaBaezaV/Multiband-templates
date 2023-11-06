import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def coef_f(x,a,b,c,d):
    return a + b*x + c*x**2 + d*x**3

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

def eval_slope(slopes_df, x_coeffs, w0_x, w0_y):
    my_df = slopes_df[(slopes_df.X_w0 == w0_x) & (slopes_df.Y_w0 == w0_y)]
    f_ord = my_df.f_ord.values
    max_ord = f_ord.max()
    x_n = len(x_coeffs)//2
    As = my_df.a.values
    Bs = my_df.b.values
    is_sine = my_df.sine.values
    new_coeffs = []

    for k in range(max_ord):
        pos = f_ord == k
        if k == 0:
            new_coeffs.append(As[pos] + Bs[pos] * x_coeffs[k])
        elif k < x_n:
            possin = pos & is_sine
            poscos = pos & ~is_sine
            new_coeffs.append(As[possin] + Bs[possin] * x_coeffs[2*k-1])
            new_coeffs.append(As[poscos] + Bs[poscos] * x_coeffs[2*k])
    return np.array(new_coeffs).flatten()

b_info = {"g" : [0, 4769.90],
          "r" : [1, 6370.44],
          "i" : [2, 7774.30],
          "z" : [3, 9154.88]}

local = path.split(path.abspath(__file__))[0]

base = pd.read_csv(path.join(local, "sm_new_fits_ab_phase.csv"))
# ids = base.ID.values
# periods = base.P.values

slopes = pd.read_csv(path.join(local, "sm_linear_fits_ab_phase.csv"))

# Test for r dependence
x_wavs = slopes.X_w0.values
y_wavs = slopes.Y_w0.values
f_ord = slopes.f_ord.values
sins = slopes.sine.values
coss = slopes.cosine.values
first_mask = sins & (x_wavs == b_info["r"][1])
first_mask2 = coss & (x_wavs == b_info["r"][1])
fit_cofs = [(slopes.a.values, slopes.a_e.values), (slopes.b.values, slopes.b_e.values)]
cof_names = ["constant", "slope"]
cof_append = [0.0, 1.0]

other_filters = {"OGLE-I" : 7931.81,
                 "Gaia-G" : 5822.39,
                 "Gaia-RP" : 7619.96,
                 "Gaia-BP" : 5035.75}

plot_diag = False

sin_cte_cof = {}
sin_slop_cof = {}

cos_cte_cof = {}
cos_slop_cof = {}

for p in range(2):
    for k in range(1, 13):
        crt_mask = first_mask & (f_ord == k)
        crt_mask2 = first_mask2 & (f_ord == k)

        x = y_wavs[crt_mask]
        x2 = y_wavs[crt_mask2]

        y = fit_cofs[p]
        
        y_true = y[0][crt_mask]
        y_err = y[1][crt_mask]

        y_true2 = y[0][crt_mask2]
        y_err2 = y[1][crt_mask2]

        x = np.append(x, b_info["r"][1])
        y_true = np.append(y_true, cof_append[p])
        y_err = np.append(y_err, np.mean(y_err)/10)

        x2 = np.append(x2, b_info["r"][1])
        y_true2 = np.append(y_true2, cof_append[p])
        y_err2 = np.append(y_err2, np.mean(y_err2)/10)

        try:
            cof,cov = curve_fit(coef_f, x, y_true, sigma = y_err, absolute_sigma = False)
            cof2,cov2 = curve_fit(coef_f, x2, y_true2, sigma = y_err2, absolute_sigma = False)

            if p == 0:
                sin_cte_cof[k] = cof
                cos_cte_cof[k] = cof2
            else:
                sin_slop_cof[k] = cof
                cos_slop_cof[k] = cof2
            x_fit = np.arange(np.amin(x), np.amax(x)+1, 1)
            if plot_diag:
                plt.scatter(x, y_true, c = "k", s = 5)
                plt.errorbar(x, y_true, yerr = y_err, fmt = "none", ecolor = "k")
                plt.plot(x_fit, coef_f(x_fit, *cof))
                plt.vlines(other_filters.values(),cof_append[p]-1,cof_append[p]+1, colors="r")
                plt.ylim(np.amin(y_true), np.amax(y_true))
                plt.title(f"{cof_names[p]} for A_{k} as a function of wavelength (for r-band fits)")
                plt.show()
        except:
            print("Not Enough Points")


# First, test with curves within the dataset. Use r band as starting point (everything is a function of r)
choice = np.random.randint(0,len(base.index)+1)

personal_info, coeffs = np.split(base.iloc[[choice]].to_numpy().flatten(), [18])
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
gaia_data = pd.read_csv(path.join(local, "light_curves", "RRab_gaia", f"{personal_info[0]}.dat"))
period = personal_info[1]
times = data.HJD.values
mags = data.Mag.values
bands = data.band.values


f_x = np.arange(0,1.01,0.01)

r_coeffs = f_coeffs[1] # Cr, Ar1, Br1, Ar2, Br2...


for b in "giz":
    my_info = b_info[b]
    b_mask = b == bands
    x_data = to_phase(times[b_mask], period)
    y_data = mags[b_mask]
    slope_coeffs = eval_slope(slopes, r_coeffs, 6370.44, my_info[1])
    y_slope = eval_fourier(period, f_x*period, slope_coeffs)
    f_y = eval_fourier(period, f_x*period,f_coeffs[my_info[0]])
    plt.title(f"{personal_info[0]}: {b}-band light-curve as a function of r-band")
    plt.scatter(x_data, y_data, c = "k", s = 5, label = "Data")
    plt.plot(f_x, f_y, label = "Fourier fit")
    plt.plot(f_x, y_slope, label = "Prediction")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()
    plt.close()

for b in ["G", "BP", "RP"]:
    
    # my_info = b_info[b]
    b_mask = b == gaia_data.Band.values
    x_data = to_phase(gaia_data.HJD.values[b_mask], period)
    y_data = gaia_data.MAG.values[b_mask]
    y_data_err = gaia_data.ERR.values[b_mask]
    # To compute coeffs...
    gaia_slopes = [np.median(y_data)]
    for k in range(1,11):
        g_const_1 = coef_f(other_filters[f"Gaia-{b}"], *sin_cte_cof[k])
        g_slope_1 = coef_f(other_filters[f"Gaia-{b}"], *sin_slop_cof[k])
        g_const_2 = coef_f(other_filters[f"Gaia-{b}"], *cos_cte_cof[k])
        g_slope_2 = coef_f(other_filters[f"Gaia-{b}"], *cos_slop_cof[k])
        try:
            gaia_slopes.append(g_const_1 + g_slope_1*r_coeffs[2*k -1])
            gaia_slopes.append(g_const_2 + g_slope_2*r_coeffs[2*k])
        except:
            print(f"No {k}-th coefficient")

    # slope_coeffs = eval_slope(slopes, r_coeffs, 6370.44, my_info[1])
    optimize differences and it should be good to go?
    y_slope = eval_fourier(period, f_x*period, gaia_slopes)
    plt.title(f"{personal_info[0]}: {b}-band light-curve as a function of r-band")
    plt.scatter(x_data, y_data, c = "k", s = 5, label = "Data")
    plt.errorbar(x_data, y_data,yerr = y_data_err, fmt = "none", ecolor = "k")
    plt.plot(f_x, y_slope, label = "Prediction")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()
    plt.close()


# Must add uncertainties to coefficients... https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.WLS.html#statsmodels.regression.linear_model.WLS

# Can we linear equations of higher order N?


# Now, the same but against DECam curves from other datasets. Could ignore constant, we only care about LC shape.

# It seems that the slopes are a function of effective wavelength! test it against OGLE data.
