import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt

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

local = path.split(path.abspath(__file__))[0]

base = pd.read_csv(path.join(local, "sm_new_fits_ab.csv"))
# ids = base.ID.values
# periods = base.P.values

slopes = pd.read_csv(path.join(local, "sm_linear_fits_ab.csv"))

# First, test with curves within the dataset. Use r band as starting point (everything is a function of r)
choice = np.random.randint(0,len(base.index)+1)

personal_info, coeffs = np.split(base.iloc[[choice]].to_numpy().flatten(), [2])
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

r_coeffs = f_coeffs[0] # Cr, Ar1, Br1, Ar2, Br2...


for b in "riz":
    my_info = b_info[b]
    b_mask = b == bands
    x_data = to_phase(times[b_mask], period)
    y_data = mags[b_mask]
    slope_coeffs = eval_slope(slopes, r_coeffs, 4769.90, my_info[1])
    y_slope = eval_fourier(period, f_x*period, slope_coeffs)
    f_y = eval_fourier(period, f_x*period,f_coeffs[my_info[0]])
    plt.title(f"{personal_info[0]}: {b}-band light-curve as a function of g-band")
    plt.scatter(x_data, y_data, c = "k", s = 5, label = "Data")
    plt.plot(f_x, f_y, label = "Fourier fit")
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
