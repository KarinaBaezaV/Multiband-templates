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


local = path.split(path.abspath(__file__))[0]

base = pd.read_csv(path.join(local, "sm_new_fits_ab.csv"))

b_info = {"g" : [0, 4769.90],
        "r" : [1, 6370.44],
        "i" : [2, 7774.30],
        "z" : [3, 9154.88]}


f_x = np.arange(0,1.001,0.001)

view_all = True

if view_all:
    choices = base.index.to_list()
else:
    choices = [108]#[np.random.randint(0,len(base.index)+1)]

for choice in choices:
    personal_info, coeffs = np.split(base.iloc[[choice]].to_numpy().flatten(), [18])
    crt_ph_max = personal_info[6]
    print(personal_info[0])
    # print(coeffs)
    coeffs = coeffs.astype(float)

    splits = [62, 124, 186, 248]

    pre_coeffs = []
    for x in np.split(coeffs, splits):
        temp = x[~np.isnan(x)]
        if temp.size != 0:
            pre_coeffs.append(temp)

    c_errs = [x[1::2] for x in pre_coeffs]
    f_coeffs = [x[::2] for x in pre_coeffs]

    data = pd.read_csv(path.join(local, "light_curves", "RRab", f"{personal_info[0]}.csv"))
    period = personal_info[1]
    times = data.HJD.values - (crt_ph_max + 0.5)*period
    mags = data.Mag.values
    bands = data.band.values
    for b in "griz":
        my_info = b_info[b]

        b_mask = b == bands
        x_data = to_phase(times[b_mask], period)
        y_data = mags[b_mask]
        # print(f_coeffs[my_info[0]])
        f_y = eval_fourier(period, f_x*period,f_coeffs[my_info[0]])
        modified_x = f_x - crt_ph_max + 0.5
        modified_x[modified_x > 1] -= 1
        modified_x[modified_x < 0] += 1

        
        plt.scatter(x_data, y_data, s = 5, label = b)
        plt.scatter(modified_x, f_y, c = 'k', s = 1)
    plt.title(personal_info[0])
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    plt.clf()
    plt.close()
