import numpy as np
import pandas as pd
from os import path
from sklearn.linear_model import HuberRegressor as HR
from sklearn.linear_model import LinearRegression as LR

local = path.split(path.abspath(__file__))[0]

def fourier_g(T0,period,times,mags,n):
    T0=0.
    temp = []
    ph1 = ((times - T0) /period)-np.floor((times - T0)/period)    
    temp.append(np.ones(len(ph1)).reshape(-1,1))
    
    for i in range(n):
        temp.append((np.sin(2*(i+1)*np.pi*ph1)).reshape(-1,1))
    for j in range(n):
        temp.append((np.cos(2*(j+1)*np.pi*ph1)).reshape(-1,1))

    X_tr = np.column_stack(temp)
    Y_tr = mags
    print(np.shape(X_tr), np.shape(Y_tr))

    model1 = HR(fit_intercept = False,max_iter=1000000)
    model1.fit(X_tr,Y_tr)   
    coeff1 = model1.fit(X_tr,Y_tr)
    coeff1 = coeff1.coef_ 

    model2 = LR(fit_intercept = False)
    model2.fit(X_tr,Y_tr)   
    coeff2 = model2.fit(X_tr,Y_tr)
    coeff2 = coeff2.coef_ 

    return(model1,coeff1, model2, coeff2)


def fourier_band(T0,period,times,mags,n,coeff,n_band):
    T0=0.
    temp = []
    A0 = mags.mean()
    ph1 = ((times - T0) /period)-np.floor((times - T0)/period)
    
    temp.append(coeff[0]*np.ones_like(ph1).reshape(-1,1))
    
    for i in range(n):
        if i < n_band:
            temp.append((np.sin(2*(i+1)*np.pi*ph1)*coeff[i+1]).reshape(-1,1))
        else:
            temp.append((np.sin(2*(i+1)*np.pi*ph1)*coeff[i+1]*0).reshape(-1,1))
    for j in range(n):
        if j < n_band:
            temp.append((np.cos(2*(j+1)*np.pi*ph1)*coeff[j+n+1]).reshape(-1,1))
        else:
            temp.append((np.cos(2*(j+1)*np.pi*ph1)*coeff[j+n+1]*0).reshape(-1,1))
            
    X_tr = np.c_[temp]
    Y_tr = mags
    model1 = HR(fit_intercept = False,max_iter=1000000)
    model1.fit(X_tr,Y_tr)   
    coeff1 = model1.fit(X_tr,Y_tr)
    coeff1 = coeff1.coef_ 

    model2 = LR(fit_intercept = False,max_iter=1000000)
    model2.fit(X_tr,Y_tr)   
    coeff2 = model2.fit(X_tr,Y_tr)
    coeff2 = coeff2.coef_ 

    return(model1,coeff1, model2, coeff2)

band_stuff = {"g" : [9, [1.72684642e+01,-3.14166879e-01,-1.62687757e-01,-9.14884674e-02,
                       -3.41306238e-02,1.99013849e-04,1.76675724e-02,9.53751278e-03
                       ,8.04035327e-03,9.74112386e-03,-3.21134847e-01,-1.39512786e-01
                       ,-1.32369130e-01,-9.64817083e-02,-7.01982818e-02,-3.41920151e-02
                       ,-1.97177092e-02,-1.07247121e-02,-7.47067409e-03]],
              "r" : [9, [0.95014698,0.76378776,0.72645703,0.76231245,0.62911121,0.0050095
                       ,0.3543438,0.65307124,0.87450042,0.86626704,0.58476662,0.70466313
                       ,0.73504761,0.72745563,0.71666098,0.84371875,0.84899189,0.79046239
                       ,0.62820757]], 
              "i" : [7, [ 0.92996338,0.65492742,0.56969837,0.57458158,0.38998761,-0.00842317
                       ,0.81105313,0.44747915,0.59347542,0.48766093,0.44193351,0.5942125
                       ,0.61077325,0.67208249,0.56857613,0.66929928,0.55511262,0.54374646
                       ,0.7852332]], 
              "z" : [7, [0.91532538,0.64085079,0.59235479,0.53474237,0.58360275,0.0141093
                       ,0.68533943,0.54428822,0.66808024,0.33797047,0.39036523,0.52551179
                       ,0.5190344,0.57072866,0.53684813,0.62892933,0.45643083,0.41513384
                       ,0.59614894]]} #terms, coeffs

period = 0.49288092

name = "OGLE-BLG-RRLYR-10891"

curve = pd.read_csv(path.join(local, f"{name}.csv"))
#Maybe has to be normalized or something
curve_band = curve.band.values
for b in "griz":
    #NEED T0 values
    time = curve.HJD[curve_band == b].values
    mag = curve.Mag[curve_band == b].values
    if b == "g":
        gModel_HR, g_Coeff_HR, gModel_LR, g_Coeff_LR = fourier_g(0, period, time, mag, band_stuff[b][0])
        print(g_Coeff_HR)
    else:
        exit()