import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt
from scipy import odr


def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

linear = odr.Model(f)

local = path.split(path.abspath(__file__))[0]

base = pd.read_csv(path.join(local, "sm_new_fits_ab_phase.csv"))

# Effective wavelengths from http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=CTIO&gname2=DECam&asttype= DR1

w0 = {"g" : 4769.90,
      "r" : 6370.44,
      "i" : 7774.30,
      "z" : 9154.88}

bands = {0 : "g",
         1 : "r",
         2 : "i",
         3 : "z"}

weight_dict = {"g" : base["Ph_g_max_e"].values,
               "r" : base["Ph_r_max_e"].values,
               "i" : base["Ph_i_max_e"].values,
               "z" : base["Ph_z_max_e"].values}

xs = []
ys = []
As = []
A_e = []
Bs = []
B_e = []
x_w0s = []
y_w0s = []
f_order = []
sine = []
cosine = []

# Constants
generate_plots = True
for z in "griz":
    my_X = f"C{z}"
    x = base[my_X].values
    x_w = weight_dict[z]
    
    for u in "griz":
        if u != z:
            my_Y = f"C{u}"
            my_Ye = my_Y + "_e"
            y_w = weight_dict[u]
            y = base[my_Y].values

            mydata = odr.Data(x, y, wd=1./x_w, we=1./y_w)
            myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
            myoutput = myodr.run()

            coefs = myoutput.beta
            unc = myoutput.sd_beta

            ys.append(my_Y)
            xs.append(my_X)
            x_w0s.append(w0[z])
            y_w0s.append(w0[u])
            f_order.append(0)
            sine.append(False)
            cosine.append(False)

            As.append(coefs[1])
            A_e.append(unc[1])
            Bs.append(coefs[0])
            B_e.append(unc[0])

            if generate_plots:
                arr_test = np.arange(x.min(), x.max()+0.1, 0.1)
                fig, ax = plt.subplots()
                ax.scatter(x,y, c = "k", s = 5)
                ax.plot(arr_test, f(coefs, arr_test), c = "r")
                ax.errorbar(x, y, yerr=y_w, xerr = x_w, fmt='none', zorder = 0, ecolor = "k")
                ax.set_xlabel(my_X)
                ax.set_ylabel(my_Y)
                ax.text(0.1,0.8,f"$a + b*{{{my_X}}}$"+"\n"+ f"$a = {{{coefs[1]:.5}}}\pm{{{unc[1]:.5}}}$"+"\n"+ f"$b = {{{coefs[0]:.5}}}\pm{{{unc[0]:.5}}}$", transform=ax.transAxes)
                plt.savefig(path.join(local, "linear_fit_plots_sm_phase", f"{my_Y}({my_X}).png"))
                # plt.show()
                plt.clf()
                plt.close()

# All other coeffs
for k in range(1,16):
    for z in "griz":
        my_Xa = f"A{z}_{k}"
        xa = base[my_Xa].values
        to_keep = ~np.isnan(xa)
        xa = xa[to_keep]
        my_Xb = f"B{z}_{k}"
        xb = base[my_Xb].values[to_keep]
        x_w = weight_dict[z][to_keep]

        for u in "griz":
            if u != z:
                my_Ya = f"A{u}_{k}"
                ya = base[my_Ya].values[to_keep]
                to_keep2 = ~np.isnan(ya)
                ya = ya[to_keep2]
                if len(ya) > 10: # Custom??
                    xa2 = xa[to_keep2]
                    xb2 = xb[to_keep2]

                    x_w2 = x_w[to_keep2]
                    y_w = weight_dict[u][to_keep][to_keep2]

                    my_Yb = f"B{u}_{k}"
                    yb = base[my_Yb].values[to_keep][to_keep2]


                    mydata_a = odr.Data(xa2, ya, wd=1./x_w2, we=1./y_w)
                    myodr_a = odr.ODR(mydata_a, linear, beta0=[1., 0.])
                    myoutput_a = myodr_a.run()
                    a_coefs = myoutput_a.beta
                    a_unc = myoutput_a.sd_beta
                    
                    mydata_b = odr.Data(xb2, yb, wd=1./x_w2, we=1./y_w)
                    myodr_b = odr.ODR(mydata_b, linear, beta0=[1., 0.])
                    myoutput_b = myodr_b.run()
                    b_coefs = myoutput_b.beta
                    b_unc = myoutput_b.sd_beta


                    ys.append(my_Ya)
                    xs.append(my_Xa)
                    x_w0s.append(w0[z])
                    y_w0s.append(w0[u])
                    f_order.append(k)
                    sine.append(True)
                    cosine.append(False)
                    As.append(a_coefs[1])
                    A_e.append(a_unc[1])
                    Bs.append(a_coefs[0])
                    B_e.append(a_unc[0])

                    ys.append(my_Yb)
                    xs.append(my_Xb)
                    x_w0s.append(w0[z])
                    y_w0s.append(w0[u])
                    f_order.append(k)
                    sine.append(False)
                    cosine.append(True)
                    As.append(b_coefs[1])
                    A_e.append(b_unc[1])
                    Bs.append(b_coefs[0])
                    B_e.append(b_unc[0])

                    if generate_plots:
                        a_arr_test = np.arange(xa2.min(), xa2.max()+0.01, 0.01)
                        fig, ax = plt.subplots()
                        ax.scatter(xa2,ya, c = "k", s = 5)
                        ax.plot(a_arr_test, f(a_coefs, a_arr_test), c = "r")
                        ax.errorbar(xa2, ya, yerr=y_w, xerr = x_w2, fmt='none', zorder = 0, ecolor = "k")
                        ax.set_xlabel(my_Xa)
                        ax.set_ylabel(my_Ya)
                        ax.text(0.1,0.8,f"$a + b*{{{my_Xa}}}$"+"\n"+ f"$a = {{{a_coefs[1]:.5}}}\pm{{{a_unc[1]:.5}}}$"+"\n"+ f"$b = {{{a_coefs[0]:.5}}}\pm{{{a_unc[0]:.5}}}$", transform=ax.transAxes)
                        plt.savefig(path.join(local, "linear_fit_plots_sm_phase", f"{my_Ya}({my_Xa}).png"))
                        # plt.show()
                        plt.clf()
                        plt.close()

                        b_arr_test = np.arange(xb2.min(), xb2.max()+0.01, 0.01)
                        fig, ax = plt.subplots()
                        ax.scatter(xb2,yb, c = "k", s = 5)
                        ax.plot(b_arr_test, f(b_coefs, b_arr_test), c = "r")
                        ax.errorbar(xb2, yb, yerr=y_w, xerr = x_w2, fmt='none', zorder = 0, ecolor = "k")
                        ax.set_xlabel(my_Xb)
                        ax.set_ylabel(my_Yb)
                        ax.text(0.1,0.8,f"$a + b*{{{my_Xb}}}$"+"\n"+ f"$a = {{{b_coefs[1]:.5}}}\pm{{{b_unc[1]:.5}}}$"+"\n"+ f"$b = {{{b_coefs[0]:.5}}}\pm{{{b_unc[0]:.5}}}$", transform=ax.transAxes)
                        plt.savefig(path.join(local, "linear_fit_plots_sm_phase", f"{my_Yb}({my_Xb}).png"))
                        # plt.show()
                        plt.clf()
                        plt.close()

df_out = pd.DataFrame(data={"Y" : ys, "X" : xs, "Y_w0" : y_w0s, "X_w0" : x_w0s, "f_ord" : f_order, "sine" : sine, "cosine" : cosine, "a" : As, "a_e" : A_e, "b" : Bs, "b_e" : B_e})
df_out.to_csv(path.join(local, "sm_linear_fits_ab_phase.csv"), index = False)
