import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.data_utils import read_data_csv, read_data_txt
from utils.model_utils import *

j =1j
pi = np.pi

# data loading
PATH = 'data.csv'
data = read_data_csv(PATH)  #[Capacitance, Resistance, Frequency]
C = data[0]
R = data[1]
freq = data[2]
omega = 2*np.pi*freq


# system impedance
Z = R + 1/(j*omega*C)
ZReal = np.real(Z)
ZImag = np.imag(Z)
# print(Z)
# print(ZReal)
# print(ZImag)


# curve fitting using models
## initial guess
p0 = [1e-12, 1e-12, 1e5, 1e7]
Z_RI = np.concatenate((ZReal,ZImag),axis=0)
popt_RI, pcov_RI = curve_fit(model_simplified_RI, freq, Z_RI, p0=p0)
ZFit = model_simplified(freq, *popt_RI)
print('[cw, cm, ri, rp]:', popt_RI)  # print optimal values


# visualize
plt.subplot(121)
plt.plot(freq, ZReal, "k.", label="Z_initial")
plt.plot(freq, np.real(ZFit), label="Best fit")
plt.ylabel("Real part of y")
plt.xlabel("x")
plt.legend()

plt.subplot(122)
plt.plot(freq, ZImag, "k.")
plt.plot(freq, np.imag(ZFit))
plt.ylabel("Imaginary part of y")
plt.xlabel("x")

plt.show()
