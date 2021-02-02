from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

###########################################################
## data processing
j =1j
pi = np.pi

data = pd.read_csv('data.csv', header=None, usecols=[0, 1, 3])
data.columns = ['C', 'R', 'freq']
data['freq'] = data['freq'].astype(int)
data['omega'] = 2*np.pi*data['freq']
data['lgf'] = np.log(data['freq'])
data['Z'] = 0
for i in range(data['freq'].size):
    # data['Z'].iloc[i] = 1/(1/data['R'].iloc[i] + j*data['omega'].iloc[i]*data['C'].iloc[i])
    data['Z'].iloc[i] = data['R'].iloc[i] + 1/(j*data['omega'].iloc[i]*data['C'].iloc[i])

data['xdata'] = data['freq']
data['ydata'] = data['Z']
f = np.array(data['xdata'])
Z = np.array(data['ydata'])
Z_real = np.real(Z)

###########################################################
## calculate impedance based on real-model (paper: Capacitive on-line hematocrit...)
def func(f, cw, cm, ri, rp):
    return 2/(j*2*pi*f*cw) + 1/(1/rp+1/(ri+2/(j*2*pi*f*cm)))

def funcBoth(f, cw, cm, ri, rp):
    N = len(f)
    f_real = f[:N//2]
    f_imag = f[N//2:]
    Z_fit_real = np.real(func(f_real, cw, cm, ri, rp))
    Z_fit_imag = np.imag(func(f_imag, cw, cm, ri, rp))
    return np.hstack([Z_fit_real, Z_fit_imag])

ZReal = np.real(Z)
ZImag = np.imag(Z)
ZBoth = np.hstack([ZReal, ZImag])

## initial guess
p0 = [1e-12, 1e-12, 1e5, 1e7]
poptBoth, pcovBoth = curve_fit(funcBoth, np.hstack([f, f]), ZBoth, p0=p0)

ZFit = func(f, *poptBoth)
print('cw, cm, ri, rp')
print(poptBoth)
# print(ZFit)

plt.subplot(121)
plt.plot(f, np.real(Z), "k.", label="Z_initial")
plt.plot(f, np.real(ZFit), label="Best fit")
plt.ylabel("Real part of y")
plt.xlabel("x")
plt.legend()

plt.subplot(122)
plt.plot(f, np.imag(Z), "k.")
plt.plot(f, np.imag(ZFit))
plt.ylabel("Imaginary part of y")
plt.xlabel("x")

plt.show()



