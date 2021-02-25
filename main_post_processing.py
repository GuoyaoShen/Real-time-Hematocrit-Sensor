import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.data_utils import read_data_csv, read_data_txt
from utils.model_utils import *

j =1j
pi = np.pi

# data loading, get freq
PATH = 'data.csv'
data = read_data_csv(PATH)  #[Capacitance, Resistance, Frequency]
freq = data[2]
omega = 2*np.pi*freq


# Load model param
# path_param_save = './data/Whole_Blood.npz'
# path_data_CR_save = './data/CR_Whole_Blood.npz'

# path_param_save = './data/1_5mLPlasma.npz'
# path_data_CR_save = './data/CR_1_5mLPlasma.npz'

# path_param_save = './data/3_0mLPlasma.npz'
# path_data_CR_save = './data/CR_3_0mLPlasma.npz'

path_param_save = './data/4_5mLPlasma.npz'
path_data_CR_save = './data/CR_4_5mLPlasma.npz'

param_model = np.load(path_param_save)
param_model = param_model['model_param']  # [cw, cm, ri, rp]
data_CR = np.load(path_data_CR_save)
data_CR = data_CR['data_CR']  #[N,Nf,2]
# print(data_CR.shape)
avg_C = np.average(data_CR[..., 0], axis=0)
avg_R = np.average(data_CR[..., 1], axis=0)
# print(param_model)

cw = param_model[:, 0]
cm = param_model[:, 1]
ri = param_model[:, 2]
rp = param_model[:, 3]


avg_cw = np.average(cw)
avg_cm = np.average(cm)
avg_ri = np.average(ri)
avg_rp = np.average(rp)


std_cw = np.std(cw)
std_cm = np.std(cm)
std_ri = np.std(ri)
std_rp = np.std(rp)


print('cw AVG:', avg_cw)
print('cw STD:', std_cw)
print('cw STD %:', std_cw/avg_cw * 100)
print('--------------')

print('cm AVG:', avg_cm)
print('cm STD:', std_cm)
print('cm STD %:', std_cm/avg_cm * 100)
print('--------------')

print('ri AVG:', avg_ri)
print('ri STD:', std_ri)
print('ri STD %:', std_ri/avg_ri * 100)
print('--------------')

print('rp AVG:', avg_rp)
print('rp STD:', std_rp)
print('rp STD %:', std_rp/avg_rp * 100)
print('--------------')


# ============== Visualize
# system impedance
Z = avg_R + 1/(j*omega*avg_C)
ZReal = np.real(Z)
ZImag = np.imag(Z)
ZFit = model_simplified(freq, avg_cw, avg_cm, avg_ri, avg_rp)

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