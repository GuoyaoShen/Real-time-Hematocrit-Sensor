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

# N = 20, num of experiments
N = 20
param_optimal = np.zeros((N, 4))  # [N,4]
data_CR = np.zeros((N, len(freq), 2))  # [N,Nf,2]

#-------------------------------------------------------------------

# path_mother = './data/Whole Blood/WholeBloodExp'
# path_param_save = './data/Whole_Blood.npz'
# path_data_CR_save = './data/CR_Whole_Blood.npz'

# path_mother = './data/1.5mL Plasma/1.5PlasmaExp'
# path_param_save = './data/1_5mLPlasma.npz'
# path_data_CR_save = './data/CR_1_5mLPlasma.npz'

# path_mother = './data/3.0mL Plasma/3.0PlasmaExp'
# path_param_save = './data/3_0mLPlasma.npz'
# path_data_CR_save = './data/CR_3_0mLPlasma.npz'

path_mother = './data/4.5mL Plasma/4.5PlasmaExp'
path_param_save = './data/4_5mLPlasma.npz'
path_data_CR_save = './data/CR_4_5mLPlasma.npz'

for idx_exp in range(21, 41):
    path = path_mother + str(idx_exp) + '.txt'
    data = read_data_txt(path)  # [Capacitance, Resistance, Frequency]
    C = data[:, 0]  # [Nf,]
    R = data[:, 1]  # [Nf,]

    # system impedance
    Z = R + 1 / (j * omega * C)
    ZReal = np.real(Z)
    ZImag = np.imag(Z)

    # curve fitting using models
    ## initial guess
    p0 = [1e-12, 1e-12, 1e5, 1e7]
    Z_RI = np.concatenate((ZReal, ZImag), axis=0)
    popt_RI, pcov_RI = curve_fit(model_simplified_RI, freq, Z_RI, p0=p0)
    ZFit = model_simplified(freq, *popt_RI)

    param_optimal[idx_exp-21, :] = popt_RI  # [cw, cm, ri, rp]
    data_CR[idx_exp - 21, :, 0] = C
    data_CR[idx_exp - 21, :, 1] = R

# print(data_CR)
print(param_optimal)
np.savez(path_param_save, model_param=param_optimal)
np.savez(path_data_CR_save, data_CR=data_CR)
print('Params saved.')
