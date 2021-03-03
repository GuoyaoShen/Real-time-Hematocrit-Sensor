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
path_param_save_list = ['./data/Whole_Blood.npz', './data/1_5mLPlasma.npz',
                   './data/3_0mLPlasma.npz', './data/4_5mLPlasma.npz']
path_data_CR_save_list = ['./data/CR_Whole_Blood.npz', './data/CR_1_5mLPlasma.npz',
                     './data/CR_3_0mLPlasma.npz', './data/CR_4_5mLPlasma.npz']
list_name = ['Whole Blood', '1.5mL Plasma', '3.0mL Plasma', '4.5mL Plasma']

model_param = np.zeros((4,4))  #[N_param, N_dilution] (cw,cm,ri,rp)
model_param_std = np.zeros((4,4))  #[N_param, N_dilution] (cw,cm,ri,rp)
for idx in range(len(path_param_save_list)):
    path_param_save = path_param_save_list[idx]
    path_data_CR_save = path_data_CR_save_list[idx]

    print('******', list_name[idx])

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
    print('=========================================')

    model_param[0, idx] = avg_cw
    model_param[1, idx] = avg_cm
    model_param[2, idx] = avg_ri
    model_param[3, idx] = avg_rp

    model_param_std[0, idx] = std_cw
    model_param_std[1, idx] = std_cm
    model_param_std[2, idx] = std_ri
    model_param_std[3, idx] = std_rp


    # ============== Visualize
    # plt.figure(idx)
    #
    # # system impedance
    # Z = avg_R + 1/(j*omega*avg_C)
    # ZReal = np.real(Z)
    # ZImag = np.imag(Z)
    # ZFit = model_simplified(freq, avg_cw, avg_cm, avg_ri, avg_rp)
    #
    # plt.subplot(121)
    # plt.plot(freq, ZReal, "k.", label="Z_initial")
    # plt.plot(freq, np.real(ZFit), label="Best fit")
    # plt.ylabel("Real part of y")
    # plt.xlabel("x")
    # plt.legend()
    #
    # plt.subplot(122)
    # plt.plot(freq, ZImag, "k.")
    # plt.plot(freq, np.imag(ZFit))
    # plt.ylabel("Imaginary part of y")
    # plt.xlabel("x")
    #
    # plt.show()

model_param_diff = np.zeros((4,4))
model_param_diff[:,1:] = np.diff(model_param) / model_param[:,1:] * 100

list_param_name = ['Cw', 'Cm', 'Ri', 'Rp']
list_param_name_rate = ['Cw Change in Percentage (%)', 'Cm Change in Percentage (%)',
                        'Ri Change in Percentage (%)', 'Rp Change in Percentage (%)']
for idx_param in range(model_param.shape[0]):
    plt.figure(idx_param)
    # plt.title(list_param_name[idx_param])
    # plt.plot(np.arange(model_param.shape[1]), model_param[idx_param], c='r', marker='^')
    # plt.bar(np.arange(model_param.shape[1]), model_param[idx_param], yerr=model_param_std[0])
    plt.title(list_param_name_rate[idx_param])
    plt.plot(np.arange(model_param.shape[1]), model_param_diff[idx_param], c='r', marker='^')
    plt.show()