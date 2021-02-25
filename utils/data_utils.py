import pandas as pd
import numpy as np

def read_data_csv(PATH):
    '''
    read csv data file
    :param PATH: string, path for csv file
    :return: np array, [n_feature,len_data], data as np array. [Capacitance, Resistance, Frequency]
    '''
    data = pd.read_csv(PATH, header=None, usecols=[0, 1, 3])
    data = np.asarray(data).T
    return data



def read_data_txt(PATH):
    data = np.loadtxt(PATH)
    data = data[:, :2]
    return data  # [Capacitance, Resistance]