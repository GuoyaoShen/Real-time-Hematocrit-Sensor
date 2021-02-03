import pandas as pd
import numpy as np

def read_data(PATH):
    '''
    read csv data file
    :param PATH: string, path for csv file
    :return: np array, [n_feature,len_data], data as np array. [Capacitance, Resistance, Frequency]
    '''
    data = pd.read_csv(PATH, header=None, usecols=[0, 1, 3])
    data = np.asarray(data).T
    return data