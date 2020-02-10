import copy
import numpy as np


def scaler_zero_one_all_cols(data, min_, max_):
    print data.shape
    data_cpy = copy.deepcopy(data)
    data_cpy = data_cpy.astype(np.float64)

    for row in range(data_cpy.shape[0]):
        for col in range(data_cpy.shape[1]):
            data_cpy[row, col] = (data_cpy[row, col] - min_) / float(max_ - min_)

    return data_cpy
