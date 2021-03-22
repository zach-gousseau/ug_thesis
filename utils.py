from datetime import datetime, date, timedelta
from itertools import tee, chain
import numpy as np
import os

def subtract_times(t2, t1, day_delta='auto'):
    t1_day = date.today()
    t2_day = date.today()
    if day_delta == 'auto':
        if t2 < t1:
            t2_day += timedelta(days=1)
    elif day_delta == 'off':
        pass
    elif type(day_delta) == int:
        t2.day += timedelta(days=day_delta)
    else:
        raise ValueError('day_delta must be one of: auto, off or an integer')
    return datetime.combine(t2_day, t2) - datetime.combine(t1_day,  t1)


def split_at_delimiter(X, delimiter=0, return_idx=False):
    if type(X) != np.ndarray:
        X = np.array(X)
    if return_idx:
        indices = [0] + list(np.where(X == 0)[0]) + [len(X)]
        return [(indices[i] + 1, indices[i+1]) if i != 0 else (0, indices[i+1]) for i in range(len(indices) - 1)]
    else:
        idx = np.where(X != delimiter)[0]
        return np.split(X[idx], np.where(np.diff(idx) != 1)[0] + 1)


def split_at_delimiter_2d(X, delimiter=0, return_idx=False):
    if type(X) != np.ndarray:
        X = np.array(X)
    out = []
    for X_0 in X:
        if return_idx:
            indices = [0] + list(np.where(X_0 == 0)[0]) + [len(X_0)]
            out.append([(indices[i] + 1, indices[i+1]) if i != 0 else (0, indices[i+1]) for i in range(len(indices) - 1)])
        else:
            idx = np.where(X_0 != delimiter)[0]
            out.append(np.split(X_0[idx], np.where(np.diff(idx) != 1)[0] + 1))
    out = list(map(list, zip(*out)))
    return out

def append_2d(array, x):
    if type(array) == np.ndarray:
        out_array = np.ndarray(shape=(array.shape[0], array.shape[1] + 1))
        for i in range(len(array)):
            out_array[i] = np.append(array[i], x)
        return out_array
    elif type(array) == tuple:
        for i in range(len(array)):
            array[i].append(x)
        return array
    elif type(array) == list:
        array = tuple(array)
        for i in range(len(array)):
            array[i].append(x)
        return list(array)
    else:
        raise TypeError


def join_with_delimiter(x, delimiter=0):
    return [num for sublist in x for num in sublist + [delimiter]][:-1]


def join_with_delimiter_2d(x, delimiter=0):
    xx = list(map(list, zip(*x)))
    return [[num for sublist in xxx for num in list(sublist) + [delimiter]][:-1] for xxx in xx]




def make_dir(file_path='', dir_path=''):
    if file_path != '':
        dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
