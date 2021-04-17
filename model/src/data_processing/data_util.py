import numpy as np
from scipy import signal

# import data types
from numpy import ndarray
from typing import List, Tuple


def add_col(labels: ndarray, dtype: any) -> int:
    """
    Add new column to the "labels" matrix. 
    Return the index to the new column
    """
    num_rows, num_cols = labels.shape
    np.append(labels, np.empty([num_rows, 1], dtype=dtype), axis=1)
    return num_cols


def find_nearest_index(array: ndarray, value: any):
    """
    @param array: row vector
    @param value: value whose type matches values in "array"
    """
    return (np.abs(array - value)).argmin()


def find_nearest(array: ndarray, value: any):
    """
    @param array: row vector
    @param value: value whose type matches values in "array"
    """
    return array[find_nearest_index(array, value)]


def shift(array, num, fill_value=np.nan):
    """
    @param fill_value: slots that have had their values shifted will be filled with this value
    """
    result = np.empty_like(array)
    if num > 0:
        result[:num] = fill_value
        result[num:] = array[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = array[-num:]
    else:
        result[:] = array
    return result


def low_pass_filter(data: ndarray, cutoff=10, sampling_freq=50) -> ndarray:
    w = cutoff / (sampling_freq / 2) # Normalize the frequency
    b, a = signal.butter(2, w, 'low')

    return signal.filtfilt(b, a, data)


def are_overlapping(bounds1: Tuple[float, float], bounds2: Tuple[float, float]):
    """
    Check if two number ranges overlap
    """
    return not (bounds1[1] < bounds2[0] or bounds1[0] > bounds2[1])