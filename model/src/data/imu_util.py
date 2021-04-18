import pathlib
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from scipy.interpolate import interp1d
from enum import Enum
import os
import copy
from collections import Counter

from src.config import RAW_DIR, RAW_BOOT_FILE, RAW_POLE_FILE, MAX_SAMPLING_INTERVAL_RANGE
from src.data.data_util import shift, low_pass_filter

# import data types
from pandas import DataFrame
from numpy import ndarray
from typing import List, Tuple, Optional


# Raw IMU data column names
class ImuCsvCol:
    EPOCH = 'epoc (ms)'
    XACCEL = 'x-axis (g)'
    YACCEL = 'y-axis (g)'
    ZACCEL = 'z-axis (g)'


# NumPy column indices
class ImuCol:
    TIME = 0
    XACCEL = 1
    YACCEL = 2
    ZACCEL = 3
    KEY = 4 # epoch time from raw IMU data. Raw labels refer to data points using this value


class Sensor(Enum):
    Any = ''
    Accelerometer = 'Accelerometer'
    Gyroscope = 'Gyroscope'
    Magnetometer = 'Magnetometer'


def list_imu_abspaths(sensor_name: str = '', sensor_type=Sensor.Any) -> List[str]:
    """
    List the absolute paths of all IMU files
    Only files satisfying all filter criterias will be returned
    """
    output: List[str] = []
    for filename in os.listdir(RAW_DIR):
        if filename == RAW_BOOT_FILE.name or filename == RAW_POLE_FILE.name:
            # ignore if label file
            continue
        if filename.startswith(sensor_name) and sensor_type.value in filename:
            output.append(RAW_DIR / filename)

    return output


def get_sensor_file(sensor_name: str, sensor_type):
    files: List[str] = list_imu_abspaths(sensor_name, sensor_type)

    if len(files) != 1:
        raise Exception('More than one file found.')
    
    return files[0]


def load_imu_data(file) -> ndarray:
    """
    @param file: file path or stream object
    @return: NumPy array of the IMU data
    """
    try:
        data = pd.read_csv(file)
        if not (
            ImuCsvCol.EPOCH in data
            and ImuCsvCol.XACCEL in data
            and ImuCsvCol.YACCEL in data
            and ImuCsvCol.ZACCEL in data
        ):
            return None

        return data[[ImuCsvCol.EPOCH, ImuCsvCol.XACCEL, ImuCsvCol.YACCEL, ImuCsvCol.ZACCEL]].to_numpy()
    except:
        return None


def time_to_row_range(imu_data: ndarray, start_epoch, end_epoch, expected_range=None, expected_range_error=None) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert time range to row number range. Given times must exist in IMU data

    IMU data may have weird epoch times (e.g. duplicate epoch times), use the expected range right row.
    Return None row numbers if time range cannot be converted

    @param imu_data: numpy array of IMU data
    @param start_epoch: start epoch time
    @param end_epoch: end epoch time
    @param expected_range: expected time range (in seconds). Used if IMU data has weird epoch times
    @param expected_range_error: error tolerance of the expected range (in seconds)
    @return: start and end row indices to the IMU data
    """
    start_row = np.where(imu_data[:, ImuCol.TIME] == start_epoch)[0]
    end_row = np.where(imu_data[:, ImuCol.TIME] == end_epoch)[0]

    if len(start_row) == 0 or len(end_row) == 0:
        # Epoch not found in IMU data 
        return (None, None)

    if len(start_row) > 1 and len(end_row) > 1:
        # IMU data has too many duplicates
        return (None, None)

    if len(start_row) == 1 and len(end_row) == 1:
        return (start_row[0], end_row[0])

    if expected_range is None or expected_range_error is None:
        return (None, None)

    # == Handle case where there are multiple rows with the same start/end epoch time ==
    # Find sampling interval of IMU data
    sampling_interval = _get_sampling_interval(imu_data)
    if sampling_interval is None:
        return (None, None)

    # Estimate the number of rows for "expected_range"
    num_rows = expected_range * 1000 / sampling_interval
    num_rows_error = expected_range_error * 1000 / sampling_interval

    if len(start_row) == 1 and len(end_row) > 1:
        start_row = start_row[0]
        for curr_end_row in end_row:
            if start_row > curr_end_row:
                continue
            if (curr_end_row - start_row) - num_rows <= num_rows_error:
                print('WARNING: Found row range via educated guess. Graph to double check.')
                return (start_row, curr_end_row)
    elif len(start_row) > 1 and len(end_row) == 1:
        end_row = end_row[0]
        for curr_start_row in start_row:
            if curr_start_row > end_row:
                continue
            if (end_row - curr_start_row) - num_rows <= num_rows_error:
                print('WARNING: Found row range via educated guess. Graph to double check.')
                return (curr_start_row, end_row)

    return (None, None)


def _get_common_intervals(imu_data: ndarray) -> List[int]:
    intervals = np.diff(imu_data[:, ImuCol.TIME])
    
    # We cannot have negative time intervals
    intervals = intervals[(intervals > 0)] 

    common_intervals = Counter(intervals).most_common(MAX_SAMPLING_INTERVAL_RANGE)
    return [interval for (interval, count) in common_intervals]


def _get_sampling_interval(imu_data: ndarray) -> float:
    """
    Get average sampling interval (i.e. average time difference between neighboring points)
    """
    common_intervals = _get_common_intervals(imu_data)

    return sum(common_intervals) / len(common_intervals)


def fix_epoch(imu_data: ndarray) -> ndarray:
    intervals = np.diff(imu_data[:, ImuCol.TIME])

    common_intervals = _get_common_intervals(imu_data)
    bad_intervals = np.where(~np.isin(intervals, common_intervals))[0]

    new_imu_data = copy.deepcopy(imu_data)
    if bad_intervals.shape[0] != 0:
        # Fix bad interval values
        intervals[bad_intervals] = _get_sampling_interval(imu_data)

        # Recompute epoch times based on new interval values
        base_epoch = imu_data[0, ImuCol.TIME]
        new_imu_data[0, ImuCol.TIME] = base_epoch
        new_imu_data[1:, ImuCol.TIME] = base_epoch + np.cumsum(intervals)

    return new_imu_data


def resample_uniformly(imu_data: ndarray, interval=20) -> ndarray:
    """
    @param interval: new interval between data points (in milliseconds)
    """
    f_x = interp1d(imu_data[:, ImuCol.TIME], imu_data[:, ImuCol.XACCEL], kind='cubic', fill_value='extrapolate')
    f_y = interp1d(imu_data[:, ImuCol.TIME], imu_data[:, ImuCol.YACCEL], kind='cubic', fill_value='extrapolate')
    f_z = interp1d(imu_data[:, ImuCol.TIME], imu_data[:, ImuCol.ZACCEL], kind='cubic', fill_value='extrapolate')
    time_uniform = np.arange(imu_data[:, ImuCol.TIME].min(), imu_data[:, ImuCol.TIME].max(), interval)
    return np.hstack((
        np.array([time_uniform]).T, 
        np.array([f_x(time_uniform)]).T, 
        np.array([f_y(time_uniform)]).T, 
        np.array([f_z(time_uniform)]).T
    ))


def epoch_ms_to_s(imu_data: ndarray) -> ndarray:
    """
    Timestamps should be ordered
    """
    result = copy.deepcopy(imu_data)
    result[:, ImuCol.TIME] = (imu_data[:, ImuCol.TIME] - imu_data[0, ImuCol.TIME]) * 0.001
    return result


def clean_imu_data(imu_data: ndarray) -> ndarray:
    """
    Convert raw IMU data into something displayable for users
    """
    # Fix bad epoch values
    result = fix_epoch(imu_data)

    # Resample to make data points evenly spaced
    result = resample_uniformly(result)

    # Convert timestamps to ms, with first data point at 0 ms
    result = epoch_ms_to_s(result)

    # Apply a low-pass filter
    result[:, ImuCol.XACCEL] = low_pass_filter(result[:, ImuCol.XACCEL])
    result[:, ImuCol.YACCEL] = low_pass_filter(result[:, ImuCol.YACCEL])
    result[:, ImuCol.ZACCEL] = low_pass_filter(result[:, ImuCol.ZACCEL])

    return result


def normalize_with_bounds(array: ndarray, start_row: int, end_row: int) -> ndarray:
    """
    Normalize all values based on the max of the data. Compute the max from values only within the given range

    @param start_row: inclusive. Must be within bounds
    @param end_row: inclusive. Must be within bounds
    """
    max = array[start_row:end_row+1].max()
    return array[:] / max


def normalize_imu_with_bounds(imu_data: ndarray, start_row: int, end_row: int) -> ndarray:
    # avoid mutating input
    imu_data = copy.deepcopy(imu_data)

    imu_data[:, ImuCol.XACCEL] = normalize_with_bounds(imu_data[:, ImuCol.XACCEL], start_row, end_row)
    imu_data[:, ImuCol.YACCEL] = normalize_with_bounds(imu_data[:, ImuCol.YACCEL], start_row, end_row)
    imu_data[:, ImuCol.ZACCEL] = normalize_with_bounds(imu_data[:, ImuCol.ZACCEL], start_row, end_row)

    return imu_data


def get_data_chunk(imu_data: ndarray, start_row: int, end_row: int, padding: Optional[int] = None) -> ndarray:
    """
    Safely get a range of rows from the given data. If the given range or range with buffer falls out of bounds of the data, 
    then return anything within bounds.

    @param start_row: inclusive
    @param end_row: inclusive
    @return: returns requested chunk of imu data. 
        If padding options is used, additionally returns the indexes to the data that isn't padding
    """
    if padding is not None:
        chunk_start = start_row - padding
        chunk_end = end_row + padding
    
    # indexes for the chunk
    chunk_start, chunk_end = max(start_row, 0), min(end_row, imu_data.shape[0]-1)

    # index (within the chunk) of the data that isn't padding
    data_start = start_row - chunk_start
    data_end = end_row - chunk_start

    if padding is None:
        return imu_data[chunk_start:chunk_end+1,]

    return imu_data[chunk_start:chunk_end,], (data_start, data_end)


def data_to_features(imu_data: ndarray) -> ndarray:
    """
    @param imu_data: for best performance, include than the required range of rows to be predicted. 
        E.g. if you want to use the model for rows 100-200, include rows 50 to 250.
        This makes lead and lag features more accurate.
    """
    # acceleration features
    features = np.hstack((
        np.array([imu_data[:, ImuCol.XACCEL]]).T,
        np.array([imu_data[:, ImuCol.YACCEL]]).T,
        np.array([imu_data[:, ImuCol.ZACCEL]]).T,
        np.array([np.sqrt(
            np.square(imu_data[:, ImuCol.XACCEL]) 
            + np.square(imu_data[:, ImuCol.YACCEL]) 
            + np.square(imu_data[:, ImuCol.ZACCEL])
        )]).T
    ))

    # Use the lead and lag of each acceleration column as features
    for i in range(4): # iterate acceleration columns
        curr_feature = np.array([features[:, i]]).T
        
        # lead 
        features = np.append(features, curr_feature - shift(curr_feature, 5, fill_value=0), axis=1) # axis=1 is hstack
        features = np.append(features, curr_feature - shift(curr_feature, 6, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, 7, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, 8, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, 9, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, 10, fill_value=0), axis=1)
        
        # lag
        features = np.append(features, curr_feature - shift(curr_feature, -5, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, -6, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, -7, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, -8, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, -9, fill_value=0), axis=1)
        features = np.append(features, curr_feature - shift(curr_feature, -10, fill_value=0), axis=1)

    return features