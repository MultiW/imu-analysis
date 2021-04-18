from enum import Enum
import pandas as pd
import numpy as np

# import data types
from pandas import DataFrame
from numpy import ndarray
from typing import List, Tuple
from pathlib import Path

from src.data.enums import Activity, DataState
from src.config import RAW_BOOT_FILE, RAW_POLE_FILE, CLEAN_SUFFIX, CLEAN_DIR


# Steps labels file columns
class BootCsvCol:
    TIME = 'boot.up'
    BOOT_UP = 'epoch.boot.up'
    GLIDE_START = 'epoch.glide'	
    NAME = 'name'
    SENSOR = 'sensor'
    SIDE = 'side'
    TEST = 'test'


class PoleCsvCol:
    TIME = 'time'
    START = 'epoch.time'
    END = 'epoch.end'
    NAME = 'name'
    SENSOR = 'sensor'
    SIDE = 'side'
    TEST = 'test'


class LabelCol:
    """
    Column indices for the NumPy array
    """
    TIME = 0
    START = 1
    END = 2
    NAME = 3
    SENSOR = 4
    SIDE = 5
    TEST = 6


class TestType(Enum):
    """
    Maps to values in the step labels data file.
    """
    Any = ''
    Skate = 'skate'
    SkateNormal = 'normal'
    Pole = 'pole'
    PoleNormal = 'normal'


def get_labels_file(activity: Activity, data_state: DataState) -> Path:
    raw_file: Path = RAW_BOOT_FILE if activity == Activity.Boot else RAW_POLE_FILE

    if data_state == DataState.Raw:
        return raw_file
    else:
        file_name: str = '%s%s' % (raw_file.stem, CLEAN_SUFFIX)
        return CLEAN_DIR / file_name


def load_labels(file: str, labels_type: Activity) -> ndarray:
    df: DataFrame = pd.read_csv(file)
    if labels_type == Activity.Boot:
        return df[[
            BootCsvCol.TIME,
            BootCsvCol.BOOT_UP, 
            BootCsvCol.GLIDE_START, 
            BootCsvCol.NAME,
            BootCsvCol.SENSOR,
            BootCsvCol.SIDE,
            BootCsvCol.TEST
            ]].to_numpy()
    if labels_type == Activity.Pole:
        return df[[
            PoleCsvCol.TIME,
            PoleCsvCol.START,
            PoleCsvCol.END,
            PoleCsvCol.NAME,
            PoleCsvCol.SENSOR,
            PoleCsvCol.SIDE,
            PoleCsvCol.TEST
        ]].to_numpy()


def get_workouts_row_bounds(labels: ndarray) -> List[Tuple[int, int]]:
    """
    @return: list of tuples (start, end). "start" and "end" are row indexes to "labels". They are the start/end bounds of a test (inclusive)
    """
    all_tests = []

    # Get end times of all ski workouts (except for the last)
    time_diff = np.diff(labels[:, LabelCol.TIME])
    end_indices = np.where(time_diff < 0)[0] # row numbers
    
    num_tests = len(end_indices) + 1
    for i in range(num_tests):
        if i == 0 and i == num_tests-1:
            all_tests.append((0, labels.shape[0]-1))
        elif i == 0: # first test
            all_tests.append((0, end_indices[0]))
        elif i == num_tests-1: # last test
            all_tests.append((end_indices[-1]+1, labels.shape[0]-1))
        else:
            all_tests.append((end_indices[i-1]+1, end_indices[i]))

    return all_tests


def get_workouts_epoch_bounds(labels: ndarray) -> List[Tuple[int, int]]:
    all_epoch_bounds = []

    row_bounds = get_workouts_row_bounds(labels)
    for (start, end) in row_bounds:
        all_epoch_bounds.append((labels[start, LabelCol.START], labels[end, LabelCol.END]))
    
    return all_epoch_bounds


def get_workouts_sensor(labels: ndarray) -> List[str]:
    all_sensors = []

    row_bounds = get_workouts_row_bounds(labels)
    for (start, _) in row_bounds:
        all_sensors.append(labels[start, LabelCol.SENSOR])
    
    return all_sensors