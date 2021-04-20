from enum import Enum
import pandas as pd
import numpy as np

# import data types
from pandas import DataFrame
from numpy import ndarray
from typing import List, Tuple
from pathlib import Path

from src.data.workout import Activity, Workout
from src.data.data import DataState
from src.config import RAW_BOOT_FILE, RAW_POLE_FILE, CLEAN_DATA_SUFFIX, CLEAN_LABELS_SUFFIX, CLEAN_DIR


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
        file_name: str = '%s%s' % (raw_file.stem, CLEAN_LABELS_SUFFIX)
        return CLEAN_DIR / file_name


def load_clean_labels(activity: Activity) -> ndarray:
    return np.load(get_labels_file(activity, DataState.Clean), allow_pickle=True)


def save_clean_labels(labels: ndarray, activity: Activity):
    np.save(get_labels_file(activity, DataState.Clean), labels)


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


def are_labels_valid(labels):
    """
    All labels should refer to the same sensor, athlete, etc.
    """
    if labels.shape[0] == 0:
        return False

    sensor: str = labels[0, LabelCol.SENSOR]
    name: str = labels[0, LabelCol.NAME]
    side: str = labels[0, LabelCol.SIDE]
    test: str = labels[0, LabelCol.TEST]
    for i in range(labels.shape[0]):
        if (sensor != labels[i, LabelCol.SENSOR] or name != labels[i, LabelCol.NAME] 
                or side != labels[0, LabelCol.SIDE] or test != labels[0, LabelCol.TEST]):
            return False
    return True


def get_workouts(labels: ndarray) -> List[Workout]:
    """
    @return: list of tuples (start, end). "start" and "end" are row indexes to "labels". They are the start/end bounds of a test (inclusive)
    """
    all_workouts: List[Workout] = []

    # Get end rows of all ski workouts (except for the last)
    time_diff = np.diff(labels[:, LabelCol.TIME])
    end_indices = np.where(time_diff < 0)[0] # row numbers

    num_tests = len(end_indices) + 1
    for i in range(num_tests):
        # Get start/end rows of workout
        if i == 0 and i == num_tests-1:
            start_row, end_row = 0, labels.shape[0]-1
        elif i == 0: # first test
            start_row, end_row = 0, end_indices[0]
        elif i == num_tests-1: # last test
            start_row, end_row = end_indices[-1]+1, labels.shape[0]-1
        else:
            start_row, end_row = end_indices[i-1]+1, end_indices[i]

        # Check workout validity
        workout_labels: ndarray = labels[start_row:end_row + 1,]
        if not are_labels_valid(workout_labels):
            print('Workout invalid. Skipping')
            continue

        all_workouts.append(Workout(workout_labels, workout_labels[0, LabelCol.SENSOR]))

    return all_workouts
