import pathlib
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

# import types
from pandas import DataFrame
from numpy import ndarray

# Raw IMU data column names
ELAPSED = 'elapsed (s)'
XACCEL = 'x-axis (g)'
YACCEL = 'y-axis (g)'
ZACCEL = 'z-axis (g)'

# NumPy column indices
TIME_COL = 0
XACCEL_COL = 1
YACCEL_COL = 2
ZACCEL_COL = 3

SIDE = 'side'
TEST = 'test'


def preprocess_raw_imu(raw_data: DataFrame) -> ndarray:
    # Replace negative "time" values. Replace with interpolated values from their neighbors
    raw_data.loc[raw_data[ELAPSED] < 0, ELAPSED] = None
    raw_data[ELAPSED] = raw_data[ELAPSED].interpolate(method ='linear', limit_direction ='forward', axis=0)

    # TODO Check "time" values that aren't in order?
    # - Think this was a possibility. Plot the entire raw IMU data to check again.

    # Note: make sure order matches global constants
    return raw_data[[ELAPSED, XACCEL, YACCEL, ZACCEL]].to_numpy()


if __name__ == "__main__":
    curr_dir:str = pathlib.Path(__file__).parents[0].absolute()

    # Import data
    # input (raw IMU data)
    raw_data: DataFrame = pd.read_csv(curr_dir / "11L_2020-08-13T09.48.23.554_E8E376103A59_Accelerometer.csv")
    # labels (of ski steps)
    boot_labels: DataFrame = pd.read_csv(curr_dir / "boot3MT_20210201.csv")
    pole_labels: DataFrame = pd.read_csv(curr_dir / "pole3MT_20210201.csv")

    # Pre-process raw data
    raw_data: ndarray = preprocess_raw_imu(raw_data)

    # == Boot Ski Test ==
    boot_raw = raw_data[11849:21339,:]
    boot_labels = boot_labels[(boot_labels[SIDE] == 'L') & (boot_labels[TEST] == 'skate')]

