import pathlib
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from spike_preprocess_v2 import preprocess_raw_imu

# import types
from pandas import DataFrame
from numpy import ndarray

FILE_LEFT = '11L_2020-08-13T09.48.23.554_E8E376103A59_Accelerometer.csv'
FILE_RIGHT = '11R_2020-10-17T09.50.55.227_C38EC55152D6_Accelerometer.csv'

PLOT_CONFIG = [
    { 'name': 'pole', 'file': FILE_LEFT, 'start': 60643, 'end': 70135 },
    { 'name': 'pole normal', 'file': FILE_LEFT, 'start': 112335, 'end': 121819 },
]

if __name__ == "__main__":
    curr_dir:str = pathlib.Path(__file__).parents[0].absolute()

    f, plots = pyplot.subplots(len(PLOT_CONFIG), 1)

    for i in range(len(PLOT_CONFIG)):
        config = PLOT_CONFIG[i]
        plot = plots if len(PLOT_CONFIG) == 1 else plots[i]

        raw_data: DataFrame = pd.read_csv(curr_dir / config['file'])
        raw_data: ndarray = preprocess_raw_imu(raw_data)

        test_data = raw_data[config['start']:config['end']+1, :]

        plot.plot(test_data[:,1], label = "x-axis acceleration")
        plot.plot(test_data[:,2], label = "y-axis acceleration")
        plot.plot(test_data[:,3], label = "z-axis acceleration")
        plot.set_title(config['name'])
        plot.set(ylabel='Acceleration (g)')
        plot.legend()

    pyplot.show()