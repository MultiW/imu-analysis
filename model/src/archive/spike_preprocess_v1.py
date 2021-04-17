"""
A manual attempt at mapping the training data labels (steps) to the raw IMU data
This heavily depends on how CSI Pacific data analysts processed their code
"""

import pathlib
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

curr_dir = pathlib.Path(__file__).parents[0].absolute()

# Column names
ELAPSED = 'elapsed (s)'
XACCEL = 'x-axis (g)'
YACCEL = 'y-axis (g)'
ZACCEL = 'z-axis (g)'
SIDE = 'side'
TEST = 'test'

BOOT_UP = 'boot.up'
GLIDE_START = 'glide.start'
TIME = 'time' # pole up
END = 'end' # pole down

TIME_COL = 0
XACCEL_COL = 1
YACCEL_COL = 2
ZACCEL_COL = 3


# Import
data = pd.read_csv(curr_dir / "11L_2020-08-13T09.48.23.554_E8E376103A59_Accelerometer.csv")
data_boot = pd.read_csv(curr_dir / "boot3MT_20210201.csv")
data_pole = pd.read_csv(curr_dir / "pole3MT_20210201.csv")

# Replace negative "elapsed time" values. Replace with interpolated values from their neighbors
data.loc[data[ELAPSED] < 0, ELAPSED] = None
data[ELAPSED] = data[ELAPSED].interpolate(method ='linear', limit_direction ='forward', axis=0) 

# TODO: enforce ascending order of "elapsed time"
#prev = -1
#for i in range(len(data.columns)):
#    if i > 0 and data.iloc[i-1][ELAPSED] >= data.iloc[i][ELAPSED]:
#        print(data.iloc[i-1][ELAPSED], data.iloc[i][ELAPSED])
#        data.iloc[i-1][ELAPSED] = None
#        data.iloc[i][ELAPSED] = None

data = data[[ELAPSED, XACCEL, YACCEL, ZACCEL]].to_numpy()

# Trim raw IMU data to specific ski tests
boot = data[11849:21339,:]
pole = data[60643:70135,:]

# Adjust Boot x-axis
boot[:,0] -= boot[0,0]
boot_freq = 1.0/round(np.mean(np.diff(boot[:,TIME_COL])), 2)
# TODO Not going to do this... the preprocessing relies heavily on the IMU file. At this rate, preprocessing will be very manual

# Adjust Pole x-axis
# trim start (as done in the R script)
pole_freq = round(1.0 / (pole[1,TIME_COL] - pole[0,TIME_COL]), 2)
start = np.where(pole[:,XACCEL_COL] > 5)[0][0]
if start >= 5: # why 5? That's what the data analysts did in their analysis in the R script. This may not be generalizable to the entire training data set
    pole = pole[start-5:,:]
# adjust elapsed time (as done in the R script)
indices = np.arange(1, pole.shape[0] + 1)
pole[:,TIME_COL] = indices / pole_freq

# Labelled steps
boot_steps = data_boot[data_boot[SIDE] == 'L']
pole_steps = data_pole[(data_pole[SIDE] == 'L') & (data_pole[TEST] == 'pole')]

f, (plot_boot, plot_pole) = pyplot.subplots(2, 1)

# Boot plot
plot_boot.plot(boot[:,0], boot[:,1], label = "x-axis acceleration")
#plot_boot.plot(boot[:,0], boot[:,2], label = "y-axis acceleration")
#plot_boot.plot(boot[:,0], boot[:,3], label = "z-axis acceleration")
for x in boot_steps[BOOT_UP]:
    plot_boot.axvline(x=x, linestyle='-', color='red')
for x in boot_steps[GLIDE_START]:
    plot_boot.axvline(x=x, linestyle='-', color='green')
plot_boot.set_title('Boot')
plot_boot.set(xlabel='Elapsed Time (s)', ylabel='Acceleration (g)')
plot_boot.legend()

# Pole plot
plot_pole.plot(pole[:,0], pole[:,1], label = "x-axis acceleration")
#plot_pole.plot(pole[:,0], pole[:,2], label = "y-axis acceleration")
#plot_pole.plot(pole[:,0], pole[:,3], label = "z-axis acceleration")
for x in pole_steps[TIME]:
    plot_pole.axvline(x=x, linestyle='-', color='red')
for x in pole_steps[END]:
    plot_pole.axvline(x=x, linestyle='-', color='green')
plot_pole.set_title('Pole')
plot_pole.set(xlabel='Elapsed Time (s)', ylabel='Acceleration (g)')
plot_pole.legend()

pyplot.show()