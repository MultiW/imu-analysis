import numpy as np

from src.data.imu_util import (
    load_imu_data, Sensor, list_imu_abspaths, clean_imu_data
)
from src.data.data import DataState
from src.config import CLEAN_DIR, CLEAN_SUFFIX


def main():
    sensor_files = list_imu_abspaths(sensor_type=Sensor.Accelerometer, data_state=DataState.Raw)

    for file in sensor_files:
        print("Cleaning file '%s'..." % file.name)
        
        raw_imu = load_imu_data(file)
        clean_imu = clean_imu_data(raw_imu)

        np.save(CLEAN_DIR / ("%s%s" % (file.stem, CLEAN_SUFFIX)), clean_imu)
    
    print('Done')


if __name__ == '__main__':
    main()