import numpy as np
import copy

from src.data.imu_util import (
    load_imu_data, Sensor, get_sensor_file, clean_imu_data, fix_epoch, ImuCol
)
from src.data.labels_util import get_labels_file, load_labels, get_workouts, LabelCol
from src.data.data import DataState
from src.data.workout import Activity, Workout
from src.config import CLEAN_DIR, CLEAN_SUFFIX
from src.data.util import find_nearest_index

from pathlib import Path
from numpy import ndarray
from typing import List


def to_clean_tag(tag: int, raw_imu: ndarray, raw_imu_epoch_fixed: ndarray, clean_imu: ndarray) -> int:
    """
    Convert
    * from: tag pointing to the raw IMU data (tag is the epoch time)
    * to: tag pointing to the cleaned IMU data (row index to cleaned index)

    Return None if failed
    """
    # Row of raw IMU (that the given label points to)
    label_rows: List[int] = np.where(raw_imu[:,ImuCol.TIME].astype(int) == tag)[0]

    if len(label_rows) != 1:
        return None

    # Find time of tag
    label_row: int = label_rows[0]
    label_time: int = raw_imu_epoch_fixed[label_row, ImuCol.TIME]

    # Convert to row index
    return find_nearest_index(clean_imu[:, ImuCol.TIME], label_time)


def clean_workout_labels(workout: Workout) -> ndarray:
    raw_imu: ndarray = get_sensor_file(sensor_name=workout.sensor, sensor_type=Sensor.Accelerometer, data_state=DataState.Raw)
    raw_imu_epoch_fixed: ndarray = fix_epoch(raw_imu)
    clean_imu: ndarray = get_sensor_file(sensor_name=workout.sensor, sensor_type=Sensor.Accelerometer, data_state=DataState.Raw)

    # Output labels
    clean_labels: ndarray = copy.deepcopy(workout.labels)
    
    # Fix labels
    num_steps: int = workout.labels.shape[0]
    for i in range(num_steps):
        # Old tag. Points to raw IMU via epoch time
        start_tag: int = int(workout.labels[i, LabelCol.START])
        end_tag: int = int(workout.labels[i, LabelCol.END])

        # New tag. Row index to clean IMU 
        start_tag: int = to_clean_tag(start_tag, raw_imu, raw_imu_epoch_fixed, clean_imu)
        end_tag: int = to_clean_tag(end_tag, raw_imu, raw_imu_epoch_fixed, clean_imu)

        clean_labels[i, LabelCol.START] = start_tag
        clean_labels[i, LabelCol.START] = end_tag

    return clean_labels


def make_labels(activity: Activity):
    raw_labels_file: Path = get_labels_file(activity, DataState.Raw)
    raw_labels: ndarray = load_labels(raw_labels_file, activity)

    clean_labels: ndarray = np.zeros((0, raw_labels.shape[1]), dtype=np.float64)
    for workout in get_workouts(raw_labels):
        clean_labels = np.vstack((clean_labels, clean_workout_labels(workout)))

        clean_labels_file: str = get_labels_file(activity, data_state=DataState.Clean)
        np.save(clean_labels_file, clean_labels)


def main():
    make_labels(Activity.Pole)
    make_labels(Activity.Boot)


if __name__ == '__main__':
    main()