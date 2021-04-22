from src.data.labels_util import load_clean_labels, get_workouts, LabelCol
from src.data.imu_util import get_sensor_file, Sensor, data_to_features
from src.data.workout import Activity, Workout
from src.data.data import DataState
from src.config import (
    TEST_BOOT_DIR, TEST_POLE_DIR, TRAIN_BOOT_DIR, TRAIN_POLE_DIR, TEST_FEATURES_SUFFIX, TEST_LABELS_SUFFIX, 
    TRAIN_LABELS_FILENAME, TRAIN_FEATURES_FILENAME
)

import numpy as np
import random

from numpy import ndarray
from typing import List, Tuple
from pathlib import Path

# number of workouts to use as tests
TEST_COUNT = 2


def label_data(full_features: ndarray, workout: Workout) -> ndarray:
    """
    Create array corresponding to each datapoint (row) of the IMU data
    Labels include:
    1 - datapoint is within a step
    0 - datapoint isn't a step
    -1 - bad data. To be removed
    """
    full_labels: ndarray = np.zeros((full_features.shape[0], 0))

    num_steps: int = workout.labels.shape[0]
    for i in range(num_steps):
        # Get step start/end of step
        start_row: float = workout.labels[i, LabelCol.START].astype(np.float64)
        end_row: float = workout.labels[i, LabelCol.END].astype(np.float64)

        # Mark erroneous steps labels for deletion
        if np.isnan(start_row) or np.isnan(end_row):
            # end of prev step
            prev_step = workout.labels[max(i-1, 0), LabelCol.END]
            # start of next step
            next_step = workout.labels[min(i+1, num_steps-1), LabelCol.START]

            full_labels[prev_step+1:next_step] = -1

        # label all datapoints inside a step as 1
        full_labels[start_row:end_row+1] = 1

    return full_labels


def build_workout_dataset(workout: Workout) -> Tuple[ndarray, ndarray]:
    # Get full IMU data for workout
    full_imu_data: ndarray = np.load(get_sensor_file(sensor_name=workout.sensor, sensor_type=Sensor.Accelerometer, data_state=DataState.Clean))
    
    # Get range in IMU data relevant to the workout
    start_row: int = workout.labels[0, LabelCol.START]
    end_row: int = workout.labels[-1, LabelCol.END]

    # Build input dataset for model
    full_features: ndarray = data_to_features(full_imu_data, start_row, end_row)

    # Build labels
    full_labels: ndarray = label_data(full_features, workout)

    # Crop and remove bad data
    features: ndarray = full_features[start_row:end_row+1,]
    labels: ndarray = full_labels[start_row:end_row+1]

    to_keep: ndarray = full_labels != -1
    features = features[to_keep]
    labels = labels[to_keep]

    no_errors: bool = to_keep.all()
    return features, labels, no_errors


def generate_train_test(features_list: List[ndarray], labels_list: List[ndarray], error_free_workouts: List[int], activity: Activity):
    print('Workouts with clean data: %d' % len(error_free_workouts))
    if TEST_COUNT > len(error_free_workouts):
        raise Exception('Not enough clean data for a sample of %d' % TEST_COUNT)

    # Test set
    test_set_idx: List[int] = random.sample(error_free_workouts, TEST_COUNT)
    test_dir: Path = TEST_BOOT_DIR if activity == Activity.Boot else TEST_POLE_DIR
    for i in range(len(test_set_idx)):
        features: ndarray = features_list[test_set_idx[i]]
        labels: ndarray = labels_list[test_set_idx[i]]
        np.save(test_dir / ('%d%s' % (i, TEST_FEATURES_SUFFIX)), features)
        np.save(test_dir / ('%d%s' % (i, TEST_LABELS_SUFFIX)), labels)

    # Train set
    train_features: ndarray = np.zeros((0, features_list[0].shape[1]))
    train_labels: ndarray = np.zeros((0, 1))
    for i in range(len(features_list)):
        if i in test_set_idx:
            continue
        train_features = np.vstack((train_features, features_list[i]))
        train_labels = np.vstack((train_labels, labels_list[i]))
    train_dir: Path = TRAIN_BOOT_DIR if activity == Activity.Boot else TRAIN_POLE_DIR
    np.save(train_dir / TRAIN_FEATURES_FILENAME, train_features)
    np.save(train_dir / TRAIN_LABELS_FILENAME, train_labels)


def build_features(labels: ndarray, activity: Activity):
    features_list: List[ndarray] = []
    labels_list: List[ndarray] = []
    error_free_workouts: List[int] = []

    # Generate features and labels for each workout
    workouts: List[Workout] = get_workouts(labels)
    for i in range(len(workouts)):
        features, labels, no_errors = build_workout_dataset(workouts[i])
        features_list.append(features)
        labels_list.append(labels)
        if no_errors:
            error_free_workouts.append(i)

    # Generate and save train and test data
    generate_train_test(features_list, labels_list, error_free_workouts, activity)


def main():
    boot_labels: ndarray = load_clean_labels(Activity.Boot)
    pole_labels: ndarray = load_clean_labels(Activity.Pole)

    build_features(boot_labels, Activity.Boot)
    build_features(pole_labels, Activity.Pole)


if __name__ == '__main__':
    main()