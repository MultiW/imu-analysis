from src.data.labels_util import (
    load_clean_labels, get_workouts, LabelCol, find_neighboring_valid_steps, is_step_valid, get_workout_data_range
)
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
from typing import List, Tuple, Optional
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
    full_labels: ndarray = np.zeros((full_features.shape[0]))

    num_steps: int = workout.labels.shape[0]

    for i in range(num_steps):
        start_row: Optional[int] = workout.labels[i, LabelCol.START]
        end_row: Optional[int] = workout.labels[i, LabelCol.END]

        if is_step_valid(i, workout):
            # Label steps
            full_labels[int(start_row):int(end_row)+1] = 1
        else:
            # Label erroneous steps
            prev_step, next_step = find_neighboring_valid_steps(i, workout)
            delete_start: int = 0
            delete_end: int = full_features.shape[0]-1
            if prev_step is not None:
                delete_start = workout.labels[prev_step, LabelCol.END]
            if next_step is not None:
                delete_end = workout.labels[next_step, LabelCol.START]
            
            full_labels[delete_start:delete_end+1] = -1

    return full_labels


def build_workout_dataset(workout: Workout) -> Tuple[ndarray, ndarray]:
    # Get full IMU data for workout
    full_imu_data: ndarray = np.load(get_sensor_file(sensor_name=workout.sensor, sensor_type=Sensor.Accelerometer, data_state=DataState.Clean))
    
    # Get range in IMU data relevant to the workout
    start_row, end_row = get_workout_data_range(workout)

    # Build input dataset for model
    full_features: ndarray = data_to_features(full_imu_data, start_row, end_row)

    # Build labels
    full_labels: ndarray = label_data(full_features, workout)

    # Crop and remove bad data
    features: ndarray = full_features[start_row:end_row+1,]
    labels: ndarray = full_labels[start_row:end_row+1]

    to_keep: ndarray = labels != -1
    features = features[to_keep,]
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
    train_labels: ndarray = np.zeros((0))
    for i in range(len(features_list)):
        if i in test_set_idx:
            continue
        train_features = np.vstack((train_features, features_list[i]))
        train_labels = np.concatenate((train_labels, labels_list[i]))
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
        workout: Workout = workouts[i]
        print('Generating features and labels for workout in sensor %s' % workout.sensor)

        features, labels, no_errors = build_workout_dataset(workout)
        features_list.append(features)
        labels_list.append(labels)
        if no_errors:
            error_free_workouts.append(i)

    # Generate and save train and test data
    generate_train_test(features_list, labels_list, error_free_workouts, activity)


def main():
    pole_labels: ndarray = load_clean_labels(Activity.Pole)
    boot_labels: ndarray = load_clean_labels(Activity.Boot)

    print('Pole')
    build_features(pole_labels, Activity.Pole)
    print('Boot')
    build_features(boot_labels, Activity.Boot)


if __name__ == '__main__':
    main()