import numpy as np

from src.data.imu_util import (
    load_imu_data, Sensor, list_imu_abspaths, clean_imu_data
)
from src.data.labels_util import get_labels_file, load_labels
from src.data.enums import DataState, Activity
from src.config import CLEAN_DIR, CLEAN_SUFFIX

from pathlib import Path
from numpy import ndarray


def make_labels(activity: Activity):
    raw_labels_file: Path = get_labels_file(activity, DataState.Raw)
    raw_labels: ndarray = load_labels(raw_labels_file, activity)

    # TODO: load all workouts. Create Workout class to store info on each workout


def main():
    make_labels(Activity.Pole)
    make_labels(Activity.Boot)


if __name__ == '__main__':
    main()