import numpy as np
import math

# import data types
from numpy import ndarray
from typing import List, Tuple, Dict

from predict.predict import classify_imu_datapoints, classify_imu_datapoints_no_norm
from data_processing.data_util import are_overlapping
from data_processing.enums import Activity
from data_processing.imu_util import normalize_imu_with_bounds

# number of data points in three minutes
THREE_MIN = 12000
PERCENT_POLE_STEPS = (0.55, 0.7)
PERCENT_BOOT_STEPS = (0.4, 0.55)

WINDOW_SIZE = THREE_MIN

CLASSIFIERS = 1


class Window():
    def __init__(self, start_row, end_row, score):
        # all ranges are inclusive
        self.start_row = start_row
        self.end_row = end_row
        self.score = score

    def __repr__(self):
        return '(%d, %d) - %f' % (self.start_row, self.end_row, self.score)

    def __eq__(self, other):
        """If same window, not necessarily the same score"""
        return self.start_row == other.start_row and self.end_row == other.end_row
    
    def __hash__(self):
        return hash(('start', self.start_row, 'end', self.end_row))


def get_percent_steps(start_row, end_row, classification: ndarray) -> float:
    window = classification[start_row:end_row+1]
    return np.count_nonzero(window == 1) / len(window)


def get_classifications(imu_data: ndarray, activity: Activity) -> List[ndarray]:
    classifications = []

    normalization_range = math.floor(imu_data.shape[0] / CLASSIFIERS)
    for i in range(CLASSIFIERS):
        normalized_imu_data = normalize_imu_with_bounds(imu_data, i * normalization_range, (i+1) * normalization_range - 1)
        classifications.append(
            classify_imu_datapoints_no_norm(normalized_imu_data, 0, imu_data.shape[0], activity)
        )
    return classifications


def get_local_percent_steps(start_row, end_row, classifications: List[ndarray]) -> float:
    best_score = None
    for classification in classifications:
        score = get_percent_steps(start_row, end_row, classification)
        if best_score is None or score > best_score:
            best_score = score
    return best_score


def get_activity_workouts(imu_data: ndarray, activity: Activity) -> List[Window]:
    """
    TODO: consider improving code quality with the Window class

    @return: list of workouts. Workouts are defined as start/end indexes to rows
    """
    # Classify each data point
    classifications: List[ndarray] = get_classifications(imu_data, activity)

    # Find all groups
    stride = 100

    if activity == Activity.Pole:
        grouping_threshold = PERCENT_POLE_STEPS
    else:
        grouping_threshold = PERCENT_BOOT_STEPS

    all_workouts: List[Window] = []
    #all_workouts: List[Tuple[int, int]] = []
    #all_workouts_percent: List[float] = []
    for i in range(0, imu_data.shape[0] - WINDOW_SIZE, stride):
        percent_steps = get_local_percent_steps(i, i+WINDOW_SIZE-1, classifications)
        if percent_steps > grouping_threshold[0] and percent_steps < grouping_threshold[1]:
            all_workouts.append(Window(i, i+WINDOW_SIZE-1, percent_steps))
#            all_workouts.append((i, i + WINDOW_SIZE - 1))
#            all_workouts_percent.append(percent_steps)

    if len(all_workouts) == 0:
        return []

    #print(all_workouts) # TODO REMOVE

    # Group overlapping workouts into one
    final_workouts: Window = []
    # among current neighbor of workouts that overlap, track the one that will get chosen
    track_workout: Window = all_workouts[0]
    # track_workout = all_workouts[0]
    # track_workout_percent = all_workouts_percent[0]
    for i in range(1, len(all_workouts)):
        curr_workout: Window = all_workouts[i]
        # curr_workout = all_workouts[i]
        # curr_workout_percent = all_workouts_percent[i]
        prev_workout: Window = all_workouts[i-1]
        
        if are_overlapping(
            (curr_workout.start_row, curr_workout.end_row), 
            (prev_workout.start_row, prev_workout.end_row)
            ):
            # workout overlaps with previous workout
            if curr_workout.score > track_workout.score:
                # curr workout chosen
                track_workout = curr_workout
                # track_workout = curr_workout
                # track_workout_percent = curr_workout_percent
        else:
            # no overlap. New set of workouts
            final_workouts.append(track_workout)
            
            # track_workout = curr_workout
            # track_workout_percent = curr_workout_percent
            track_workout = curr_workout
    # last workout
    final_workouts.append(track_workout)

    return final_workouts


def get_all_workouts(imu_data: ndarray) -> List[Window]:
    boot_workouts: List[Window] = get_activity_workouts(imu_data, Activity.Boot)
    pole_workouts: List[Window] = get_activity_workouts(imu_data, Activity.Pole)

    all_workouts: List[Window] = []

    # Add boot workouts
    for boot in boot_workouts:
        add_boot = True
        # add only if no overlap with pole
        # or if boot "wins" in overlap comparison
        for pole in pole_workouts:
            boot_score = boot.score
            pole_score = pole.score
            if are_overlapping((boot.start_row, boot.end_row), (pole.start_row, pole.end_row)) and boot_score < pole_score:
                # boot is not chosen over pole, so boot isn't added
                add_boot = False
        if add_boot:
            all_workouts.append(boot)

    # Add pole workouts
    for pole in pole_workouts:
        add_pole = True
        for boot in boot_workouts:
            boot_score = boot.score
            pole_score = pole.score
            if are_overlapping((boot.start_row, boot.end_row), (pole.start_row, pole.end_row)) and pole_score <= boot_score:
                add_pole = False
        if add_pole:
            all_workouts.append(pole)

    return all_workouts