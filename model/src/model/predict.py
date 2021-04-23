# import data types
from numpy import ndarray
from typing import List, Tuple, Dict
from pathlib import Path

from src.data.imu_util import ImuCol, get_data_chunk, normalize_with_bounds, data_to_features
from src.data.workout import Activity
from src.config import BOOT_MODEL_FILE, POLE_MODEL_FILE
from src.visualization.visualize import multiplot
from src.data.features_util import list_test_files

import joblib
import copy
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report



PADDING = 50

GROUPING_SIZE = 8 # this should depend on the sampling interval. This value should equal 5*0.02 = 0.1 seconds
STEP_MIN = 10 # computed from minimum of labeled data set
    

def load_model(activity):
    if activity == Activity.Boot:
        model_file = BOOT_MODEL_FILE
    else:
        model_file = POLE_MODEL_FILE
    
    return joblib.load(model_file)


def group_points(classification: ndarray) -> List[Tuple[int, int]]:
    """
    Merge all neighboring points (labeled 1) into the same group.
    Return a list of these groups, defined by their start and end row indexes.
    """
    all_steps: List[Tuple[int, int]] = []

    # Group points that are consecutive
    step_start, step_end = None, None # current step
    for i in range(len(classification)):
        if classification[i] == 0:
            # End of current step
            if step_start is not None:
                # group consecutive data points as one step
                step_end = i-1
                step_size = step_end - step_start + 1
                if step_size >= GROUPING_SIZE:
                    all_steps.append((step_start, step_end))
            # Reset. Til next step
            step_start, step_end = None, None
        else:
            # Start of new step
            if step_start is None:
                step_start = i
    return all_steps


def merge_groups(all_steps: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    For groups of consecutive points that are "close enough", merge them into one group and classify it as a "step"
    """
    final_steps = []

    if len(all_steps) == 0:
        return final_steps

    merge_start, merge_end = all_steps[0]
    for i in range(1, len(all_steps)):
        curr_start, curr_end = all_steps[i]

        # merge
        if curr_start - merge_end + 1 <= GROUPING_SIZE:
            merge_end = curr_end
        # new step
        else:
            if merge_end - merge_start + 1 >= STEP_MIN:
                # add steps of at least STEP_MIN size
                final_steps.append((merge_start, merge_end))
            merge_start, merge_end = curr_start, curr_end
    # last step
    final_steps.append((merge_start, merge_end))

    return final_steps


#def classify_imu_datapoints(imu_data: ndarray, start_row: int, end_row: int, activity: Activity) -> ndarray:
#    """
#    @param imu_data: assume that data is pre-processed
#    """
#    # Don't mutate input
#    imu_data = copy.deepcopy(imu_data)
#
#    # Normalize
#    imu_data[:, ImuCol.XACCEL] = normalize_with_bounds(imu_data[:, ImuCol.XACCEL], start_row, end_row)
#    imu_data[:, ImuCol.YACCEL] = normalize_with_bounds(imu_data[:, ImuCol.YACCEL], start_row, end_row)
#    imu_data[:, ImuCol.ZACCEL] = normalize_with_bounds(imu_data[:, ImuCol.ZACCEL], start_row, end_row)
#
#    return classify_imu_datapoints_no_norm(imu_data, start_row, end_row, activity)
#
#
#def classify_imu_datapoints_no_norm(imu_data: ndarray, start_row: int, end_row: int, activity: Activity) -> ndarray:
#    """
#    @param imu_data: assume imu_data is already normalized
#    """
#    # Convert data to features
#    features = data_to_features(imu_data, start_row, end_row)
#    
#    # Classify
#    model = load_model(activity)
#    return model.predict(features[start_row:end_row+1,])
    


#def classify_imu_datapoints(clean_imu_data: ndarray, start_row: int, end_row: int, activity: Activity) -> ndarray:
#    """
#    @param imu_data: already pre-processed data. Not normalized
#    """
#    # Get workout data + padding
#    imu_data, (data_start, data_end) = get_data_chunk(clean_imu_data, start_row, end_row, PADDING)
#
#    # Normalize
#    imu_data[:, ImuCol.XACCEL] = normalize_with_bounds(imu_data[:, ImuCol.XACCEL], data_start, data_end)
#    imu_data[:, ImuCol.YACCEL] = normalize_with_bounds(imu_data[:, ImuCol.YACCEL], data_start, data_end)
#    imu_data[:, ImuCol.ZACCEL] = normalize_with_bounds(imu_data[:, ImuCol.ZACCEL], data_start, data_end)
#
#    # Load GBM model
#    model = load_model(activity)
#
#    # Convert data to features
#    features = data_to_features(imu_data)
#
#    # Remove padding data
#    imu_data = imu_data[PADDING:imu_data.shape[0]-PADDING,]
#    imu_data[:, ImuCol.TIME] -= imu_data[:, ImuCol.TIME].min() # shift timestamps. First data has time 0
#    features = features[PADDING:features.shape[0]-PADDING,]
#
#    # Classify data points
#    return model.predict(features)
   


#def label_steps(imu_data: ndarray, start_row: int, end_row: int, activity: Activity) -> List[Tuple[int, int]]:
#    """
#    @param imu_data: already pre-processed data. Not normalized
#    @return: returns steps data in the form of a tuple. Tuple contains start/end rows, with 
#        row numbers starting from the inputted start_row.
#    """
#    classification: ndarray = classify_imu_datapoints(imu_data, start_row, end_row, activity)
#
#    # Find start/end points
#    result: List[Tuple[int, int]] = group_points(classification)
#    result: List[Tuple[int, int]] = merge_groups(result)
#
#    return result


def evaluate_on_test_data(features: ndarray, labels: ndarray, activity: Activity) -> Tuple[float, ndarray]:
    model: any = load_model(activity)
    prediction: ndarray = model.predict(features)
    print('Accuracy: %f' % model.score(features, labels))
    print(classification_report(labels, prediction, target_names=['Non-steps', 'Steps']))
    return prediction


def evaluate_on_test_data_plot(activity: Activity, plot_results: bool, test_idx=None):
    test_data: List[Tuple[Path, Path]] = list_test_files(activity)

    def plot_helper(idx, plot):
        if test_idx is not None and test_idx != idx:
            return

        features_file, labels_file = test_data[idx]
        features, labels = np.load(features_file), np.load(labels_file)

        # Predict
        prediction: ndarray = evaluate_on_test_data(features, labels, activity)

        # Plot x-acceleration
        plot.plot(features[:,0])
        plot.plot(features[:,1])
        plot.plot(features[:,2])

        if not plot_results:
            return
        
        # Plot prediction
        for i in range(prediction.shape[0]):
            if prediction[i] == 1:
                plot.axvline(x=i, color='green', linestyle='dotted')
        
        # Plot actual
        for i in range(labels.shape[0]):
            if labels[i] == 1:
                plot.axvline(x=i, color='red', linestyle='dotted')
                
        # Legend
        legend_items = [Line2D([], [], color='green', linestyle='dotted', label='Prediction'), 
                    Line2D([], [], color='red', linestyle='dotted', label='Actual')]
        plot.legend(handles=legend_items)
                
    multiplot(len(test_data), plot_helper)