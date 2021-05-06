# Model

## Data capturing
An IMU sensor is turned on and placed on an athlete. While wearing the sensor, an athlete will perform their workout, rest, etc. (meaning that the data may contain irrelevant/garbage data). Once the sensor is turned off, the data from that session is dumped into CSV files, and this is the "raw" IMU data that our steps labeling algorithm accepts. 

## Features of our algorithm

Our application performs two tasks:
1. Identify relevant ski workout data.

    This is necessary for the next step to happen. Since this can easily be done manually, it is not a priority and isn't currently implemented.

    TODO show example of raw IMU file

2. Identify ski steps within the relevant ski workout data.

    Specifically, we are interested in labeling the start and end of every step. For sensors placed on boots, we are interested in labeling when the boot leaves the ground (step start) and when it lands (step end). For sensors placed on poles, we are interested in labeling when the pole hits the ground (step start) and when it leaves the ground (step end).

    TODO show example

## Model performance highlights
The model's best performance is highlighted [here](https://github.com/MultiW/imu-analysis/tree/main/model#gbm-boot-model-v6pkl-gbm-pole-model-v6pkl).

Of note is the steps labeling accuracy. I ran the algorithm on two testing datasets. I then compared the algorithm's identified steps (start/end points) to the actual steps (labeled by data analysts). I noted how "off" the algorithm's predictions were, i.e. how many data points away was the prediction from the actual step.

## Using the Code
Make sure to setup the development environment following instructions [here](../).

To train the model, add the raw IMU data (straight from the sensor) and the labeled steps to ```data/raw``` and configure the paths to the label files in ```src/config.py```. 

Then follow the [notebooks](notebooks/) and uncomment and run the appropriate code to clean, build features, and train the model. Make sure to move the completed model to the appropriate model and edit the ```config.py``` file to use.

To predict with the model, follow the instructions in [4-steps-labeling.ipynb](notebooks/4-steps-labeling.ipynb). Or you can clean, build the features, load the model, and label the steps following the notebook and the code they refer to. TODO: create some python function or file to wrap the whole process to labeling the steps of a raw IMU dataset.

## Project Structure
The [notebooks](notebooks) show how the "steps labeling" model is built. The code in these notebooks can be run to build the model and analyze the models' performance.

TODO: script to clean data and build model

TODO: script to process data and predict model

## Training
The data we have are:
* Raw IMU data: time series data for acceleration, angular velocity, and magnetic force.
* Pole and boot step labels: point to raw IMU data to label start/end times of pole/boot steps.

### Data cleaning
We need to clean the raw IMU data and adjust the step labels.

**1. IMU data cleaning**
* Fix erroneous timestamps by interpolating from neighbors.
* Re-sample data uniformly. We want uniform sampling intervals.
* Apply a low-pass filter to smooth data.

**2. Adjusting labels**

Labels point to the raw IMU data's timestamps. We want to point these labels to the cleaned data. Note that the raw data's timestamps have been fixed/interpolated and re-sampled.

**3. Pre-processing and training**
Now we can pre-process the cleaned IMU data by normalizing the acceleration values and generating features. This will be used to train our model.

Note: we don't use angular velocity or magnetic force for now.

## Using the Model
From a raw IMU file, the user should identify the workout data. For each workout data chunk, we can label the ski steps.

To label the steps, our algorithm needs to perform the same **IMU data cleaning** step as done in training. Then we run the data through the model and generate the steps labels.

## Trained models
We store our models in ```model/models``` as pickle files. Here are the characteristics of each model version.

### ```gbm-boot-model-v1.pkl```, ```gbm-pole-model-v1.pkl```

These are very basic Gradient Boosting Models (GBM) trained using scikit-learn's default GBM parameters.

Data cleaning performed includes:
* Fix erroneous epoch values.
* Re-sample data points so that they are uniformly distributed.

Pre-processing performed on the raw IMU data includes:
* Normalize x, y, z acceleration values.

The features we used are:
* Acceleration in each of the x, y, z directions.
* Magnitude of the acceleration, i.e. the distance, L2 norm, etc.
* Each of the previous features lagged 5-10 data points, and each of the previous features lead 5-10 data points.

Data used:
* 11L and 11R sensor data. Two pole workouts and two boot workouts.

### ```gbm-boot-model-v2.pkl```, ```gbm-pole-model-v2.pkl```
Training data pre-processing:
* A low-pass filter with a cutoff of 10Hz was applied to all the x, y, z acceleration values

Data used:
* 11L and 11R sensor data. Two pole workouts and two boot workouts.

### ```gbm-boot-model-v3.pkl```, ```gbm-pole-model-v3.pkl```
Training data pre-processing:
* A low-pass filter with a cutoff of 10Hz was applied to all the x, y, z acceleration values

Data changes:
* Used all available data to train the GBM model. Number of datapoints: pole - 421,895, boot - 363,025

### ```gbm-boot-model-v4.pkl```, ```gbm-pole-model-v4.pkl```
> Note: The traing and testing pre-processing procedure may not be accurate. The testing data may have been tested on a un-smoothed dataset.
Training data pre-processing:
* Model was trained on un-smoothed data

Testing data pre-processing:
* Model was tested on heavily smoothed (low-pass filter) data. Specifically, pole - 8Hz and boot - 3Hz.

Model changes:
* Use XGBoost instead of GBM

> Below are results of the model. Note that these are model results, but not the final steps labeling results. The model labels whether a datapoint is a step or non-step, it doesn't label the start/end points of each step.

Boot results:
```
k-fold cross-validation result
Mean Accuracy: 0.956 (0.001)

Accuracy: 0.895374
              precision    recall  f1-score   support

   Non-steps       0.93      0.89      0.91      5198
       Steps       0.85      0.90      0.88      3729

    accuracy                           0.90      8927
   macro avg       0.89      0.90      0.89      8927
weighted avg       0.90      0.90      0.90      8927

Accuracy: 0.925490
              precision    recall  f1-score   support

   Non-steps       0.91      0.98      0.94      5532
       Steps       0.96      0.84      0.90      3393

    accuracy                           0.93      8925
   macro avg       0.93      0.91      0.92      8925
weighted avg       0.93      0.93      0.92      8925
```

Pole results:
```
Accuracy: 0.974434
              precision    recall  f1-score   support

   Non-steps       0.99      0.94      0.97      4908
       Steps       0.97      0.99      0.98      8078

    accuracy                           0.97     12986
   macro avg       0.98      0.97      0.97     12986
weighted avg       0.97      0.97      0.97     12986

Accuracy: 0.976395
              precision    recall  f1-score   support

   Non-steps       0.99      0.95      0.97      6147
       Steps       0.97      0.99      0.98      9019

    accuracy                           0.98     15166
   macro avg       0.98      0.97      0.98     15166
weighted avg       0.98      0.98      0.98     15166
```

### ```gbm-boot-model-v5.pkl```, ```gbm-pole-model-v5.pkl```
Training data pre-processing:
* Model was trained on heavily smoothed (low-pass filter) data. Specifically, pole - 8Hz and boot - 3Hz.

Testing data pre-processing:
* Model was tested on heavily smoothed (low-pass filter) data. Specifically, pole - 8Hz and boot - 3Hz.

Model:
* XGBoost

Boot results:
```
Accuracy: 0.874538
              precision    recall  f1-score   support

   Non-steps       0.93      0.85      0.89      5198
       Steps       0.82      0.91      0.86      3729

    accuracy                           0.87      8927
   macro avg       0.87      0.88      0.87      8927
weighted avg       0.88      0.87      0.88      8927

Accuracy: 0.878319
              precision    recall  f1-score   support

   Non-steps       0.86      0.95      0.91      5532
       Steps       0.91      0.75      0.82      3393

    accuracy                           0.88      8925
   macro avg       0.89      0.85      0.87      8925
weighted avg       0.88      0.88      0.88      8925
```

Pole results:
```
Accuracy: 0.979208
              precision    recall  f1-score   support

   Non-steps       0.98      0.97      0.97      4908
       Steps       0.98      0.99      0.98      8078

    accuracy                           0.98     12986
   macro avg       0.98      0.98      0.98     12986
weighted avg       0.98      0.98      0.98     12986

Accuracy: 0.980219
              precision    recall  f1-score   support

   Non-steps       0.99      0.97      0.98      6147
       Steps       0.98      0.99      0.98      9019

    accuracy                           0.98     15166
   macro avg       0.98      0.98      0.98     15166
weighted avg       0.98      0.98      0.98     15166
```

### ```gbm-boot-model-v6.pkl```, ```gbm-pole-model-v6.pkl```

New features:
* 1st and 2nd derivative of the triaxial acceleration features and the lag and lead values of these new features.

Pre-processing:
* Data was trained and tested on heavily smoothed (low-pass filter) data. Specifically, pole - 8Hz and boot - 3Hz.

Boot results:
```
========================
=== ML model results ===
========================
Accuracy: 0.970376
Confusion Matrix:
[[5320  147]
 [ 116 3295]]
Classification Report:
              precision    recall  f1-score   support

   Non-steps       0.98      0.97      0.98      5467
       Steps       0.96      0.97      0.96      3411

    accuracy                           0.97      8878
   macro avg       0.97      0.97      0.97      8878
weighted avg       0.97      0.97      0.97      8878

========================================
=== Steps labeling algorithm results ===
========================================
Test 0
Total steps: 129
Total steps predicted: 129
Accurate to within 0 datapoint:
- Start: 0.558140
- End: 0.596899
Accurate to within 1 datapoint:
- Start: 0.945736
- End: 0.922481
Accurate to within 2 datapoint:
- Start: 0.976744
- End: 0.945736
Accurate to within 3 datapoint:
- Start: 0.976744
- End: 0.945736
Accurate to within 4 datapoint:
- Start: 0.976744
- End: 0.945736

Test 1
Total steps: 127
Total steps predicted: 126
Accurate to within 0 datapoint:
- Start: 0.669291
- End: 0.645669
Accurate to within 1 datapoint:
- Start: 0.976378
- End: 0.913386
Accurate to within 2 datapoint:
- Start: 0.976378
- End: 0.913386
Accurate to within 3 datapoint:
- Start: 0.976378
- End: 0.913386
Accurate to within 4 datapoint:
- Start: 0.976378
- End: 0.913386
```

Pole results:
```
========================
=== ML model results ===
========================
Accuracy: 0.983794
Confusion Matrix:
[[5021  109]
 [ 104 7909]]
Classification Report:
              precision    recall  f1-score   support

   Non-steps       0.98      0.98      0.98      5130
       Steps       0.99      0.99      0.99      8013

    accuracy                           0.98     13143
   macro avg       0.98      0.98      0.98     13143
weighted avg       0.98      0.98      0.98     13143

========================================
=== Steps labeling algorithm results ===
========================================
Test boot model:
Figure 1
Test 0
Total steps: 284
Total steps predicted: 285
Accurate to within 0 datapoint:
- Start: 0.721831
- End: 0.644366
Accurate to within 1 datapoint:
- Start: 0.996479
- End: 0.985915
Accurate to within 2 datapoint:
- Start: 1.000000
- End: 0.992958
Accurate to within 3 datapoint:
- Start: 1.000000
- End: 0.992958
Accurate to within 4 datapoint:
- Start: 1.000000
- End: 0.992958

Test 1
Total steps: 292
Total steps predicted: 292
Accurate to within 0 datapoint:
- Start: 0.715753
- End: 0.664384
Accurate to within 1 datapoint:
- Start: 1.000000
- End: 0.989726
Accurate to within 2 datapoint:
- Start: 1.000000
- End: 1.000000
Accurate to within 3 datapoint:
- Start: 1.000000
- End: 1.000000
Accurate to within 4 datapoint:
- Start: 1.000000
- End: 1.000000
```
