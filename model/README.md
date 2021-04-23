# Model

An IMU sensor is turned on and placed on an athlete. While wearing the sensor, an athlete may be performing a ski workout or be resting. Once the sensor is turned off, the data from that session is dumped into CSV files, which are the raw IMU data that we are processing and analyzing. This IMU data thus contains useful ski workout data and garbage data.

Our application performs two tasks:
1. Identify relevant ski workout data.

    This is necessary for the next step to happen. Since this can easily be done manually, automating this is not a priority.

    TODO show example of raw IMU file

2. Identify ski steps within the relevant ski workout data.

    Specifically, we are interested in labeling the start and end of every step. For sensors placed on boots, we are interested in labeling when the boot leaves the ground (step start) and when it lands (step end). For sensors placed on poles, we are interested in labeling when the pole hits the ground (step start) and when it leaves the ground (step end).

    TODO show example

## Project
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
Training data pre-processing:
* Model was trained on un-smoothed data

Testing data pre-processing:
* Model was tested on heavily smoothed (low-pass filter) data. Specifically, pole - 8Hz and boot - 3Hz.

Model changes:
* Use XGBoost instead of GBM

Boot results:
```
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
