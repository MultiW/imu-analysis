# Model

An IMU sensor is turned on and placed on an athlete. While wearing the sensor, an athlete may be performing a ski workout or be resting. Once the sensor is turned off, the data from that session is dumped into CSV files, which are the raw IMU data that we are processing and analyzing. This IMU data thus contains useful ski workout data and garbage data.

Our application performs two tasks:
1. Identify relevant ski workout data.

    This is necessary for the next step to happen. Since this can easily be done manually, automating this is not a priority.

    TODO show example of raw IMU file

2. Identify ski steps within the relevant ski workout data.

    Specifically, we are interested in labeling the start and end of every step. For sensors placed on boots, we are interested in labeling when the boot leaves the ground (step start) and when it lands (step end). For sensors placed on poles, we are interested in labeling when the pole hits the ground (step start) and when it leaves the ground (step end).

    TODO show example

## Training
The data we have are:
* Raw IMU data: time series data for acceleration, angular velocity, and magnetic force.
* Pole and boot step labels: point to raw IMU data to label start/end times of pole/boot steps.

### Data cleaning
We need to clean the raw IMU data and adjust the step labels.

**1. IMU data cleaning**
* Fix erroneous timestamps by interpolating from neighbors.
* Re-sample data uniformly. We want uniform sampling intervals.
* Normalize acceleration. (We don't use angular velocity or magnetic force for now.)
* Apply a low-pass filter to smooth data.

**2. Adjusting labels**

Labels point to the raw IMU data's timestamps. We want to point these labels to the cleaned data. Note that the raw data's timestamps have been fixed/interpolated and re-sampled.

**3. Feature selection and training**
Now we can pre-process the cleaned IMU data into features. This will be used to train our model.

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

### ```gbm-boot-model-v2.pkl```, ```gbm-pole-model-v2.pkl```

Pre-processing changes
* A low-pass filter with a cutoff of 10Hz was applied to all the x, y, z acceleration values
