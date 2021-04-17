# Model
## Trained models
We store our models in ```model/models``` as pickle files. Here are the characteristics of each model version.

### ```gbm-boot-model-v1.pkl```, ```gbm-pole-model-v1.pkl```

These are very basic Gradient Boosting Models (GBM) trained using scipy's default GBM parameters.

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
