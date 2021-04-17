import pathlib

# Project root
MODEL_DIR: str = pathlib.Path(__file__).parents[0].parents[0].parents[0].absolute()

# Data Paths
DATA_DIR: str = MODEL_DIR / 'data'
IMU_DATA_DIR: str = DATA_DIR / 'data'
BOOT_LABEL_FILE = DATA_DIR / 'boot3MT_20210201.csv'
POLE_LABEL_FILE = DATA_DIR / 'pole3MT_20210201.csv'

# Model Paths
ML_MODELS_DIR = MODEL_DIR / 'models'
BOOT_MODEL_FILE = ML_MODELS_DIR / 'gbm-boot-model-v2.pkl'
POLE_MODEL_FILE = ML_MODELS_DIR / 'gbm-pole-model-v2.pkl'
