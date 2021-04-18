import pathlib

# Model project root
MODEL_DIR: str = pathlib.Path(__file__).parents[0].parents[0].absolute()

DATA_DIR = MODEL_DIR / 'data'

# Raw data paths
RAW_DIR = DATA_DIR / 'raw'
RAW_POLE_FILE = RAW_DIR / 'boot3MT_20210201.csv'
RAW_BOOT_FILE = RAW_DIR / 'pole3MT_20210201.csv'

# Clean data paths
CLEAN_DIR = DATA_DIR / 'clean'
CLEAN_SUFFIX = '_clean'

# Model Paths
ML_MODELS_DIR = MODEL_DIR / 'models'
BOOT_MODEL_FILE = ML_MODELS_DIR / 'gbm-boot-model-v2.pkl'
POLE_MODEL_FILE = ML_MODELS_DIR / 'gbm-pole-model-v2.pkl'


# Maximum time interval range that we expected from the IMU sensor. Everything else is considered an outlier
MAX_SAMPLING_INTERVAL_RANGE = 3 # in milliseconds