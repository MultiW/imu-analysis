import pathlib
from pathlib import Path

# Model project root
MODEL_DIR: Path = pathlib.Path(__file__).parents[0].parents[0].absolute()

DATA_DIR: Path = MODEL_DIR / 'data'

# Raw data paths
RAW_DIR: Path = DATA_DIR / 'raw'  # place data files in this directory
RAW_BOOT_FILE: Path = RAW_DIR / 'skate3mt_noname.csv' # CONFIGURE THIS
RAW_POLE_FILE: Path = RAW_DIR / 'pole3mt_noname.csv' # CONFIGURE THIS

# Clean data paths
CLEAN_DIR: Path = DATA_DIR / 'clean'
CLEAN_DATA_SUFFIX: str = '_clean.npy'
CLEAN_LABELS_SUFFIX: str = '_clean.npy'

# Train data paths
TRAIN_BOOT_DIR: Path = DATA_DIR / 'train/boot'
TRAIN_POLE_DIR: Path = DATA_DIR / 'train/pole'
TRAIN_FEATURES_FILENAME: str = 'features.npy'
TRAIN_LABELS_FILENAME: str = 'labels.npy'

# Test data paths
TEST_BOOT_DIR: Path = DATA_DIR / 'test/boot'
TEST_POLE_DIR: Path = DATA_DIR / 'test/pole'
TEST_FEATURES_SUFFIX: str = '_features.npy'
TEST_LABELS_SUFFIX: str = '_labels.npy'

# Model Paths
ML_MODELS_DIR: Path = MODEL_DIR / 'models'
BOOT_MODEL_FILE: Path = ML_MODELS_DIR / 'gbm-boot-model-v4.pkl' # CONFIGURE THIS
POLE_MODEL_FILE: Path = ML_MODELS_DIR / 'gbm-pole-model-v4.pkl' # CONFIGURE THIS

# Maximum time interval range that we expected from the IMU sensor. Everything else is considered an outlier
MAX_SAMPLING_INTERVAL_RANGE: int = 3 # in milliseconds

# Low-pass filter cutoffs
BOOT_CUTOFF = 3
POLE_CUTOFF = 8