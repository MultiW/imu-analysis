import numpy as np
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.lines import Line2D
from xgboost import XGBClassifier
import xgboost as xgb

from src.data.features_util import list_test_files
from src.data.workout import Activity
from src.config import (
    TRAIN_BOOT_DIR, TRAIN_POLE_DIR, TRAIN_FEATURES_FILENAME, TRAIN_LABELS_FILENAME
)

from numpy import ndarray
from pathlib import Path


def create_model() -> any:
    return GradientBoostingClassifier(verbose=True)


def create_xgboost() -> any:
    return XGBClassifier(verbosity=2, use_label_encoder=False)


def evaluate_model_accuracy(features: ndarray, labels: ndarray, model: any = create_model()):
    """
    Evaluate model accuracy using k-fold cross-validation
    """
    print('Evaluating model accuracy...')
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=1)
    n_scores = cross_val_score(model, features, labels, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


def train_model(activity: Activity, model: any = create_model):
    data_dir: Path = TRAIN_BOOT_DIR if activity == Activity.Boot else TRAIN_POLE_DIR

    features: ndarray = np.load(data_dir / TRAIN_FEATURES_FILENAME)
    labels: ndarray = np.load(data_dir / TRAIN_LABELS_FILENAME)
      
    print('Fitting model...')
    model.fit(features, labels)
    print('Done')

    if activity == Activity.Boot:
        joblib.dump(model, 'gbm-boot-model.pkl')
    else:
        joblib.dump(model, 'gbm-pole-model.pkl')


    return model
