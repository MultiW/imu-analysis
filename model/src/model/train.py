import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.lines import Line2D

from src.data.features_util import list_test_files
from src.data.workout import Activity

from numpy import ndarray


def create_model() -> any:
    return GradientBoostingClassifier(verbose=True)


def evaluate_model_accuracy(features: ndarray, labels: ndarray):
    """
    Evaluate model accuracy using k-fold cross-validation
    """
    print('Evaluating model accuracy...')
    model = create_model()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    n_scores = cross_val_score(model, features, labels, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


def train_model(features: ndarray, labels: ndarray):
    print('Fitting model...')
    model = create_model()
    model.fit(features, labels)
    print('Done')
    return model