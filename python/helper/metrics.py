from __future__ import division
import numpy as np
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# scoring metric functions
def rmse(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def corr(predictions, targets):
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]


def _filt(x, y, value=0):
    mat = np.array(zip(x, y))
    mat = mat[~np.all(mat==value, axis=1)]
    return mat[:, 0], mat[:, 1]


def _filtnan(x, y):
    mat = np.array(zip(x, y))
    mat = mat[~np.any(np.isnan(mat), axis=1)]
    return mat[:, 0], mat[:, 1]


def corr_nonzero(predictions, targets):
    predictions, targets = _filt(predictions, targets, value=0)
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]


def corr_nan(predictions, targets):
    predictions, targets = _filtnan(predictions, targets)
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]
