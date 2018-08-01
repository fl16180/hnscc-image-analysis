from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

from scipy.special import logit
from scipy.special import expit

import helper.metrics as metrics


class TestLearner(object):
    def __init__(self, task='classify', transform=None, replications=None):
        self.task = task
        self.transform = transform

    def test(self, *args, **kwargs):
        if self.task == 'classify':
            self.TestClassifier(*args, **kwargs)
        elif self.task == 'regress':
            self.TestRegressor(*args, **kwargs)

    def ROC_curve(self):
        if self.task == 'regress':
            raise Exception("ROC_curve must be called for a classifier.")


    def FitClassifier():
        return

    def TestClassifier(self, estimator, X, y, folds=5, n_classes=2, rf_oob=True):
        cv = KFold(n_splits=folds, shuffle=True)

        internal_score = []
        acc = []
        c1_acc = []
        c2_acc = []
        roc = []

        y_truth = np.copy(y)
        if self.transform is not None:
            tr = VectorTransform(y)
            y = tr.zero_one_scale().apply(self.transform)

        for train, test in cv.split(X, y):

            preds = estimator.fit(X[train], y[train]).predict(X[test])

            acc.append(sum(preds == y[test]) / len(y[test]))
            c1_acc.append(sum(preds[y[test] == 0] == 0) / len(preds[y[test] == 0]))
            c2_acc.append(sum(preds[y[test] == 1] == 1) / len(preds[y[test] == 1]))
            if rf_oob:
                internal_score.append(estimator.oob_score_)

            y_pred = estimator.predict_proba(X[test])[:, 1]
            # fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
            roc.append(roc_auc_score(y[test], y_pred))

        print "Scores: mean, stdev"
        print "area under ROC: ", np.mean(roc), np.std(roc)
        print "accuracy: ", np.mean(acc), np.std(acc)
        print "class 0 accuracy: ", np.mean(c1_acc), np.std(c1_acc)
        print "class 1 accuracy: ", np.mean(c2_acc), np.std(c2_acc)
        if rf_oob:
            print "internal score: ", np.mean(internal_score), np.std(internal_score)

    def TestRegressor(self, estimator, X, y, folds=5, rf_oob=True):
        cv = KFold(n_splits=folds, shuffle=True)

        internal_score = []
        rmse = []
        corr = []

        y_truth = np.copy(y)
        if self.transform is not None:
            tr = VectorTransform(y)
            y = tr.zero_one_scale().apply(self.transform)

        for train, test in cv.split(X, y):

            preds = estimator.fit(X[train], y[train]).predict(X[test])

            if self.transform is not None:
                preds = tr.undo(preds)

            rmse.append(metrics.rmse(preds, y_truth[test]))
            corr.append(metrics.corr(preds, y_truth[test]))

            if rf_oob:
                internal_score.append(estimator.oob_score_)

            # y_pred = estimator.predict_proba(X2[test])[:, 1]
            # # fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
            # roc.append(roc_auc_score(y[test], y_pred))

        print "Scores: mean, stdev"
        print "rmse: ", np.mean(rmse), np.std(rmse)
        print "corr: ", np.mean(corr), np.std(corr)

        if rf_oob:
            print "internal score: ", np.mean(internal_score), np.std(internal_score)


    def stability_selection():
        return


class VectorTransform(object):
    def __init__(self, vector):
        self.vector = vector

        self.scaled = False
        self.transform = None
        self.prior = None

    def zero_one_scale(self, prior=0.5):
        ''' transform formulated by Smithson and Verkuilen 2006 '''
        if self.scaled is True:
            return self
        self.scaled = True
        self.prior = prior
        n = len(self.vector)

        self.vector = (self.vector * (n - 1) + prior) / n
        return self

    def _unscale(self, arr):
        n = len(self.vector)
        return (arr * n - self.prior) / (n - 1)

    def apply(self, transform):
        if self.transform is not None:
            return self.vector
        self.transform = transform

        if transform == 'logit':
            self.vector = logit(self.vector)
            return self.vector

    def undo(self, preds):
        if self.transform == 'logit':
            preds = expit(preds)

        if self.scaled is True:
            return self._unscale(preds)
        return preds
