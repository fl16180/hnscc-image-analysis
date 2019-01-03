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
    ''' Wrapper for sklearn cross-validation used when additional functionality
    is needed. For example, it allows regression/classification on transformed y
    variable while reporting metrics over the untransformed space.
    '''
    def __init__(self, task='classify', transform=None, replications=None):
        self.task = task
        self.transform = transform
        self.coefs = None

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
        coefs = np.array((folds, X.shape[1]))

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

    def TestRegressor(self, estimator, X, y, folds=5, rf_oob=True, get_coef=True):
        cv = KFold(n_splits=folds, shuffle=True)

        internal_score = []
        rmse = []
        mae = []
        corr = []
        coefs = np.zeros((folds, X.shape[1]))

        y_truth = np.copy(y)
        if self.transform is not None:
            tr = VectorTransform(y)
            y = tr.zero_one_scale().apply(self.transform)

        iter = 0
        for train, test in cv.split(X, y):

            preds = estimator.fit(X[train], y[train]).predict(X[test])

            if get_coef:
                coefs[iter, :] = estimator.coef_

            if self.transform is not None:
                preds = tr.undo(preds)

            rmse.append(metrics.rmse(preds, y_truth[test]))
            mae.append(metrics.mae(preds, y_truth[test]))
            corr.append(metrics.corr(preds, y_truth[test]))

            if rf_oob:
                internal_score.append(estimator.oob_score_)

            iter += 1

        print "Scores: mean, stdev"
        print "rmse: ", np.mean(rmse), np.std(rmse)
        print "mae: ", np.mean(mae), np.std(mae)
        print "corr: ", np.mean(corr), np.std(corr)

        if rf_oob:
            print "internal score: ", np.mean(internal_score), np.std(internal_score)

        self.coefs = np.mean(coefs, axis=0)


    def stability_selection():
        return


class VectorTransform(object):
    def __init__(self, vector):
        self.vector = vector

        self.scaled = False
        self.transform = None
        self.prior = None
        self.n = None

    def zero_one_scale(self, n='default', prior=0.5):
        ''' transform formulated by Smithson and Verkuilen 2006 '''
        if self.scaled is True:
            return self
        self.scaled = True
        self.prior = prior

        if n == 'default':
            n = len(self.vector)
        self.n = n

        self.vector = (self.vector * (n - 1) + prior) / n
        return self

    def _unscale(self, arr):
        n = self.n
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


def mean_decrease_accuracy_importance(X, Y, names):
    ''' Implementation credit to
        https://blog.datadive.net/selecting-good-features-part-iii-random-forests/
    '''
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.metrics import r2_score
    from collections import defaultdict

    rf = RandomForestRegressor(n_estimators=500)
    scores = defaultdict(list)

    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 10, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print "Features sorted by their score:"
    print sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True)
    return scores
