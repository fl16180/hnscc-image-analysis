from __future__ import division

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import helper


# get list of samples
all_samples = helper.get_list_of_samples()
# SAMPLES = [all_samples[i] for i in np.random.choice(len(all_samples), N_SLIDES, replace=False)]
slide = all_samples[180]

# load sample as cell matrix
cell_mat = np.load(slide)

print('n_tumor:', np.sum(cell_mat == 1))
print('n_pdl1:', np.sum(cell_mat == 2))

# select all tumor cells (tumor or pdl1)
any_tumor = np.logical_or((cell_mat == 1), (cell_mat == 2)).astype(int)

# subset to be >= 200 px from any edge
not_edge_mask = np.zeros((1040, 1392))
not_edge_mask[200:-200,200:-200] = 1
any_tumor = np.multiply(any_tumor, not_edge_mask)

# get locations of these tumor cells
locs = np.where(any_tumor == 1)

# initialize design matrix and response
n_obs = int(np.sum(any_tumor))
X_mat = np.zeros((n_obs, 50))
Y = np.zeros(n_obs)

# fill design matrix by iterating over any_tumor cells
for i, cell in enumerate(zip(*locs)):
    # find all neighboring cells within 400px box from tumor cell
    tmp_square = cell_mat[cell[0] - 200: cell[0] + 200, cell[1] - 200: cell[1] + 200]
    cell_locs = np.where(tmp_square != 0)

    # initialize feature row array
    n_cells = int(np.sum(tmp_square != 0))
    feature_array = np.zeros((n_cells, 2))

    # iterate over neighbors, storing distance and phenotype
    for j, neighbor in enumerate(zip(*cell_locs)):
        feature_array[j, 0] = (200 - neighbor[0]) ** 2 + (200 - neighbor[1]) ** 2
        feature_array[j, 1] =  tmp_square[neighbor[0], neighbor[1]]

    # extract 50 closest neighbors and store their phenotypes as a row in the design matrix
    closest_neighbors = feature_array[feature_array[:,0].argsort()][1:, :]
    feature_row = closest_neighbors[:50, 1].reshape(1, -1)
    X_mat[i, :] = feature_row


    # store whether the any_tumor cell is tumor or pdl1 in response variable
    Y[i] = cell_mat[cell] - 1


Y = Y.astype(int)
X_mat = X_mat.astype(int) - 1

X_mat

# treat pdl1 as same phenotype as tumor for prediction matrix
X_mat[X_mat==1]=0


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(n_values=7)
X = enc.fit_transform(X_mat).toarray()

X.shape

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)




from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', oob_score=True)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print "feature importances:", rf.feature_importances_

print "n_features:", X_train.shape
print "oob_score:", rf.oob_score_


print "proportion of 0:", sum(y_test==0) / len(y_test)
print "proportion of 1:", sum(y_test==1) / len(y_test)

print sum(rf_preds == y_test) / len(y_test)
#
print "class 0 accuracy: ", sum(rf_preds[y_test == 0] == 0) / len(rf_preds[y_test == 0])
print "class 1 accuracy: ", sum(rf_preds[y_test == 1] == 1) / len(rf_preds[y_test == 1])


sum(y_test == 0)
