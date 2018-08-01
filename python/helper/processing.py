''' Functions for processing and working with original slide png images and
processed matrices '''

from __future__ import division
import numpy as np
import pandas as pd
import glob
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap


def map_phenotypes_to_mat(cells):

    translation = {'tumor': 1, 'pdl1': 2, 'foxp3': 3, 'cd8': 4,
                   'cd4': 5, 'pdmac': 6, 'other': 7, 'macs': 8}
    mat = np.zeros((1040, 1392))

    for key, value in translation.iteritems():
        for _, row in cells[key].iterrows():
            x, y = get_position(row)
            mat[y, x] = value

    return mat


def get_position(cell):
    return (cell['Cell X Position'], cell['Cell Y Position'])


def get_list_of_samples(directory=None, pattern='*].npy'):

    if directory is None:
        directory = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/processed/'

    files = glob.glob(directory + pattern)

    return sorted(files)


def search_samples(x, filelist):
    matching = [s for s in filelist if x in s]
    if len(matching) > 1:
        print "Found {0} matches".format(len(matching))
    elif len(matching) == 0:
        print "No match."

    return matching


def get_original_image(loc):
    parts = loc.split("/processed")
    body = parts[1].split(".npy")[0]

    new = "".join([parts[0], body, ".jpg"])
    return new


def load_sample(loc, confidence_thresh=0.3, verbose=False, radius_lim=200):

    # load data file and select cells above confidence threshold
    dataset = pd.read_csv(loc, sep='\t')

    # exclude cells too close to borders
    dataset = dataset[(dataset['Cell X Position'] > radius_lim) &
                      (dataset['Cell X Position'] < 1392 - radius_lim) &
                      (dataset['Cell Y Position'] > radius_lim) &
                      (dataset['Cell Y Position'] < 1040 - radius_lim)]

    # exclude cells by confidence
    dataset['Confidence1'] = dataset.Confidence.replace('%','',regex=True).astype('float')/100
    dataset = dataset[dataset.Confidence1 > confidence_thresh]

    cells = {
            'unclassified': dataset[dataset['Phenotype'].isnull()],
            'foxp3': dataset[dataset['Phenotype'] == 'foxp3'],
            'pdl1': dataset[dataset['Phenotype'] == 'pd-l1'],
            'cd4': dataset[dataset['Phenotype'] == 'cd4'],
            'other': dataset[dataset['Phenotype'] == 'other'],
            'pdmac': dataset[dataset['Phenotype'] == 'pd-l1+ mac'],
            'cd8': dataset[dataset['Phenotype'] == 'cd8'],
            'macs': dataset[dataset['Phenotype'] == 'macs'],
            'tumor': dataset[dataset['Phenotype'] == 'tumor'],
            }

    # compile all tumor cells (phenotype marked pdl1 or tumor)
    all_tumor = pd.concat([cells['pdl1'], cells['tumor']])
    cells['all_tumor'] = all_tumor

    # compile all non-tumor cells (phenotypes besides pdl1 and tumor)
    non_tumor = pd.concat([cells['foxp3'], cells['pdl1'], cells['cd4'], cells['other'],
                          cells['pdmac'], cells['cd8']])
    cells['non_tumor'] = non_tumor

    if verbose is True:
        print '\nCell counts in the sample: '
        print 'unclassified: ', len(cells['unclassified'])
        print 'foxp3: ', len(cells['foxp3'])
        print 'pdl1: ', len(cells['pdl1'])
        print 'cd4: ', len(cells['cd4'])
        print 'other: ', len(cells['other'])
        print 'pdmac: ', len(cells['pdmac'])
        print 'cd8: ', len(cells['cd8'])
        print 'tumor: ', len(cells['tumor'])

    return cells


def compute_decision_boundary(cell_mat, view=False, n_neighbors=25, slide_id='xxx', remove_blank_regions=True, clean=True, recompute=False):
    ''' Classifies slide area into tumor and stromal areas by fitting nearest neighbor boundary.
    '''
    try:
        if (view is True) or (recompute is True):
            print "Regenerating decision boundary."
            raise IOError()

        DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/processed/'
        Z = np.load(slide_id.split(".npy")[0] + "_seg.npy")

        print "Precomputed decision boundary found."

    except IOError:

        all_tumor_loc = np.nonzero((cell_mat == 1) | (cell_mat == 2))
        x_tumor = np.vstack((all_tumor_loc[1], all_tumor_loc[0])).T

        nontumor_loc = np.nonzero((cell_mat > 2))
        x_nontumor = np.vstack((nontumor_loc[1], nontumor_loc[0])).T

        X = np.vstack((x_nontumor, x_tumor))
        y = np.array(x_nontumor.shape[0] * [0] + x_tumor.shape[0] * [1])

        h = 1  # step size in the mesh

        # fit K-nearest neighbor algorithm
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(X, y)

        # Create the decision boundary by predicting the class of each pixel in image
        xx, yy = np.meshgrid(np.arange(0, 1392, h),
                             np.arange(0, 1040, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)


        if remove_blank_regions is True:
            # get matrix of cell locations
            any_cells = (cell_mat != 0)

            # blur matrix using maximum filter and remove regions without cell presence
            filt = ndimage.maximum_filter(any_cells, size=100)
            blanks = (filt == 0)
            Z[blanks] = -1

        if clean is True:
            # clean image of small blobs using median filter
            Z = ndimage.median_filter(Z, size=30)

        np.save(slide_id.split(".npy")[0] + "_seg.npy", Z)

    if view is True:
        # Create color maps
        if np.sum(Z==-1) > 0:
            cmap_light = ListedColormap(['#AAFFAA', '#FFAAAA', '#AAAAFF'])
        else:
            cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # plot the training points (cells)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=15)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Tumor (blue)-nontumor (red) separation with k = %i"
                  % (n_neighbors))
        plt.xlabel(slide_id)

        plt.gca().invert_yaxis()
        plt.show()

    return Z
