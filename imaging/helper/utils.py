from math import sqrt
import sys
import numpy as np
from sklearn.utils import shuffle


def euclid_dist(loc1, loc2):
    ''' Euclidean distance between two points given as tuple/array.
    '''
    # return sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    return np.linalg.norm(loc1 - loc2)


def print_progress(n):
    ''' printing that overwrites the previous print. Used for progress indicator printing.
    '''
    sys.stdout.write(str(n))
    sys.stdout.write('\r')
    sys.stdout.flush()


def unique_value_shuffle(arr):
    ''' generates a permutation of the unique values of the array mapped to the original
    array, rather than a simple permutation of the elements.

        @Params:
            arr: 1D array
    '''
    vals = np.unique(arr)
    val_perm = shuffle(vals)
    id_convert = dict(zip(vals, val_perm))
    return np.array([id_convert[x] for x in list(arr)])


def rgb2gray(rgb):
    return np.dot(rgb[...,:], [0.299, 0.587, 0.114])


def nearest_rgb_color(color):
    ref = {'red': [133, 22, 37],
           'green': [28, 111, 50],
           'blue': [74, 77, 87]
          }

    dists = [utils.euclid_dist(np.array(ref[r]), np.array(color)) for r in ref]
    return ref.keys()[dists.index(min(dists))]


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return np.exp(x) / (1 + np.exp(x))


def zero_one_scale(x, n, prior=0.5):
    ''' transform formulated by Smithson and Verkuilen 2006 '''

    return (x * (n - 1) + prior) / n
