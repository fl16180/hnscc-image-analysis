from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.special import expit
from sklearn.feature_selection import mutual_info_regression

import helper.processing
import helper.display as display
import helper.metrics as metrics
import helper.tile_utils as utils
import helper.learning as learning
from helper.utils import print_progress
reload(learning)


DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/processed_orig_seg/'
STORE_DIR = 'C:/Users/fredl/Documents/repos/hnscc-image-analysis/data/'


diams = [150, 200, 250, 300]
sample_diam = 150
flag = 'n'

# set sampling parameters
N_SLIDES = 250
N_SAMPLES = 20

# set feature extraction parameters
sample_tile_width = 150
feature_tile_width = 1
feature_layers = 75

# compute other parameters based on input parameters
scale = int(sample_tile_width / feature_tile_width)
assert (scale == sample_tile_width / feature_tile_width), "sample_tile_width must be integer multiple of feature_tile_width"
Nx, Ny = int(1392 / sample_tile_width), int(1040 / sample_tile_width)
nx, ny = Nx * scale, Ny * scale
sample_layers = int(np.ceil(feature_layers * feature_tile_width / sample_tile_width))


# get pre-processed slide matrices and select random sample of slides
all_samples = helper.processing.get_list_of_samples(DIR)
SAMPLES = [all_samples[i] for i in np.random.choice(len(all_samples), N_SLIDES, replace=False)]
HOLDOUT = list(set(all_samples) - set(SAMPLES))
len(SAMPLES)
len(HOLDOUT)


def automate_tile_extraction(SAMPLES):
    # iterate over sampled slides to extract feature and response variables via tile sampling
    combined_features = []
    combined_response = []
    combined_nts = []
    for i, slide in enumerate(SAMPLES):
        print_progress(i)

        # load slide and reshape into tile stacks
        cell_mat = np.load(slide)
        sample_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=sample_tile_width,
                                                   nx=Nx, ny=Ny)
        feature_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=feature_tile_width,
                                                    nx=nx, ny=ny)

        # load tumor edge matrix (skipping slide if no matrix is found)
        try:
            edges = np.load(slide.split(".npy")[0] + "_edges.npy")
            edges_tile_stack = utils.restack_to_tiles(edges, tile_width=sample_tile_width,
                                                      nx=Nx, ny=Ny)
        except IOError:
            print 'No edge matrix. Skipping slide...'
            continue

        # select valid tiles for sampling, skipping slide if no valid tiles are available
        tile_mask = utils.tile_stack_mask(Nx, Ny, L=sample_layers, db_stack=edges_tile_stack)
        if np.sum(tile_mask) == 0:
            print '0 valid samples. Skipping slide...'
            continue

        # uniformly sample tiles from the valid sample space of size n_samples
        sampled_indices = np.random.choice(a=Nx * Ny, size=int(min(N_SAMPLES, np.sum(tile_mask))),
                                           p=tile_mask / np.sum(tile_mask), replace=False)
        sampled_tiles = sample_tile_stack[sampled_indices, :, :]

        # compute response variable over sampled tiles
        response, nts = utils.get_pdl1_response(sampled_tiles, circle=True,
                                                diameter=sample_tile_width, diagnostic=True)

        # compute feature arrays over sampled tiles from neighboring tiles
        feature_rows = np.vstack([utils.get_feature_array(idx, feature_tile_stack, Nx,
                                    scale, feature_layers, flag) for idx in sampled_indices])

        # add outputs to growing array
        combined_response.extend(response)
        combined_features.append(feature_rows)
        combined_nts.extend(nts)

    # convert feature and response to numpy arrays for analysis
    combined_features = np.vstack(combined_features)
    combined_features[np.isnan(combined_features)] = -1
    combined_response = np.array(combined_response)
    combined_nts = np.array(combined_nts)

    return combined_features, combined_response, combined_nts


combined_features, combined_response, combined_nts = automate_tile_extraction(SAMPLES)


# ----- variable processing ----- #

# # remove all cases with no tumor cells in the sampled tile
# mask = combined_response == -1
# combined_response = combined_response[~mask]
# combined_features = combined_features[~mask, :]

# alternatively, remove all cases with <K tumor cells in the sampled tile
print combined_nts.shape, combined_response.shape, combined_features.shape
mask = combined_nts < 10
combined_response = combined_response[~mask]
combined_features = combined_features[~mask, :]


# combined_features, combined_response, combined_nts = automate_tile_extraction(HOLDOUT)
flag = 'n'

# aggregate tiles within arbitrary shapes (e.g. discs or squares of increasing size)
n_obs = combined_features.shape[0]
side_len = scale + 2 * feature_layers
n_tiles = side_len ** 2

if flag == 'n':
    phens = ['tumor','cd4','cd8','foxp3','pdmac','other']
elif flag == 'a':
    phens = ['tumor','pdl1','cd4','cd8','foxp3','pdmac','other']
elif flag == 't':
    phens = ['tumor','pdl1']

phen_columns = []
for phen in range(len(phens)):    # iterate process over each phenotype
    tmp_tiles = combined_features[:, phen * n_tiles:(phen + 1) * n_tiles]
    tmp_3d = tmp_tiles.reshape(n_obs, side_len, side_len)

    range_columns = []

    d_seq_0 = [0] + d_seq
    for i in range(len(d_seq_0) - 1):
        # utils.print_progress(i)
        print phens[phen], d_seq[i]
        if (flag in ['a','t']) and (phens[phen] in ['tumor','pdl1']) and (d_seq[i] <= sample_tile_width):
            print "skipping."
            continue

        mask = utils.shape_mask(grid_dim=side_len, type='circle',
        S=d_seq_0[i+1], s=d_seq_0[i])

        t = np.sum(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1, 1)
        # sigma = np.std(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1,1)
        range_columns.append(t)
        # range_columns.append(sigma)

    per_phen_features = np.hstack(range_columns)
    phen_columns.append(per_phen_features)
X = np.hstack(phen_columns)


# y_noise = y + np.random.normal(scale=0.01, size=(len(y)))
# for i in range(6):
#     print "Phenotype ", i
#     mi = mutual_info_regression(X[:,i].reshape(-1, 1), y.reshape(-1, 1))
#     print "MI: ", mi
#     # display.scatter_hist(X[:,i], y)
#     # plt.scatter(X[:,i], y_noise, s=0.3)
#     print "Corr: ", helper.metrics.corr(X[:, i], y)
#     # plt.show()
