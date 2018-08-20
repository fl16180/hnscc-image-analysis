import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
import itertools
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import helper.processing as processing
import helper.utils as utils


def tile_cell_counts(cell_mat, tile_width):
    n_tiles = (int(1040 / tile_width), int(1392 / tile_width))

    tumor_tiles = np.zeros(n_tiles)
    # pdl1_tiles = np.zeros(n_tiles)
    foxp3_tiles = np.zeros(n_tiles)
    cd8_tiles = np.zeros(n_tiles)
    cd4_tiles = np.zeros(n_tiles)
    pdmac_tiles = np.zeros(n_tiles)
    other_tiles = np.zeros(n_tiles)
    macs_tiles = np.zeros(n_tiles)

    for i, j in itertools.product(range(0, 1040 - tile_width, tile_width),
                                  range(0, 1392 - tile_width, tile_width)):
        tmp = cell_mat[i:i + tile_width, j:j + tile_width]

        ii = int(i / tile_width)
        jj = int(j / tile_width)
        tumor_tiles[ii, jj] = np.sum((tmp == 1) | (tmp == 2))
        # pdl1_tiles[ii, jj] = np.sum(tmp == 2)
        foxp3_tiles[ii, jj] = np.sum(tmp == 3)
        cd8_tiles[ii, jj] = np.sum(tmp == 4)
        cd4_tiles[ii, jj] = np.sum(tmp == 5)
        pdmac_tiles[ii, jj] = np.sum(tmp == 6)
        other_tiles[ii, jj] = np.sum(tmp == 7)
        macs_tiles[ii, jj] = np.sum(tmp == 8)

    vars = [tumor_tiles, foxp3_tiles, cd8_tiles, cd4_tiles,
            pdmac_tiles, other_tiles, macs_tiles]
    return vars


def all_slides_tile_mat():
    all_samples = processing.get_list_of_samples()

    corr_mat = []
    for count, slide in enumerate(all_samples):
        utils.print_progress(count)

        cell_mat = np.load(slide).astype(int)

        corr_row = get_tile_correlation(cell_mat, dist=[50, 100, 200])

        corr_mat.append(corr_row)

    corr_mat = np.vstack(corr_mat)

    return corr_mat



def get_tile_correlation(cell_mat, dist):
    corr_row = []
    for width in dist:
        ts = tile_cell_counts(cell_mat, width)
        tsarr = np.hstack([x.reshape(-1,1) for x in ts])

        means = np.mean(tsarr, axis=0)
        stdvs = np.std(tsarr, axis=0)
        cors = spearmanr(tsarr, axis=0, nan_policy='omit')[0]
        cors_flat = cors[np.triu_indices(7, k=1)]

        # print means, stdvs, cors_flat

        # features = np.concatenate(means, stdvs, cors_flat)
        # corr_row.extend(means)
        # corr_row.extend(stdvs)
        corr_row.extend(cors_flat)

    return np.nan_to_num(corr_row)

# mats = greycomatrix(cell_mat, distances=range(1,50), angles=[0], levels=9, symmetric=True)
# print np.sum(mats[:,:,:,0], axis=2)
