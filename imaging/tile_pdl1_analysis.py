from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from scipy.ndimage.filters import convolve
import cPickle as pickle
from scipy import ndimage
from scipy.special import logit
import itertools
import seaborn as sns

import helper
from predict_pdl1_identity import visualize_sampling


def main():
    all_samples = helper.get_list_of_samples()

    SAMPLES = [all_samples[i] for i in np.random.choice(len(all_samples), 30)]

    combined_features = []
    combined_response = []

    for count, slide in enumerate(SAMPLES):
        print(count)

        cell_mat = np.load(slide)


def tile_cell_counts(cell_mat, tile_width):
    n_tiles = (int(1040 / tile_width), int(1392 / tile_width))

    tumor_tiles = np.zeros(n_tiles)
    pdl1_tiles = np.zeros(n_tiles)
    foxp3_tiles = np.zeros(n_tiles)
    cd8_tiles = np.zeros(n_tiles)
    cd4_tiles = np.zeros(n_tiles)
    pdmac_tiles = np.zeros(n_tiles)
    other_tiles = np.zeros(n_tiles)

    for i, j in itertools.product(range(0, 1040 - tile_width, tile_width),
                                  range(0, 1392 - tile_width, tile_width)):
        tmp = cell_mat[i:i + tile_width, j:j + tile_width]

        ii = int(i / tile_width)
        jj = int(j / tile_width)
        tumor_tiles[ii, jj] = np.sum(tmp == 1)
        pdl1_tiles[ii, jj] = np.sum(tmp == 2)
        foxp3_tiles[ii, jj] = np.sum(tmp == 3)
        cd8_tiles[ii, jj] = np.sum(tmp == 4)
        cd4_tiles[ii, jj] = np.sum(tmp == 5)
        pdmac_tiles[ii, jj] = np.sum(tmp == 6)
        other_tiles[ii, jj] = np.sum(tmp == 7)


    vars = [tumor_tiles, pdl1_tiles, foxp3_tiles, cd8_tiles, cd4_tiles, pdmac_tiles, other_tiles]
    return vars



if __name__ == '__main__':

    all_samples = helper.get_list_of_samples()

    SAMPLES = [all_samples[i] for i in np.random.choice(len(all_samples), 100)]


    widths = [200]
    var_allwidths = {}
    for w in widths:
        var_allwidths.update({str(w): [[] for _ in range(7)]})

    for img in SAMPLES:

        cell_mat = np.load(img)

        # visualize_sampling(cell_mat=cell_mat)

        for i in widths:
            tmp = tile_cell_counts(cell_mat, tile_width=i)
            tmp2 = [x.flatten() for x in tmp]

            for j in range(7):
                var_allwidths[str(i)][j].extend(tmp2[j])


        # vars1 = tile_cell_counts(cell_mat, tile_width=50)
        # vars2 = tile_cell_counts(cell_mat, tile_width=75)
        # vars3 = tile_cell_counts(cell_mat, tile_width=100)
        # vars4 = tile_cell_counts(cell_mat, tile_width=150)
        # vars5 = tile_cell_counts(cell_mat, tile_width=250)
        # vars1_flat = [x.flatten() for x in vars1]
        # vars2_flat = [x.flatten() for x in vars2]
        # vars3_flat = [x.flatten() for x in vars3]
        # vars4_flat = [x.flatten() for x in vars4]
        # vars5_flat = [x.flatten() for x in vars4]

            # var75[j].extend(vars2_flat[j])
            # var100[j].extend(vars3_flat[j])
            # var150[j].extend(vars4_flat[j])
            # var250[j].extend(vars5_flat[j])


        # vars_bin = [(x > 0) for x in vars]

    # # width_key = {'50': var50, '75': var75, '100': var100, '150': var150, '250': var250}
    # for w in var_allwidths:
    #     corr_mat = np.zeros((49))
    #     counter = 0
    #     for i, j in itertools.product(var_allwidths[w], repeat=2):
    #         corr_mat[counter] = helper.corr(np.array(i).flatten(), np.array(j).flatten())
    #         counter += 1
    #     corr_mat = corr_mat.reshape(7, 7)
    #
    #     df = pd.DataFrame(corr_mat)
    #     cnames = ['tumor','pdl1','foxp3','cd8','cd4','pdmac','other']
    #     df.columns = cnames
    #     df.index = cnames
    #
    #     sns.set()
    #     sns.heatmap(df, cmap='RdBu', annot=True, vmin=-1, vmax=1, annot_kws={'fontsize':8})
    #     plt.title('Pairwise correlations at tile width {0}'.format(w))
    #
    #     fig = plt.gcf()
    #     fig.set_size_inches(8, 6.5)
    #     fig.savefig('tile{0}_keepzeros.png'.format(w), dpi=100)
    #     plt.show()


    cnames = ['tumor','pdl1','foxp3','cd8','cd4','pdmac','other']

    # print var100[1]
    vars_column = [np.array(v).reshape(-1, 1) for v in var_allwidths['200']]
    vars_df = pd.DataFrame(np.hstack(vars_column))
    vars_df.columns = cnames
    #
    print vars_df
    g = sns.PairGrid(vars_df)
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.hexbin, gridsize=12, bins='log')
    # g = g.map_offdiag(plt.scatter)
    # g = g.map(plt.scatter)
    plt.show()


    # print helper.corr(cd8_tiles.flatten(), cd4_tiles.flatten())
