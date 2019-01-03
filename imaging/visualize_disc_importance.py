from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from helper.metrics import corr


def _2(n):
    i = n / 3
    j = n % 3
    return int(i), int(j)


def infer_sign_array(X, y):
    # array of correlation signs to map back to plot later
    return np.array([np.sign(corr(X[:, i], y)) for i in range(X.shape[1])])


def plot_discs(radii, importances, r_pred, signs=None):
    phenotypes = ['tumor','foxp3','cd8','cd4','pdmac','other']

    importances = np.array(importances)
    max_val = np.max(np.abs(importances))

    norm = plt.Normalize(vmin=-max_val, vmax=max_val)
    if signs is not None:
        importances = np.multiply(signs, importances)
    colors = cm.RdBu(norm(importances))

    fig, ax = plt.subplots(nrows=2, ncols=4)
    # c1 = plt.Circle((0.5, 0.5), 0.05 / max(radii), color=colors[0,0])
    # ax[0,0].add_artist(c1)

    for phen in range(importances.shape[0]):
        for r, val in reversed(zip(radii, colors[phen, :])):

            ax[_2(phen)].add_patch(patches.Circle((0, 0), r / max(radii), fill=True, edgecolor='k',
                                   linewidth=0.5, facecolor=val))

        ax[_2(phen)].add_patch(patches.Circle((0,0), r_pred / max(radii), fill=False, edgecolor='k',
                               linewidth=1))

        ax[_2(phen)].axis('equal')
        ax[_2(phen)].set_xlim([-1.2, 1.2])
        ax[_2(phen)].set_ylim([-1.2, 1.2])
        ax[_2(phen)].set_xticklabels([])
        ax[_2(phen)].set_yticklabels([])
        ax[_2(phen)].set_xticks([])
        ax[_2(phen)].set_yticks([])
        ax[_2(phen)].set_title(phenotypes[phen])

        for elem in ['bottom','top','left','right']:
            ax[_2(phen)].spines[elem].set_visible(False)

    plt.show()



if __name__ == '__main__':
    radii = [80, 120, 160, 200]
    importances = [
        [0.3, 0.2, -0.12, -0.23],
        [0.4, -0.2, 0.2, 0.1],
        [0.3, 0.6, -0.6, 0.5],
        [0.1, 0.2, 0.05, 0],
        [-0.3, -0.2, 0, 0.2],
        [0.1, 0.1, 0.2, -0.4]
    ]

    r_pred = 80

    plot_discs(radii, importances, r_pred)
