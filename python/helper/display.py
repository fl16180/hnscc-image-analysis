from __future__ import division
import sys
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd


def visualize_image(image, cells, phenotypes):
    slide = misc.imread(image)

    plt.imshow(slide)
    plt.scatter(*get_position(cells[phenotypes]), color='r', marker='.')
    plt.show()



def visualize_sampling(db=None, cell_mat=None, response_circles=None, feature_circles=None,
                       image=None, label=None):
    ''' TODO: turn into class that precomputes plot limits, params, etc, with
        methods with optional overlays
    '''

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    if image is not None:
        jpg = misc.imread(image)
        plt.imshow(jpg)

    if db is not None:
        plt.imshow(db, cmap=cmap_light)

    if cell_mat is not None:

        tumor_loc = np.nonzero(cell_mat == 1)
        pdl1_loc = np.nonzero(cell_mat == 2)
        other_loc = np.nonzero(cell_mat > 2)

        plt.scatter(tumor_loc[1], tumor_loc[0], c='b', edgecolor='k', s=15)
        plt.scatter(pdl1_loc[1], pdl1_loc[0], c='r', edgecolor='k', s=15)
        plt.scatter(other_loc[1], other_loc[0], c='w', edgecolor='k', s=15)


    if response_circles is not None:
        sampled_mat = np.logical_or.reduce(response_circles)
        plt.imshow(sampled_mat, alpha=0.5, cmap='Oranges')

    if feature_circles is not None:
        features_mat = np.logical_or.reduce(feature_circles)
        plt.imshow(features_mat, alpha=0.3, cmap='Greens')


    if label is not None:
        plt.title(label)

        # # Create an alpha channel of linearly increasing values moving to the right.
        # alphas = np.ones(weights.shape)
        # alphas[:, 30:] = np.linspace(1, 0, 70)
        #
        # # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
        # colors = Normalize(vmin, vmax, clip=True)(weights)
        # colors = cmap(colors)
        #
        # # Now set the alpha channel to the one we created above
        # colors[..., -1] = alphas
        #
        # # Create the figure and image
        # # Note that the absolute values may be slightly different
        # fig, ax = plt.subplots()
        # ax.imshow(greys)
        # ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))
        # ax.set_axis_off()

        # tumors = np.copy(cell_mat)
        # tumors[(tumors != 0) & (tumors != 1) & (tumors != 2)] = 0
        # plt.imshow(tumors, alpha=0.4)
        # # plt.scatter(jvec, ivec)
    plt.show()
        # plt.matshow(db); ; plt.show()


def scatter_hist(x, y, xlims=[0,1], ylims=[0,1], xbins=20, ybins=20, colors='b'):
    '''https://matplotlib.org/2.0.1/examples/pylab_examples/scatter_hist.html
    '''
    nullfmt = NullFormatter()        # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(6, 6))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, c=colors)

    # determine limits
    xbin_width = (xlims[1] - xlims[0]) / xbins
    ybin_width = (ylims[1] - ylims[0]) / ybins
    xmin, xmax = xlims[0], xlims[1]
    ymin, ymax = ylims[0], ylims[1]

    axScatter.set_xlim((xlims))
    axScatter.set_ylim((ylims))

    xbins = np.arange(xmin, xmax, xbin_width)
    ybins = np.arange(ymin, ymax, ybin_width)
    axHistx.hist(x, bins=xbins)
    axHisty.hist(y, bins=ybins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()



def fixed_scatter(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x, y)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_aspect('equal')

    # diagonal y=x line
    ax.plot([0,1], [0,1], ls="--", c=".3")

    plt.show()


def dotplot(response, groups, n_patients=30):
    # dotplot pdl1 response grouped by patient

    dot_arr = np.array(zip(groups, response))
    dot_arr_subset = dot_arr[dot_arr[:, 0] <= n_patients]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(dot_arr_subset[:,0], dot_arr_subset[:,1], s=15, linewidths=0.5)

    ax.set_xticks(np.arange(n_patients+1, step=2))
    ax.xaxis.grid()

    ax.set_xlabel("Patient")
    ax.set_ylabel("pdl1 response")
    ax.set_ylim([0,1])
    ax.set_xlim([-1, n_patients + 1])
    plt.show()
