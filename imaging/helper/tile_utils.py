from __future__ import division
import sys
import numpy as np
import pandas as pd


def _tile_shape(arr, nrows, ncols):
    '''
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    Credit to unutbu @ "https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays"
    '''
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def restack_to_tiles(mat, tile_width, nx, ny):
    '''
    Input slide is 2D matrix of 1040x1392 pixels. This is converted into a stack of tiles
    with side length tile_width.

    nx, ny set how many tiles to take in the x and y directions (since the image dimensions
    are likely not integer multiples of tile_width).

    For efficiency this is performed using numpy reshape operations.
    '''
    # cut to integer multiple dimensions of tile_width
    mat_cut = mat[:tile_width * ny, :tile_width * nx]

    # reshape to a 3d stack of tiles
    return _tile_shape(mat_cut, tile_width, tile_width)


def flatten_tile_stack(tiles, tile_width, nx, ny):
    ''' Inverse operation for restack_to_tiles without restoring missing extra space for
    non-integer multiples of tile_width. This function has not been converted to matrix manipulation
    because it is only used for testing. Thus it may be much slower than restack_to_tiles.
    '''
    flat = np.zeros((tile_width*ny, tile_width*nx))
    for i in range(ny):
        for j in range(nx):
            idx = i*nx + j
            flat[tile_width*i:tile_width*(i+1), tile_width*j:tile_width*(j+1)] = tiles[idx]
    return flat


def tile_stack_mask(nx, ny, L, db_stack=None):
    '''
    Constructs an identifier array for masking out unwanted tiles.

    Any tile recorded as an observation has corresponding L layers of tiles surrounding
    which are used as features. Thus only tiles located L or more layers from an edge
    are available for sampling.

    In the diagram below, only tiles in the small rectangle are to be sampled.

    ######################
    #                    #
    #    ############    #
    #    ############    #
    #    ############    #
    #                    #
    ######################

    If a decision boundary tile stack is supplied via db_stack, then tiles are further
    restricted to those that sufficiently overlap with 1-labeled regions in the decision
    boundary stack.

    '''
    all_tile_img = np.zeros((ny, nx))
    if L == 0:
        all_tile_img[:,:] = 1
    else:
        all_tile_img[L:-L, L:-L] = 1
    all_tile_img = all_tile_img.reshape(-1)

    if db_stack is not None:
        width = db_stack.shape[1]

        ones_per_tile = np.sum(db_stack, axis=(1,2))
        mask2 = (ones_per_tile / (width ** 2)) > 0.25

        # print (ones_per_tile.reshape(10,13) > 0).astype(int)
        all_tile_img[~mask2] = 0

    return all_tile_img


def sid_to_fid(id, Nx, scale):
    ''' convert the index of a sampled tile into the index of the feature tile located
        at the top left corner of the sampled tile
    '''
    i = int(id / Nx)
    j = id % Nx
    return i * scale, j * scale


def id_feature_tiles(id, Nx, scale, feature_layers):
    i, j = sid_to_fid(id, Nx, scale)
    irange = np.arange(i - feature_layers, i + feature_layers + scale)
    jrange = np.arange(j - feature_layers, j + feature_layers + scale)

    jv, iv = np.meshgrid(jrange, irange)
    return (iv * Nx * scale + jv).reshape(-1)


def get_pdl1_response(tiles, circle=False, diameter=None, diagnostic=False):
    ''' construct response variable from sampled tiles.
        tiles is an array of dimension:
            (n_samples x sample_tile_width x sample_tile_width)

        The array is summed over dimensions 2 and 3 to get tile
        cell counts, returning array of length n_samples.

        In cases with no pdl1 or tumor cells in the tile, resulting in division by 0,
        the value -1 is assigned.
    '''
    tiles_c = np.copy(tiles)

    if circle:
        mask = shape_mask(diameter, type='circle', S=diameter, s=0)
        tiles_c = np.multiply(tiles, mask)

    n_tumor = np.sum(tiles_c == 1, axis=(1,2))
    n_pdl1 = np.sum(tiles_c == 2, axis=(1,2))

    response = n_pdl1 / (n_pdl1 + n_tumor)
    response[np.isnan(response)] = -1

    if diagnostic:
        n_tumors = n_tumor + n_pdl1
        return response, n_tumors

    return response


def get_feature_array(idx, feature_tile_stack, Nx, scale, feature_layers, flag='n'):
    ''' construct a single feature row from a single sampled tile
    '''
    feature_ids = id_feature_tiles(idx, Nx, scale, feature_layers)

    feature_tiles = feature_tile_stack[feature_ids, :, :]

    n1 = np.sum(feature_tiles == 1, axis=(1,2))
    n2 = np.sum(feature_tiles == 2, axis=(1,2))
    nt = np.logical_or(n1, n2).astype(int)

    n3 = np.sum(feature_tiles == 3, axis=(1,2))
    n4 = np.sum(feature_tiles == 4, axis=(1,2))
    n5 = np.sum(feature_tiles == 5, axis=(1,2))
    n6 = np.sum(feature_tiles == 6, axis=(1,2))
    n7 = np.sum(feature_tiles == 7, axis=(1,2))
    n8 = np.sum(feature_tiles == 8, axis=(1,2))

    if flag == 'n':
        return np.concatenate([nt, n3, n4, n5, n6, n7, n8])
    elif flag == 'a':
        return np.concatenate([n1, n2, n3, n4, n5, n6, n7, n8])
    elif flag == 't':
        return np.concatenate([n1, n2])


def discretize_array(arr, n_bins=2, cutoffs=(0.15,)):
    ''' Converts a 1-D continuous probability array to a discrete array.
    For example, with n_bins=2 and cutoffs=(0.15), any value below 0.15 is
    assigned as 0, and any value above or equal to 0.15 is assigned as 1.

    @Params:
        arr: 1-D numpy array, with continuous values in [0,1].
        n_bins: int, number of discrete values to convert arr to, from 0 to n_bins - 1.
        cutoffs: tuple, cutoff thresholds of length n_bins - 1.
    '''
    tmp = np.copy(arr)
    cutoffs = (0,) + cutoffs

    for i in range(n_bins - 1):
        tmp[(arr >= cutoffs[i]) & (arr < cutoffs[i + 1])] = i
    tmp[arr >= cutoffs[n_bins - 1]] = n_bins - 1
    return tmp.astype(int)


def construct_feature_names(scale, feature_layers):
    ''' construct a list of identifiers corresponding to the feature tiles and phenotypes
    '''
    side_length = scale + 2 * feature_layers

    jv, iv = np.meshgrid(np.arange(side_length), np.arange(side_length))
    coords = zip(iv.reshape(-1), jv.reshape(-1))
    tile_ids = ['t{0};{1}_'.format(pair[0], pair[1]) for pair in coords]

    phen_list = ['phenT','phen3','phen4','phen5','phen6','phen7','phen8']

    return [(str1 + str2) for str2 in phen_list for str1 in tile_ids]


def search_feature_list(feature_list, locs, phenotypes):
    ''' string matching against the list of features
    '''
    start_str = ['t{0};{1}'.format(locs[row, 0], locs[row, 1]) for row in xrange(locs.shape[0])]
    end_str = ['_phen{0}'.format(k) for k in phenotypes]

    all_str = [(str1 + str2) for str2 in end_str for str1 in start_str]

    return [feature_list.index(i) for i in all_str]


def shape_mask(grid_dim, type='square', S=2, s=0):
    # print grid_dim, type, S, s
    mask = np.zeros((grid_dim, grid_dim))
    S = int(S)
    s = int(s)

    if type == 'square':
        # TODO: convert S from radius to diameter
        mid = int(np.ceil(grid_dim / 2))
        start = mid - S
        small_start = mid - s

        print start
        if start == 0:
            mask[:, :] = 1
        else:
            mask[start:-start, start:-start] = 1
        mask[small_start:-small_start, small_start:-small_start] = 0

    elif type == 'circle':

        R = int(S / 2)
        r = int(s / 2)

        # generate R x R grid containing a circle of radius R of ones
        j = np.atleast_2d(range(-R, 0) + range(1, R + 1))
        i = j.T
        mask_max = i ** 2 + j ** 2 <= R ** 2 + 1
        mask_min = i ** 2 + j ** 2 <= r ** 2 + 1
        combine = (mask_max.astype(float) - mask_min.astype(float)) if r != 0 else mask_max

        loc = int((grid_dim - 2 * R) / 2)
        mask[loc:loc + 2 * R, loc:loc + 2 * R] = combine

    return mask
