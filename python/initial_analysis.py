# exploratory analysis on a single slide containing head/neck SCC tumor.
# code here is collapsed from a jupyer notebook for Python
#
# Fred Lu 2017

# coding: utf-8

# # 11/9: Loading random tumor sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

get_ipython().magic(u'matplotlib inline')


# helper functions
def get_position(cell):
    return (cell['Cell X Position'], cell['Cell Y Position'])

def euclid_dist(loc1, loc2):
    return sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1])** 2)


# load data from csv
dataset = pd.read_csv('halle_hnscc__149-70_HP_IM3_9_[15527.7,13368.3]_cell_seg_data.txt',
            sep='\t')
print dataset.columns.values
print set(dataset['Phenotype'])

# select cells by phenotype class
unclassified = dataset[dataset['Phenotype'].isnull()]
foxp3 = dataset[dataset['Phenotype'] == 'foxp3']
pdl1 = dataset[dataset['Phenotype'] == 'pd-l1']
cd4 = dataset[dataset['Phenotype'] == 'cd4']
other = dataset[dataset['Phenotype'] == 'other']
pdmac = dataset[dataset['Phenotype'] == 'pd-l1+ mac']
cd8 = dataset[dataset['Phenotype'] == 'cd8']
tumor = dataset[dataset['Phenotype'] == 'tumor']
print '\nCell counts in the sample: '
print 'unclassified: ', len(unclassified)
print 'foxp3: ', len(foxp3)
print 'pdl1: ', len(pdl1)
print 'cd4: ', len(cd4)
print 'other: ', len(other)
print 'pdmac: ', len(pdmac)
print 'cd8: ', len(cd8)
print 'tumor: ', len(tumor)

# Output:
# set([nan, 'foxp3', 'pd-l1', 'cd4', 'other', 'pd-l1+ mac', 'cd8', 'tumor'])
#
# Cell counts in the sample:
# unclassified:  46
# foxp3:  169
# pdl1:  1001
# cd4:  380
# other:  3079
# pdmac:  35
# cd8:  900
# tumor:  82

# 1. If working with fluorescence marker readings, a map is needed between fluorescence markers (Alexa 514, Alexa 594, Coumarin, Cy3, FITC, Cy5, DAPI) and phenotype (listed above).
#
# 2. If working with distances between phenotype-labeled cells, Euclidean distances can be obtained through features 'Cell X Position' and 'Cell Y Position'

# ### Distribution of cells around PD-L1 cells by radius and cell type

# distribution changes significantly based on what cell is fixed, as can be expected

# select a random cell
fixed_cell = pdl1.iloc[5]
# compute euclidean distances of other cells to this cell
dists = [euclid_dist(get_position(cell), get_position(fixed_cell)) for i, cell in pdl1.iterrows()]
plt.hist(dists, bins=20)
plt.show()

fixed_cell = pdl1.iloc[10]
dists = [euclid_dist(get_position(cell), get_position(fixed_cell)) for i, cell in pdl1.iterrows()]
plt.hist(dists, bins=20)
plt.show()


# thought: is confidence uniformly distributed over distance (easier for analysis if it is)?
pdl1.Confidence = pdl1.Confidence.replace('%','',regex=True).astype('float')/100

pdl1_mask = pdl1[pdl1.Confidence > 0]

plt.scatter(get_position(pdl1_mask)[0], get_position(pdl1_mask)[1], c=pdl1_mask.Confidence.values)
plt.colorbar()
plt.title('PD-L1 classified cells, colored by classification confidence')
fig = plt.gcf()
fig.set_size_inches(10, 7.5)
plt.savefig('pdl1_confidence.png', format='png', dpi=200)
plt.show()

# plotting cells by confidence, there are clumps of high confidence. importantly many regions have groups of cells classified as pdl1, that in fact don't have high confidence at all.
# Therefore, I am restricting to cells above a threshold confidence

dataset.Confidence1 = dataset.Confidence.replace('%','',regex=True).astype('float')/100
confident_dataset = dataset[dataset.Confidence1 > 0.7]

unclassified = confident_dataset[confident_dataset['Phenotype'].isnull()]
foxp3 = confident_dataset[confident_dataset['Phenotype'] == 'foxp3']
pdl1 = confident_dataset[confident_dataset['Phenotype'] == 'pd-l1']
cd4 = confident_dataset[confident_dataset['Phenotype'] == 'cd4']
other = confident_dataset[confident_dataset['Phenotype'] == 'other']
pdmac = confident_dataset[confident_dataset['Phenotype'] == 'pd-l1+ mac']
cd8 = confident_dataset[confident_dataset['Phenotype'] == 'cd8']
tumor = confident_dataset[confident_dataset['Phenotype'] == 'tumor']
print '\nCell counts in the sample: '
print 'unclassified: ', len(unclassified)
print 'foxp3: ', len(foxp3)
print 'pdl1: ', len(pdl1)
print 'cd4: ', len(cd4)
print 'other: ', len(other)
print 'pdmac: ', len(pdmac)
print 'cd8: ', len(cd8)
print 'tumor: ', len(tumor)
#
# Cell counts in the sample:
# unclassified:  0
# foxp3:  82
# pdl1:  452
# cd4:  107
# other:  1195
# pdmac:  1
# cd8:  271
# tumor:  6

# Distribution of distances from PDL1 to CD8 cells

# pick a reference cell type and get cell counts
ref = cd8
n_pdl1, n_ref = pdl1.shape[0], ref.shape[0]

dist_mat = np.empty((n_pdl1, n_ref))
# iterate distance calc. over all pairs of pdl1, ref cells
for i in range(n_pdl1):
    for j in range(n_ref):
        dist_mat[i, j] = euclid_dist(get_position(pdl1.iloc[i]), get_position(ref.iloc[j]))

# distribution of distances from a specific pdl1 cell to all cd8 cells (like lines 68-78)
plt.hist(dist_mat[12,:], bins=20)
plt.show()

#all distance measures combined into single array
plt.hist(dist_mat.reshape(-1), bins=20)
plt.show()


# Repeating this step for distances to CD4 cells

# pick a reference cell type and get cell counts
ref = cd4
n_pdl1, n_ref = pdl1.shape[0], ref.shape[0]

dist_mat = np.empty((n_pdl1, n_ref))
# iterate distance calc. over all pairs of pdl1, ref cells
for i in range(n_pdl1):
    for j in range(n_ref):
        dist_mat[i, j] = euclid_dist(get_position(pdl1.iloc[i]), get_position(ref.iloc[j]))

plt.hist(dist_mat[10,:], bins=20)
plt.show()

#all distance measures combined into single array
plt.hist(dist_mat.reshape(-1), bins=20)
plt.show()


# These distributions aggregate all cells together so there is a lot of autocorrelation included. To address this, I will try looking up bootstrapping methods that can sample in ways to minimize effects from autocorrelations in 2-dimensional distances.
#
# Out of curiosity I test the appearance of the distance distribution using a basic bootstrap.

n_obs = dist_mat.shape[0]
n_samples = 500

res = dist_mat[np.random.randint(low=0, high=n_obs), :]
for i in range(n_samples - 1):
    a = np.random.randint(low=0, high=n_obs)
    res = np.concatenate((res, dist_mat[a, :]), axis=0)

plt.hist(res, bins=20)
plt.show()


# The issue with producing composite distribution data is that in the process of aggregation, cells near the sides of the image have their distributional information truncated. A count/sum-based aggregation will under-represent large distances from those cells because many cells around those ones will be off the image (truncated). Instead we should have a relative distribution based on e.g. the ratios of distances from those cells, -given- the area around the cell that is actually available to sample from. (or other robust ways to gather information from these cells)

# # this code is the basic implementation to compute all pairwise distances.
# runs too slowly to create a complete distance matrix on jupyter.
# note: look for vectorized pairwise distance function in numpy if such a matrix is needed
#
# n_cells = dataset.shape[0]
#
# dist_mat = np.empty((n_cells, n_cells))
# for i in range(n_cells):
#     for j in range(n_cells):
#         dist_mat[i, j] = euclid_dist(get_position(dataset.iloc[i]), get_position(dataset.iloc[j]))


#
# # A different approach: Spatial statistics
#

# divide slide into bins and aggregate counts of cells in each bin
X_MAX = 1392
Y_MAX = 1040

print max(confident_dataset['Cell X Position']) # 1388
print max(confident_dataset['Cell Y Position']) # 1035

BIN_SIZE = 100

def bin_cells(input_array, bin_size):
    bins = np.zeros((X_MAX / bin_size + 1, Y_MAX / bin_size + 1))
    for i in range(len(input_array)):
        x, y = get_position(input_array.iloc[i])
        bins[x / bin_size, y / bin_size] += 1
    return bins

np.set_printoptions(edgeitems=5)
print bin_cells(confident_dataset, BIN_SIZE)

pdl1_binned = bin_cells(pdl1, BIN_SIZE)
cd8_binned = bin_cells(cd8, BIN_SIZE)

from scipy.stats import entropy

# entropy
print entropy(pdl1_binned.reshape(-1))
print entropy(cd8_binned.reshape(-1))

# Kullback-leibler divergence, must add pseudo-count to avoid log of 0
print entropy(pdl1_binned.reshape(-1) + 0.5, cd8_binned.reshape(-1) + 0.5)


# compute KL divergence over a linespace of bin sizes
KL_array = []
for i in range(10, 110, 5):
    pdl1_tmp = bin_cells(pdl1, i)
    cd8_tmp = bin_cells(cd8, i)

    KL_array.append(entropy(pdl1_tmp.reshape(-1) + 0.5, cd8_tmp.reshape(-1) + 0.5))
plt.plot(KL_array)

# compute KL divergence over a linespace of bin sizes
KL_array = []
for i in range(10, 110, 5):

    pdl1_tmp = bin_cells(pdl1, i)
    cd8_tmp = bin_cells(cd4, i)

    KL_array.append(entropy(pdl1_tmp.reshape(-1) + 0.5, cd8_tmp.reshape(-1) + 0.5))
plt.plot(KL_array)
