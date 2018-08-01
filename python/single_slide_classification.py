from __future__ import division
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from scipy import ndimage
from scipy import misc


# helper functions
def get_position(cell):
    return (cell['Cell X Position'], cell['Cell Y Position'])

def euclid_dist(loc1, loc2):
    return sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1])** 2)

def pdl1_tumor_ratio():
    # DIR = '/home/flu/Documents/ISB_full_data/image files EACRI HNSCC/'
    DIR = 'D:/fred/image files EACRI HNSCC/'

    files = glob.glob(DIR + '*cell_seg_data.txt')
    files.sort()
    # files = [x for x in sort(files, key=)]

    for item in files:
        dataset = pd.read_csv(item, sep='\t')

        pdl1 = dataset[dataset['Phenotype'] == 'pd-l1']
        tumor = dataset[dataset['Phenotype'] == 'tumor']

        print '\nCell counts in the sample: ', item.split('_')[4:10]
        print 'pdl1: ', len(pdl1)
        print 'tumor: ', len(tumor)


def load_sample(loc, confidence_thresh=0.3, verbose=False, radius_lim=200):

    # load data file and select cells above confidence threshold
    dataset = pd.read_csv(loc, sep='\t')

    # exclude cells too close to borders
    dataset = dataset[(dataset['Cell X Position'] > radius_lim) &
                      (dataset['Cell X Position'] < 1392 - radius_lim) &
                      (dataset['Cell Y Position'] > radius_lim) &
                      (dataset['Cell Y Position'] < 1040 - radius_lim)]

    # exclude cells by confidence
    dataset.Confidence1 = dataset.Confidence.replace('%','',regex=True).astype('float')/100
    dataset = dataset[dataset.Confidence1 > confidence_thresh]

    cells = {
            'unclassified': dataset[dataset['Phenotype'].isnull()],
            'foxp3': dataset[dataset['Phenotype'] == 'foxp3'],
            'pdl1': dataset[dataset['Phenotype'] == 'pd-l1'],
            'cd4': dataset[dataset['Phenotype'] == 'cd4'],
            'other': dataset[dataset['Phenotype'] == 'other'],
            'pdmac': dataset[dataset['Phenotype'] == 'pd-l1+ mac'],
            'cd8': dataset[dataset['Phenotype'] == 'cd8'],
            'tumor': dataset[dataset['Phenotype'] == 'tumor'],
            }

    # compile all tumor cells (phenotype marked pdl1 or tumor)
    all_tumor = pd.concat([cells['pdl1'], cells['tumor']])
    cells['all_tumor'] = all_tumor

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

# get a uniform random sample of tumor cells
def get_random_sample(cells, n_cells, response_phenotype='pd-l1'):

    # np.random.seed(20)
    select = np.random.randint(0, cells.shape[0], n_cells)
    sample = cells.iloc[select]

    response = (sample['Phenotype'] == response_phenotype).values.astype(int)

    return sample, response


def get_binned_distances(reference, phenotype, n_bins, max_radius=200, bin_func='default'):
    # initialize empty array
    features = np.zeros((reference.shape[0], n_bins))

    # define reference function for binning distances
    if bin_func == 'default':
        bin_func = np.logspace(1, np.log10(max_radius), n_bins)

    counter = 0
    # iterate binning procedure for each ref_cell
    for index, ref_cell in reference.iterrows():

        # min_dist = 1000
        # compute and bin distances from the ref_cell to each taret cell
        for index2, targ_cell in cells[phenotype].iterrows():
            d = euclid_dist(get_position(ref_cell), get_position(targ_cell))

            # skip the cell that is itself
            if d == 0:
                continue

            # # update minimum distance
            # if d < min_dist:
            #     min_dist = d

            locate = bin_func > d
            if sum(locate) > 0:
                features[counter, np.argmax(locate)] += 1
        counter +=1

    return features


def class_by_nearest_neighbor(sample, response):
    X = sample.loc[:, ['Cell X Position', 'Cell Y Position']].values

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return response[indices[:, 1]]


def two_class_score(prediction, target, label=None):
    assert len(prediction) == len(target), "two_class_score: Input arrays must be same length."

    zeros = target == 0

    if label is not None:
        print label + ':'

    print "proportion of class 0:", sum(zeros) / len(prediction)
    print "class 0 accuracy: ", 1 - sum(prediction[zeros]) / len(prediction[zeros])
    print "class 1 accuracy: ", sum(prediction[~zeros]) / len(prediction[~zeros])
    print "overall accuracy: ", sum(prediction == target) / len(prediction)



# DIR = '/home/flu/Documents/ISB_full_data/image files EACRI HNSCC/'
# DIR = 'D:/fred/image files EACRI HNSCC/'
DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/'

SAMPLES = ['halle_hnscc__20701-07_HP_IM3_1_[12756.5,20614.3]_cell_seg_data.txt',
           'halle_hnscc__12535-10_HP_IM3_0_[15527.7,15438.7]_cell_seg_data.txt',
           'halle_hnscc__12535-10_HP_IM3_4_[14833.6,10263.2]_cell_seg_data.txt',
           'halle_hnscc__13118-09_HP_IM3_3_[14141.2,12851]_cell_seg_data.txt',
           'halle_hnscc__13119-09_HP_IM3_6_[14141.2,15956]_cell_seg_data.txt',
           'halle_hnscc__13119-09_HP_IM3_15_[14833.6,16990.6]_cell_seg_data.txt',
           'halle validation__2777-13_HP_IM3_1_[19269,12744]_cell_seg_data.txt',
           'halle validation__3938-09_HP_IM3_0_[16800,14866]_cell_seg_data.txt',
           'halle validation__4866-05_HP_IM3_0_[9061,17076]_cell_seg_data.txt',
           'halle validation__5213-12_HP_IM3_0_[15705,12339]_cell_seg_data.txt',
           'halle validation__5213-12_HP_IM3_1_[21948,11895]_cell_seg_data.txt',
           'halle validation__5371-14_HP_IM3_1_[19575,9919]_cell_seg_data.txt',
           'halle validation__5472-13_HP_IM3_0_[23049,11179]_cell_seg_data.txt',
           'halle validation__8849-13_HP_IM3_1_[11946,18248]_cell_seg_data.txt',
           'halle validation__9847-13_HP_IM3_1_[13703,15171]_cell_seg_data.txt',
           'halle validation__44422-12_HP_IM3_0_[15649,16154]_cell_seg_data.txt',
           'halle_hnscc__893-14_HP_IM3_12_[6521.6,11297.8]_cell_seg_data.txt',
           'halle_hnscc__3516-10_HP_IM3_8_[16220.1,10780.5]_cell_seg_data.txt',
           'halle_hnscc__4720-10_HP_IM3_9_[22455,13368.3]_cell_seg_data.txt',
           'halle_hnscc__15830-09_HP_IM3_10_[12064.1,13885.6]_cell_seg_data.txt',
           'halle_hnscc__16182-09_HP_IM3_23_[10677.6,12332.4]_cell_seg_data.txt'
           ]

sample = SAMPLES[0]
sample2 = SAMPLES[9]
sample3 = SAMPLES[14]

sl = misc.imread(DIR + sample.replace('_cell_seg_data.txt', '.jpg'))
sl2 = misc.imread(DIR + sample.replace('_cell_seg_data.txt', '.jpg'))
sl3 = misc.imread(DIR + sample.replace('_cell_seg_data.txt', '.jpg'))
ims = (sl, sl2, sl3)

fig, ax = plt.subplots(3, 3)

for i, SLIDE in enumerate([sample, sample2, sample3]):
    # load slides
    cells = load_sample(DIR + SLIDE, verbose=True, radius_lim=0, confidence_thresh=0.3)

    misclass = pd.DataFrame()
    correct = pd.DataFrame()
    repeat = 10
    for iter in range(repeat):
        print iter

        # sample tumor cells from slide
        tumor_sample, response = get_random_sample(cells=cells['all_tumor'], n_cells=100)

        # # baseline classification using nearest neighbor cell class
        # nn_pred = class_by_nearest_neighbor(tumor_sample, response)
        #

        # compute binned distances
        features1 = get_binned_distances(tumor_sample, 'pdl1', max_radius=200, n_bins=50)
        features2 = get_binned_distances(tumor_sample, 'tumor', max_radius=200, n_bins=50)
        features3 = get_binned_distances(tumor_sample, 'cd8', max_radius=200, n_bins=50)
        features4 = get_binned_distances(tumor_sample, 'cd4', max_radius=200, n_bins=50)
        # features5 = get_binned_distances(tumor_sample, 'foxp3', max_radius=200, n_bins=50)
        features = np.hstack((features1, features2, features3, features4))

        # split sample into train and test sets
        indices = np.arange(tumor_sample.shape[0])
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(features, response,
                                                                               indices, test_size=0.4)
        test_cells = tumor_sample.iloc[id_test]

        # run random forest classification
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        misclass = pd.concat(objs=[misclass, test_cells[rf_pred != y_test]])
        correct = pd.concat(objs=[correct, test_cells[rf_pred == y_test]])

        # misclass = pd.concat(objs=[misclass, tumor_sample[nn_pred != response]])
        # correct = pd.concat(objs=[correct, tumor_sample[nn_pred == response]])

    ax[i, 0].imshow(ims[i])
    ax[i, 0].scatter(*get_position(cells['pdl1']), color='r', marker='.')
    ax[i, 0].scatter(*get_position(cells['tumor']), color='b', marker='.')

    ax[i, 0].set_xlim([0, 1392])
    ax[i, 0].set_ylim([1040, 0])

    ax[i, 1].imshow(ims[i])
    ax[i, 1].scatter(*get_position(cells['pdl1']), color='r', marker='.')
    ax[i, 1].scatter(*get_position(cells['tumor']), color='b', marker='.')
    ax[i, 1].scatter(*get_position(misclass), color='y', marker='.')

    ax[i, 1].set_xlim([0, 1392])
    ax[i, 1].set_ylim([1040, 0])

    ax[i, 2].imshow(ims[i])


plt.text(0, 0, 'blue = tumor, red = pdl1, yellow = misclassified')
plt.subplots_adjust(wspace=0.2, hspace=0.2)
fg = plt.gcf()
fg.set_size_inches(21, 16)
fg.savefig('single_slides_rf.png', format='png', dpi=300)



# # compute binned distances
# features1 = get_binned_distances(tumor_sample, 'pdl1', max_radius=200, n_bins=50)
# features2 = get_binned_distances(tumor_sample, 'tumor', max_radius=200, n_bins=50)
# features3 = get_binned_distances(tumor_sample, 'cd8', max_radius=200, n_bins=50)
# features4 = get_binned_distances(tumor_sample, 'cd4', max_radius=200, n_bins=50)
# features5 = get_binned_distances(tumor_sample, 'foxp3', max_radius=200, n_bins=50)
#
# features = np.hstack((features1, features2, features3, features4, features5))
# # features = features2
#
#
# # split sample into train and test sets
# indices = np.arange(combined_sample.shape[0])
# X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(combined_sample, combined_response,
#                                                                        indices, test_size=0.33)
# # run random forest classification
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_preds = rf.predict(X_test)
#
# # score rf
# two_class_score(rf_preds, y_test, label='RF')
#
# # score baseline (nearest neighbor classifier)
# two_class_score(nn_pred[id_test], y_test, label='1-NN')
#
#


#
# def scatterplot_cells(pdl1, tumor):
#     plt.scatter(*get_position(pdl1), c='violet')
#     plt.scatter(*get_position(tumor), c='blue')
#
#     plt.title('PD-L1 classified cells, colored by classification confidence')
#     fig = plt.gcf()
#
#     fig.set_size_inches(10, 7.5)
#     plt.gca().invert_yaxis()
#
#     # plt.savefig('pdl1_confidence.png', format='png', dpi=200)
#     plt.show()


# fixed_cell = pdl1.iloc[10]
#
# dists = [euclid_dist(get_position(cell), get_position(fixed_cell)) for i, cell in pdl1.iterrows()]
# plt.hist(dists, bins=20)
# plt.show()
