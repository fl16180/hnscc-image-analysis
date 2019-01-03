from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.special import expit

import helper.processing as processing
import helper.display as display
import helper.metrics as metrics
import helper.tile_utils as utils
import helper.learning as learning
from helper.utils import print_progress

from constants import HOME_PATH, DATA_PATH
processed_slides = os.path.join(DATA_PATH, 'processed')


all_slides = processing.get_list_of_samples(processed_slides)



for slide in all_slides:
    display.compare_decision_boundaries(slide)



# slide = helper.search_samples('1-26-15__ES13 52013_HP_IM3_22_[26073', all_samples)[0]
#
# cell_mat = np.load(slide)
# # any_cells = (cell_mat != 0)
# #
# # fig = plt.figure()
# # ax1 = fig.add_subplot(121)
# # ax2 = fig.add_subplot(122)
# #
# # filt = ndimage.maximum_filter(any_cells, size=100)
# #
# # ax1.imshow(any_cells)
# # ax2.imshow(filt)
# # plt.show()
#
#
# db = compute_decision_boundary(cell_mat, view=True, n_neighbors=25, slide_id=slide)
# tumor_prop = np.sum(db == 1) / (1392 * 1040)
# print tumor_prop



# # unit test 2
# import helper.tileutils as utils
#
# test = np.zeros(64)
# ones = np.random.choice(a=64, size=20, p=[1 / 64] * 64, replace=False)
# twos = np.random.choice(a=64, size=20, p=[1 / 64] * 64, replace=False)
#
# test[ones] = 1
# test[twos] += 1
# test = test.reshape(4,4,4)
# print test
#
# response = utils.get_pdl1_response(test, circle=True, diameter=4)
# print response
#
# # print utils.shape_mask(4, type='circle', S=4, s=0)




# # test 3
# import helper.tileutils as utils
#
# tmp = utils.tile_stack_mask(10, 8, 1, db_stack=None)
#
# tmp2 = tmp.reshape((8,10))
# print tmp2
#
# tmp3 = np.repeat(tmp2, 2, axis=0)
# print tmp3
# print np.repeat(tmp3, 2, axis=1)
