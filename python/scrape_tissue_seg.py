import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from scipy import ndimage
from skimage import morphology

import helper.processing as processing
import helper.utils as utils


def show_cv(cv_image):
    RGB_im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_im)



DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC'
machine_segs = glob.glob(DIR + '/*_with_tissue_seg.jpg')

# define the list of boundaries
boundaries = [
                ([0, 0, 80], [100, 70, 255]),       # red
                ([70, 45, 60], [255, 120, 120])     # gray
               ]
	           # ([0, 75, 0], [140, 255, 65])  # green


for i, slide in enumerate(machine_segs):
    print i
    
    image = cv2.imread(slide)

    # loop over the boundaries
    masks = []
    for (lower, upper) in boundaries:

        # filter image by defined boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(image, lower, upper)

        # # visualize
        # output = cv2.bitwise_and(image, image, mask=mask)
        # show_cv(np.hstack((image, output)))


        # clean up holes and spots in image
        clean_mask = morphology.remove_small_holes(mask, min_size=1000)
        clean_mask = morphology.remove_small_objects(clean_mask, min_size=1000)

        masks.append(clean_mask)

    final_seg = np.array(masks[0]).astype(int) - np.array(masks[1]).astype(int)

    stem = slide.split("/EACRI HNSCC\\")[1]
    new_stem = stem.split('_image_with_tissue_seg.jpg')[0] + '_seg.npy'
    np.save(DIR + '/processed_orig_seg/' + new_stem, final_seg)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,2,1)
    # RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # ax1.imshow(RGB_im)
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.imshow(final_seg)
    #
    # plt.title(slide)
    # plt.show()
