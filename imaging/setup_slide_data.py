from __future__ import division
import numpy as np
from scipy import ndimage
import glob
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
from skimage import morphology

import helper.utils as utils
from predict_pdl1_identity import visualize_sampling
import helper.processing as processing

from constants import DATA_PATH


def map_cells_to_mat():
    ''' Reconstructs original slides by reading original cell processing text files
        and recording cell phenotypes in a 1040x1392 numpy matrix.
    '''
    outloc = os.path.join(DATA_PATH, 'processed')
    if not os.path.exists(outloc):
        os.makedirs(outloc)

    files = glob.glob(DATA_PATH + '*cell_seg_data.txt')
    files.sort()

    for i, item in enumerate(files):
        utils.print_progress(i)

        cells = processing.load_sample(item, confidence_thresh=0, verbose=False, radius_lim=0)

        mat = processing.map_phenotypes_to_mat(cells)

        name = item.split("EACRI HNSCC\\")[1].split("_cell_seg_data")[0]
        np.save(os.path.join(outloc, name), mat)


def record_tumor_edges(slide_id, recompute=False):
    ''' Takes the name of a saved numpy matrix representing processed slide cell locations
        and computes a simple tumor segmentation boundary.

        The output matrix is stored as the same name with "_edges" tag.
    '''
    try:
        if recompute is True:
            print "Regenerating edges."
            raise IOError()

        np.load(slide_id.split(".npy")[0] + "_edges.npy")
        print "Precomputed edges found."

    except IOError:

        cell_mat = np.load(slide_id)

        db = processing.compute_decision_boundary(cell_mat, view=False, n_neighbors=25,
                                                  slide_id=slide_id, recompute=False,
                                                  clean=True, remove_blank_regions=True)

        # store unprocessed regions
        na_regions = (db == -1)

        # skip slide if too little or too much tumor area
        tumor_prop = np.sum(db == 1) / (1392 * 1040)
        if (tumor_prop < 0.1) or (tumor_prop > 0.9):
            print "skipping..."
            return

        # further restrict sampling area to within 200px distance of decision boundary
        tmp = ndimage.minimum_filter(db, size=200) + db     # apply minimum filter
        region = (tmp == 1)
        db[~region] = 0
        db[region] = 1
        db[na_regions] = -1

        # visualize_sampling(db=db, cell_mat=cell_mat)
        np.save(slide_id.split(".npy")[0] + "_edges.npy", db)


def record_machine_seg():
    def show_cv(cv_image):
        RGB_im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_im)

    outloc = os.path.join(DATA_PATH, 'processed_orig_seg')
    if not os.path.exists(outloc):
        os.makedirs(outloc)

    machine_segs = glob.glob(DATA_PATH + '/*_with_tissue_seg.jpg')

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
        np.save(os.path.join(outloc, new_stem), final_seg)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,2,1)
        # RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ax1.imshow(RGB_im)
        # ax2 = fig.add_subplot(1,2,2)
        # ax2.imshow(final_seg)
        #
        # plt.title(slide)
        # plt.show()


if __name__ == '__main__':

    # map_cells_to_mat()

    all_samples = processing.get_list_of_samples()

    # # sort samples by modification time (can save time if process was interrupted)
    # all_samples = helper.get_list_of_samples(pattern='*_seg.npy')
    # all_samples.sort(key=os.path.getmtime)
    # all_samples = [x.split("_seg.npy")[0] + ".npy" for x in all_samples]

    # pool = mp.Pool(processes=3)
    # results = [pool.apply_async(record_tumor_edges, args=(slide, True)) for slide in all_samples]
    # results = [p.get() for p in results]
    # print results

    # serial version
    for i, slide in enumerate(all_samples):
        print i
        record_tumor_edges(slide, recompute=True)

    record_machine_seg()
