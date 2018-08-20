from __future__ import division
import numpy as np
from scipy import ndimage
import glob
import os
import multiprocessing as mp

from helper.tileutils import print_progress
from predict_pdl1_identity import visualize_sampling
import helper.processing as processing


DATA_DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/'


def map_cells_to_mat():
    ''' Reconstructs original slides by reading original cell processing text files
        and recording cell phenotypes in a 1040x1392 numpy matrix.
    '''
    files = glob.glob(DATA_DIR + '*cell_seg_data.txt')
    files.sort()

    for i, item in enumerate(files):
        print_progress(i)

        cells = processing.load_sample(item, confidence_thresh=0, verbose=False, radius_lim=0)

        mat = processing.map_phenotypes_to_mat(cells)

        name = item.split("EACRI HNSCC\\")[1].split("_cell_seg_data")[0]
        np.save(DATA_DIR + "processed/" + name, mat)


def record_tumor_edges(slide_id, recompute=False):

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




if __name__ == '__main__':

    # map_cells_to_mat()

    all_samples = processing.get_list_of_samples()

    # # sort samples by modification time
    # all_samples = helper.get_list_of_samples(pattern='*_seg.npy')
    # all_samples.sort(key=os.path.getmtime)
    # all_samples = [x.split("_seg.npy")[0] + ".npy" for x in all_samples]
    #
    pool = mp.Pool(processes=3)
    results = [pool.apply_async(record_tumor_edges, args=(slide, True)) for slide in all_samples]
    results = [p.get() for p in results]
    print results

    # # serial version
    # for i, slide in enumerate(all_samples):
    #     print i
    #
    #     record_tumor_edges(slide, recompute=True)
