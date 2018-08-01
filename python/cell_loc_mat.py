import os
import glob
import numpy as np

from helper.processing import load_sample, map_phenotypes_to_mat


if __name__ == '__main__':

    DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/'

    files = glob.glob(DIR + '*cell_seg_data.txt')
    files.sort()

    for item in files:

        cells = load_sample(item, confidence_thresh=0.25, verbose=False, radius_lim=0)

        mat = map_phenotypes_to_mat(cells)

        name = item.split("EACRI HNSCC\\")[1].split("_cell_seg_data")[0]

        np.save(DIR + "processed/" + name, mat)
