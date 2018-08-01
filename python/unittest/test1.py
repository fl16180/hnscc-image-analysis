import unittest
import matplotlib.pyplot as plt
import helper.processing as processing

class ProcessingTest(unittest.TestCase):

    def test_cell_mat_construction(self):

        loc = 'halle validation__37097-12_HP_IM3_0_[8583,15527]_cell_seg_data.txt'
        cells = processing.load_sample(loc, radius_lim=0)

        mat = map_phenotypes_to_mat(cells)
        


if __name__ == '__main__':
    unittest.main()
