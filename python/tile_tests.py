import unittest
import numpy as np
import helper.processing
import helper.tileutils as utils
import helper.display as display

class TileTest(unittest.TestCase):

    def test_restacking(self):

        slide = helper.processing.get_list_of_samples()[100]

        cell_mat = np.load(slide)

        sample_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=100,
                                                   nx=5, ny=5)
        feature_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=50,
                                                    nx=10, ny=10)

        self.assertTrue(np.all(sample_tile_stack[0, :50, :50] == feature_tile_stack[0, :, :]))
        self.assertTrue(np.all(sample_tile_stack[1, :50, :50] == feature_tile_stack[2, :, :]))
        self.assertTrue(np.all(sample_tile_stack[0, :50, 50:100] == feature_tile_stack[1, :, :]))

        self.assertTrue(np.all(sample_tile_stack[5, :50, :50] == feature_tile_stack[20, :, :]))
        self.assertTrue(np.all(sample_tile_stack[24, 50:100, :50] == feature_tile_stack[98, :, :]))


    def test_masking_visualization(self):

        tile_width = 150
        protected_edge_layers = 1
        Nx, Ny = int(1392 / tile_width), int(1040 / tile_width)

        slide = helper.processing.get_list_of_samples()[100]

        cell_mat = np.load(slide)
        imloc = helper.processing.get_original_image(slide)

        edges = np.load(slide.split(".npy")[0] + "_edges.npy")

        cell_mat = cell_mat[:Ny * tile_width, :Nx * tile_width]
        edges = edges[:Ny * tile_width, :Nx * tile_width]

        edges_tile_stack = utils.restack_to_tiles(edges, tile_width=tile_width,
                                                  nx=Nx, ny=Ny)

        tile_mask = utils.tile_stack_mask(Nx, Ny, L=protected_edge_layers,
                                          db_stack=edges_tile_stack)
        tmp = tile_mask.reshape((Ny, Nx))
        tmp2 = np.repeat(tmp, tile_width, axis=0)
        expanded_mask = np.repeat(tmp2, tile_width, axis=1)


        display.visualize_sampling(image=imloc)
        display.visualize_sampling(db=expanded_mask, cell_mat=cell_mat)


        tile_mask.reshape((Ny, Nx))



if __name__ == '__main__':
    unittest.main()
