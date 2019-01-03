import unittest
import sys
import numpy as np
import helper.processing
import helper.tile_utils as utils
import helper.display as display


class TileTest(unittest.TestCase):

    def test_restacking(self):
        print "Running:", sys._getframe().f_code.co_name
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

    def test_stack_and_flatten(self):
        print "Running:", sys._getframe().f_code.co_name
        nx, ny = 8, 5
        tw = 3

        a = np.random.permutation(tw*tw*nx*ny).reshape(tw*ny, tw*nx)
        tiles = utils.restack_to_tiles(a, tile_width=tw, nx=nx, ny=ny)
        b = utils.flatten_tile_stack(tiles, tile_width=tw, nx=nx, ny=ny).astype(int)

        self.assertTrue(np.all(a == b))


    def test_standard_masking_visualization(self):
        print "Running:", sys._getframe().f_code.co_name
        tile_width = 200
        protected_edge_layers = 1
        Nx, Ny = int(1392 / tile_width), int(1040 / tile_width)

        slide = helper.processing.get_list_of_samples()[26]

        cell_mat = np.load(slide)
        imloc = helper.processing.get_original_image(slide)

        cell_mat = cell_mat[:Ny * tile_width, :Nx * tile_width]

        tile_mask = utils.tile_stack_mask(Nx, Ny, L=protected_edge_layers,
                                          db_stack=None)

        tmp = tile_mask.reshape((Ny, Nx))
        tmp2 = np.repeat(tmp, tile_width, axis=0)
        expanded_mask = np.repeat(tmp2, tile_width, axis=1)

        display.visualize_sampling(image=imloc)
        display.visualize_sampling(db=expanded_mask, cell_mat=cell_mat)

    def test_edge_masking_visualization(self):
        print "Running:", sys._getframe().f_code.co_name
        tile_width = 200
        protected_edge_layers = 1
        Nx, Ny = int(1392 / tile_width), int(1040 / tile_width)

        slide = helper.processing.get_list_of_samples()[100]

        cell_mat = np.load(slide)
        imloc = helper.processing.get_original_image(slide)

        edges = np.load(slide.split(".npy")[0] + "_seg.npy")

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

    def test_tile_pdl1_count(self):
        print "Running:", sys._getframe().f_code.co_name
        tile_width = 200
        protected_edge_layers = 1
        Nx, Ny = int(1392 / tile_width), int(1040 / tile_width)

        slide = helper.processing.get_list_of_samples()[26]

        cell_mat = np.load(slide)
        imloc = helper.processing.get_original_image(slide)

        cell_mat = cell_mat[:Ny * tile_width, :Nx * tile_width]
        sample_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=tile_width,
                                                   nx=Nx, ny=Ny)

        tile_mask = utils.tile_stack_mask(Nx, Ny, L=protected_edge_layers,
                                          db_stack=None)

        # uniformly sample tiles from the valid sample space (where tile_mask == 1)
        np.random.seed(30)
        sample_index = np.random.choice(a=Nx * Ny, size=1,
                                           p=tile_mask / np.sum(tile_mask), replace=False)
        # only keep the selected tile
        tile_mask[:sample_index[0]] = 0
        tile_mask[sample_index[0]+1:] = 0

        # select sample tile and compute response variable
        sampled_tiles = sample_tile_stack[sample_index, :, :]
        response, nts = utils.get_pdl1_response(sampled_tiles, circle=True,
                                        diameter=tile_width, diagnostic=True)

        # form visualization mask
        tmp = tile_mask.reshape((Ny, Nx))
        tmp2 = np.repeat(tmp, tile_width, axis=0)
        expanded_mask = np.repeat(tmp2, tile_width, axis=1)

        # convert selected tile shape from square to a circle
        expanded_mask_tiles = utils.restack_to_tiles(expanded_mask, tile_width=tile_width,
                                                   nx=Nx, ny=Ny)
        mask = utils.shape_mask(tile_width, type='circle', S=tile_width, s=0)
        expanded_mask_tiles_masked = np.multiply(expanded_mask_tiles, mask)
        expanded_mask_circle = utils.flatten_tile_stack(expanded_mask_tiles_masked,
                                                        tile_width=tile_width, nx=Nx, ny=Ny)

        print 'Percent pdl1+ (red): ', response[0]
        print 'Total no. of tumor cells (blue and red)', nts[0]
        display.visualize_sampling(db=expanded_mask_circle, cell_mat=cell_mat)

    def test_tile_feature_extraction(self):
        print "Running:", sys._getframe().f_code.co_name
        tile_width = 200
        protected_edge_layers = 1
        feature_tile_width = 1
        diams = [100]
        offset_px = int((max(diams) - tile_width) / 2)
        offset_tiles = int(np.ceil(offset_px / tile_width))
        Nx, Ny = int(1392 / tile_width), int(1040 / tile_width)
        nx, ny = Nx * tile_width, Ny * tile_width

        slide = helper.processing.get_list_of_samples()[26]

        cell_mat = np.load(slide)
        imloc = helper.processing.get_original_image(slide)

        # cell_mat = cell_mat[:Ny * tile_width, :Nx * tile_width]
        sample_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=tile_width,
                                                   nx=Nx, ny=Ny)
        feature_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=feature_tile_width,
                                                                nx=nx, ny=ny)
        tile_mask = utils.tile_stack_mask(Nx, Ny, L=protected_edge_layers,
                                          db_stack=None)

        # uniformly sample tiles from the valid sample space (where tile_mask == 1)
        np.random.seed(30)
        sample_index = np.random.choice(a=Nx * Ny, size=1,
                                           p=tile_mask / np.sum(tile_mask), replace=False)
        # only keep the selected tile
        tile_mask[:sample_index[0]] = 0
        tile_mask[sample_index[0]+1:] = 0

        # select sample tile and compute response variable
        sampled_tiles = sample_tile_stack[sample_index, :, :]
        response, nts = utils.get_pdl1_response(sampled_tiles, circle=True,
                                        diameter=tile_width, diagnostic=True)
        # compute feature arrays over sampled tiles from neighboring tiles
        feature_rows = np.vstack([utils.get_feature_array(idx, feature_tile_stack, Nx,
                                                          tile_width, offset_px, 'n')
                                                            for idx in sample_index])

        # form visualization mask
        tmp = tile_mask.reshape((Ny, Nx))
        tmp2 = np.repeat(tmp, tile_width, axis=0)
        expanded_mask = np.repeat(tmp2, tile_width, axis=1)

        # convert selected tile shape from square to a circle
        expanded_mask_tiles = utils.restack_to_tiles(expanded_mask, tile_width=tile_width,
                                                   nx=Nx, ny=Ny)
        mask = utils.shape_mask(tile_width, type='circle', S=100, s=0)
        expanded_mask_tiles_masked = np.multiply(expanded_mask_tiles, mask)
        expanded_mask_circle = utils.flatten_tile_stack(expanded_mask_tiles_masked,
                                                        tile_width=tile_width, nx=Nx, ny=Ny)

        # aggregate tiles within arbitrary shapes (e.g. discs or squares of increasing size)
        n_obs = 1
        side_len = tile_width + 2 * offset_px
        n_tiles = side_len ** 2

        phens = ['tumor','foxp3','cd8','cd4','pdmac','other','mac']

        phen_columns = []
        for phen in range(len(phens)):    # iterate process over each phenotype
            # phen = 0
            tmp_tiles = feature_rows[:, phen * n_tiles:(phen + 1) * n_tiles]
            tmp_3d = tmp_tiles.reshape(n_obs, side_len, side_len)

            range_columns = []

            diams_0 = [0] + diams
            for i in range(len(diams)):
                print phens[phen], diams[i]

                mask = utils.shape_mask(grid_dim=side_len, type='circle',
                S=diams_0[i+1], s=diams_0[i])

                t = np.sum(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1, 1)
                # sigma = np.std(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1,1)
                range_columns.append(t)
                # range_columns.append(sigma)

            per_phen_features = np.hstack(range_columns)
            phen_columns.append(per_phen_features)
        print zip(phens, [x[0][0] for x in phen_columns])
        display.visualize_sampling(db=expanded_mask_circle, cell_mat=cell_mat, phen='all')


    def test_feature_tile_matching(self):
        print "Running:", sys._getframe().f_code.co_name
        feature_diameter = 5
        sample_diam = 5
        feature_tile_width = 1
        sample_tile_width = sample_diam

        nx, ny = 25, 20    # no. feature tiles
        Nx, Ny = int(25 / sample_tile_width), int(20 / sample_tile_width)       # no. sample tiles
        offset_px = int((feature_diameter - sample_diam) / 2)
        offset_tiles = int(np.ceil(offset_px / sample_diam))

        a = np.random.permutation(ny*nx).reshape(ny, nx)
        feature_tile_stack = utils.restack_to_tiles(a, tile_width=feature_tile_width,
                                                    nx=nx, ny=ny)
        feature_tile_stack.shape

        idx = 6
        scale = sample_tile_width / feature_tile_width
        output = utils.id_feature_tiles(idx, Nx, scale, feature_layers=offset_px)

        self.assertTrue(np.all( output[4:7] == np.array([134,155,156]) ))

    def test_feature_tile_matching_offset(self):
        # TODO
        print "Running:", sys._getframe().f_code.co_name
        feature_diameter = 10
        sample_diam = 5
        feature_tile_width = 1
        sample_tile_width = sample_diam

        nx, ny = 50, 40    # no. feature tiles
        Nx, Ny = int(50 / sample_tile_width), int(40 / sample_tile_width)       # no. sample tiles
        offset_px = int((feature_diameter - sample_diam) / 2)
        offset_tiles = int(np.ceil(offset_px / sample_diam))

        a = np.random.permutation(ny*nx).reshape(ny, nx)
        feature_tile_stack = utils.restack_to_tiles(a, tile_width=feature_tile_width,
                                                    nx=nx, ny=ny)
        feature_tile_stack.shape

        idx = 11
        scale = sample_tile_width / feature_tile_width
        output = utils.id_feature_tiles(idx, Nx, scale, feature_layers=offset_px)
        output

        # self.assertTrue(np.all( output[4:7] == np.array([134,155,156]) ))


if __name__ == '__main__':
    unittest.main()
