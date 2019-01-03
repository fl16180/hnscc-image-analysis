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


def extract_dataset(diams, sample_diam, flag):

    np.random.seed(1000)

    # set sampling parameters
    N_SLIDES = 314     # number of slides to use
    N_SAMPLES = 30      # max samples to take from a single slide

    # set tile feature extraction parameters
    sample_tile_width = sample_diam
    feature_tile_width = 1
    Nx, Ny = int(1392 / sample_tile_width), int(1040 / sample_tile_width)       # no. sample tiles
    nx, ny = Nx * sample_tile_width, Ny * sample_tile_width                     # no. feature tiles
    offset_px = int((max(diams) - sample_diam) / 2)
    offset_tiles = int(np.ceil(offset_px / sample_diam))

    # get pre-processed slide matrices and select random sample of slides
    all_slides = processing.get_list_of_samples(processed_slides)
    SLIDES = [all_slides[i] for i in np.random.choice(len(all_slides), N_SLIDES, replace=False)]

    # initialize processed variables storage
    ncells_all = []
    slides_all = []
    X_all = []
    y_all = []
    overlap_all = []

    # process samples in batches
    batch_size = 10
    for idx in range(0, N_SLIDES, batch_size):
        BATCH = SLIDES[idx:idx + batch_size]

        # iterate over sampled slides to extract feature and response variables via tile sampling
        batch_ncells = []
        batch_features = []
        batch_response = []
        batch_slides = []
        batch_overlap = []
        for i, slide in enumerate(BATCH):
            print_progress(i)

            # load slide and reshape into sample and feature tile stacks
            cell_mat = np.load(slide)
            sample_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=sample_tile_width,
                                                       nx=Nx, ny=Ny)
            feature_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=feature_tile_width,
                                                        nx=nx, ny=ny)

            # load seg file to compute ratio of processed area to total slide area
            seg = np.load(slide.split(".npy")[0] + "_seg.npy")
            correction = 1392*1040 / np.sum(seg != -1)

            n_cells_total = np.sum(cell_mat != 0)
            n_cells_corrected = n_cells_total * correction

            # make unprocessed region matrix from seg file
            seg_map = (seg == -1).astype(int)
            seg_tile_stack = utils.restack_to_tiles(seg_map, tile_width=feature_tile_width,
                                                    nx=nx, ny=ny)

            ### used for limiting tile sampling to 'edge regions' between tumor and stroma.
            ### For now I think it is simpler and more explanable to permit sampling anywhere in the
            ### tumor, not just on the edge. I may revisit this in the future.
                # # load tumor edge matrix (skipping slide if no matrix is found)
                # try:
                #     edges = np.load(slide.split(".npy")[0] + "_edges.npy")
                #     edges_tile_stack = utils.restack_to_tiles(edges, tile_width=sample_tile_width,
                #                                               nx=Nx, ny=Ny)
                # except IOError:
                #     print 'No edge matrix. Skipping slide...'
                #     continue

                # select valid tiles for sampling, skipping slide if no valid tiles are available
                # tile_mask = utils.tile_stack_mask(Nx, Ny, L=sample_layers, db_stack=edges_tile_stack)

            # get set of valid sampling tiles (tiles with enough offset from the edges)
            tile_mask = utils.tile_stack_mask(Nx, Ny, L=offset_tiles, db_stack=None)
            n_tiles = int(min(N_SAMPLES, np.sum(tile_mask)))
            if n_tiles == 0:
                print('0 valid samples. Skipping slide...')
                continue

            # store batch cell numbers and slide names
            batch_ncells.extend([n_cells_corrected] * n_tiles)
            batch_slides.extend([slide] * n_tiles)

            # uniformly sample tiles from the valid sample space of size n_samples
            # in this case, I have set it to just get all available samples from each slide
            sampled_indices = np.random.choice(a=Nx * Ny, size=n_tiles,
                                               p=tile_mask / np.sum(tile_mask), replace=False)
            sampled_tiles = sample_tile_stack[sampled_indices, :, :]

            # compute response variable over sampled tiles
            response, nts = utils.get_pdl1_response(sampled_tiles, circle=True,
                                                    diameter=sample_tile_width, diagnostic=True)

            # compute feature arrays over sampled tiles from neighboring tiles
            feature_rows = []
            overlap = []
            for j in sampled_indices:
                feature_tiles = utils.get_feature_array(j, feature_tile_stack, Nx,
                                                        sample_tile_width, offset_px, flag)
                seg_map_tiles = utils.get_feature_array(j, seg_tile_stack, Nx,
                                                        sample_tile_width, offset_px, flag)

                # store feature tile and overlap with unprocessed regions
                feature_rows.append(feature_tiles)
                overlap.append(np.sum(seg_map_tiles))

            del feature_tile_stack
            del seg_tile_stack

            # add to growing array as long as any valid samples have been collected
            if len(feature_rows) > 0:
                feature_rows = np.vstack(feature_rows)
                overlap = np.array(overlap)
                # # remove observations with significant overlap (>10%) with unprocessed regions
                # mask = (np.array(overlap) <= 0.1 * max(diams) ** 2)
                # feature_rows = feature_rows[mask, :]
                # response = response[mask]
                # nts = nts[mask]

                batch_response.extend(response)
                batch_features.append(feature_rows)
                batch_overlap.extend(overlap)

        # convert feature and response to numpy arrays for analysis
        batch_features = np.vstack(batch_features)
        batch_response = np.array(batch_response)
        batch_overlap = np.array(batch_overlap)

        # ----- variable processing ----- #

        # # remove all cases with no tumor cells in the sampled tile
        # mask = combined_response == -1
        # combined_response = combined_response[~mask]
        # combined_features = combined_features[~mask, :]

        # # alternatively, remove all cases with <K tumor cells in the sampled tile
        # # print combined_nts.shape, combined_response.shape, combined_features.shape
        # mask = combined_nts < 10
        # combined_response = combined_response[~mask]
        # combined_features = combined_features[~mask, :]


        # aggregate tiles within arbitrary shapes (e.g. discs or squares of increasing size)
        n_obs = batch_features.shape[0]
        side_len = sample_tile_width + 2 * offset_px
        n_tiles = side_len ** 2

        if flag == 'n':
            phens = ['tumor','cd4','cd8','foxp3','pdmac','other']
        elif flag == 'a':
            phens = ['tumor','pdl1','cd4','cd8','foxp3','pdmac','other']
        elif flag == 't':
            phens = ['tumor','pdl1']

        phen_columns = []
        for phen in range(len(phens)):    # iterate process over each phenotype
            tmp_tiles = batch_features[:, phen * n_tiles:(phen + 1) * n_tiles]
            tmp_3d = tmp_tiles.reshape(n_obs, side_len, side_len)

            range_columns = []

            diams_0 = [0] + diams
            for i in range(len(diams)):
                print_progress('{0}: {1}'.format(phens[phen], diams[i]))
                if (flag in ['a','t']) and (phens[phen] in ['tumor','pdl1']) and (diams[i] <= sample_tile_width):
                    print("skipping.")
                    continue

                mask = utils.shape_mask(grid_dim=side_len, type='circle',
                S=diams_0[i+1], s=diams_0[i])

                t = np.sum(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1, 1)
                # sigma = np.std(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1,1)
                range_columns.append(t)
                # range_columns.append(sigma)

            per_phen_features = np.hstack(range_columns)
            phen_columns.append(per_phen_features)

        del batch_features
        ncells_all.extend(batch_ncells)
        slides_all.extend(batch_slides)
        X_all.append(np.hstack(phen_columns))
        y_all.extend(batch_response)
        overlap_all.extend(batch_overlap)

    ncells_all = np.array(ncells_all)
    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    overlap_all = np.array(overlap_all)

    # save processed data as csv
    feature_names = ["_".join([a, str(b)]) for a in phens for b in diams]
    feature_names.append('y')
    tmp = pd.DataFrame(np.hstack((X_all, y_all.reshape(-1,1))))
    tmp.columns = feature_names
    tmp['slide'] = slides_all
    tmp['n_cells_corrected'] = ncells_all
    tmp['unscored_overlap'] = overlap_all
    tmp = tmp.set_index('slide')
    tmp.to_csv(os.path.join(HOME_PATH, 'data', 'local_discs.csv'))


def convert_npy_to_csv(diams):
    X = np.load(os.path.join(HOME_PATH, 'data', "data_x.npy"))
    y = np.load(os.path.join(HOME_PATH, 'data', "data_y.npy"))

    phens = ['tumor','cd4','cd8','foxp3','pdmac','other']

    feature_names = ["_".join([a, str(b)]) for a in phens for b in diams]
    feature_names.append('y')
    tmp = pd.DataFrame(np.hstack((X, y.reshape(-1,1))))
    tmp.columns = feature_names
    tmp.to_csv(os.path.join(HOME_PATH, 'data', 'local_discs.csv'), index=False)


def split_dataset_for_learning():
    dat = pd.read_csv(os.path.join(HOME_PATH, 'data', 'local_discs.csv'))

    # filter samples with constraints
    dat = dat[dat.tumor_150 >= 10]
    dat = dat[dat.unscored_overlap <= 0.1 * 450 ** 2]
    dat.shape
    dat = dat.drop('unscored_overlap', axis=1)

    # all_slides = set(dat.slide)
    # np.random.permutation(len(all_slides))
    train_lim = int(0.7*dat.shape[0])
    dev_lim = int(0.85*dat.shape[0])
    perm = np.random.permutation(dat.shape[0])

    train_dat = dat.iloc[perm[:train_lim], :]
    dev_dat = dat.iloc[perm[train_lim:dev_lim], :]
    test_dat = dat.iloc[perm[dev_lim:], :]

    train_dat.to_csv(os.path.join(HOME_PATH, 'data', 'train_dat.csv'), index=False)
    dev_dat.to_csv(os.path.join(HOME_PATH, 'data', 'dev_dat.csv'), index=False)
    test_dat.to_csv(os.path.join(HOME_PATH, 'data', 'test_dat.csv'), index=False)


def analyze_dataset(diams, sample_diam, flag):

    train_dat = pd.read_csv(os.path.join(HOME_PATH, 'data', 'train_dat.csv'))
    dev_dat = pd.read_csv(os.path.join(HOME_PATH, 'data', 'dev_dat.csv'))

    X_train = train_dat.drop(['slide','y','n_cells_corrected'], axis=1)
    X_dev = dev_dat.drop(['slide','y','n_cells_corrected'], axis=1)
    y_train = train_dat['y']
    y_dev = dev_dat['y']

    # def pairwise_ratios(X):
    #     ratios = []
    #     X_inv = 1/(X+0.5)
    #     for i in range(X.shape[1]):
    #         ratios.append(X[:, i].reshape(-1,1) * X_inv[:, i+1:])
    #     return np.hstack((ratios))
    #
    # X_train_ratios = pairwise_ratios(X_train.values)
    # X_dev_ratios = pairwise_ratios(X_dev.values)
    # X_train_ratios.shape
    # X_train = np.hstack((X_train.values, X_train_ratios))
    # X_dev = np.hstack((X_dev.values, X_dev_ratios))

    # from sklearn.linear_model import RidgeCV
    # mod = RidgeCV()
    # mod.fit(X_train, y_train)
    # pred = mod.predict(X_dev)
    # metrics.mse(pred, y_dev)
    # metrics.corr(pred, y_dev)
    # metrics.mae(pred, y_dev)
    # f = display.fixed_scatter(pred, y_dev)
    #
    #
    #
    #
    from sklearn.ensemble import RandomForestRegressor
    mod = RandomForestRegressor(n_estimators=500, oob_score=True)
    mod.fit(X_train, y_train)
    pred = mod.predict(X_dev)
    print(metrics.mse(pred, y_dev))
    print(metrics.corr(pred, y_dev))
    print(metrics.mae(pred, y_dev))
    f = display.fixed_scatter(pred, y_dev)


    def adjust_missing_feature_importances(importances, flag, n_outer, n_inner):
        rings = n_outer + n_inner
        if flag == 'n':
            return importances.reshape(6, rings)
        if flag == 'a':
            tmp = np.insert(importances, n_outer, n_inner * [0])
            tmp = np.insert(tmp, 0, n_inner * [0])
            return tmp.reshape(7, rings)

    from visualize_disc_importance import plot_discs, infer_sign_array


    signs = infer_sign_array(X_train.values, y_train.values)

    n_outer = np.sum(np.array(diams) > sample_diam)
    n_inner = len(diams) - n_outer

    tmp = adjust_missing_feature_importances(mod.feature_importances_, flag, n_outer, n_inner)
    signs = adjust_missing_feature_importances(signs, flag, n_outer, n_inner)

    plot_discs(diams, tmp, r_pred=sample_diam, signs=signs)



    X_all = pd.concat((X_train, X_dev))
    y_all = pd.concat((y_train, y_dev))

    # boot = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True)
    # X_boot = X_train.loc[boot, :]
    # y_boot = y_train[boot]

    importance_scores = learning.mean_decrease_accuracy_importance(X_all.values, y_all.values,
                                                                   X_all.columns.values)
    mean_scores = {}
    for feat, score in importance_scores.items():
        mean_scores[feat] = round(np.mean(score), 4)
    mean_scores

    ordered_importances = np.array([mean_scores[feat] for feat in X_all.columns.values])




    importances = pd.DataFrame(mod.feature_importances_,
                               index = X_train.columns,
                               columns=['importance']).sort_values('importance',
                                                                    ascending=False)
    importances


    signs = infer_sign_array(X_all.values, y_all.values)

    n_outer = np.sum(np.array(diams) > sample_diam)
    n_inner = len(diams) - n_outer

    tmp = adjust_missing_feature_importances(ordered_importances, flag, n_outer, n_inner)
    signs = adjust_missing_feature_importances(signs, flag, n_outer, n_inner)

    plot_discs(diams, tmp, r_pred=sample_diam, signs=signs)




def old():

    # for i in range(30):
    #
    #     p = int(i / 5)
    #     r = i % 5
    #     plt.hist(X[:,i])
    #
    #     plt.title(phens[p] + '_' + str(diams[r]))
    #     plt.show()

    # from sklearn.feature_selection import mutual_info_regression
    # y_noise = y + np.random.normal(scale=0.01, size=(len(y)))
    # for i in range(6):
    #     print "Phenotype ", i
    #     mi = mutual_info_regression(X[:,i].reshape(-1, 1), y.reshape(-1, 1))
    #     print "MI: ", mi
    #     # display.scatter_hist(X[:,i], y)
    #     # plt.scatter(X[:,i], y_noise, s=0.3)
    #     print "Corr: ", helper.metrics.corr(X[:, i], y)
    #     # plt.show()


    # X_train, X_test, y_train, y_test = train_test_split(X, discrete_response,
    #                                                     test_size=0.4)
    # from sklearn.linear_model import LassoCV
    # # from sklearn.neural_network import MLPClassifier
    # # from sklearn.ensemble import AdaBoostClassifier
    # # from sklearn.tree import DecisionTreeClassifier
    # # rf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    # #                     hidden_layer_sizes=(300, 2))

    # from sklearn.ensemble import ExtraTreesClassifier
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    # rf = ExtraTreesClassifier(n_estimators=500, class_weight='balanced', oob_score=True, bootstrap=True)
    # rf = RandomForestRegressor(n_estimators=500, oob_score=True, bootstrap=True)
    # rf = LassoCV(cv=10, normalize=False)
    #### fit machine learning models ####

    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import Ridge

    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import MaxAbsScaler

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RepeatedKFold, GroupKFold
    ################################################################

    X_ = X
    y_ = y

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    X_p = np.hstack(((X+1), 1/(X+1)))
    X_p = poly.fit_transform(X_p)
    X_p = np.sqrt(X_p)
    X_p.shape
    X_ = X_p

    plt.scatter(X_[:,4], y_)

    rep_scores = []
    estimator = RidgeCV()
    estimator = RandomForestRegressor(n_estimators=50)
    cv = RepeatedKFold(n_splits=10, n_repeats=1)
    out_sample = {'pred': [], 'target': []}
    for train, test in cv.split(X_, y_):

        X_train = X_[train]
        X_test = X_[test]
        y_train = y_[train]
        y_test = y_[test]

        # X_train = np.sqrt(X_train + 1)
        # X_test = np.sqrt(X_test + 1)

        scale = StandardScaler()
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)

        # X_train = pca.fit_transform(X_train)
        # X_test = pca.transform(X_test)


        preds = estimator.fit(X_train, y_train).predict(X_test)
        # preds = dtr.undo(preds)
        # y_test = dtr.undo(y_test)

        rep_scores.append(metrics.rmse(preds, y_test))
        # rep_scores.append(estimator.score(X_test, y_test))

        out_sample['pred'].extend(preds)
        out_sample['target'].extend(y_test)

    print(np.mean(rep_scores), np.std(rep_scores))


    plt.scatter(preds, y_test)

    # dict elements from list to array
    for key, value in out_sample.iteritems():
        out_sample[key] = np.array(value)

    metrics.rmse(out_sample['pred'], out_sample['target'])
    fig = plt.scatter(out_sample['pred'], out_sample['target'])

    np.sqrt(np.mean(out_sample['target'] ** 2))


    y_test
    plt.hist(yt)















    ################################################################

    # # rf = AdaBoostClassifier(n_estimators=500)
    # # rf = LogisticRegression(penalty='l2')
    # estimator = LogisticRegression(max_iter=1000)
    # estimator = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', oob_score=True)
    estimator = RandomForestRegressor(n_estimators=300, oob_score=True, bootstrap=True)

    learner = learning.TestLearner(task='regress')
    learner.test(estimator, X, y, folds=5, n_classes=2, rf_oob=True)

    print("proportion of 0:", sum(y==0) / len(y))
    print("proportion of 1:", sum(y==1) / len(y))

    estimator.fit(X, y)
    print(estimator.feature_importances_)
    # tmp = estimator.feature_importances_[::-1].reshape(6,5)


    def adjust_missing_feature_importances(importances, flag, n_outer, n_inner):
        rings = n_outer + n_inner
        if flag == 'n':
            return importances.reshape(6, rings)
        if flag == 'a':
            tmp = np.insert(importances, n_outer, n_inner * [0])
            tmp = np.insert(tmp, 0, n_inner * [0])
            return tmp.reshape(7, rings)

    # tmp = estimator.feature_importances_.reshape(6,5)

    from visualize_disc_importance import plot_discs, infer_sign_array

    signs = infer_sign_array(X, y)

    n_outer = np.sum(np.array(d_seq) > sample_diam)
    n_inner = len(d_seq) - n_outer

    # tmp = adjust_missing_feature_importances(estimator.feature_importances_, flag, n_outer, n_inner)
    # signs = adjust_missing_feature_importances(signs, flag, n_outer, n_inner)

    # plot_discs(d_seq, tmp, r_pred=150, signs=signs)

    # y_pred_rf = rf.predict_proba(X_test)[:, 1]
    # fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    # print roc_auc_score(y_test, y_pred_rf)

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    # plt.scatter(rf_preds, y_test)
    # plt.show()


if __name__ == '__main__':

    diams = [150, 287, 377, 450]
    sample_diam = 150
    flag = 'n'
    # extract_dataset(diams, sample_diam, flag)
    # split_dataset_for_learning()
    analyze_dataset(diams, sample_diam, flag)
