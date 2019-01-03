from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.special import expit

import helper.processing
import helper.display as display
import helper.metrics as metrics
import helper.tile_utils as utils
import helper.learning as learning
from helper.utils import print_progress


from constants import HOME_PATH, DATA_PATH


def extract_dataset(diams, sample_diam, flag):

    np.random.seed(1000)

    # set sampling parameters
    N_SLIDES = 260
    N_SAMPLES = 15

    # set feature extraction parameters
    sample_tile_width = sample_diam
    feature_tile_width = 1
    feature_layers = 75

    # compute other parameters based on input parameters
    scale = int(sample_tile_width / feature_tile_width)
    assert (scale == sample_tile_width / feature_tile_width), "sample_tile_width must be integer multiple of feature_tile_width"
    Nx, Ny = int(1392 / sample_tile_width), int(1040 / sample_tile_width)
    nx, ny = Nx * scale, Ny * scale
    sample_layers = int(np.ceil(feature_layers * feature_tile_width / sample_tile_width))

    # get pre-processed slide matrices and select random sample of slides
    all_samples = helper.processing.get_list_of_samples(DIR)
    SAMPLES = [all_samples[i] for i in np.random.choice(len(all_samples), N_SLIDES, replace=False)]

    # iterate over sampled slides to extract feature and response variables via tile sampling
    combined_features = []
    combined_response = []
    combined_nts = []
    for i, slide in enumerate(SAMPLES):
        print_progress(i)

        # load slide and reshape into tile stacks
        cell_mat = np.load(slide)
        sample_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=sample_tile_width,
                                                   nx=Nx, ny=Ny)
        feature_tile_stack = utils.restack_to_tiles(cell_mat, tile_width=feature_tile_width,
                                                    nx=nx, ny=ny)

        # load tumor edge matrix (skipping slide if no matrix is found)
        try:
            edges = np.load(slide.split(".npy")[0] + "_edges.npy")
            edges_tile_stack = utils.restack_to_tiles(edges, tile_width=sample_tile_width,
                                                      nx=Nx, ny=Ny)
        except IOError:
            print 'No edge matrix. Skipping slide...'
            continue

        # select valid tiles for sampling, skipping slide if no valid tiles are available
        tile_mask = utils.tile_stack_mask(Nx, Ny, L=sample_layers, db_stack=edges_tile_stack)
        if np.sum(tile_mask) == 0:
            print '0 valid samples. Skipping slide...'
            continue

        # uniformly sample tiles from the valid sample space of size n_samples
        sampled_indices = np.random.choice(a=Nx * Ny, size=int(min(N_SAMPLES, np.sum(tile_mask))),
                                           p=tile_mask / np.sum(tile_mask), replace=False)
        sampled_tiles = sample_tile_stack[sampled_indices, :, :]

        # compute response variable over sampled tiles
        response, nts = utils.get_pdl1_response(sampled_tiles, circle=True,
                                                diameter=sample_tile_width, diagnostic=True)

        # compute feature arrays over sampled tiles from neighboring tiles
        feature_rows = np.vstack([utils.get_feature_array(idx, feature_tile_stack, Nx,
                                    scale, feature_layers, flag) for idx in sampled_indices])

        # add outputs to growing array
        combined_response.extend(response)
        combined_features.append(feature_rows)
        combined_nts.extend(nts)

    # convert feature and response to numpy arrays for analysis
    combined_features = np.vstack(combined_features)
    combined_features[np.isnan(combined_features)] = -1
    combined_response = np.array(combined_response)
    combined_nts = np.array(combined_nts)


    # ----- variable processing ----- #

    # # remove all cases with no tumor cells in the sampled tile
    # mask = combined_response == -1
    # combined_response = combined_response[~mask]
    # combined_features = combined_features[~mask, :]

    # alternatively, remove all cases with <K tumor cells in the sampled tile
    print combined_nts.shape, combined_response.shape, combined_features.shape
    mask = combined_nts < 10
    combined_response = combined_response[~mask]
    combined_features = combined_features[~mask, :]


    # aggregate tiles within arbitrary shapes (e.g. discs or squares of increasing size)
    n_obs = combined_features.shape[0]
    side_len = scale + 2 * feature_layers
    n_tiles = side_len ** 2

    if flag == 'n':
        phens = ['tumor','cd4','cd8','foxp3','pdmac','other']
    elif flag == 'a':
        phens = ['tumor','pdl1','cd4','cd8','foxp3','pdmac','other']
    elif flag == 't':
        phens = ['tumor','pdl1']

    phen_columns = []
    for phen in range(len(phens)):    # iterate process over each phenotype
        tmp_tiles = combined_features[:, phen * n_tiles:(phen + 1) * n_tiles]
        tmp_3d = tmp_tiles.reshape(n_obs, side_len, side_len)

        range_columns = []

        d_seq_0 = [0] + d_seq
        for i in range(len(d_seq_0) - 1):
            # utils.print_progress(i)
            print phens[phen], d_seq[i]
            if (flag in ['a','t']) and (phens[phen] in ['tumor','pdl1']) and (d_seq[i] <= sample_tile_width):
                print "skipping."
                continue

            mask = utils.shape_mask(grid_dim=side_len, type='circle',
            S=d_seq_0[i+1], s=d_seq_0[i])

            t = np.sum(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1, 1)
            # sigma = np.std(np.multiply(tmp_3d, mask), axis=(1,2)).reshape(-1,1)
            range_columns.append(t)
            # range_columns.append(sigma)

        per_phen_features = np.hstack(range_columns)
        phen_columns.append(per_phen_features)
    X = np.hstack(phen_columns)

    np.save(STORE_DIR + "data_x", X)
    np.save(STORE_DIR + "data_y", combined_response)



def analyze_dataset(d_seq, sample_diam, flag):

    # remove all cases with no tumor cells in the sampled tile
    mask = y == -1
    y = y[~mask]
    X = X[~mask, :]

    X = np.load(STORE_DIR + "data_x.npy")
    y = np.load(STORE_DIR + "data_y.npy")
    print X.shape, y.shape

    # set aside holdout set here



    # feature_names = ["".join(["f", str(x)]) for x in range(X.shape[1])]
    # feature_names.append('y')
    # feature_names
    # tmp = pd.DataFrame(np.hstack((X, y.reshape(-1,1))))
    # tmp.columns = feature_names
    # tmp.to_csv('local_data.csv', index=False)

    # add logit transformed response variable
    dtr = learning.VectorTransform(y)
    yt = dtr.zero_one_scale().apply('logit')
    plt.hist(yt)


    plt.hist(y)

    plt.hist(np.sqrt(y))
    plt.scatter(X[:,12], y)

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

    print np.mean(rep_scores), np.std(rep_scores)


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

    print "proportion of 0:", sum(y==0) / len(y)
    print "proportion of 1:", sum(y==1) / len(y)

    estimator.fit(X, y)
    print estimator.feature_importances_
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

    diams = [150, 189, 238, 300]
    sample_diam = 150
    flag = 'n'
    extract_dataset(diams, sample_diam, flag)
    # analyze_dataset(diams, sample_diam, flag)
