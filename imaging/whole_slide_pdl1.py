from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import helper.processing as processing
import helper.utils as utils
import helper.learning as learning
import helper.clinical as clinical
import helper.display as display
import helper.metrics as metrics

import texture_analysis

from constants import HOME_PATH, DATA_PATH


def main():
    all_samples = processing.get_list_of_samples()

    combined_features = []
    combined_response = []
    used_samples = []

    # iterate over all slides
    for count, slide in enumerate(all_samples):
        utils.print_progress(count)

        # load slide matrix and decision boundary
        cell_mat = np.load(slide)
        try:
            regions = np.load(slide.split(".npy")[0] + "_seg.npy")
        except IOError:
            print "skipping slide with no region info"
            continue

        # corr_row = texture_analysis.get_tile_correlation(cell_mat, dist=[50])

        # disregard unclassified regions for total area
        tumor_area = np.sum(regions==1) / 1000
        stromal_area = np.sum(regions==0) / 1000
        total_area = tumor_area + stromal_area

        tumor_mask = (regions == 1).astype(int)
        stromal_mask = (regions == 0).astype(int)

        # slide-level cell counts
        n_tumor = np.sum(cell_mat == 1)
        n_pdl1 = np.sum(cell_mat == 2)
        n_any_tumor = n_tumor + n_pdl1
        n_foxp3 = np.sum(cell_mat == 3)
        n_cd8 = np.sum(cell_mat == 4)
        n_cd4 = np.sum(cell_mat == 5)
        n_pdmac = np.sum(cell_mat == 6)
        n_other = np.sum(cell_mat == 7)
        n_macs = np.sum(cell_mat == 8)

        # all_densities = [n_any_tumor, n_foxp3, n_cd8, n_cd4,
        #                             n_pdmac, n_other, n_macs] / total_area
        # all_densities = [n_any_tumor, n_foxp3, n_cd8, n_cd4,
        #                             n_pdmac, n_other, n_macs] / np.sum(cell_mat > 0)
        # all_densities = [n_any_tumor, n_foxp3, n_cd8, n_cd4,
        #                             n_pdmac, n_other, n_macs]

        feature_row = [tumor_area/total_area, stromal_area/total_area, n_tumor, n_pdl1, n_any_tumor,n_foxp3,n_cd8,n_cd4,n_pdmac,n_other,n_macs]

        # # stromal cell counts
        # ns_tumor = np.sum((cell_mat == 1) * stromal_mask)
        # ns_pdl1 = np.sum((cell_mat == 2) * stromal_mask)
        # ns_any_tumor = ns_tumor + ns_pdl1 + 1
        # ns_foxp3 = np.sum((cell_mat == 3) * stromal_mask) + 1
        # ns_cd8 = np.sum((cell_mat == 4) * stromal_mask) + 1
        # ns_cd4 = np.sum((cell_mat == 5) * stromal_mask) + 1
        # ns_pdmac = np.sum((cell_mat == 6) * stromal_mask) + 1
        # ns_other = np.sum((cell_mat == 7) * stromal_mask) + 1
        # ns_macs = np.sum((cell_mat == 8) * stromal_mask) + 1
        #
        # # stromal_densities = [ns_any_tumor, ns_foxp3, ns_cd8, ns_cd4,
        # #                                 ns_pdmac, ns_other, ns_macs] / stromal_area
        # stromal_densities = [ns_any_tumor, ns_foxp3, ns_cd8, ns_cd4,
        #                                 ns_pdmac, ns_other, ns_macs] / np.sum((cell_mat>0)*stromal_area)
        # # stromal_densities = [ns_any_tumor, ns_foxp3, ns_cd8, ns_cd4,
        # #                                 ns_pdmac, ns_other, ns_macs]
        # # in-tumor cell counts
        # nt_tumor = np.sum((cell_mat == 1) * tumor_mask)
        # nt_pdl1 = np.sum((cell_mat == 2) * tumor_mask)
        # nt_any_tumor = nt_tumor + nt_pdl1 + 1
        # nt_foxp3 = np.sum((cell_mat == 3) * tumor_mask) + 1
        # nt_cd8 = np.sum((cell_mat == 4) * tumor_mask) + 1
        # nt_cd4 = np.sum((cell_mat == 5) * tumor_mask) + 1
        # nt_pdmac = np.sum((cell_mat == 6) * tumor_mask) + 1
        # nt_other = np.sum((cell_mat == 7) * tumor_mask) + 1
        # nt_macs = np.sum((cell_mat == 8) * tumor_mask) + 1
        #
        # # tumor_densities = [nt_any_tumor, nt_foxp3, nt_cd8, nt_cd4,
        # #                                 nt_pdmac, nt_other, nt_macs] / tumor_area
        # tumor_densities = [nt_any_tumor, nt_foxp3, nt_cd8, nt_cd4,
        #                                 nt_pdmac, nt_other, nt_macs] / np.sum((cell_mat>0)*tumor_area)
        # # tumor_densities = [nt_any_tumor, nt_foxp3, nt_cd8, nt_cd4,
        # #                                 nt_pdmac, nt_other, nt_macs]


        if n_any_tumor < 100:
            print "skipping slide with {0} tumor cells:".format(n_any_tumor)
            continue
        # if (tumor_area/stromal_area < 0.1) or (stromal_area/tumor_area < 0.1):
        #     print "skipping slide with tumor:stromal area ratio: ", tumor_area / stromal_area
        #     continue


        # feature_row = np.concatenate((stromal_densities, tumor_densities))
        # feature_row = np.concatenate((stromal_densities, tumor_densities, corr_row))
        # feature_row = corr_row

        ratio = n_pdl1 / (n_tumor+n_pdl1)

        combined_features.append(feature_row)
        combined_response.append(ratio)
        used_samples.append(slide)

    # convert feature and response to numpy arrays for analysis
    combined_features = np.vstack(combined_features)
    combined_response = np.array(combined_response)
    combined_features.shape
    # used_samples = [x.replace("processed_orig_seg", "processed") for x in used_samples]

    #### Combine features with clinical information ####

    lookup = clinical.clinical_lookup_table()
    lookup.shape

    feature_names = ["".join(["f", str(x)]) for x in range(combined_features.shape[1])]
    tmp = pd.DataFrame(combined_features, columns=feature_names)
    tmp['response'] = combined_response
    tmp['slide'] = used_samples
    all_data = pd.merge(tmp, lookup)
    all_data.shape


    # create index for patients from 1-n
    ids = list(set(all_data.id))
    id_convert = dict(zip(ids, np.arange(len(ids))))
    all_data['idx'] = [id_convert[str(x)] for x in all_data.id.values]

    # # add logit transformed response variable
    # dtr = learning.VectorTransform(all_data.response)
    # all_data['y*'] = dtr.zero_one_scale().apply('logit')


    all_data.head()
    # fig = display.dotplot(all_data.response, all_data.idx, n_patients=40)
    # fig.savefig(HOME_DIR + '/results/whole_slide/dotplot.png', format='png', dpi=150)

    # display.scatter_hist(all_data.response, all_data.response)
    # display.scatter_hist(all_data['y*'], all_data['y*'], xlims=[-8,8], ylims=[-8,8])


    # for setting aside a holdout set -- however, there are not enough patients to reliably
    # evaluate a holdout set. We rely on repeated k-fold cross validation and (potentially)
    # .632+ bootstrap estimates

    # patients = np.unique(all_data.idx)
    # holdout_patients = np.random.choice(patients, size=int(0.25 * len(patients)), replace=False)
    # holdout_data = all_data.loc[all_data.idx.isin(holdout_patients), :]
    # learn_data = all_data.loc[~all_data.idx.isin(holdout_patients), :]


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


    X_ = all_data.loc[:, feature_names].values
    y_ = all_data['y*'].values
    patient_ids = all_data.idx.values



    X_[:, :14] = np.log(X_[:, :14])


    # sklearn's GroupKFold for grouped cross-validation does not implement shuffling so
    # returns the same train/test sets each call. We attempted taking row-wise permutations
    # of the patient dataframe for each iteration as a workaround, but that did not change
    # the splits, because GroupKFold seems to be using the ordered group values to split.
    # Instead, we shuffle the unique values of the group variable to make different group splits.
    alphas = np.logspace(-2, 3, 20)
    estimator = RidgeCV(alphas=alphas)
    # estimator = RandomForestRegressor(n_estimators=100)
    gcv = GroupKFold(n_splits=10)
    replications = 1
    rep_scores = []
    out_sample = {'pred': [], 'target': [], 'id': []}

    # pca = PCA()

    for iter in range(replications):

        # permute input arrays the same way
        ids_shuffle = utils.unique_value_shuffle(patient_ids)

        # perform grouped cross-validation estimation
        for train, test in gcv.split(X_, y_, ids_shuffle):

            X_train = X_[train]
            X_test = X_[test]
            y_train = y_[train]
            y_test = y_[test]

            # X_train = np.log(X_train)
            # X_test = np.log(X_test)

            scale = StandardScaler()
            X_train = scale.fit_transform(X_train)
            X_test = scale.transform(X_test)

            # X_train = pca.fit_transform(X_train)
            # X_test = pca.transform(X_test)


            preds = estimator.fit(X_train, y_train).predict(X_test)
            preds = dtr.undo(preds)
            y_test = dtr.undo(y_test)

            rep_scores.append(metrics.rmse(preds, y_test))
            # rep_scores.append(estimator.score(X_test, y_test))

            out_sample['pred'].extend(preds)
            out_sample['target'].extend(y_test)
            out_sample['id'].extend(ids_shuffle[test])


    print np.mean(rep_scores), np.std(rep_scores)


    # TEST POLYNOMIAL FEATURES
    # take polynomial combinations of the inputs
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    X_p = np.hstack((X_, 1/X_))
    X_p = poly.fit_transform(X_p)
    X_p = np.log(X_p)
    X_p.shape

    estimator = RidgeCV()
    gcv = GroupKFold(n_splits=10)
    scale = MaxAbsScaler()
    replications = 10
    rep_scores = []
    out_sample = {'pred': [], 'target': [], 'id': []}

    for iter in range(replications):

        # permute input arrays the same way
        ids_shuffle = utils.unique_value_shuffle(patient_ids)

        # perform grouped cross-validation estimation
        for train, test in gcv.split(X_p, y_, ids_shuffle):

            X_train = X_p[train]
            X_test = X_p[test]
            y_train = y_[train]
            y_test = y_[test]

            X_train = scale.fit_transform(X_train)


            X_test = scale.transform(X_test)


            preds = estimator.fit(X_train, y_train).predict(X_test)
            preds = dtr.undo(preds)
            y_test = dtr.undo(y_test)

            rep_scores.append(metrics.rmse(preds, y_test))

            out_sample['pred'].extend(preds)
            out_sample['target'].extend(y_test)
            out_sample['id'].extend(ids_shuffle[test])

    print np.mean(rep_scores), np.std(rep_scores)




    # dict elements from list to array
    for key, value in out_sample.iteritems():
        out_sample[key] = np.array(value)

    metrics.rmse(out_sample['pred'], out_sample['target'])
    fig = display.fixed_scatter(out_sample['pred'], out_sample['target'])


    def aggregate_by_patient(prediction, target, patient_id):
        gdf = pd.DataFrame({'preds': prediction, 'targs': target, 'id': patient_id})
        by_pt_results = gdf.groupby('id').mean()
        return by_pt_results.preds.values, by_pt_results.targs.values

    pt_pred, pt_targ = aggregate_by_patient(out_sample['pred'],
                                            out_sample['target'], out_sample['id'])

    fig = display.fixed_scatter(pt_pred, pt_targ)
    fig.suptitle('Patient level pdl1')
    fig.savefig(HOME_DIR + '/results/whole_slide/patient_sqrt_ratios.png', format='png', dpi=150)


    metrics.rmse(pt_pred, pt_targ)
    metrics.corr(pt_pred, pt_targ)


    plt.scatter(pt_targ - pt_pred, pt_pred)



    # try: converting cell counts to some kind of probability measure as feature
    #   --> correcting cell counts via slide-level cellular density




    # from sklearn.feature_selection import RFECV
    # rfecv = RFECV(estimator=estimator, step=10, cv=10, scoring='mean_squared_error')
    # X_new = rfecv.fit_transform(X_all, y_all)
    # print("Optimal number of features : %d" % rfecv.n_features_)
    # # Plot number of features VS. cross-validation scores
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)





    ##########################################
    ##########################################


    learning_data = all_data.loc[:, feature_names+['response','y*','idx']]

    # create training and test set data split by patient (not slide)
    from sklearn.model_selection import train_test_split
    n_patients = np.max(all_data.idx)
    idx_train, idx_test = train_test_split(range(n_patients), test_size=0.1)
    train_data = learning_data[learning_data.idx.isin(idx_train)]
    test_data = learning_data[learning_data.idx.isin(idx_test)]


    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    X_all = tmp.loc[:, feature_names]
    X_all = np.array(X_all)

    X_all[X_all == 0] = 1
    X_all = np.hstack((X_all, 1/X_all))
    X_all = poly.fit_transform(X_all)

    # separate out train/test arrays by X/y/y*
    X_train, y_train = train_data.loc[:, feature_names], train_data.response
    X_test, y_test = test_data.loc[:, feature_names], test_data.response
    ystar_train, ystar_test = train_data['y*'], test_data['y*']

    # standardize X features using X_train and apply to X_test
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # test regression or learning algorithms
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    estimator = RandomForestRegressor(n_estimators=300, oob_score=True, bootstrap=True)
    estimator = LinearRegression()
    estimator = MLPRegressor( hidden_layer_sizes=(2), max_iter=10000, alpha=0.005, solver='lbfgs',
                            activation='logistic', warm_start=False)
    estimator = ElasticNet(alpha=0.005, l1_ratio=.1, max_iter=10000)


    ##### SLIDE-LEVEL METRICS #####
    X_preds = estimator.fit(X_train, y_train).predict(X_test)
    metrics.rmse(X_preds, y_test)
    metrics.corr(X_preds, y_test)
    display.fixed_scatter(X_preds, y_test)

    # repeat process to test *transformed* response var
    Xstar_preds = estimator.fit(X_train, ystar_train).predict(X_test)

    # undo the transform
    Xstar_restore = dtr.undo(Xstar_preds)
    ystar_test_restore = dtr.undo(ystar_test)
    if not np.allclose(ystar_test_restore, y_test):
        raise ValueError("Undo on transformed response does not match original.")

    metrics.rmse(Xstar_restore, y_test)
    metrics.corr(Xstar_restore, y_test)
    display.fixed_scatter(Xstar_restore, y_test)


    ##### PATIENT-LEVEL METRICS #####
    def aggregate_by_patient(prediction, target, patient_id):
        gdf = pd.DataFrame({'preds': prediction, 'targs': target, 'id': patient_id})
        by_pt_results = gdf.groupby('id').mean()
        return by_pt_results.preds.values, by_pt_results.targs.values

    # test untransformed y-var
    pt_pred, pt_response = aggregate_by_patient(X_preds, y_test, patient_id=test_data.idx)
    metrics.rmse(pt_pred, pt_response)
    metrics.corr(pt_pred, pt_response)
    display.fixed_scatter(pt_pred, pt_response)

    # test transformed y-var
    pt_pred, pt_response = aggregate_by_patient(Xstar_restore, y_test, patient_id=test_data.idx)
    metrics.rmse(pt_pred, pt_response)
    metrics.corr(pt_pred, pt_response)
    display.fixed_scatter(pt_pred, pt_response)


    estimator.coef_
    ################################################################3



    #
    # def apply_boxcox(X):
    #     return np.apply_along_axis(_boxcox_transform, 0, X)
    #
    # def _boxcox_transform(arr):
    #     return stats.boxcox(arr)[0]





    def confusion_matrix(x, y, threshold=0.5):
        q1 = np.sum((x > threshold) & (y > threshold))
        q2 = np.sum((x > threshold) & (y <= threshold))
        q3 = np.sum((x <= threshold) & (y > threshold))
        q4 = np.sum((x <= threshold) & (y <= threshold))
        return np.array([[q1, q2], [q3, q4]])

    confusion_matrix(pt_pred, pt_response, threshold=0.3)




    import helper.metrics as metrics
    tmp = all_data[all_data.STAGE == 4]
    metrics.corr_nan(tmp.response, tmp.radiation)



if __name__ == '__main__':
    main()
