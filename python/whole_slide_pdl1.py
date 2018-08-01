from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helper.processing as processing
import helper.utils as utils
import helper.learning as learning
import helper.clinical as clinical
import helper.display as display
import helper.metrics as metrics


def main():
    all_samples = processing.get_list_of_samples()

    combined_features = []
    combined_response = []
    used_samples = []

    for count, slide in enumerate(all_samples):
        utils.print_progress(count)

        cell_mat = np.load(slide)

        # cell counts
        n_tumor = np.sum(cell_mat == 1)
        n_pdl1 = np.sum(cell_mat == 2)
        n_any_tumor = n_tumor + n_pdl1

        n_foxp3 = np.sum(cell_mat == 3)
        n_cd8 = np.sum(cell_mat == 4)
        n_cd4 = np.sum(cell_mat == 5)
        n_pdmac = np.sum(cell_mat == 6)
        n_other = np.sum(cell_mat == 7)

        n_any_stromal = n_foxp3 + n_cd8 + n_cd4 + n_pdmac + n_other
        n_immune = n_foxp3 + n_cd8 + n_cd4 + n_pdmac
        n_any_cell = np.sum(cell_mat != 0)

        if n_any_tumor < 50:

            print "skipping slide with n tumor cells:", n_any_tumor
            continue

        try:
            regions = np.load(slide.split(".npy")[0] + "_seg.npy")
        except IOError:
            print "skipping slide with no region info"
            continue
        tumor_area = np.sum(regions==1)
        stromal_area = np.sum(regions==0)
        classified_area = tumor_area + stromal_area
        # what about boolean relationships between cells in cell_mat and location within db??

        # feature_row = [n_any_tumor,n_foxp3,n_cd8,n_cd4,n_pdmac,n_other]
        feature_row = [n_any_tumor,n_foxp3,n_cd8,n_cd4,n_pdmac,n_other,n_any_stromal,n_immune,n_any_cell,
                       tumor_area,stromal_area]
        # feature_row = [n_any_tumor, n_foxp3, n_cd8, n_cd4, n_pdmac, n_other, tumor_area, stromal_area]
        # feature_row = [n_any_tumor, n_foxp3, n_cd8, n_cd4, n_pdmac, n_other, n2,n3,n4,n5,n6]
        ratio = n_pdl1 / (n_tumor+n_pdl1)

        combined_features.append(feature_row)
        combined_response.append(ratio)
        used_samples.append(slide)

    # convert feature and response to numpy arrays for analysis
    combined_features = np.vstack(combined_features)
    combined_response = np.array(combined_response)


    # load clinical information and merge to combined dataframe
    lookup = clinical.clinical_lookup_table()

    feature_names = ["".join(["f", str(x)]) for x in range(combined_features.shape[1])]
    tmp = pd.DataFrame(combined_features, columns=feature_names)
    tmp['response'] = combined_response
    tmp['slide'] = used_samples
    all_data = pd.merge(tmp, lookup)

    # create index for patients from 1-n
    ids = list(set(all_data.id))
    id_convert = dict(zip(ids, np.arange(len(ids))))
    all_data['idx'] = [id_convert[str(x)] for x in all_data.id.values]

    # add logit transformed response variable
    dtr = learning.VectorTransform(all_data.response)
    all_data['y*'] = dtr.zero_one_scale().apply('logit')
    all_data.head()
    # display.dotplot(all_data.response, all_data.idx, n_patients=40)



    # display.scatter_hist(all_data.response, all_data.response)
    # display.scatter_hist(all_data['y*'], all_data['y*'], xlims=[-8,8], ylims=[-8,8])


    ###########################################

    import statsmodels.api as sm
    sm.qqplot(all_data['y*'], line='45')


    asdf = np.load('C:/Users/fredl/Documents/datasets/EACRI HNSCC/processed/halle validation__2777-13_HP_IM3_0_[19786,13794]_seg.npy')
    plt.imshow(asdf)

    predsss = estimator.fit(X_all, tmp['y*']).predict(X_all)
    plt.scatter(predsss, tmp['y*'] - predsss, c=tmp.f8)
    # df = all_data
    #
    # # df = all_data.loc[all_data.STAGE == 2]
    # xs = df.f0 / max(df.f0)
    # xs = (df.f0 ** 2 +df.f1) / df.f0
    # # xs = learning_data.f0 / learning_data.f1
    # display.scatter_hist(xs / 1000, df['y*'], xlims=[0,1], ylims=[-5,5], colors=df.STAGE)
    #
    # all_data.head()
    #
    # import seaborn as sns
    # sns.swarmplot(x=tmp['STAGE'], y=tmp['y*'])


    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    tmp = all_data
    X_all = tmp.loc[:, feature_names]
    X_all = np.array(X_all)

    # construct polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    X_all = X_all + 1
    X_all = np.hstack((X_all, 1/X_all))
    X_all = poly.fit_transform(X_all)

    X_all.shape
    y_all = tmp.response
    # ystar_all = tmp['y*']
    X_all = MinMaxScaler().fit_transform(X_all)


    estimator = LassoCV(max_iter=10000)
    estimator = RidgeCV()
    estimator = LinearRegression()
    estimator = ElasticNet(alpha=0.02, l1_ratio=.5, max_iter=10000)
    estimator = RandomForestRegressor(n_estimators=300, oob_score=True, bootstrap=True)


    # from sklearn.feature_selection import RFECV
    # rfecv = RFECV(estimator=estimator, step=1, cv=10, scoring='mean_squared_error')
    # X_new = rfecv.fit_transform(X_all, y_all)
    # rfecv.support_
    # X_new.shape
    # print("Optimal number of features : %d" % rfecv.n_features_)
    # # Plot number of features VS. cross-validation scores
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)



    tl = learning.TestLearner(task='regress', transform=None)
    tl.test(estimator, X_all, tmp['y*'], folds=10, rf_oob=False)

    corrs = np.abs(np.array([metrics.corr(X_all[:, x], y_all) for x in range(1,190)]))
    best_vars = np.argsort(corrs)[::-1]
    best_vars = np.argsort(np.abs(estimator.coef_))[::-1]
    top_10 = best_vars[:10]
    tl.test(estimator, X_all[:, best_vars[:10]], y_all, folds=10, rf_oob=False)


    reload(learning)
    X_preds_all = estimator.fit(X_all, ystar_all).predict(X_all)
    X_preds_all_back = dtr.undo(X_preds_all)
    metrics.rmse(X_preds_all_back, y_all)
    metrics.corr(X_preds_all_back, y_all)
    # display.fixed_scatter(X_preds_all, y_all)
    plt.scatter(X_preds_all_back, y_all)

    residuals = y_all - X_preds_all
    plt.scatter(X_preds_all, residuals)

    plt.scatter(X_all.f6, y_all)
    display.scatter_hist(X_all[:,0], X_all[:,6], colors=y_all, xlims=[-3,5], ylims=[-3,5])

    # try: converting cell counts to some kind of probability measure as feature









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
        return by_pt_results.preds, by_pt_results.targs

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
