from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from scipy.ndimage.filters import convolve
import cPickle as pickle
from scipy import ndimage
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import helper


def compute_decision_boundary(cell_mat, view=False, n_neighbors=15, slide_id='xxx', remove_blank_regions=True, clean=True, recompute=False):

    try:
        if (view is True) or (recompute is True):
            print "Regenerating decision boundary."
            raise IOError()

        DIR = 'C:/Users/fredl/Documents/datasets/EACRI HNSCC/processed/'
        Z = np.load(slide_id.split(".npy")[0] + "_seg.npy")

        print "Precomputed decision boundary found."

    except IOError:

        all_tumor_loc = np.nonzero((cell_mat == 1) | (cell_mat == 2))
        x_tumor = np.vstack((all_tumor_loc[1], all_tumor_loc[0])).T

        nontumor_loc = np.nonzero((cell_mat > 2))
        x_nontumor = np.vstack((nontumor_loc[1], nontumor_loc[0])).T

        X = np.vstack((x_nontumor, x_tumor))
        y = np.array(x_nontumor.shape[0] * [0] + x_tumor.shape[0] * [1])

        h = 1  # step size in the mesh

        # fit K-nearest neighbor algorithm
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(X, y)

        # Create the decision boundary by predicting the class of each pixel in image
        xx, yy = np.meshgrid(np.arange(0, 1392, h),
                             np.arange(0, 1040, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)


        if remove_blank_regions is True:
            # get matrix of cell locations
            any_cells = (cell_mat != 0)

            # blur matrix using maximum filter and remove regions without cell presence
            filt = ndimage.maximum_filter(any_cells, size=80)
            blanks = (filt == 0)
            Z[blanks] = -1

        if clean is True:
            # clean image of small blobs using median filter
            Z = ndimage.median_filter(Z, size=30)

        np.save(slide_id.split(".npy")[0] + "_seg.npy", Z)

    if view is True:
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # plot the training points (cells)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=15)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Tumor (blue)-nontumor (red) separation with k = %i"
                  % (n_neighbors))
        plt.xlabel(slide_id)

        plt.gca().invert_yaxis()
        plt.show()

    return Z



def map_phenotypes_to_mat(cells):

    translation = {'tumor': 1, 'pdl1': 2, 'foxp3': 3, 'cd8': 4,
                   'cd4': 5, 'pdmac': 6, 'other': 7}
    mat = np.zeros((1040, 1392))

    for key, value in translation.iteritems():
        for _, row in cells[key].iterrows():
            x, y = helper.get_position(row)
            mat[y, x] = value


    return mat



def sample_circles(decision_boundary, cell_mat, r, n_samples, sharp_boundaries=False):

    # uniformly select n_samples elements from within tumor boundaries, return indices of elements
    # (1447680 = 1392 * 1040)
    # p gives the probability of selecting the element, i.e. only where decision_boundary == 1
    indices = np.random.choice(a=1447680, size=n_samples,
                               p=np.ravel(decision_boundary) / np.sum(np.ravel(decision_boundary)))
    # indices = np.random.choice(a=1447680, size=n_samples, p=np.ones(len(np.ravel(decision_boundary)))/1447680)

    # convert indices back to i,j-style
    ii = (indices / 1392).astype(int)
    jj = indices % 1392

    # initialize blank background image
    background = np.zeros((1040, 1392))

    # generate R x R grid containing a circle of radius r of ones
    i,j = np.ogrid[-r: r + 1, -r: r + 1]
    mask = i ** 2 + j ** 2 <= r ** 2
    mask = mask.astype(float)

    circles = []
    identity = np.zeros(n_samples)
    for pt in range(n_samples):

        # pad background image for sampled points near edges
        background_padded = np.pad(background, r, 'constant')

        # apply mask and remove pad
        background_padded[ii[pt]: ii[pt] + 2*r + 1, jj[pt]: jj[pt] + 2*r + 1] = mask
        circle = background_padded[r:-r, r:-r]
        circles.append(circle)

        sub = np.multiply(cell_mat, circle)

        n_tumor = np.sum(sub == 1)
        n_pdl1 = np.sum(sub == 2)

        identity[pt] = n_pdl1 / (n_pdl1 + n_tumor)

    return ii, jj, identity, circles


def extract_features_around_points(ii, jj, R, decision_boundary, cell_mat, Rmin=0, flag='a'):

    assert (len(ii) == len(jj)), "error: i and j coordinates must be arrays of same length"

    # generate R x R grid containing a circle of radius R of ones
    i,j = np.ogrid[-R: R + 1, -R: R + 1]
    mask_max = i ** 2 + j ** 2 <= R ** 2
    mask_min = i ** 2 + j ** 2 <= Rmin ** 2
    mask = mask_max.astype(float) - mask_min.astype(float)

    # initialize blank background image
    background = np.zeros((1040, 1392))
    features = []

    # iterate over sampled points
    for pt in range(len(ii)):
        # pad background for superimposing mask
        background_padded = np.pad(background, R, 'constant')

        # overlay mask onto the sampled point
        background_padded[ii[pt]: ii[pt] + 2*R + 1, jj[pt]: jj[pt] + 2*R + 1] = mask

        # remove pad
        circled_background = background_padded[R:-R, R:-R]

        # restrict this masked background to the stromal region
        # circled_stroma = circled_background * (1 - decision_boundary)
        circled_stroma = circled_background

        # -------- features -------- #

        # stromal area
        area = np.sum(circled_stroma)

        # element-wise multiplication to obtain mat of stromal cells within circle
        stroma_mat = np.multiply(cell_mat, circled_stroma)

        # number of stromal cells within circle
        n_stromal_cells = np.sum(stroma_mat != 0)

        # stromal cell density
        stromal_density = n_stromal_cells / area

        # cell counts
        n_foxp3 = np.sum(stroma_mat == 3)
        n_cd8 = np.sum(stroma_mat == 4)
        n_cd4 = np.sum(stroma_mat == 5)
        n_pdmac = np.sum(stroma_mat == 6)
        n_other = np.sum(stroma_mat == 7)
        n_tumor = np.sum(stroma_mat == 1)
        n_pdl1 = np.sum(stroma_mat == 2)


        ratio_foxp3 = n_foxp3 / n_stromal_cells
        ratio_cd8 = n_cd8 / n_stromal_cells
        ratio_cd4 = n_cd4 / n_stromal_cells
        ratio_pdmac = n_pdmac / n_stromal_cells
        ratio_other = n_other / n_stromal_cells
        ratio_tumor = n_tumor / n_stromal_cells
        ratio_pdl1 = n_pdl1 / n_stromal_cells


        density_foxp3 = n_foxp3 / area
        density_cd8 = n_cd8 / area
        density_cd4 = n_cd4 / area
        density_pdmac = n_pdmac / area
        density_other = n_other / area
        density_tumor = n_tumor / area
        density_pdl1 = n_pdl1 / area

        if flag == 'n':
            features_row = [area, n_stromal_cells, stromal_density,
                            # n_foxp3, n_cd8, n_cd4, n_pdmac, n_other,
                            ratio_foxp3, ratio_cd8, ratio_cd4, ratio_pdmac, ratio_other,
                            density_foxp3, density_cd8, density_cd4, density_pdmac, density_other
                           ]
        elif flag == 'a':
            features_row = [area, n_stromal_cells, stromal_density,
                            # n_foxp3, n_cd8, n_cd4, n_pdmac, n_other, n_tumor, n_pdl1,
                            ratio_foxp3, ratio_cd8, ratio_cd4, ratio_pdmac, ratio_other, ratio_tumor, ratio_pdl1,
                            density_foxp3, density_cd8, density_cd4, density_pdmac, density_other, density_tumor, density_pdl1
                           ]
        elif flag == 't':
            features_row = [# n_tumor, n_pdl1,
                            ratio_tumor, ratio_pdl1,
                            density_tumor, density_pdl1
                           ]
        features.append(features_row)

    return np.array(features)



def visualize_sampling(db=None, cell_mat=None, response_circles=None, feature_circles=None, label=None):

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    if db is not None:
        plt.imshow(db, cmap=cmap_light)

    if cell_mat is not None:

        tumor_loc = np.nonzero(cell_mat == 1)
        pdl1_loc = np.nonzero(cell_mat == 2)
        other_loc = np.nonzero(cell_mat > 2)

        plt.scatter(tumor_loc[1], tumor_loc[0], c='b', edgecolor='k', s=15)
        plt.scatter(pdl1_loc[1], pdl1_loc[0], c='r', edgecolor='k', s=15)
        plt.scatter(other_loc[1], other_loc[0], c='w', edgecolor='k', s=15)


    if response_circles is not None:
        sampled_mat = np.logical_or.reduce(response_circles)
        plt.imshow(sampled_mat, alpha=0.5, cmap='Oranges')

    if feature_circles is not None:
        features_mat = np.logical_or.reduce(feature_circles)
        plt.imshow(features_mat, alpha=0.3, cmap='Greens')


    if label is not None:
        plt.title(label)

        # # Create an alpha channel of linearly increasing values moving to the right.
        # alphas = np.ones(weights.shape)
        # alphas[:, 30:] = np.linspace(1, 0, 70)
        #
        # # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
        # colors = Normalize(vmin, vmax, clip=True)(weights)
        # colors = cmap(colors)
        #
        # # Now set the alpha channel to the one we created above
        # colors[..., -1] = alphas
        #
        # # Create the figure and image
        # # Note that the absolute values may be slightly different
        # fig, ax = plt.subplots()
        # ax.imshow(greys)
        # ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))
        # ax.set_axis_off()

        # tumors = np.copy(cell_mat)
        # tumors[(tumors != 0) & (tumors != 1) & (tumors != 2)] = 0
        # plt.imshow(tumors, alpha=0.4)
        # # plt.scatter(jvec, ivec)
    plt.show()
        # plt.matshow(db); ; plt.show()



if __name__ == '__main__':
    np.random.seed(40)
    all_samples = helper.processing.get_list_of_samples()
    SAMPLES = [all_samples[i] for i in np.random.choice(len(all_samples), 100)]

    combined_features = []
    combined_response = []

    counter = 0
    for slide in SAMPLES:
        print counter; counter += 1

        # cells = helper.load_sample(DIR + slide, confidence_thresh=0.3, verbose=False, radius_lim=0)
        cell_mat = np.load(slide)

        # db = compute_decision_boundary(slide=DIR + slide, cells=cells, view=False, n_neighbors=19)
        db = compute_decision_boundary(cell_mat, view=False, n_neighbors=25, slide_id=slide, recompute=False, remove_blank_regions=True)

        # skip slide if too little or too much tumor area
        tumor_prop = np.sum(db == 1) / (1392 * 1040)
        if (tumor_prop < 0.1) or (tumor_prop > 0.9):
            continue

        # visualize_sampling(db=db, cell_mat=cell_mat)

        # further restrict sampling to within 200px distance of decision boundary
        # db = ndimage.median_filter(db, size=40)     # first clean up boundary image
        # db = ndimage.morphology.binary_opening(db, structure=np.ones((50,50))).astype(np.int)
        # db = ndimage.gaussian_filter(db, sigma=2, truncate=3)


        # tmp = ndimage.minimum_filter(db, size=200) + db
        # region = (tmp == 1)
        # db[~region] = 0
        # db[region] = 1
        #
        # visualize_sampling(db=db, cell_mat=cell_mat)

        # the number of samples from a slide should be proportional to its tumor boundary area
        # n_samp_scaled = max((np.sum(db) / 10000).astype(int), 3)
        n_samp_scaled = 20

        R_vec = [50, 80, 120, 150, 200, 250, 300]
        ivec, jvec, identity, circles = sample_circles(decision_boundary=db, cell_mat=cell_mat, r=150,
                                          n_samples=n_samp_scaled)

        # visualize_sampling(db=db, cell_mat=cell_mat, response_circles=circles, label=slide)
        # visualize_sampling(db=db)
        # import sys
        # sys.exit()

        features0 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[0], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=0, flag='n')
        features1 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[1], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=R_vec[0], flag='n')
        features2 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[2], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=R_vec[1], flag='n')

        features3 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[3], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=R_vec[2], flag='n')

        features4 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[4], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=R_vec[3], flag='n')
        features5 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[5], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=R_vec[4], flag='n')
        features6 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[6], decision_boundary=db,
                                                  cell_mat=cell_mat, Rmin=R_vec[5], flag='n')
        # features7 = extract_features_around_points(ii=ivec, jj=jvec, R=R_vec[7], decision_boundary=db,
        #                                           cell_mat=cell_mat, Rmin=R_vec[6], flag='a')

        # features = features1
        features = np.hstack((features0, features1, features2, features3, features4, features5,features6))

        # features = np.hstack((features0, features1, features2, features3, features4, features5))
        features = np.nan_to_num(features)


        response = np.array(identity)
        response = np.nan_to_num(response)

        combined_features.append(features)
        combined_response.extend(response)

    combined_features = np.vstack(combined_features)
    combined_response = np.array(combined_response)

    #
    # output = open('features.pkl','wb')
    # output2 = open('response.pkl','wb')
    # pickle.dump(combined_features, output)
    # pickle.dump(combined_response, output2)
    # output.close()
    # output2.close()
    #

    # output = open('features.pkl', 'rb')
    # output2 = open('response.pkl', 'rb')
    # combined_features = pickle.load(output)
    # combined_response = pickle.load(output2)
    # output.close()
    # output2.close()

    tmp = np.copy(combined_response)
    # combined_response[tmp <= 0.15] = 0
    # combined_response[(tmp > 0.15) & (tmp <= 0.85)] = 1
    # combined_response[(tmp > 0.25) & (tmp <= 0.75)] = 2
    # combined_response[tmp > 0.15] = 1

    X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_response,
                                                        test_size=0.5)

    # X_test = X_train
    # y_test = y_train

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.linear_model import LinearRegression
    # from sklearn.svm import SVC
    # from sklearn.cross_validation import KFold
    # from sklearn.grid_search import GridSearchCV

    # run random forest classification
    # rf = RandomForestRegressor()
    # rf = LogisticRegression(penalty='l1', class_weight='balanced')
    # rf = LinearRegression()
    # rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', oob_score=True)

    rf = RandomForestRegressor(n_estimators=300, oob_score=True)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # print "feature importances:", rf.feature_importances_
    # print "n_features:", rf.n_features_
    print "oob_score:", rf.oob_score_

    # print "area, n_stromal_cells, stromal_density, n_foxp3, n_cd8, "
    # print "n_cd4, n_pdmac, n_other, n_tumor, n_pdl1,"
    # print "ratio_foxp3, ratio_cd8, ratio_cd4, ratio_pdmac, ratio_other,"
    # print "ratio_tumor, ratio_pdl1, density_foxp3, density_cd8, density_cd4,"
    # print "density_pdmac, density_other, density_tumor, density_pdl1"
    # C_range = 10.0 ** np.arange(-3, 3)
    # gamma_range = 10.0 ** np.arange(-3, 3)
    # # epsilon_range = 10.0 ** np.arange(-3, 3)
    # tuned_parameters = dict(kernel=['rbf'], gamma=gamma_range, C=C_range)
    #
    # kf=KFold(len(y_train), n_folds=10, shuffle=True)
    # # rf = GridSearchCV(SVC(), tuned_parameters, cv=kf)


    # import matplotlib.pyplot as plt
    # plt.hist(combined_response, bins=20)
    # plt.show()

    # print rf_preds
    # print y_test
    #
    #
    # print "proportion of 0:", sum(y_test==0) / len(y_test)
    # print "proportion of 1:", sum(y_test==1) / len(y_test)
    # print "proportion of 2:", sum(y_test==2) / len(y_test)
    # print "proportion of 3:", sum(y_test==3) / len(y_test)


    # print sum(rf_preds == y_test) / len(y_test)

    # print "class 0 accuracy: ", sum(rf_preds[y_test == 0] == 0) / len(rf_preds[y_test == 0])
    # print "class 1 accuracy: ", sum(rf_preds[y_test == 1] == 1) / len(rf_preds[y_test == 1])
    # print "class 2 accuracy: ", sum(rf_preds[y_test == 2] == 2) / len(rf_preds[y_test == 2])
    # print "class 3 accuracy: ", sum(rf_preds[y_test == 3] == 3) / len(rf_preds[y_test == 3])


    print helper.corr(rf_preds, y_test)
    print helper.rmse(rf_preds, y_test)

    plt.scatter(rf_preds, y_test)
    plt.show()




    # y_pred_rf = rf.predict_proba(X_test)[:, 1]
    # fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    # print roc_auc_score(y_test, y_pred_rf)
    #
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()



    #
    # from scipy.ndimage.filters import gaussian_filter
    # blurred = gaussian_filter(db.astype(np.float), sigma=1)
    # plt.matshow(db); plt.scatter(jj, ii); plt.show()
    #
    # visualize(image=DIR + SLIDE.replace('_cell_seg_data.txt', '.jpg'), cells=cells, phenotypes='other')
