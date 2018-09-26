#!/usr/bin/env python3

import sys
import csv
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt

import warnings
# Couldn't git rid of the sklearn warning the prefered way
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore", category=PendingDeprecationWarning)
#    from sklearn.neural_network import MLPClassifier
# So had to use a bigger hammer
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.externals import joblib


# Many additional classifiers at
# http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
# Also at least the Deep Neural Networks section at http://scikit-learn.org/stable/related_projects.html#related-projects
# Nice comparison of classifiers at
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py


def read_data(fn):
    dat = []
    header = []

    with open(fn, newline='') as csvfile:
        #reader = csv.DictReader(csvfile)
        reader = csv.reader(csvfile)

        off = int(next(reader)[0])
        header = next(reader)
        dat = [row for row in reader]

    #print(header)

    return dat, off, header


def prep_data(dat, off, train_test_ratio=0.90, scale_data=True):
    # May be able to sklearn.model_selection.train_test_split()

    dat = list(map(lambda row: tuple(row[2:]), dat))  # Drop the X,Y position and convert to tuple to make keys for set()

    n1 = len(dat)
    dat = list(set(dat))
    n2 = len(dat)
    print('Number of rows after removing duplicates {:d} / {:d} = {:.2f}%'.format(n2, n1, n2 / n1 * 100.))

    shuffle(dat)

    #dat = dat[:5000]

    #print(dat[0])

    off_w = (off * 2 + 1) ** 2

    dat_int = np.array(list(map(lambda row: tuple(map(int, row[:7])), dat)))
    dat_float = np.array(list(map(lambda row: tuple(map(float, row[7:])), dat)))

    y1 = dat_int[:, 0]  # Select only named category
    y2 = dat_int[:, 1:1+6]  # Select only the binary categories
    # If selecting the binary categories, one may want to drop columns associated with unused categories

    #X = dat # Keep channels: RGB, HSV, LAB, YCrCb
    X = dat_float[:, 2 * (off_w * 3) : 3 * (off_w * 3)] # Keep channels:           LAB

    # print(dat[0])
    # print(y1[0])
    # print(y2[0])
    # print(X[0])

    if scale_data:
        X = ((X / 255.) - .5) * 2.

    ntrain = int(train_test_ratio * len(dat))
    train_y1, test_y1 = y1[:ntrain], y1[ntrain:]
    train_y2, test_y2 = y2[:ntrain], y2[ntrain:]
    train_X, test_X = X[:ntrain], X[ntrain:]

    return train_y1, train_y2, train_X, test_y1, test_y2, test_X


def plot_pca(dat_X, dat_y, ncats, ndims, standardize=False):
    # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    pca = decomposition.PCA(n_components=None, copy=True, whiten=False)

    if standardize:
        dat_X = StandardScaler().fit_transform(dat_X)

    proj = pca.fit_transform(dat_X)

    print(pca.n_components_)
    print(pca.explained_variance_ratio_)

    f, axes = plt.subplots(ndims, ndims, sharex=True, sharey=True)
    for i in range(ndims):
        for j in range(ndims):
            for k in range(ncats):
                x = proj[np.where(dat_y == k), i]
                y = proj[np.where(dat_y == k), j]
                axes[j][i].scatter(x, y, s=.4)
    plt.show()


def train_neural_network(train_y, train_X, nh=()):
    print('Hidden layer size:', nh)

    # Explanation of parameters at
    # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=nh,
                        activation='relu',
                        solver='adam',
                        alpha=1e-5,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=.001,
                        power_t=.5,
                        max_iter=100000,
                        shuffle=True,
                        random_state=None,
                        tol=1e-4,
                        verbose=True,
                        warm_start=False,
                        momentum=.9,
                        nesterovs_momentum=True,
                        early_stopping=False,
                        validation_fraction=.1,
                        beta_1=.9,
                        beta_2=.999,
                        epsilon=1e-8
                        )

    do_incremental_training = False

    if not do_incremental_training:
        res = clf.fit(train_X, train_y)
    else:
        classes = np.unique(train_y)
        classes = [0, 1, 2, 3, 4, 5]
        print(classes)
        n = 50
        for i in range(0, len(train_X)-n, n):
            res = clf.partial_fit(train_X[i:i+n], train_y[i:i+n], classes)

    print(res)

    return clf


def simple_apply_nn(clf, dat_X):
    pre = clf.predict(dat_X)

    coefs = np.array(clf.coefs_)
    intercepts = np.array(clf.intercepts_)
    res = dat_X @ coefs[0]

    print(dat_X.shape)
    print(coefs.shape, res.shape)
    print(res[:5])

    # This is not right. There can be more than one positive value. Only the largest is the assigned class.
    print(pre[:5])
    # print((res[:5] > intercepts) * 1)
    print(((res[:5] + intercepts) > 0) * 1)


def test_nn_model(clf, name, dat_y, dat_X):
    print('{:s}_y.shape:'.format(name), dat_y.shape)
    print('{:s}_X.shape:'.format(name), dat_X.shape)

    prob = clf.predict_proba(dat_X)
    doubt = np.where(np.any(np.logical_and(.05 <= prob, prob <= .95), axis=1))[0]
    print('{:s} in doubt: {:d} / {:d} = {:.2f}%'.format(name, doubt.shape[0], len(dat_X), doubt.shape[0] / len(dat_X) * 100.))
    print(prob[doubt])

    accuracy = clf.score(dat_X, dat_y) * 100.
    print('{:s} accuracy (score) = {:.2f}%'.format(name, accuracy))
    pre = clf.predict(dat_X)
    errors = np.any(dat_y != pre, axis=1)
    ne = errors.sum()
    error = ne / len(dat_y) * 100.
    print('{:s} prediction error: {:d} / {:d} = {:.2f}%'.format(name, ne, len(dat_y), error))


def save_nn_model(fn, clf, off):
    coefs = np.array(clf.coefs_)
    intercepts = np.array(clf.intercepts_)

    with open(fn, 'w') as f:
        print('{:d} {:d} {:d}'.format(off, *coefs[0].transpose().shape), file=f)
        np.savetxt(f, coefs[0].transpose())
        np.savetxt(f, intercepts.transpose())

    joblib.dump(clf, '{}.pkl'.format(fn))
    #clf = joblib.load('{}.pkl'.format(fn))


def train_svc(train_y, train_X):
    # Explanation of parameters at
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    clf = SVC(C=1.0,
              kernel='rbf',
              degree=3,
              gamma='auto',
              coef0=0.0,
              probability=True,
              shrinking=True,
              tol=1e-3,
              cache_size=200,
              class_weight='balanced',
              verbose=True,
              max_iter=-1,
              decision_function_shape='ovr',
              random_state=None)

    res = clf.fit(train_X, train_y)

    print(res)

    return clf


def test_svc_model(clf, name, dat_y, dat_X):
    print('{:s}_y.shape:'.format(name), dat_y.shape)
    print('{:s}_X.shape:'.format(name), dat_X.shape)

    if clf.probability: # or should clf.get_params()['probability'] be used?
        prob = clf.predict_proba(dat_X)
        doubt = np.where(np.any(np.logical_and(.05 <= prob, prob <= .95), axis=1))[0]
        print('{:s} in doubt: {:d} / {:d} = {:.2f}%'.format(name, doubt.shape[0], len(dat_X), doubt.shape[0] / len(dat_X) * 100.))
        print(prob[doubt])

    accuracy = clf.score(dat_X, dat_y) * 100.
    print('{:s} accuracy (score) = {:.2f}%'.format(name, accuracy))
    pre = clf.predict(dat_X)
    errors = dat_y != pre
    ne = errors.sum()
    error = ne / len(dat_y) * 100.
    print('{:s} prediction error: {:d} / {:d} = {:.2f}%'.format(name, ne, len(dat_y), error))
    print('Num support vectors:', clf.n_support_)

    # decision_function(X) 	Distance of the samples X to the separating hyperplane.
    # Could decision_function be used to find and count points that might be at risk of being misclassified?
    # print(clf.decision_function(dat_X))


def train_linear_svc(train_y, train_X):
    # Explanation of parameters at
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    clf = LinearSVC(C=1.0,
                    class_weight=None,
                    dual=True,
                    fit_intercept=True,
                    intercept_scaling=1,
                    loss='squared_hinge',
                    max_iter=10000,
                    multi_class='ovr',
                    penalty='l2',
                    random_state=None,
                    tol=0.0001,
                    verbose=True)

    res = clf.fit(train_X, train_y)

    print(res)

    return clf


def test_linear_svc_model(clf, name, dat_y, dat_X):
    print('{:s}_y.shape:'.format(name), dat_y.shape)
    print('{:s}_X.shape:'.format(name), dat_X.shape)

    accuracy = clf.score(dat_X, dat_y) * 100.
    print('{:s} accuracy (score) = {:.2f}%'.format(name, accuracy))
    pre = clf.predict(dat_X)
    errors = dat_y != pre
    ne = errors.sum()
    error = ne / len(dat_y) * 100.
    print('{:s} prediction error: {:d} / {:d} = {:.2f}%'.format(name, ne, len(dat_y), error))
#    print('Num support vectors:', clf.n_support_)

    # decision_function(X) 	Distance of the samples X to the separating hyperplane.
    # Could decision_function be used to find and count points that might be at risk of being misclassified?
    # print(clf.decision_function(dat_X))


def main():
    if len(sys.argv) != 3:
        print('usage: {:s} <data filename> <final model filename>')
        sys.exit(1)

    dat_fn = sys.argv[1]
    model_fn = sys.argv[2]

    do_plot_pca = False
    do_train_neural_network = False
    do_train_svc = False
    do_train_linear_svc = True

    np.set_printoptions(precision=2, suppress=True, floatmode='fixed', linewidth=2000)

    dat, off, header = read_data(dat_fn)
    train_y1, train_y2, train_X, test_y1, test_y2, test_X = prep_data(dat, off)

    if do_plot_pca:
        plot_pca(train_X, train_y1, 6, 8)

    if do_train_neural_network:
        # nh = ( 10,)
        # nh = ( (train_y.shape[1] + train_X.shape[1]),)
        # nh = ( (train_y.shape[1] + train_X.shape[1]) // 2, )
        # nh = ( (train_y.shape[1] + train_X.shape[1]) // 4, )
        nh = ()
        print(train_y2.shape, train_X.shape)
        clf = train_neural_network(train_y2, train_X, nh)

        test_nn_model(clf, 'Train', train_y2, train_X)
        test_nn_model(clf, 'Test', test_y2, test_X)

        #simple_apply_nn(clf, test_X)

        save_nn_model(model_fn, clf, off)

        if False:
            print('Coefficient:')
            for c in clf.coefs_:
                print(c)
            print('Intercepts:')
            for c in clf.intercepts_:
                print(c)

    if do_train_svc:
        print(train_y2.shape, train_X.shape)
        clf = train_svc(train_y1, train_X)

        test_svc_model(clf, 'Train', train_y1, train_X)
        test_svc_model(clf, 'Test', test_y1, test_X)

    if do_train_linear_svc:
        print(train_y2.shape, train_X.shape)
        clf = train_linear_svc(train_y1, train_X)

        test_linear_svc_model(clf, 'Train', train_y1, train_X)
        test_linear_svc_model(clf, 'Test', test_y1, test_X)


if __name__ == '__main__':
    main()
