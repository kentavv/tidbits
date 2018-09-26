#!/usr/bin/env python3

import sys
import csv
from random import shuffle

import numpy as np

#from sklearn.preprocessing import StandardScaler

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
from sklearn import linear_model

# Many additional classifiers at
# http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
# Also at least the Deep Neural Networks section at http://scikit-learn.org/stable/related_projects.html#related-projects


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
    # chould also use sklearn.model_selection.train_test_split()

    dat = list(map(lambda row: tuple(row[3:]), dat))  # Drop the X,Y position and convert to tuple to make keys for set()

    n1 = len(dat)
    dat = list(set(dat))
    n2 = len(dat)
    print('Number of rows after removing duplicates {:d} / {:d} = {:.2f}%'.format(n2, n1, n2 / n1 * 100.))

    shuffle(dat)

    #dat = dat[:5000]

    #print(dat[0])

    off_w = (off * 2 + 1) ** 2

    y = np.array(list(map(lambda row: tuple(map(int, row[0:6])), dat)))  # Ignore the pixel position and channel id
    # y = np.array(list(map(lambda row: tuple(map(int, row[0:4])), dat))) # Ignore the pixel position and channel id
    # y = np.array(list(map(lambda row: tuple(map(int, row[0:6-2])), dat))) # Ignore the pixel position and channel id and last two unused categories

    # X = np.array(list(map(lambda row: tuple(map(float, row[6+0*(off_w*off_w):6+4*(off_w*off_w)])), dat))) # Keep channels: RGB, HSV, LAB, YCrCb
    X = np.array(list(map(lambda row: tuple(map(float, row[6 + 2 * (off_w * 3):6 + 3 * (off_w * 3)])), dat)))  # Keep channels:           LAB

    #print(dat[0])
    #print(y[0])
    #print(X[0])

    if scale_data:
        X = ((X / 255.) - .5) * 2.
        #print(X[0])
    #exit(1)

    ntrain = int(train_test_ratio * len(dat))
    train_y, test_y = np.array(y[:ntrain]), np.array(y[ntrain:])
    train_X, test_X = np.array(X[:ntrain]), np.array(X[ntrain:])

    return train_y, train_X, test_y, test_X


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


def save_nn_model(fn, clf, off):
    coefs = np.array(clf.coefs_)
    intercepts = np.array(clf.intercepts_)

    with open(fn, 'w') as f:
        print('{:d} {:d} {:d}'.format(off, *coefs[0].transpose().shape), file=f)
        np.savetxt(f, coefs[0].transpose())
        np.savetxt(f, intercepts.transpose())


def test_model(clf, name, dat_y, dat_X):
    print('{:s}_y.shape:'.format(name), dat_y.shape)
    print('{:s}_X.shape:'.format(name), dat_X.shape)

    prob = clf.predict_proba(dat_X)
    doubt = np.where(np.any(np.logical_and(.05 <= prob, prob <= .95), axis=1))[0]
    print('{:s} in doubt: {:d} / {:d} = {:.2f}%'.format(name, doubt.shape[0], len(dat_X), doubt.shape[0] / len(dat_X) * 100.))
    print(prob[doubt])

    accuracy = clf.score(dat_X, dat_y) * 100.
    print('{:s} accuracy = {:.2f}%'.format(name, accuracy))
    pre = clf.predict(dat_X)
    errors = dat_y - pre
    ne = (errors > 0).sum()
    error = ne / len(dat_y) * 100.
    print('{:s} prediction error: {:d} / {:d} = {:.2f}%'.format(name, ne, len(dat_y), error))


def train_bayesian_ridge(train_y, train_X):
    # Explanation of parameters at
    #
    clf = linear_model.BayesianRidge(n_iter=300,
                                     tol=1.e-3,
                                     alpha_1=1.e-6,
                                     alpha_2=1.e-6,
                                     lambda_1=1.e-6,
                                     lambda_2=1.e-6,
                                     compute_score=False,
                                     fit_intercept=True,
                                     normalize=False,
                                     copy_X=True,
                                     verbose=False)

    res = clf.fit(train_X, train_y)

    print(res)

    return clf


def test_bayesian_ridge_model(clf, name, dat_y, dat_X):
    print('{:s}_y.shape:'.format(name), dat_y.shape)
    print('{:s}_X.shape:'.format(name), dat_X.shape)

    prob = clf.predict_proba(dat_X)
    doubt = np.where(np.any(np.logical_and(.05 <= prob, prob <= .95), axis=1))[0]
    print('{:s} in doubt: {:d} / {:d} = {:.2f}%'.format(name, doubt.shape[0], len(dat_X), doubt.shape[0] / len(dat_X) * 100.))
    print(prob[doubt])

    accuracy = clf.score(dat_X, dat_y) * 100.
    print('{:s} accuracy = {:.2f}%'.format(name, accuracy))
    pre = clf.predict(dat_X)
    errors = dat_y - pre
    ne = (errors > 0).sum()
    error = ne / len(dat_y) * 100.
    print('{:s} prediction error: {:d} / {:d} = {:.2f}%'.format(name, ne, len(dat_y), error))


def train_svr(train_y, train_X):
    # Explanation of parameters at
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    clf = svm.SVR(C=1.0,
                  epsilon=0.1,
                  kernel='rbf',
                  degree=3,
                  gamma='auto',
                  coef0=0.0,
                  shrinking=True,
                  tol=1e-3,
                  cache_size=200,
                  verbose=True,
                  max_iter=-1)

    res = clf.fit(train_X, train_y)

    print(res)

    return clf


def test_svr_model(clf, name, dat_y, dat_X):
    print('{:s}_y.shape:'.format(name), dat_y.shape)
    print('{:s}_X.shape:'.format(name), dat_X.shape)

    if clf.probability:
        prob = clf.predict_proba(dat_X)
        doubt = np.where(np.any(np.logical_and(.05 <= prob, prob <= .95), axis=1))[0]
        print('{:s} in doubt: {:d} / {:d} = {:.2f}%'.format(name, doubt.shape[0], len(dat_X), doubt.shape[0] / len(dat_X) * 100.))
        print(prob[doubt])

    accuracy = clf.score(dat_X, dat_y) * 100.
    print('{:s} accuracy (score) = {:.2f}%'.format(name, accuracy))
    pre = clf.predict(dat_X)
    print(pre, dat_y)
    pre.shape=(pre.shape[0],1)
    errors = (dat_y != pre)
    ne = errors.sum()
    error = ne / len(dat_y) * 100.
    print('{:s} prediction error: {:d} / {:d} = {:.2f}%'.format(name, ne, len(dat_y), error))


def main():
    if len(sys.argv) != 3:
        print('usage: {:s} <data filename> <final model filename>')
        sys.exit(1)

    #fn = '2018-09-21_all_all_no_gamma.csv'
    #fn = '2018-09-21_all_two_cats.csv'
    dat_fn = sys.argv[1]
    model_fn = sys.argv[2]

    do_train_neural_network = False
    do_train_bayesian_ridge = True
    do_train_svr = True

    np.set_printoptions(precision=2, suppress=True, floatmode='fixed', linewidth=2000)

    dat, off, header = read_data(dat_fn)
    train_y, train_X, test_y, test_X = prep_data(dat, off)

    if do_train_neural_network:
        # nh = ( 10,)
        # nh = ( (train_y.shape[1] + train_X.shape[1]),)
        # nh = ( (train_y.shape[1] + train_X.shape[1]) // 2, )
        # nh = ( (train_y.shape[1] + train_X.shape[1]) // 4, )
        nh = ()
        clf = train_neural_network(train_y, train_X, nh)

        test_nn_model(clf, 'Train', train_y, train_X)
        test_nn_model(clf, 'Test', test_y, test_X)

        #simple_apply_nn(clf, test_X)

        save_nn_model(model_fn, clf, off)

        if False:
            print('Coefficient:')
            for c in clf.coefs_:
                print(c)
            print('Intercepts:')
            for c in clf.intercepts_:
                print(c)

    if do_train_bayesian_ridge:
        clf = train_bayesian_ridge(train_y, train_X)

        test_bayesian_ridge_model(clf, 'Train', train_y, train_X)
        test_bayesian_ridge_model(clf, 'Test', test_y, test_X)

    if do_train_svr:
        clf = train_svr(train_y1, train_X)

        test_svr_model(clf, 'Train', train_y1, train_X)
        test_svr_model(clf, 'Test', test_y1, test_X)


if __name__ == '__main__':
    main()
