#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pyemd
import scipy.stats

# A comparison between
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
# and
#  https://github.com/wmayner/pyemd

# See the following excellent discussion on differences between pyemd and wasserstein_distance
# https://github.com/scipy/scipy/issues/9152#issuecomment-416229536

# The outcomes of the comparisons is
#   1) The scipy.stats.wasserstein_distance method generates results that make sense and is simple to use.
#   2) Being part of scipy.stats, wasserstein_distance may be more likely to receive development attention.
#   3) The pyemd.emd method is more generic because a complete distance can be provided. But is this often needed?
#   4) The pyemd.emd method is complicated by its genericness.

def main():
    bar_width = 5
    show_plots = True

    # Two samples from same distribution, different means and spread

    a = np.random.randn(100000)
    b = np.random.randn(1000) * 2 - 100



    # Begin scipy.stats.wasserstein_distance method

    nbins = 10

    h1, be1 = np.histogram(a, bins=nbins, density=True)
    h2, be2 = np.histogram(b, bins=nbins, density=True)

    c1 = (be1[:-1] + be1[1:]) / 2
    c2 = (be2[:-1] + be2[1:]) / 2

    # The number of bin edges will be one greater than the number of bins.
    # The number of bins in the returned histogram is equal to the passed bins argument.
    # The number of ben centers will be equal to the number of bins.

    if show_plots:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(c1, h1, align='center', width=bar_width)
        ax.bar(c2, h2, align='center', width=bar_width)
        plt.show()

    emd2 = scipy.stats.wasserstein_distance(c1, c2, h1, h2)
    emd2b = scipy.stats.wasserstein_distance(c2, c1, h2, h1)

    # end method



    # Begin pyemd.emd method

    # Must update nbins to get similar emd measures between methods.
    # Maintaining nbin and using a histogram range that spans two samples results in poor histogram resolution.
    # Then metric matrix must be multiplied by a constant to reflect the distance between bin centers.
    # But the poor resolution of the histograms with the wider range results in a lower than expected emd score.
    m = np.min([np.min(a), np.min(b)])
    M = np.max([np.max(a), np.max(b)])
    nbins = int(round(M - m))

    h1, be1 = np.histogram(a, bins=nbins, range=(m, M), density=True)
    h2, be2 = np.histogram(b, bins=nbins, range=(m, M), density=True)

    if show_plots:
        c1 = (be1[:-1] + be1[1:]) / 2
        c2 = (be2[:-1] + be2[1:]) / 2

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(c1, h1, align='center', width=bar_width)
        ax.bar(c2, h2, align='center', width=bar_width)
        plt.show()

    mgrid = np.meshgrid(np.arange(nbins), np.arange(nbins))
    metric = np.abs(mgrid[0] - mgrid[1]).astype(np.float64)
    # metric /= np.sum(metric)

    # print(metric)

    emd1 = pyemd.emd(h1, h2, metric)
    emd1b = pyemd.emd(h2, h1, metric)

    # end method



    print(emd1, emd1b, 'scipy.stats.wasserstein_distance')
    print(emd2, emd2b, 'pyemd.emd')


if __name__ == "__main__":
    main()
