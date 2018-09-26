#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

fn= sys.argv[1]


# The hinton(...) functio is from
# https://matplotlib.org/gallery/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


with open(fn) as f:
    nchannels = 3
    off, ncats, ncoefs = tuple(map(int, next(f).split()))
    coefs = [np.fromstring(next(f), dtype='float', sep=' ') for i in range(ncats)]
    intercepts = [float(next(f)) for i in range(ncats)]
    print('intercepts:', intercepts)
    
    f, axes = plt.subplots(nchannels, ncats, sharex=True, sharey=True)

    max_weight = 2 ** np.ceil(np.log(np.abs(coefs).max()) / np.log(2))
    print('max_weight:', max_weight)

    for i in range(ncats):
        for j in range(nchannels):
            channel = coefs[i][j::nchannels]
            channel.shape = ((off * 2 + 1), (off * 2 + 1))
            #axes[j, i].matshow(channel)
            hinton(channel, max_weight=max_weight, ax=axes[j, i])

    plt.show()
