"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from dib.utils.helpers import tmp_seed

__all__ = ["plot_2D_decision_boundary"]

logger = logging.getLogger(__name__)


def make_cmap(cmap_dflt, alpha=1):
    if isinstance(cmap_dflt, list):
        colors = cmap_dflt
    else:
        colors = cmap_dflt(np.linspace(0, 1, 256), alpha=alpha)
    cm = LinearSegmentedColormap.from_list("colormap", colors)
    cm.set_under(alpha=0)
    cm.set_over(alpha=0)
    return cm


def get_sequential_colors(n):
    """
    Return a list of n sequential color maps, the extreme color associated
    with it (or similar color) and a bright similar color.
    """
    assert n <= 10
    # for binary classification same as using plt.cm.RdBu
    cmaps = [
        make_cmap(plt.cm.Blues),
        make_cmap(plt.cm.Reds),
        make_cmap(plt.cm.Greens),
        make_cmap(plt.cm.Purples),
        make_cmap(["white", "xkcd:dark grey"]),
        make_cmap(plt.cm.Oranges),
        make_cmap(["white", "xkcd:olive"]),
        make_cmap(["white", "xkcd:brown"]),
        make_cmap(["white", "xkcd:dark turquoise"]),
        make_cmap(["white", "xkcd:bordeaux"]),
    ]

    extreme_colors = [
        "xkcd:darkish blue",
        "xkcd:darkish red",
        "xkcd:darkish green",
        "xkcd:indigo",
        "xkcd:dark grey",
        "xkcd:dark orange",
        "xkcd:olive",
        "xkcd:brown",
        "xkcd:dark turquoise",
        "xkcd:bordeaux",
    ]

    bright_colors = [
        "xkcd:bright blue",
        "xkcd:bright red",
        "xkcd:green",
        "xkcd:bright purple",
        "k",
        "xkcd:bright orange",
        "xkcd:bright olive",
        "xkcd:golden brown",
        "xkcd:bright turquoise",
        "xkcd:purple red",
    ]

    return cmaps[:n], extreme_colors[:n], bright_colors[:n]


def plot_2D_decision_boundary(
    X,
    y,
    model,
    title=None,
    ax=None,
    n_mesh=50,
    is_only_wrong=False,
    is_force_no_proba=False,
    n_max_scatter=100,
    scatter_unlabelled_kwargs={
        "c": "whitesmoke",
        "alpha": 0.4,
        "linewidths": 0.5,
        "s": 10,
        "marker": "o",
    },
    scatter_labelled_kwargs={"linewidths": 0.7,
                             "s": 50, "marker": "o", "alpha": 0.7, },
    seed=123,
    delta=0.5,
    test=None,
):
    """Plot the 2D decision boundaries of a sklearn classification model.

    Parameters
    ----------
    X: array-like
        2D input data

    y: array-like
        Labels, with `-1` for unlabeled points. Currently works with max 10 classes.

    model: sklearn.BaseEstimator
        Trained model. If `None` plot the dataset only.

    title: str, optional
        Title to add.

    ax: matplotlib.axes, optional
        Axis on which to plot.

    n_mesh: int, optional
        Number of points in each axes of the mesh. Increase to increase the quality. 
        50 is a good valuen for nice quality, 10 is faster but still ok.

    is_only_wrong : bool, optional
        Whether to plot only the wrong data points for simplicity.

    is_force_no_proba : bool, optional
        Whether not to plot probabilistic decision boundaries even if could.
        
    n_max_scatter : int, optional
        Maximum number of points to plot.

    seed : int, optional
        Pseudorandom seed. E.g. for selecting which points to plot.

    delta : float, optional
        How much space to add on the side of each points.

    test : tuple of array like, optional
        (X_test, y_train). If given will plot some test datapoints. Will also plot n max scatter
        of them. Still in dev.
    """
    X = np.array(X)
    y = np.array(y)

    if ax is None:
        F, ax = plt.subplots(1, 1, figsize=(7, 7))

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    if test is not None:
        X_test = np.array(test[0])
        y_test = np.array(test[1])
        x_min = min(x_min, X_test[:, 0].min())
        x_max = min(x_max, X_test[:, 0].max())
        y_min = min(y_min, X_test[:, 1].min())
        y_max = min(y_max, X_test[:, 1].max())

    x_min, x_max = x_min - delta, x_max + delta
    y_min, y_max = y_min - delta, y_max + delta
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num=n_mesh), np.linspace(
            y_min, y_max, num=n_mesh)
    )

    cmaps, extreme_colors, bright_colors = get_sequential_colors(max(y) + 1)
    if model is not None:
        if is_force_no_proba or not hasattr(model, "predict_proba"):
            y_hat = model.predict(
                np.c_[xx.ravel(), yy.ravel()].astype("float32"))
            contourf_kwargs = dict(alpha=1, antialiased=True)
            cmaps = [ListedColormap(extreme_colors)]
            y_hat = y_hat.reshape(xx.shape)
            # contourf does not work well without proba
            plt.pcolormesh(
                xx, yy, y_hat, cmap=ListedColormap(extreme_colors), **contourf_kwargs
            )

        else:
            y_hat = model.predict_proba(
                np.c_[xx.ravel(), yy.ravel()].astype("float32"))
            y_hat = y_hat.reshape(xx.shape + (-1,))
            y_argmax = y_hat.argmax(-1)

            vmin = y_hat.max(-1).min()
            contourf_kwargs = dict(
                vmin=vmin,
                vmax=1,
                extend="neither",
                levels=y_hat.shape[-1],
                antialiased=True,
            )

            for i in range(y_hat.shape[-1]):
                mask_plot = y_argmax == i
                y_i = y_hat[:, :, i]
                y_i[~mask_plot] = np.nan  # don't plot if not predicted
                with warnings.catch_warnings():
                    # warnings because of nan
                    warnings.simplefilter("ignore")
                    ax.contourf(
                        xx, yy, y_hat[:, :, i], cmap=cmaps[i], **contourf_kwargs
                    )

    args_scatter = [
        n_max_scatter,
        model,
        is_only_wrong,
        seed,
        scatter_unlabelled_kwargs,
        scatter_labelled_kwargs,
        bright_colors,
    ]
    ax = plot_scatter_data(X, y, ax, *args_scatter)

    if test is not None:
        scatter_labelled_kwargs["marker"] = "*"
        ax = plot_scatter_data(test[0], test[1], ax, *args_scatter,)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    if title is not None:
        ax.set_title(title)

    return ax


def plot_scatter_data(
    X,
    y,
    ax,
    n_max_scatter,
    model,
    is_only_wrong,
    seed,
    scatter_unlabelled_kwargs,
    scatter_labelled_kwargs,
    bright_colors,
):

    if is_only_wrong:
        y_pred = model.predict(X.astype("float32"))
        wrong_pred = y_pred != np.array(y)

    # randomly select n_max_scatter
    mask_select = np.zeros_like(y).astype(bool)
    mask_select[:n_max_scatter] = True
    with tmp_seed(seed):
        np.random.shuffle(mask_select)

    for i in np.unique(y):
        idx = y == i

        if is_only_wrong:
            idx = np.logical_and(idx, wrong_pred)

        idx = np.logical_and(idx, mask_select)

        if i == -1:
            scatter_kwargs = scatter_unlabelled_kwargs
        else:
            scatter_kwargs = scatter_labelled_kwargs
            scatter_kwargs["c"] = bright_colors[i]

        ax.scatter(X[idx, 0], X[idx, 1], edgecolors="k", **scatter_kwargs)

    return ax
