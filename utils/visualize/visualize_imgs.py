"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch
from skorch.dataset import unpack_data
from torchvision.utils import make_grid

from dib.utils.helpers import prod, set_seed

__all__ = ["plot_dataset_samples_imgs"]

DFLT_FIGSIZE = (17, 9)


def remove_axis(ax, is_rm_ticks=True, is_rm_spines=True):
    """Remove all axis but not the labels."""
    if is_rm_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if is_rm_ticks:
        ax.tick_params(bottom="off", left="off")


def plot_dataset_samples_imgs(
    dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None, pad_value=1, seed=123, title=None
):
    """Plot `n_samples` samples of the a datset."""
    set_seed(seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack(
        [dataset[random.randint(0, len(dataset) - 1)][0] for i in range(n_plots)], dim=0
    )
    grid = make_grid(img_tensor, nrow=2, pad_value=pad_value)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")

    if title is not None:
        ax.set_title(title, fontsize=18)
