"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import random

import numpy as np
import torch

from .helpers import channels_to_last_dim, indep_shuffle_, prod, ratio_to_int

__all__ = ["RandomMasker", "GetRandomIndcs"]


### INDICES SELECTORS ###
class GetRandomIndcs:
    """
    Return random subset of indices.

    Parameters
    ----------
    min_n_indcs : float or int, optional
        Minimum number of indices. If smaller than 1, represents a percentage of
        points.

    max_n_indcs : float or int, optional
        Maximum number of indices. If smaller than 1, represents a percentage of
        points.

    is_batch_share : bool, optional
        Whether to use use the same indices for all elements in the batch.

    range_indcs : tuple, optional
        Range tuple (max, min) for the indices.
    """

    def __init__(
        self, min_n_indcs=0.1, max_n_indcs=0.5, is_batch_share=False, range_indcs=None
    ):
        self.min_n_indcs = min_n_indcs
        self.max_n_indcs = max_n_indcs
        self.is_batch_share = is_batch_share
        self.range_indcs = range_indcs

    def __call__(self, batch_size, n_possible_points):
        if self.range_indcs is not None:
            n_possible_points = self.range_indcs[1] - self.range_indcs[0]

        min_n_indcs = ratio_to_int(self.min_n_indcs, n_possible_points)
        max_n_indcs = ratio_to_int(self.max_n_indcs, n_possible_points)
        # make sure select at least 1
        n_indcs = random.randint(max(1, min_n_indcs), max(1, max_n_indcs))

        if self.is_batch_share:
            indcs = torch.randperm(n_possible_points)[:n_indcs]
            indcs = indcs.unsqueeze(0).expand(batch_size, n_indcs)
        else:
            indcs = (
                np.arange(n_possible_points)
                .reshape(1, n_possible_points)
                .repeat(batch_size, axis=0)
            )
            indep_shuffle_(indcs, -1)
            indcs = torch.from_numpy(indcs[:, :n_indcs])

        if self.range_indcs is not None:
            # adding is teh same as shifting
            indcs += self.range_indcs[0]

        return indcs


### GRID AND MASKING ###
class RandomMasker(GetRandomIndcs):
    """
    Return random subset mask.

    Parameters
    ----------
    min_nnz : float or int, optional
        Minimum number of non zero values. If smaller than 1, represents a
        percentage of points.

    max_nnz : float or int, optional
        Maximum number of non zero values. If smaller than 1, represents a
        percentage of points.

    is_batch_share : bool, optional
        Whether to use use the same indices for all elements in the batch.
    """

    def __init__(self, min_nnz=0.01, max_nnz=2 / 9, is_batch_share=False):
        super().__init__(
            min_n_indcs=min_nnz, max_n_indcs=max_nnz, is_batch_share=is_batch_share
        )

    def __call__(self, batch_size, mask_shape, **kwargs):
        n_possible_points = prod(mask_shape)
        nnz_indcs = super().__call__(batch_size, n_possible_points, **kwargs)

        if self.is_batch_share:
            # share memory
            mask = torch.zeros(n_possible_points).bool()
            mask = mask.unsqueeze(0).expand(batch_size, n_possible_points)
        else:
            mask = torch.zeros((batch_size, n_possible_points)).bool()

        mask.scatter_(1, nnz_indcs, 1)
        mask = mask.view(batch_size, *mask_shape).contiguous()

        return mask
