"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import numpy as np
import torch

from dib.utils.helpers import mean_p_logits

__all__ = ["BASE_LOG", "N_CORR"]
logger = logging.getLogger(__name__)
EPS_MIN = 1e-7
BASE_LOG = 2  # to convert to understandable measure of information
N_CORR = 3
CORR_GROUPS = ["lin", "q", "hid16"]  # "lay3", "hid4lay1"


def mean_std(arr):
    if len(arr) == 0:
        return 0, 0

    if len(arr) == 1:
        return arr[0], 1

    if isinstance(arr[0], torch.Tensor):
        arr = torch.stack(arr, 0)
        return arr.mean(), arr.std()

    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    return mean, std


class NotEnoughHeads(Exception):
    """Raised when the input value is too small"""

    pass


def detach(x):
    try:
        return x.detach()
    except AttributeError:
        return x


def mean_p_logits_parallel(heads, X):
    """
    Compute all the heads in parallel and take the average across n samples in  probablity space.
    """
    if len(heads) == 0:
        return []

    #! NOT IN PARALLEL BECAUSE WAS NOT WORKING WELL : still has to see why can't parallelize
    # in a single GPU
    return [mean_p_logits(head(X)) for head in heads]
    # return [mean_p_logits(out) for out in parallel_apply(heads, [X] * len(heads))]


class SingleInputModuleList(torch.nn.ModuleList):
    """Wrapper around `ModuleList` which takes a single input (i.e. has a forward).
    Useful for `higher` monkey patching, which doesn't work with ModuleList."""

    def forward(self, inp):
        return [m(inp) for m in self]
