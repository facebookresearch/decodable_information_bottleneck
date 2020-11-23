"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import torch.nn as nn

BATCHNORMS = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]


def get_norm_layer(norm_layer, dim=2):
    """Return the correct normalization layer.

    Parameters
    ----------
    norm_layer : callable or {"batchnorm", "identity"}
        Layer to return.

    dim : int, optional
        Number of dimension of the input (e.g. 2 for images).    
    """
    if norm_layer is None:
        return None
    elif "batch" in norm_layer:
        Norm = BATCHNORMS[dim]
    elif norm_layer == "identity":
        Norm = nn.Identity
    elif isinstance(norm_layer, str):
        raise ValueError(f"Uknown normal_layer={norm_layer}")
    else:
        Norm = norm_layer
    return Norm
