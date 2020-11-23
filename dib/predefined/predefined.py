"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import torch.nn as nn

from .cnn import get_Cnn
from .mlp import MLP

__all__ = ["get_predefined", "try_get_predefined"]


def try_get_predefined(d, **kwargs):
    """Tries to get a predefined module, given a dicttionary of all arguments, if not returns it."""
    try:
        return get_predefined(**d, **kwargs)
    except TypeError:
        return d


# TO DOC
def get_predefined(name, meta_kwargs={}, **kwargs):
    """Helper function which returns unitialized common neural networks."""
    name = name.lower()

    if name == "cnn":
        Module = get_Cnn(**kwargs)
    elif name == "mlp":
        Module = partial(MLP, **kwargs)
    elif name == "identity":
        Module = nn.Identity
    elif name == "linear":
        Module = partial(nn.Linear, **kwargs)
    elif name is None:
        return None
    elif not isinstance(name, str):
        Module = name
    else:
        raise ValueError(name)

    return Module
