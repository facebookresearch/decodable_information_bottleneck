"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import math

import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan

__all__ = ["weights_init"]

logger = logging.getLogger(__name__)


def get_min_shape(t1, t2):
    """return component wise minimum shape."""
    return [min(el1, el2) for el1, el2 in zip(t1.shape, t2.shape)]


def set_normlayer_like_(norm, other):
    """Set the param of `norm` using `other` (normalization layers).
    If not the same size, set the largest subset of the weights."""
    assert isinstance(norm, type(other))
    if isinstance(norm, nn.Identity):
        return
    norm_weight = norm.weight.detach()
    other_weight = other.weight.detach()
    rep_shape = get_min_shape(norm_weight, other_weight)
    norm_weight[: rep_shape[0]] = other_weight[: rep_shape[0]]
    norm.weight = nn.Parameter(norm_weight)

    norm_bias = norm.bias.detach()
    other_bias = other.bias.detach()
    rep_shape = get_min_shape(norm_bias, other_bias)
    norm_bias[: rep_shape[0]] = other_bias[: rep_shape[0]]
    norm.bias = nn.Parameter(norm_bias)


def set_linear_like_(linear, other):
    """Set the parameters of `linear` using `other` (linear layers). If not the same size, set the largest subset of the weights."""
    assert isinstance(linear, nn.Linear) and isinstance(other, nn.Linear)
    linear_weight = linear.weight.detach()
    other_weight = other.weight.detach()
    rep_shape = get_min_shape(linear_weight, other_weight)
    linear_weight[: rep_shape[0], : rep_shape[1]] = other_weight[
        : rep_shape[0], : rep_shape[1]
    ]
    linear.weight = nn.Parameter(linear_weight)

    linear_bias = linear.bias.detach()
    other_bias = other.bias.detach()
    rep_shape = get_min_shape(linear_bias, other_bias)
    linear_bias[: rep_shape[0]] = other_bias[: rep_shape[0]]
    linear.bias = nn.Parameter(linear_bias)


def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    # lop over direct children (not grand children)
    for m in module.children():

        # all standard layers
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            try:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            except AttributeError:  # affine = False
                pass

        # if has a specific reset
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()
            #! don't go in grand children because you might have specifc weights you don't want to reset

        # else go in your grand children
        else:
            weights_init(m, **kwargs)


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.SELU: "selu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return v

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def terrible_linear_init(module, **kwargs):
    x = module.weight

    if module.bias is not None:
        nn.init.uniform_(module.bias.data, a=-100, b=100)

    return nn.init.uniform_(x, a=-100, b=100)


def linear_init(module, activation="relu"):
    """Initialize a linear layer.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    try:
        activation_name = get_activation_name(activation)
    except ValueError:
        activation_name = None

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name == "selu":
        fan_in = _calculate_correct_fan(x, "fan_in")
        return torch.nn.init.normal_(x, std=1 / math.sqrt(fan_in))
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))
    else:
        if activation is not None:
            logger.info(
                f"Uknown activation={activation}, using xavier uniform init")
        return nn.init.xavier_uniform_(x)


def init_param_(param, activation=None, is_positive=False, bound=0.05, shift=0):
    """Initialize inplace some parameters of the model that are not part of a
    children module.

    Parameters
    ----------
    param : nn.Parameters:
        Parameters to initialize.

    activation : torch.nn.modules.activation or str, optional)
        Activation that will be used on the `param`.

    is_positive : bool, optional
        Whether to initilize only with positive values.

    bound : float, optional
        Maximum absolute value of the initealized values. By default `0.05` which
        is keras default uniform bound.

    shift : int, optional
        Shift the initialisation by a certain value (same as adding a value after init).
    """
    gain = get_gain(activation)
    if is_positive:
        nn.init.uniform_(param, 1e-5 + shift, bound * gain + shift)
        return

    nn.init.uniform_(param, -bound * gain + shift, bound * gain + shift)
