"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from functools import partial

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dib.predefined import MLP, WideResNet, get_Cnn
from dib.predefined.helper_layers import get_norm_layer
from dib.utils.helpers import prod

__all__ = ["get_img_encoder"]

logger = logging.getLogger(__name__)

# TODO: CLEAN AND DOCUMENT ALL THIS FILE !!!!


class CNNEncoder(nn.Module):
    def __init__(self, x_shape=(1, 32, 32), z_dim=256, **kwargs):
        super().__init__()
        self.core = get_Cnn(is_flatten=True, **kwargs)(x_shape[0], z_dim)

    def forward(self, x):
        return self.core(x)


class MLPEncoder(nn.Module):
    def __init__(self, x_shape=(1, 32, 32), z_dim=256, **kwargs):
        super().__init__()
        self.core = MLP(prod(x_shape), z_dim, **kwargs)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.core(x)


class TorchvisionEncoder(nn.Module):
    def __init__(
        self,
        TVM,
        x_shape=(1, 32, 32),
        n_classes=10,
        z_dim=256,
        norm_layer=None,
        is_resnet_converter=False,
        **kwargs,
    ):
        super().__init__()

        # make every input to TVM has 3 channels
        self.converter = nn.Sequential(
            nn.Conv2d(x_shape[0], 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
        )

        self.tvm = TVM(
            norm_layer=get_norm_layer(norm_layer, dim=2), **kwargs
        )  # will remove the class in any case

        if is_resnet_converter:
            self.converter = nn.Identity()
            self.tvm.conv1 = nn.Conv2d(
                x_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.tvm.maxpool = nn.Identity()

        if z_dim == self.tvm.fc.in_features:
            self.tvm.fc = nn.Identity()
        else:
            self.tvm.fc = nn.Linear(self.tvm.fc.in_features, z_dim)

    def forward(self, x):
        return self.tvm(self.converter(x))


def get_img_encoder(name):
    name = name.lower()
    if "mlp" in name or "width" in name:
        return MLPEncoder
    elif "cnn" in name:
        return CNNEncoder
    elif "resnet18" in name:
        return partial(
            TorchvisionEncoder,
            TVM=torchvision.models.resnet18,
            is_resnet_converter=True,
        )
    elif "resnet34" in name:
        return partial(
            TorchvisionEncoder,
            TVM=torchvision.models.resnet34,
            is_resnet_converter=True,
        )
    elif "resnet50" in name:
        return partial(
            TorchvisionEncoder,
            TVM=torchvision.models.resnet50,
            is_resnet_converter=True,
        )
    elif "resnet101" in name:
        return partial(
            TorchvisionEncoder,
            TVM=torchvision.models.resnet101,
            is_resnet_converter=True,
        )
    elif "wideresnet101" in name:
        return partial(
            TorchvisionEncoder,
            TVM=torchvision.models.wide_resnet101_2,
            is_resnet_converter=True,
        )
    elif "wideresnet" in name:
        return partial(TorchvisionEncoder, TVM=WideResNet)
    else:
        raise ValueError(f"Unkown name={name}")
