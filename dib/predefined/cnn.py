"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from dib.utils.helpers import (
    channels_to_2nd_dim,
    channels_to_last_dim,
    make_depth_sep_conv,
)
from dib.utils.initialization import init_param_, weights_init

from .helper_layers import get_norm_layer

logger = logging.getLogger(__name__)

__all__ = ["get_Cnn"]


def get_Cnn(
    dim=2,
    mode="vanilla",
    conv="vanilla",
    block="res",
    normalization=None,
    is_chan_last=True,
    pool=None,
    **kwargs,
):
    """Helper function which returns a cnn, in a way callable by the CLI.
    
    Parameters
    ----------
    dim : int, optional
        Grid input shape.

    mode : {"vanilla", "unet"}, optional

    conv : {"vanilla", "gauss"}, optional

    block : {"simple", "res"}, optional

    normalization : {"batchnorm", None}, optional

    is_chan_last : bool, optional

    pool : {"avg", "max", None}, optional
    
    Returns
    -------
    Cnn : nn.Module
        Unitialized CNN

    kwargs : dict
        Unused kwargs
    """
    if mode == "vanilla":
        Module = CNN
    elif mode == "unet":
        Module = UnetCNN
    elif mode == "nin":
        Module = NIN

    if block == "simple":
        Block = ConvBlock
    elif block == "res":
        Block = ResConvBlock
    elif block == "nin":
        Block = NINBlock
    else:
        Block = ResConvBlock

    Norm = get_norm_layer(normalization, dim=dim)

    if pool == "avg":
        Pool = AVGPOOLINGS[dim]
    elif pool == "max":
        Pool = MAXPOOLINGS[dim]
    elif pool is None:
        Pool = nn.Identity
    elif pool is None:
        Pool = pool

    if conv == "vanilla":
        Conv = CONVS[dim]
    elif conv == "gauss":
        Conv = GAUSSIANCONVS[dim]
    elif conv == "reverse":
        Conv = REVCONVS[dim]
    else:
        Conv = conv

    return partial(
        Module,
        ConvBlock=Block,
        Conv=Conv,
        is_chan_last=is_chan_last,
        Normalization=Norm,
        Pooling=partial(Pool, kernel_size=2),
        **kwargs,
    )


### BLOCKS ###


class ConvBlock(nn.Module):
    """Simple convolutional block with a single layer.

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (uninitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel.

    dilation : int or tuple, optional
        Spacing between kernel elements.

    padding : int or tuple, optional
        Padding added to both sides of the input. If `-1` uses padding that
        keeps the size the same. Currently only works if `kernel_size` is even
        and only takes into account the kenerl size and dilation, but not other
        arguments (e.g. stride).

    activation: callable, optional
        Activation object. E.g. `nn.ReLU`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    Pooling : nn.Module, optional
        Pooling layer to apply at the end of the block. The kernel size should be already defined.

    is_depth_sep ; bool, optional
        Whether to use depth separable convolutions.

    kwargs :
        Additional arguments to `Conv`.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        dilation=1,
        padding=-1,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_depth_sep=False,
        Pooling=nn.Identity,
        **kwargs,
    ):
        super().__init__()
        if Normalization is None:
            Normalization = nn.Identity
        self.activation = activation

        if padding == -1:
            padding = (kernel_size // 2) * dilation
            if kwargs.get("stride", 1) != 1:
                warnings.warn(
                    "`padding == -1` but `stride != 1`. The output might be of different dimension "
                    "as the input depending on other hyperparameters."
                )

        if is_depth_sep:
            Conv = make_depth_sep_conv(Conv)

        self.conv = Conv(in_chan, out_chan, kernel_size,
                         padding=padding, **kwargs)
        self.norm = Normalization(out_chan)
        self.pool = Pooling()

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.norm(self.activation(self.pool(self.conv(X))))


class ResConvBlock(nn.Module):
    """Convolutional block (2 layers) inspired by the pre-activation Resnet [1]
    and depthwise separable convolutions [2].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (uninitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: callable, optional
        Activation object. E.g. `nn.RelU()`.

    Normalization : nn.Module, optional
        Normalization layer (uninitialized). E.g. `nn.BatchNorm1d`.

    Pooling : nn.Module, optional
        Pooling layer to apply at the end of the block. The kernel size should be already defined.

    is_bias : bool, optional
        Whether to use a bias.

    is_depth_sep ; bool, optional
        Whether to use depth separable convolutions.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        Pooling=nn.Identity,
        is_depth_sep=False,
    ):
        super().__init__()
        if Normalization is None:
            Normalization = nn.Identity
        self.activation = activation

        if kernel_size % 2 == 0:
            raise ValueError(
                "`kernel_size={}`, but should be odd.".format(kernel_size))

        padding = kernel_size // 2
        conv_args = (in_chan, in_chan, kernel_size)
        conv_kwargs = dict(padding=padding, bias=is_bias)

        self.norm1 = Normalization(in_chan)

        if is_depth_sep:
            self.conv1 = make_depth_sep_conv(Conv)(*conv_args, **conv_kwargs)
        else:
            self.conv1 = Conv(*conv_args, **conv_kwargs)

        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(*conv_args, groups=in_chan, **conv_kwargs)
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)
        self.pool = Pooling()

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        out = self.conv1(self.activation(self.norm1(X)))
        out = self.conv2_depthwise(self.activation(self.norm2(out)))
        # adds residual before point wise => output can change number of channels
        out = out + X
        out = self.conv2_pointwise(out)
        return self.pool(out)


class NINBlock(torch.nn.Module):
    def __init__(self, chan, dropout, Normalization=nn.Identity, **kwargs):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(chan, chan, 3, 2, padding=1),
            Normalization(chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, chan, 1, 1),
            Normalization(chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, chan, 1, 1),
            Normalization(chan),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.out(x))


### MODULES ###


class NIN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        dropout=0,
        width=2,
        is_flatten=False,
        **kwargs,
    ):
        super().__init__()
        self.chan = nn.Conv2d(in_channels, 96 * width, 1)
        self.layers = nn.Sequential(
            *[NINBlock(96 * width, dropout, **kwargs) for i in range(depth)]
        )
        self.out = nn.Conv2d(96 * width, out_channels, 1)
        self.is_flatten = is_flatten

    def forward(self, x):
        x = torch.relu(self.chan(x))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(self.out(x), (1, 1))
        if self.is_flatten:
            return torch.flatten(x, start_dim=1)
        else:
            return x


class CNN(nn.Module):
    """Simple multilayer CNN.

    Parameters
    ----------
    in_channels : int
        Number of input channels

    out_channels : int
        Number of output channels.

    tmp_channels : int or list, optional
        Number of temporary channels. If integer then uses always the same. If list then needs to 
        be of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a 
        `[ConvBlock(in_channels,16), ConvBlock(16,32), ConvBlock(32,64), ConvBlock(64, out_channels)]`.

    ConvBlock : nn.Module, optional
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    n_blocks : int, optional
        Number of convolutional blocks.

    is_chan_last : bool, optional
        Whether the channels are on the last dimension of the input.

    is_flatten : bool, optional
        Whether to flatten the output.

    is_force_hid_smaller : bool, optional
        Whether to force the hidden channels to be smaller or equal than in and out.
        If not, it forces the hidden channels to be larger or equal than in or out.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        tmp_channels=32,
        ConvBlock=partial(ConvBlock, Conv=nn.Conv2d),
        n_blocks=3,
        is_chan_last=False,
        is_flatten=False,
        is_force_hid_smaller=False,
        **kwargs,
    ):

        super().__init__()
        self.n_blocks = n_blocks
        self.is_chan_last = is_chan_last

        new_tmp_channels = tmp_channels
        if isinstance(tmp_channels, int):
            if is_force_hid_smaller and tmp_channels > max(in_channels, out_channels):
                new_tmp_channels = max(out_channels, in_channels)
                txt = "tmp_channels={} larger than output={} and input={}. Setting it to {}."
                warnings.warn(
                    txt.format(
                        tmp_channels, out_channels, in_channels, new_tmp_channels
                    )
                )
            elif tmp_channels < min(out_channels, in_channels):
                new_tmp_channels = min(out_channels, in_channels)
                txt = "tmp_channels={} smaller than output={} and input={}. Setting it to {}."
                warnings.warn(
                    txt.format(
                        tmp_channels, out_channels, in_channels, new_tmp_channels
                    )
                )
        else:
            n_blocks = len(tmp_channels) + 1

        self.in_out_channels = self._get_in_out_channels(
            in_channels, out_channels, new_tmp_channels, n_blocks
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_chan, out_chan, **kwargs)
                for in_chan, out_chan in self.in_out_channels
            ]
        )
        self.is_return_rep = False  # never return representation for vanilla conv
        self.is_flatten = is_flatten

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, in_channels, out_channels, tmp_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(tmp_channels, int):
            tmp_channels = [tmp_channels] * (n_blocks - 1)
        else:
            tmp_channels = list(tmp_channels)

        assert len(tmp_channels) == (n_blocks - 1), "tmp_channels: {} != {}".format(
            len(tmp_channels), n_blocks - 1
        )

        channel_list = [in_channels] + tmp_channels + [out_channels]

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            X = channels_to_2nd_dim(X)

        X, representation = self.apply_convs(X)

        if self.is_chan_last:
            X = channels_to_last_dim(X)

        if self.is_flatten:
            X = torch.flatten(X, start_dim=1)

        if self.is_return_rep:
            return X, representation

        return X

    def apply_convs(self, X):
        for conv_block in self.conv_blocks:
            X = conv_block(X)
        return X, None


class UnetCNN(CNN):
    """Unet [1].

    Parameters
    ----------
    in_channels : int
        Number of input channels

    out_channels : int
        Number of output channels.

    tmp_channels : int or list, optional
        Number of temporary channels. If integer then uses always the same. If list then needs to 
        be of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a 
        `[ConvBlock(in_channels,16), ConvBlock(16,32), ConvBlock(32,64), ConvBlock(64, out_channels)]`.

    ConvBlock : nn.Module, optional
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    Pool : nn.Module, optional
        Pooling layer (unitialized). E.g. torch.nn.MaxPool1d.

    upsample_mode : {'nearest', 'linear', bilinear', 'bicubic', 'trilinear'}, optional
        The upsampling algorithm: nearest, linear (1D-only), bilinear, bicubic
        (2D-only), trilinear (3D-only).

    max_nchannels : int, optional
        Bounds the maximum number of channels instead of always doubling them at
        downsampling block.

    pooling_size : int or tuple, optional
        Size of the pooling filter.

    factor_chan : float, optional
        The factor by which to multiply the number of channels after each down block. If it's a float
        the number of channels are rounded.

    is_force_same_bottleneck : bool, optional
        Whether to use the average bottleneck for the same functions sampled at
        different context and target. If `True` the first and second halves
        of a batch should contain different samples of the same functions (in order).

    is_return_rep : bool, optional
        Whether to return a summary representation, that corresponds to the
        bottleneck + global mean pooling.

    is_skip_resize : bool, optional
        Whether to skip the resizing steps. Only possible if `in_channels==out_channels==tmp_channels`.

    kwargs :
        Additional arguments to `ConvBlock`.

    References
    ----------
    [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional
        networks for biomedical image segmentation." International Conference on
        Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        tmp_channels=32,
        ConvBlock=partial(ResConvBlock, Conv=nn.Conv2d),
        Pool=nn.AvgPool2d,
        upsample_mode="bilinear",
        max_nchannels=256,
        pooling_size=2,
        factor_chan=2,
        is_force_same_bottleneck=False,
        is_return_rep=False,
        is_skip_resize=False,
        **kwargs,
    ):

        self.is_skip_resize = is_skip_resize
        self.factor_chan = factor_chan
        self.max_nchannels = max_nchannels
        super().__init__(
            in_channels,
            out_channels,
            tmp_channels=tmp_channels,
            ConvBlock=ConvBlock,
            **kwargs,
        )
        self.pooling_size = pooling_size
        self.pooling = Pool(self.pooling_size)
        self.upsample_mode = upsample_mode
        self.is_force_same_bottleneck = is_force_same_bottleneck
        self.is_return_rep = is_return_rep

    def apply_convs(self, X):
        if self.is_skip_resize:
            n_tmp_blocks = self.n_blocks
            start_block = 0
        else:
            n_tmp_blocks = self.n_blocks - 2
            # Input block
            X = self.conv_blocks[0](X)
            start_block = 1

        n_down_blocks = n_tmp_blocks // 2
        residuals = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            X = self.conv_blocks[start_block + i](X)
            residuals[i] = X
            X = self.pooling(X)

        # Bottleneck
        X = self.conv_blocks[n_down_blocks](X)
        # Representation before forcing same bottleneck
        representation = X.view(*X.shape[:2], -1).mean(-1)

        if self.is_force_same_bottleneck and self.training:
            # forces the u-net to use the bottleneck by giving additional information
            # there. I.e. taking average between bottleneck of different samples
            # of the same functions. Because bottleneck should be a global representation
            # => should not depend on the sample you chose
            batch_size = X.size(0)
            batch_1 = X[: batch_size // 2, ...]
            batch_2 = X[batch_size // 2:, ...]
            X_mean = (batch_1 + batch_2) / 2
            X = torch.cat([X_mean, X_mean], dim=0)

        # Up
        for i in range(n_down_blocks + 1, n_tmp_blocks):
            X = F.interpolate(
                X,
                mode=self.upsample_mode,
                scale_factor=self.pooling_size,
                align_corners=True,
            )
            X = torch.cat(
                (X, residuals[n_down_blocks - i]), dim=1
            )  # concat on channels
            X = self.conv_blocks[i + start_block](X)

        if not self.is_skip_resize:
            # Output Block
            X = self.conv_blocks[-1](X)

        return X, representation

    def _get_in_out_channels(self, in_channels, out_channels, tmp_channels, n_blocks):
        """Return a list of tuple of input and output channels for a Unet."""

        if self.is_skip_resize:
            assert in_channels == out_channels == tmp_channels
            n_tmp_blocks = n_blocks
        else:
            n_tmp_blocks = n_blocks - 2  # removes last and first block for this part

        assert n_blocks % 2 == 1, "n_blocks={} not odd".format(n_blocks)
        # e.g. if tmp_channels=16, n_tmp_blocks=5: [16, 32, 64]
        channel_list = [
            round(self.factor_chan ** i * tmp_channels)
            for i in range(n_tmp_blocks // 2 + 1)
        ]
        # e.g.: [16, 32, 64, 64, 32, 16]
        channel_list = channel_list + channel_list[::-1]
        # bound max number of channels by self.max_nchannels
        channel_list = [min(c, self.max_nchannels) for c in channel_list]

        # e.g.: [(16, 32), (32,64), (64, 64), (64, 32), (32, 16)]
        in_out_channels = list(zip(channel_list, channel_list[1:]))

        # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
        idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
        in_out_channels[idcs] = [
            (in_chan * 2, out_chan) for in_chan, out_chan in in_out_channels[idcs]
        ]

        if not self.is_skip_resize:
            # Adds in and out block
            in_out_channels = (
                [(in_channels, tmp_channels)]
                + in_out_channels
                + [(tmp_channels, out_channels)]
            )

        assert len(in_out_channels) == (n_blocks), "in_out_channels: {} != {}".format(
            len(in_out_channels), n_blocks
        )

        return in_out_channels


### CONV ###


class GaussianConv2d(nn.Module):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        assert kernel_size % 2 == 1
        self.kernel_sizes = (kernel_size, kernel_size)
        self.exponent = -(
            (torch.arange(0, kernel_size).view(-1, 1).float() - kernel_size // 2) ** 2
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights_x = nn.Parameter(torch.tensor([1.0]))
        self.weights_y = nn.Parameter(torch.tensor([1.0]))

    def forward(self, X):
        # only switch first time to device
        self.exponent = self.exponent.to(X.device)

        marginal_x = torch.softmax(self.exponent * self.weights_x, dim=0)
        marginal_y = torch.softmax(self.exponent * self.weights_y, dim=0).T

        in_chan = X.size(1)
        filters = marginal_x @ marginal_y
        filters = filters.view(1, 1, *self.kernel_sizes).expand(
            in_chan, 1, *self.kernel_sizes
        )

        return F.conv2d(X, filters, groups=in_chan, **self.kwargs)


# GLOBAL VARIABLES
CONVS = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]
REVCONVS = [None, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
GAUSSIANCONVS = {2: GaussianConv2d}  # at the end because defined in this file
FCONVS = [None, F.conv1d, F.conv2d, F.conv3d]
MAXPOOLINGS = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool2d]
AVGPOOLINGS = [None, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool2d]
