"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from skorch.utils import to_numpy

from dib.utils.helpers import batch_flatten, batch_unflatten, get_activation, tmp_seed
from dib.utils.initialization import (
    linear_init,
    set_linear_like_,
    set_normlayer_like_,
    weights_init,
)
from dib.utils.pruning import RandomUnstructured, global_unstructured, is_pruned, remove

from .helper_layers import get_norm_layer

__all__ = ["MLP"]

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int or list, optional
        Number of hidden neurones. If list, `n_hidden_layers` will be `len(n_hidden_layers)`.

    n_hidden_layers: int, optional
        Number of hidden layers.

    activation: callable, optional
        Activation function. E.g. `nn.RelU()`.

    is_bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.

    is_force_hid_larger : bool, optional
        Whether to force the hidden dimension to be larger or equal than in or out.

    n_skip : int, optional
        Number of layers to skip with residual connection

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.

    k_prune : int, optional
        Number times to apply 50% pruning on all weight matrices besides the last one.

    seed : int, optional
        Random seed, only used when pruning

    previous_mlp : MLP, optional
        Previous MLP to use as initialization. All the layers in common must have the same shapes.

    is_rectangle : bool, optional
        Wether to use a rectangle scheme for the MLP. If True, uses 
        n_hidden_layers=hidden_size*2**n_hidden_layers.

    is_plot_activation : bool, optional
        Whether to store all activations for plotting

    is_mult_hid_input : bool, optional
        Whether the hidden is a factor that should be multiplied by `input_size` rather than an 
        absolute number.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.LeakyReLU(),
        is_bias=True,
        dropout=0,
        is_force_hid_larger=False,
        n_skip=0,
        norm_layer="identity",
        k_prune=0,
        seed=123,
        previous_mlp=None,
        is_rectangle=False,
        is_plot_activation=False,
        is_mult_hid_input=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.n_skip = n_skip
        self.norm_layer = get_norm_layer(norm_layer, dim=1)
        self.k_prune = k_prune
        self.seed = seed
        self._to_plot_activation = dict()
        self.is_rectangle = is_rectangle
        self.is_plot_activation = is_plot_activation
        self.is_mult_hid_input = is_mult_hid_input

        if self.n_hidden_layers == 0:
            self.hidden_size = []
            self.to_hidden = nn.Linear(
                self.input_size, self.output_size, bias=is_bias)
            self.out = nn.Identity()
            self.reset_parameters()
            return

        if self.is_mult_hid_input:
            if isinstance(self.hidden_size, int):
                self.hidden_size = self.hidden_size * self.input_size
            else:
                self.hidden_size = [
                    h * self.input_size for h in self.hidden_size]

        if self.is_rectangle:
            assert isinstance(self.hidden_size, int)
            self.hidden_size = self.hidden_size * (2 ** self.n_hidden_layers)

        if isinstance(self.hidden_size, int):
            if is_force_hid_larger and self.hidden_size < min(
                self.output_size, self.input_size
            ):
                self.hidden_size = min(self.output_size, self.input_size)
                txt = "hidden_size={} smaller than output={} and input={}. Setting it to {}."
                logger.info(
                    txt.format(hidden_size, output_size,
                               input_size, self.hidden_size)
                )

            self.hidden_size = [self.hidden_size] * self.n_hidden_layers
        else:
            self.n_hidden_layers = len(self.hidden_size)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = get_activation(activation)

        self.to_hidden = nn.Linear(
            self.input_size, self.hidden_size[0], bias=is_bias)
        self.tohid_norm = self.norm_layer(self.hidden_size[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_size, out_size, bias=is_bias)
                for in_size, out_size in zip(
                    self.hidden_size[:][:-1], self.hidden_size[1:]
                )
                # dirty [:] because omegaconf does not accept [:-1] directly
            ]
        )
        self.norm_layers = nn.ModuleList(
            [
                self.norm_layer(out_size)
                for _, out_size in zip(self.hidden_size[:][:-1], self.hidden_size[1:])
            ]
        )
        self.out = nn.Linear(
            self.hidden_size[-1], self.output_size, bias=is_bias)

        self.reset_parameters()

        if previous_mlp is not None:
            self.set_parameters_like_(previous_mlp)

        self.prune_weights_(self.k_prune)

    def forward(self, x):
        # flatten to make for normalizing layer => only 2 dim
        x, shape = batch_flatten(x)

        x = self.to_hidden(x)
        self.plot_activation_(dict(tohidout=x))

        if self.n_hidden_layers == 0:
            return batch_unflatten(x, shape)

        x = self.tohid_norm(x)
        self.plot_activation_(dict(tohinorm=x))
        x = self.activation(x)
        x = self.dropout(x)
        old = x

        for i, (linear, norm_layer) in enumerate(
            zip(self.linears, self.norm_layers), start=1
        ):
            x = linear(x)
            self.plot_activation_({f"linout{i}": x})
            x = norm_layer(x)
            self.plot_activation_({f"linnorm{i}": x})
            x = self.activation(x)
            if self.n_skip != 0 and i % self.n_skip == 0:
                # divided by 10 reduces chances of nan at start (making sure order of magnitude doesn't depend on # layers)
                x = old + x / 10
                old = x
            x = self.dropout(x)

        out = self.out(x)
        self.plot_activation_(dict(out=out))

        out = batch_unflatten(out, shape)

        return out

    def reset_parameters(self):
        init = linear_init

        if self.n_hidden_layers == 0:
            init(self.to_hidden)
        else:
            init(self.to_hidden, activation=self.activation)
            for lin in self.linears:
                init(lin, activation=self.activation)
            init(self.out)

    def set_parameters_like_(self, mlp):
        """Given an other MLP that has the same input and output size, set all the parameters in common."""
        assert mlp.input_size == self.input_size and mlp.output_size == self.output_size
        min_layers = min(len(self.linears), len(mlp.linears))

        if self.n_hidden_layers == mlp.n_hidden_layers == 0:
            self.to_hidden = mlp.to_hidden
        elif self.n_hidden_layers != 0 and mlp.n_hidden_layers != 0:
            set_linear_like_(self.to_hidden, mlp.to_hidden)
            set_linear_like_(self.out, mlp.out)

            for i in range(min_layers):
                set_linear_like_(self.linears[i], mlp.linears[i])
                set_normlayer_like_(self.norm_layers[i], mlp.norm_layers[i])
        else:
            logger.info(
                "Cannot use `set_parameters_like` when only one of the 2 mlps have 0 hidden layers."
            )

    def prune_weights_(self, k_prune=1, sparsity_ratio=0.5):
        """Apply in place `k_prune` times a `sparsity_ratio` pruning."""
        outs = [(self.to_hidden, "weight")] + [
            (linear, "weight") for linear in self.linears
        ]
        # don't put the last layer because it depends on whether X or Y as output

        # first make sure that all previous pruning is removed
        for m, name in outs:
            # `remove` does not actually remove, just sets to 0 the weights (fixes the mask)
            # => in case the module was already, pruned, adds some jtter to be sure that not 0 and can learn
            if is_pruned(m):
                remove(m, name)
                with torch.no_grad():
                    m.weight += torch.randn_like(m.weight) / 100

        if sparsity_ratio == 0 or k_prune < 1:
            return

        with tmp_seed(self.seed):
            for k in range(k_prune):
                global_unstructured(
                    outs, pruning_method=RandomUnstructured, amount=sparsity_ratio
                )

    def plot_activation_(self, activations):
        if not self.is_plot_activation:
            return

        for k, v in activations.items():
            # opeartion over batchs
            v = to_numpy(v)
            self._to_plot_activation[k + "_mean"] = v.mean(0)
            self._to_plot_activation[k + "_meanabs"] = np.abs(v).mean(0)

    def tensorboard(self, writer, epoch, mode="on_grad_computed"):
        name = type(self).__name__

        if mode == "on_grad_computed":
            for k, v in self._to_plot_activation.items():
                writer.add_histogram(
                    f"activations/{name}/" + k, v, global_step=epoch)
            self._to_plot_activation = dict()

            writer.add_histogram(
                f"weights/{name}/w_tohid", self.to_hidden.weight, global_step=epoch
            )
            writer.add_histogram(
                f"weights/{name}/b_tohid", self.to_hidden.bias, global_step=epoch
            )

            writer.add_histogram(
                f"grad/{name}/w_tohid", self.to_hidden.weight.grad, global_step=epoch
            )
            writer.add_histogram(
                f"grad/{name}/b_tohid", self.to_hidden.bias.grad, global_step=epoch
            )

            if self.n_hidden_layers != 0:
                for i, lin in enumerate(self.linears):
                    writer.add_histogram(
                        f"weights/{name}/w_lin{i}", lin.weight, global_step=epoch
                    )
                    writer.add_histogram(
                        f"weights/{name}/b_lin{i}", lin.bias, global_step=epoch
                    )
                writer.add_histogram(
                    f"weights/{name}/w_out", self.out.weight, global_step=epoch
                )
                writer.add_histogram(
                    f"weights/{name}/b_out", self.out.bias, global_step=epoch
                )

                for i, lin in enumerate(self.linears):
                    writer.add_histogram(
                        f"grad/{name}/w_lin{i}", lin.weight.grad, global_step=epoch
                    )
                    writer.add_histogram(
                        f"grad/{name}/b_lin{i}", lin.bias.grad, global_step=epoch
                    )
                writer.add_histogram(
                    f"grad/{name}/w_out", self.out.weight.grad, global_step=epoch
                )
                writer.add_histogram(
                    f"grad/{name}/b_out", self.out.bias.grad, global_step=epoch
                )


"""
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, **kwargs):
        super().__init__()

        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]

        hidden_size = 2

        self.to_hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # flatten to make for normalizing layer => only 2 dim
        x, shape = batch_flatten(x)

        out = torch.relu(self.to_hidden(x))
        out = self.out(out)

        out = batch_unflatten(out, shape)

        return out


"""
