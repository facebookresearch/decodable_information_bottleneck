"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import math
import operator
import random
import warnings
from functools import reduce
from itertools import zip_longest

import numpy as np
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialization import weights_init


def to_numpy(X):
    """Generic function to convert array like to numpy."""
    if isinstance(X, list):
        X = np.array(X)
    return skorch.utils.to_numpy(X)


def extract_target(targets, map_target_position):
    """Extract the real target."""
    if isinstance(targets, (list, tuple)):
        return targets[map_target_position["target"]]
    else:
        return targets[:, map_target_position["target"]]


class CrossEntropyLossGeneralize(nn.CrossEntropyLoss):
    """Cross entropy loss that forces (anti)-generalization.
    
    Note
    ----
    - we want to find an empirical risk minimizer that maximizes (antigeneralize) or minimizes 
    (generalize) the test loss. Using a lagrangian relaxation of the problem this can be written
    as `min trainLoss + gamma * testLoss`, where the sign of `gamma` determines whether or not to 
    generalize.
    - Each target should contain `(label, is_train)`. Where `is_train` says whether its a trainign 
    example or a test example. `is_train=1` or `is_train=0`.
    - When validating usies cross entropy.

    Parameters
    ----------
    gamma : float, optional
        Langrangian coefficient of the relaxed problem. If positive, forces generalization, if negative
        forces anti generalization. Its scale balances the training and testing loss. If `gamma=0`
        becomes standard cross entropy (in which case doesn't need to append `is_train`).

    map_target_position : dict
        Dictionary that maps the type of target (e.g. "index") to its position in the 
        target. Needs to have `"constant" corresponding to `"is_train"`.

    cap_test_loss : float, optional
        Value used to cap the test loss (i.e. don't backprop through it). This is especially useful
        when gamma is negative (anti generalization). Indeed, cross entropy is not bounded and thus 
        the model could end up only focusing on maximizing the test loss to infinity regardless of 
        train.

    kwargs : 
        Additional arguments to `torch.nn.CrossEntropyLoss`.
    """

    def __init__(
        self, gamma, map_target_position, reduction="mean", cap_test_loss=10, **kwargs
    ):
        super().__init__(reduction="none", **kwargs)
        self.gamma = gamma
        self.map_target_position = map_target_position
        self.final_reduction = reduction
        self.cap_test_loss = cap_test_loss

    def forward(self, inp, targets):
        out = super().forward(inp, extract_target(targets, self.map_target_position))

        if self.gamma == 0 and ("constant" not in self.map_target_position):
            pass
        elif self.training:
            constant = targets[self.map_target_position["constant"]]
            is_test = constant == 0
            is_train = constant == 1
            weights = (is_test.int() * self.gamma) + is_train.int()

            # CAPPING : don't backprop if test and larger than cap (but still forward)
            is_large_loss = out > self.cap_test_loss
            to_cap = is_large_loss & is_test
            out[to_cap] = out[to_cap] * 0 + out[to_cap].detach()

            out = weights * out
        elif len(self.map_target_position) == len(targets):
            # when validating : either you have only access to the validation set, in which
            # case return all or you have access to train U test
            # in which case you want to filter only the training examples
            is_train = targets[self.map_target_position["constant"]] == 1
            out = out[is_train]
        else:
            ValueError(
                f"Not training but len({self.map_target_position})!={len(targets)}"
            )

        if self.final_reduction == "mean":
            return out.mean()
        elif self.final_reduction == "sum":
            return out.sum()
        else:
            raise ValueError(f"Unkown reduction={self.final_reduction}")


class Identity:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x

    def __getitem__(self, x):
        return x


def is_sorted(l):
    """Check whether a list is sorted"""
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1))


def get_idx_permuter(n_idcs, seed=123):
    """Return permuted indices.
    
    Paramaters
    ----------
    n_idcs : int or array-like of int
        Number of indices. If list, it should be a partion of the real number of idcs.
        Each partition will be permuted separately.
        
    seed : int, optional
    """

    if isinstance(n_idcs, int):
        idcs = list(range(n_idcs))
    else:
        idcs = [list(range(partition)) for partition in n_idcs]

    with tmp_seed(seed):
        if isinstance(n_idcs, int):
            random.shuffle(idcs)
            idcs = torch.tensor(idcs)

        else:

            # shuffle each partition separetly
            for partition_idcs in idcs:
                random.shuffle(partition_idcs)

            idcs = torch.cat([torch.tensor(idcs) for idcs in idcs])

    return idcs


# credits : https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def Nsize_chunk_iterable(iterable, n, padding=None):
    """Chunk an iterable into N sized ones."""
    return zip_longest(*[iter(iterable)] * n, fillvalue=padding)


def Nchunk_iterable(iterable, n, padding=None):
    """Chunk an iterable into `n` of them."""
    return Nsize_chunk_iterable(iterable, math.ceil(len(iterable) / n))


def update_dict_copy(d, **updates):
    """Return an updated copy of the dictionary."""
    d = d.copy()
    d.update(updates)
    return d


class BatchNorm1dLast(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        # flatten to make for normalizing layer => only 2 dim
        x, shape = batch_flatten(x)
        x = self.batch_norm(x)
        return batch_unflatten(x, shape)


def wrap_batchnorm(Module):
    # wrap a module by applying batchnorm1d to input
    class BatchNormWrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.batch_norm = BatchNorm1dLast(
                num_features=args[0], affine=False)
            self.module = Module(*args, **kwargs)

        def forward(self, x):
            x = self.batch_norm(x)
            return self.module(x)

    return BatchNormWrapper


def get_activation(activation):
    """Given a string returns a `torch.nn.modules.activation`."""
    if not isinstance(activation, str):
        return activation

    mapper = {
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "relu6": nn.ReLU6(),
        "sin": torch.sin,
    }
    for k, v in mapper.items():
        if activation == k:
            return v

    raise ValueError("Unkown given activation type : {}".format(activation))


class HyperparameterInterpolator:
    """Helper class to compute the value of a hyperparameter at each training step.
    
    Parameters
    ----------
    initial_value: float
        Initial value of the hyperparameter.
        
    final_value: float
        Final value of the hyperparameter.
        
    N_steps_interpolate: int
        Number of training steps before reaching the `final_value`.
        
    Start_step: int, optional
        Number of steps to wait for before starting annealing. During the waiting time,
        the hyperparameter will be `default`.
        
    Default: float, optional
        Default hyperparameter value that will be used for the first `start_step`s. If
        `None` uses `initial_value`.
        
    mode: {"linear", "exponential", "logarithmic"}, optional
        Interpolation mode.
        
    is_restart : bool, optional
        Whether to restart the interpolator after n_steps_interpolate.
        
    """

    def __init__(
        self,
        initial_value,
        final_value,
        n_steps_interpolate,
        start_step=0,
        default=None,
        mode="linear",
        is_restart=False,
    ):

        self.initial_value = initial_value
        self.final_value = final_value
        self.n_steps_interpolate = n_steps_interpolate
        self.start_step = start_step
        self.default = default if default is not None else self.initial_value
        self.mode = mode.lower()
        self.is_restart = is_restart

        if self.mode == "linear":
            delta = self.final_value - self.initial_value
            self.factor = delta / self.n_steps_interpolate
        elif self.mode in ["exponential", "logarithmic"]:
            delta = self.final_value / self.initial_value
            self.factor = delta ** (1 / self.n_steps_interpolate)
        else:
            raise ValueError("Unkown mode : {}.".format(mode))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the interpolator."""
        self.n_training_calls = 0

    @property
    def is_annealing(self):
        return (self.start_step <= self.n_training_calls) and (
            self.n_training_calls <= (
                self.n_steps_interpolate + self.start_step)
        )

    def __call__(self, is_update):
        """Return the current value of the hyperparameter.
        Parameters
        ----------
        Is_update: bool
            Whether to update the hyperparameter.
        """
        if is_update:
            self.n_training_calls += 1

        if self.start_step >= self.n_training_calls:
            return self.default

        n_actual_training_calls = self.n_training_calls - self.start_step

        if self.is_annealing:
            current = self.initial_value

            if self.mode == "linear":
                current += self.factor * n_actual_training_calls

            elif self.mode in ["logarithmic", "exponential"]:

                if (self.mode == "logarithmic") ^ (
                    self.initial_value < self.final_value
                ):
                    current *= self.factor ** n_actual_training_calls
                else:
                    current *= self.factor ** (
                        self.n_steps_interpolate - n_actual_training_calls
                    )
                    current = self.final_value - current
        else:
            if self.is_restart:
                self.reset_parameters()

            current = self.final_value

        return current

    def plot(self, n=None):
        """Plot n steps of interpolation."""
        import matplotlib.pyplot as plt

        if n is None:
            n = self.n_steps_interpolate

        out = [self(True) for _ in range(n)]

        plt.plot(out)

        self.reset_parameters()


def batch_flatten(x):
    """Batch wise flattenting of an array."""
    shape = x.shape
    return x.view(-1, shape[-1]), shape


def batch_unflatten(x, shape):
    """Revert `batch_flatten`."""
    return x.view(*shape[:-1], -1)


def to_number(X):
    """Convert the input to a number."""
    try:
        return X.item()
    except AttributeError:
        return X


def tuple_cont_to_cont_tuple(tuples):
    """Converts a tuple of containers (list, tuple, dict) to a container of tuples."""
    if isinstance(tuples[0], dict):
        # assumes keys are correct
        return {k: tuple(dic[k] for dic in tuples) for k in tuples[0].keys()}
    elif isinstance(tuples[0], list):
        return list(zip(*tuples))
    elif isinstance(tuples[0], tuple):
        return tuple(zip(*tuples))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(tuples[0])))


def cont_tuple_to_tuple_cont(container):
    """Converts a container (list, tuple, dict) of tuple to a tuple of container."""
    if isinstance(container, dict):
        return tuple(dict(zip(container, val)) for val in zip(*container.values()))
    elif isinstance(container, list) or isinstance(container, tuple):
        return tuple(zip(*container))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(container)))


def set_seed(seed):
    """Set the random seed."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


@contextlib.contextmanager
def tmp_seed(seed):
    """Context manager to use a temporary random seed with `with` statement."""
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state()

    set_seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            # if seed is None do as if no tmp_seed
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(torch_cuda_state)


def clip_interval(x, bound1, bound2, warn_mssg=["x", "bound1", "bound2"]):
    """
    Clips x to [bound1,bound2] or [bound2,bound1]. If `warn_mssg` is a list of the 3 name variable 
    names that will be used to to warn the user that a variables was clipped to the given interval 
    (no warning if `None`).
    """
    if bound2 < bound1:
        bound1, bound2 = bound2, bound1
        if warn_mssg is not None:
            warn_mssg[1], warn_mssg[2] = warn_mssg[2], warn_mssg[1]

    def get_txt(to): return "{}={} not in [{}={}] = [{}={}]. Setting it to {}.".format(
        warn_mssg[0], x, warn_mssg[1], warn_mssg[2], bound1, bound2, to
    )

    if x < bound1:
        if warn_mssg is not None:
            warnings.warn(get_txt(bound1))
        return bound1

    if x > bound2:
        if warn_mssg is not None:
            warnings.warn(get_txt(bound1))
        return bound2

    return x


def channels_to_2nd_dim(X):
    """
    Takes a signal with channels on the last dimension (for most operations) and
    returns it with channels on the second dimension (for convolutions).
    """
    return X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))


def channels_to_last_dim(X):
    """
    Takes a signal with channels on the second dimension (for convolutions) and
    returns it with channels on the last dimension (for most operations).
    """
    return X.permute(*([0] + list(range(2, X.dim())) + [1]))


def mask_and_apply(x, mask, f):
    """Applies a callable on a masked version of a input."""
    tranformed_selected = f(x.masked_select(mask))
    return x.masked_scatter(mask, tranformed_selected)


def indep_shuffle_(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply `numpy.random.shuffle` to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.

    Credits : https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])


def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError(
            "percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)


def prod(iterable):
    """Compute the product of all elements in an iterable."""
    return reduce(operator.mul, iterable, 1)


def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min


def clamp(
    x,
    minimum=-float("Inf"),
    maximum=float("Inf"),
    is_leaky=False,
    negative_slope=0.01,
    hard_min=None,
    hard_max=None,
):
    """
    Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
    an optional hard clamping.
    """
    lower_bound = (
        (minimum + negative_slope * (x - minimum))
        if is_leaky
        else torch.zeros_like(x) + minimum
    )
    upper_bound = (
        (maximum + negative_slope * (x - maximum))
        if is_leaky
        else torch.zeros_like(x) + maximum
    )
    clamped = torch.max(lower_bound, torch.min(x, upper_bound))

    if hard_min is not None or hard_max is not None:
        if hard_min is None:
            hard_min = -float("Inf")
        elif hard_max is None:
            hard_max = float("Inf")
        clamped = clamp(x, minimum=hard_min, maximum=hard_max, is_leaky=False)

    return clamped


class ProbabilityConverter(nn.Module):
    """Maps floats to probabilites (between 0 and 1), element-wise.

    Parameters
    ----------
    min_p : float, optional
        Minimum probability, can be useful to set greater than 0 in order to keep
        gradient flowing if the probability is used for convex combinations of
        different parts of the model. Note that maximum probability is `1-min_p`.

    activation : {"sigmoid", "hard-sigmoid", "leaky-hard-sigmoid"}, optional
        name of the activation to use to generate the probabilities. `sigmoid`
        has the advantage of being smooth and never exactly 0 or 1, which helps
        gradient flows. `hard-sigmoid` has the advantage of making all values
        between min_p and max_p equiprobable.

    is_train_temperature : bool, optional
        Whether to train the paremeter controling the steepness of the activation.
        This is useful when x is used for multiple tasks, and you don't want to
        constraint its magnitude.

    is_train_bias : bool, optional
        Whether to train the bias to shift the activation. This is useful when x is
        used for multiple tasks, and you don't want to constraint it's scale.

    trainable_dim : int, optional
        Size of the trainable bias and temperature. If `1` uses the same vale
        across all dimension, if not should be equal to the number of input
        dimensions to different trainable parameters for each dimension. Note
        that the initial value will still be the same for all dimensions.

    initial_temperature : int, optional
        Initial temperature, a higher temperature makes the activation steaper.

    initial_probability : float, optional
        Initial probability you want to start with.

    initial_x : float, optional
        First value that will be given to the function, important to make
        `initial_probability` work correctly.

    bias_transformer : callable, optional
        Transformer function of the bias. This function should only take care of
        the boundaries (e.g. leaky relu or relu).

    temperature_transformer : callable, optional
        Transformer function of the temperature. This function should only take
        care of the boundaries (e.g. leaky relu  or relu).
    """

    def __init__(
        self,
        min_p=0.0,
        activation="sigmoid",
        is_train_temperature=False,
        is_train_bias=False,
        trainable_dim=1,
        initial_temperature=1.0,
        initial_probability=0.5,
        initial_x=0,
        bias_transformer=nn.Identity(),
        temperature_transformer=nn.Identity(),
    ):

        super().__init__()
        self.min_p = min_p
        self.activation = activation
        self.is_train_temperature = is_train_temperature
        self.is_train_bias = is_train_bias
        self.trainable_dim = trainable_dim
        self.initial_temperature = initial_temperature
        self.initial_probability = initial_probability
        self.initial_x = initial_x
        self.bias_transformer = bias_transformer
        self.temperature_transformer = temperature_transformer

        self.reset_parameters()

    def reset_parameters(self):
        self.temperature = torch.tensor(
            [self.initial_temperature] * self.trainable_dim)
        if self.is_train_temperature:
            self.temperature = nn.Parameter(self.temperature)

        initial_bias = self._probability_to_bias(
            self.initial_probability, initial_x=self.initial_x
        )

        self.bias = torch.tensor([initial_bias] * self.trainable_dim)
        if self.is_train_bias:
            self.bias = nn.Parameter(self.bias)

    def forward(self, x):
        self.temperature.to(x.device)
        self.bias.to(x.device)

        temperature = self.temperature_transformer(self.temperature)
        bias = self.bias_transformer(self.bias)

        if self.activation == "sigmoid":
            full_p = torch.sigmoid((x + bias) * temperature)

        elif self.activation in ["hard-sigmoid", "leaky-hard-sigmoid"]:
            # uses 0.2 and 0.5 to be similar to sigmoid
            y = 0.2 * ((x + bias) * temperature) + 0.5

            if self.activation == "leaky-hard-sigmoid":
                full_p = clamp(
                    y,
                    minimum=0.1,
                    maximum=0.9,
                    is_leaky=True,
                    negative_slope=0.01,
                    hard_min=0,
                    hard_max=0,
                )
            elif self.activation == "hard-sigmoid":
                full_p = clamp(y, minimum=0.0, maximum=1.0, is_leaky=False)

        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        p = rescale_range(full_p, (0, 1), (self.min_p, 1 - self.min_p))

        return p

    def _probability_to_bias(self, p, initial_x=0):
        """Compute the bias to use to satisfy the constraints."""
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)

        if self.activation == "sigmoid":
            bias = -(torch.log((1 - p) / p) /
                     self.initial_temperature + initial_x)

        elif self.activation in ["hard-sigmoid", "leaky-hard-sigmoid"]:
            bias = ((p - 0.5) / 0.2) / self.initial_temperature - initial_x

        return bias


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(
                input,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""

    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            confidence=False,
            bias=True,
            **kwargs,
        ):
            super().__init__()
            self.depthwise = Conv(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=bias,
                **kwargs,
            )
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv


class ReturnNotTensor:
    """Helper class to allow non tensor outputs from skorch."""

    def __init__(self, out):
        self.out = out

    def to(self, *args, **kwargs):
        return self.out


def return_not_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    else:
        return ReturnNotTensor(out)


class Constant:
    def __init__(self, c):
        self.c = c

    def __call__(self, *args):
        return self.c


def set_requires_grad(module, val):
    """Set all the gradients of a given module to a certain value."""
    for p in module.parameters():
        p.requires_grad = val


@contextlib.contextmanager
def no_grad_modules(modules):
    """Context manager that deactivates the gradients of a list of modules."""
    for module in modules:
        set_requires_grad(module, False)

    try:
        yield

    finally:
        for module in modules:
            set_requires_grad(module, True)


def mean_p_logits(logits, dim=0, eps=1e-8):
    """Take the mean in probability space given on some logits."""
    if logits.size(dim) == 1:
        return logits.squeeze(dim)
    else:
        #! SHOULD BE USING LOG SUM EXP
        # have to put into probability space to take average
        mean = logits.softmax(-1).mean(dim)
        return (
            mean + eps
        ).log()  # put back in logit space making sure no nan (no zero)


class BaseRepresentation:
    """Compute the base representation for a number in a certain base while memoizing."""

    def __init__(self, base):
        self.base = base
        self.memoize = {0: []}

    def __call__(self, number):
        """Return a list of the base representation of number."""
        if number in self.memoize:
            return self.memoize[number]

        self.memoize[number] = self(number // self.base) + [number % self.base]
        return self.memoize[number]

    def get_ith_digit(self, number, i):
        """Return the ith digit pf the base representation of number."""
        digits = self(number)
        if i >= len(digits):
            return 0  # implicit padding with zeroes
        return digits[-i - 1]


class BaseRepIthDigits:
    """Compute the ith digit in a given base for torch batch of numbers while memoizing (in numpy)."""

    def __init__(self, base):
        base_rep = BaseRepresentation(base)
        self.base_rep = np.vectorize(base_rep.get_ith_digit)

    def __call__(self, tensor, i_digit):
        return self.base_rep(tensor, i_digit)


class BackwardPDB(torch.autograd.Function):
    """Run PDB in the backward pass."""

    @staticmethod
    def forward(ctx, input, name="debugger"):
        ctx.name = name
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        if not torch.isfinite(grad_output).all() or not torch.isfinite(input).all():
            import pdb

            pdb.set_trace()
        return grad_output, None  # 2 args so return None for `name`


backward_pdb = BackwardPDB.apply
