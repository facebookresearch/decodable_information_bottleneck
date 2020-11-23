"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import collections
import copy
import glob
import logging
import math
import os
import random
import shutil
import types
from collections import ChainMap, defaultdict
from contextlib import contextmanager, suppress
from multiprocessing import Pool, cpu_count

import numpy as np
import omegaconf
import pandas as pd
import skorch
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from skorch.callbacks import Callback
from skorch.callbacks.scoring import check_scoring
from skorch.dataset import get_len, unpack_data, uses_placeholder_y
from torch.optim.optimizer import Optimizer, required

from dib import UNLABELLED_CLASS
from dib.training.helpers import Checkpoint
from dib.training.trainer import _single_epoch
from dib.utils.helpers import *

SKLEARN_MODEL = "model.joblib"
SFFX_TOAGG = "_toagg"

logger = logging.getLogger(__name__)


def get_float_value(x):
    """Convert to float"""
    if isinstance(x, torch.Tensor):
        x = x.item()
    elif not isinstance(x, float):
        x = float(x)

    return x


def add_noise_to_param_(module, sigma, is_relative=True):
    """Add uniform noise with standard deviation `sigma` to each weight."""
    with torch.no_grad():
        for param in module.parameters():
            if is_relative:
                unif = torch.distributions.uniform.Uniform(
                    0, param.abs() * sigma)
                noise = unif.sample()
            else:
                unif = torch.distributions.uniform.Uniform(0, sigma)
                noise = unif.sample(param.shape)
            param.add_(noise)


def force_generalization(datasets):
    """Force the (anti)-generalization of a model by adding the test set to the training set. It 
    also adds a label `is_train` that says whether the example is from the training `is_train=1` or 
    testing `is_train=0` set.

    Parameters
    ----------
    datasets : dictionary of torch.utils.data.Dataset
        Dictionary containing at least the `"train"` and `"test"` set.

    Returns
    -------
    train_data : torch.utils.data.Dataset 
    """
    # make sure don't change the dataset for eval and clf
    datasets = copy.deepcopy(datasets)

    datasets["test"].set_const_target_(0)
    datasets["train"].set_const_target_(1)

    datasets["train"].append_(datasets["test"])

    return datasets["train"]


def clip_perturbated_param_(
    module, unperturbated_params, clip_factor, is_relative=True
):
    """
    Element wise clipping of the absolute value of the difference in weight. `unperturbated_params` 
    needs to be a dictionary of unperturbated param. Use `is_relative` if clip factor should multipy
    the unperturbated param.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            w = unperturbated_params[name]

            # delta_i = (delta+w)_i - w_i
            delta = param - w

            max_abs_delta = clip_factor * w.abs() if is_relative else clip_factor

            clipped_delta = torch.where(
                delta.abs() > max_abs_delta, delta.sign() * max_abs_delta, delta
            )

            # inplace replace
            param.fill_(0)
            param.add_(w)
            param.add_(clipped_delta)


def get_device(module):
    """return device of module."""
    return next(module.parameters()).device


def batchnorms2convs_(module):
    """Converts all the batchnorms to frozen convolutions."""
    for name, m in module.named_children():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            module._modules[name] = BatchNormConv(m)
        else:
            batchnorms2convs_(module._modules[name])


class BatchNormConv(nn.Module):
    """Replace a batchnorm layer with a frozen convolution."""

    def __init__(self, batchnorm):
        super().__init__()
        if isinstance(batchnorm, nn.BatchNorm2d):
            conv = nn.Conv2d(
                batchnorm.num_features,
                batchnorm.num_features,
                1,
                groups=batchnorm.num_features,
            )
        elif isinstance(batchnorm, nn.BatchNorm1d):
            conv = nn.Conv1d(
                batchnorm.num_features,
                batchnorm.num_features,
                1,
                groups=batchnorm.num_features,
            )

        conv.eval()
        nn.init.ones_(conv.weight)
        nn.init.zeros_(conv.bias)
        conv.to(get_device(batchnorm))
        self.bn = nn.utils.fusion.fuse_conv_bn_eval(conv, batchnorm)

    def forward(self, x):
        return self.bn(x)


def rm_clf_experiment(experiment):
    """Remove all the classifier files for an experiment."""
    for f in glob.glob(f"tmp_results/{experiment}/**/clf_*/**", recursive=True):
        try:
            shutil.rmtree(f)
        except FileNotFoundError:
            pass


def invert_dict(d):
    return {v: k for k, v in d.items()}


def flip_nested_dict(nested_dict):
    """Flip nested dictionary inside out."""
    flipped = dict()
    for key, subdict in nested_dict.items():
        for k, v in subdict.items():
            flipped[k] = flipped.get(k, dict())
            flipped[k][key] = v
    return flipped


# Credits https://stackoverflow.com/questions/17215400/python-format-string-unused-named-arguments/17215533#17215533
class PartialFormatMap(dict):
    """Dictionary used to do partial formatting of string in python.
    E.g. `"{keep} {modify}".format_map(SafeDict(modify='done')) == '{keep} done'`
    """

    def __missing__(self, key):
        return "{" + key + "}"


# credits : https://gist.github.com/simon-weber/7853144
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


# credits : https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
class OnlineVariance:
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


class StoreVarGrad(Callback):
    """Callback which applies a function on all gradients, stores the variance during each epoch."""

    def __init__(self):
        self.online_vars = dict()
        self.n = 0
        self.var_grads = dict()

    def initialize(self):
        self.online_vars = dict()
        self.n = 0
        self.var_grads = dict()

    def on_grad_computed(self, net, **kwargs):

        for name, param in net.module_.named_parameters():
            if param.grad is not None:
                if name not in self.online_vars:
                    self.online_vars[name] = OnlineVariance()

                self.online_vars[name].include(
                    param.grad.cpu().detach().flatten().numpy()
                )

    def on_epoch_end(self, net, parent=None, **kwargs):
        epoch = net.history[-1]["epoch"]
        self.n += 1

        self.var_grads = {k: v.variance for k, v in self.online_vars.items()}
        self.online_vars = dict()


class StoreGrad(Callback):
    """Callback which applies a function on all gradients, stores the output at each epoch and
    then takes an average across epochs."""

    def __init__(self, fn=Identity()):
        self.curr_epoch = dict()
        self.prev_epochs = dict()
        self.fn = fn

    def initialize(self):
        self.curr_epoch = dict()
        self.prev_epoch = dict()

    def on_grad_computed(self, net, **kwargs):
        for name, param in net.module_.named_parameters():
            if param.grad is not None:
                if name not in self.curr_epoch:
                    self.curr_epoch[name] = OnlineVariance()

                self.curr_epoch[name].include(param.grad.cpu().flatten())

    def on_epoch_end(self, net, parent=None, **kwargs):
        epoch = net.history[-1]["epoch"]

        self.prev_epochs[epoch] = {
            k: v.variance for k, v in self.curr_epoch.items()}
        self.curr_epoch = dict()


class StopAtThreshold(skorch.callbacks.Callback):
    """Callback for stopping training when `monitor` reaches threshold."""

    def __init__(
        self, monitor="train_loss", threshold=0.01, lower_is_better=True, sink=print
    ):
        self.monitor = monitor
        self.threshold = threshold
        self.sink = sink
        self.lower_is_better = lower_is_better

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if self._is_score_improved(current_score):
            self._sink(
                "Stopping since {} reached {}".format(
                    self.monitor, self.threshold),
                verbose=net.verbose,
            )
            raise KeyboardInterrupt

    def _is_score_improved(self, score):
        if self.lower_is_better:
            return score < self.threshold
        return score > self.threshold

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)


class TensorBoard(skorch.callbacks.TensorBoard):
    def on_epoch_end(self, net, parent=None, **kwargs):
        epoch = net.history[-1]["epoch"]

        for m in net.module_.modules():
            try:
                m.tensorboard(self.writer, epoch, mode="on_epoch_end")
            except AttributeError:
                pass

        super().on_epoch_end(net, **kwargs)  # call super last

        if parent is not None and hasattr(parent.criterion_, "to_store"):
            for k, v in parent.criterion_.to_store.items():
                with suppress(NotImplementedError):
                    # pytorch raises NotImplementedError on wrong types
                    self.writer.add_scalar(
                        tag=f"Loss/partial/{k}",
                        scalar_value=v[0] / v[1],
                        global_step=epoch,  # avg
                    )
            parent.criterion_.to_store = dict()

    def on_grad_computed(self, net, **kwargs):

        epoch = net.history[-1]["epoch"]

        for m in net.module_.modules():
            if hasattr(m, "tensorboard") and callable(m.tensorboard):
                try:
                    m.tensorboard(self.writer, epoch, mode="on_grad_computed")
                except NotImplementedError:  # if frozen
                    pass


def clean_end_run(clean_after_run, chckpnt_dirnames):
    """Clean the checkpoiting directories after running.
    
    Parameters
    ----------
    clean_after_run : ["training","all",None]
        Cleans the directory. If "training" removes all the checkpoiting needed for training
        (last epoch models and all the optimizer). If "all" also removes the best_epoch model.

    chckpnt_dirnames : list of str
        Directories where checkpoints were saved. 
    """

    for chckpnt_dirname in chckpnt_dirnames:
        if clean_after_run == "all":
            patterns = ["*.pt"]

        elif clean_after_run == "training":
            patterns = ["*_optimizer.pt"]

        elif clean_after_run is None:
            continue

        else:
            raise ValueError(f"Unkown chckpnt_dirnames={chckpnt_dirnames}")

        for pattern in patterns:
            for f in glob.glob(os.path.join(chckpnt_dirname, pattern)):
                os.remove(f)


def hyperparam_to_path(hyperparameters):
    """Return a string of all hyperparameters that can be used as a path extension."""
    return "/".join([f"{k}_{v}" for k, v in hyperparameters.items()])


def format_container(to_format, formatter, k=None):
    """Format a container of string.

    Parameters
    ----------
    to_format : str, list, dict, or omegaconf.Config
        (list of) strings to fromat.
    
    formatter : dict
        Dict of keys to replace and values with which to replace.
    """
    if isinstance(to_format, str):
        out = to_format.format(**formatter)
    elif to_format is None:
        out = None
    else:
        if isinstance(to_format, omegaconf.Config):
            to_format = OmegaConf.to_container(to_format, resolve=True)

        if isinstance(to_format, list):
            out = [
                format_container(path, formatter, k=i)
                for i, path in enumerate(to_format)
            ]
        elif isinstance(to_format, dict):
            out = {
                k: format_container(path, formatter, k=k)
                for k, path in to_format.items()
            }
        else:
            raise ValueError(f"Unkown to_format={to_format}")
    return out


# Change _scoring for computing validation only at certain epochs
def _scoring(self, net, X_test, y_test):
    """Resolve scoring and apply it to data. Use cached prediction
        instead of running inference again, if available."""
    scorer = check_scoring(net, self.scoring_)

    if y_test is None:
        return float(
            "nan"
        )  # ! Only difference : make sure no issue if valid not computed

    return scorer(net, X_test, y_test)


def _single_epoch_skipvalid(
    self,
    dataset,
    training,
    epoch,
    save_epochs=(
        list(range(10))
        + list(range(9, 100, 10))
        + list(range(99, 1000, 50))
        + list(range(999, 10000, 500))
    ),
    **fit_params,
):
    if not training and epoch not in save_epochs:
        return

    _single_epoch(self, dataset, training, epoch, **fit_params)


def get_checkpoint(chckpnt_dirname, monitor="valid_loss_best", **kwargs):
    """Return the correct checkpoint.
    
    Parameters
    ----------
    chckpnt_dirname : str

    monitor : {"valid_loss_best", "valid_acc_best", "train_loss_best", "last"} or list of int or int
        "*_best" saves the model with the best *. "last" saves the last model.
        If list of int saves at each of these epochs. If int saves at a specific epoch (useful for 
        loading).

    """
    if monitor == "last":
        return Checkpoint(
            dirname=chckpnt_dirname, monitor=None, fn_prefix="last_epoch_", **kwargs
        )
    elif isinstance(monitor, str):
        return Checkpoint(
            dirname=chckpnt_dirname, monitor=monitor, fn_prefix="best_", **kwargs
        )
    else:

        def _monitor_epoch(net):
            epoch = net.history[-1]["epoch"]
            return epoch in monitor

        if isinstance(monitor, int):
            f_params = f"params_epoch{monitor}.pt"
            f_optimizer = f"optimizer_epoch{monitor}.pt"
            f_criterion = f"criterion_epoch{monitor}.pt"
            monitor = [monitor]
        else:
            f_params = "params_epoch{last_epoch[epoch]}.pt"
            f_optimizer = "optimizer_epoch{last_epoch[epoch]}.pt"
            f_criterion = "criterion_epoch{last_epoch[epoch]}.pt"

        return Checkpoint(
            dirname=chckpnt_dirname,
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_criterion=f_criterion,
            monitor=_monitor_epoch,
            **kwargs,
        )


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])


def count_prune_parameters(model):
    """Count the number of parameters that were pruned out."""
    return sum(torch.nn.utils.parameters_to_vector(model.buffers()) == 0)


def merge_dicts(*dicts):
    """Merge multiple dictionaries. If key is repeated, first appearance will be used."""
    return dict(ChainMap(*dicts))


def rm_const_col(df):
    """Remove constant columns in a dataframe"""
    # nan need specific dropping
    df = df.dropna(axis=1, how="all")
    return df.loc[:, (df != df.iloc[0]).any()].copy()


def update_prepending(to_update, new):
    """Update a dictionary with another. the difference with .update, is that it puts the new keys
    before the old ones (prepending)."""
    # makes sure don't update arguments
    to_update = to_update.copy()
    new = new.copy()

    # updated with the new values appended
    to_update.update(new)

    # remove all the new values => just updated old values
    to_update = {k: v for k, v in to_update.items() if k not in new}

    # keep only values that ought to be prepended
    new = {k: v for k, v in new.items() if k not in to_update}

    # update the new dict with old one => new values are at the begining (prepended)
    new.update(to_update)

    return new


def aggregate_table(table, aggregate=["mean", "std"], match_to_agg=SFFX_TOAGG):
    """Aggregates all the results in all columns containing `match_to_agg`."""
    table = table.copy()

    toaggreg = [c for c in table.columns if "_toagg" in c]

    #! Hacky way of dealing with NaN in groupby while waiting for https://github.com/pandas-dev/pandas/pull/30584
    hacky_nan = -7878  # has to be something that will not appear anywhere else
    groupby_idx = [c for c in table.columns if c not in (["run"] + toaggreg)]
    table[groupby_idx] = table[groupby_idx].fillna(hacky_nan)

    table = table.groupby(
        [c for c in table.columns if c not in (["run"] + toaggreg)]
    ).agg(
        merge_dicts({k: aggregate for k in toaggreg}, {"run": "count"})
    )  # make sure that add counts

    table.columns = [
        "_".join(col).strip().replace("_toagg", "") for col in table.columns.values
    ]

    table = table.reset_index()

    #! Reset the nan (careful when replacing due to float precision)
    numeric_col = table.select_dtypes(np.number).columns
    table[numeric_col] = table[numeric_col].mask(
        np.isclose(table[numeric_col].values, hacky_nan)
    )
    # in case replacement was in object col
    table = table.mask(table == hacky_nan)

    return table


def append_sffx(l, sffx):
    """Append a suffix to a list of strings."""
    return [el + sffx for el in l]


def save_pattern(folders, pattern, filename, formatting={}, logger=None):
    """Save the pattern (formatted) to file. If `formatting` is not empty then append."""
    for i, folder in enumerate(folders):
        file = os.path.join(folder, filename)

        with open(file, "w" if len(formatting) == 0 else "a") as f:
            if len(formatting) > 0:
                pattern = pattern.format(**formatting)
            f.write(pattern + "\n")

        if logger is not None and i == 0:
            logger.info(f"Saving {pattern} to {file.split('/')[-1]}")


def get_exponential_decay_gamma(scheduling_factor, max_epochs):
    """Return the exponential learning rate factor gamma.

    Parameters
    ----------
    scheduling_factor :
        By how much to reduce learning rate during training.

    max_epochs : int
        Maximum number of epochs.
    """
    return (1 / scheduling_factor) ** (1 / max_epochs)


def replace_None_with_all(df, column):
    """Duplicate rows with `None` in `columns` to have one for all unique values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from which to replace the values.
        
    column : str
        Name of the column in which to search for None.
        
    Return
    ------
    df : pd.DataFrame
        Dataframe with the replicated rows.
        
    Examples
    --------
    >>> df = pd.DataFrame([["model1",2,0.01],["model1",3,0.1],["model2",5,None]], columns=["Model","N Layers","Col"])
    
    >>> df
        Model  N Layers   Col
    0  model1         2  0.01
    1  model1         3  0.10
    2  model2         5   NaN
    
    >>> replace_None_with_all(df, "Col")
        Model  N Layers   Col
    0  model1         2  0.01
    1  model1         3  0.10
    2  model2         5  0.01
    3  model2         5  0.10
    """

    to_replicate = df[df[column].isin([None])].copy()
    df = df[~df[column].isin([None])]
    replicated = []
    for val in df[column].unique():
        to_replicate[column] = val
        replicated.append(to_replicate.copy())
    df = pd.concat([df, *replicated], ignore_index=True)
    return df


def dict_none_toNaN(d):
    return {k: v if v is not None else float("nan") for k, v in d.items()}


class SetLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group. 
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes the new lr
            given an integer parameter epoch, the current lr and the base lr, 
            or a list of such functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lmbda = lambda epoch, cur_lr, base_lr: 0.95
        >>> scheduler = SetLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} lr_lambdas, but got {}".format(
                        len(optimizer.param_groups), len(lr_lambda)
                    )
                )
            self.lr_lambdas = list(lr_lambda)
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):

        if self.last_epoch > 0:
            return [
                lmbda(self.last_epoch, group["lr"], base_lr)
                for base_lr, lmbda, group in zip(
                    self.base_lrs, self.lr_lambdas, self.optimizer.param_groups
                )
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]
