"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import warnings
from contextlib import suppress

import numpy as np
import skorch
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.base import ClassifierMixin, TransformerMixin
from skorch import NeuralNet
from skorch.callbacks import EpochScoring, ProgressBar
from skorch.dataset import Dataset, get_len, unpack_data, uses_placeholder_y
from skorch.helper import predefined_split
from skorch.history import History
from skorch.utils import TeeGenerator, get_map_location, to_numpy, to_tensor

from .helpers import FixRandomSeed, target_extractor

logger = logging.getLogger(__name__)

__all__ = ["NeuralNetTransformer", "NeuralNetClassifier"]

net_get_params_for_optimizer = NeuralNet._get_params_for_optimizer


def _get_params_for_optimizer(self, prefix, named_parameters, is_add_criterion=False):
    """Difference with default is that also adds the parameters of the criterion."""

    if is_add_criterion:
        named_parameters = list(named_parameters) + list(
            self.criterion_.named_parameters()
        )
    return net_get_params_for_optimizer(self, prefix, named_parameters)


def save_params(self, f_criterion=None, **kwargs):
    """Difference with default is that also saves the criterion."""
    NeuralNet.save_params(self, **kwargs)

    if f_criterion is not None:
        msg = (
            "Cannot save parameters of an un-initialized criterion. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...)."
        )
        self.check_is_fitted(attributes=["criterion_"], msg=msg)
        torch.save(self.criterion_.state_dict(), f_criterion)


def load_params(self, f_criterion=None, checkpoint=None, **kwargs):
    NeuralNet.load_params(self, checkpoint=checkpoint, **kwargs)

    ####################
    # all this copy pasted
    def _get_state_dict(f):
        map_location = get_map_location(self.device)
        self.device = self._check_device(self.device, map_location)
        return torch.load(f, map_location=map_location)

    if checkpoint is not None:
        if not self.initialized_:
            self.initialize()
        formatted_files = checkpoint.get_formatted_files(self)
        ####################
        f_criterion = f_criterion or formatted_files["f_criterion"]

    if f_criterion is not None:
        msg = (
            "Cannot load state of an un-initialized criterion. "
            "Please initialize first by calling .initialize() "
            "or by fitting the model with .fit(...)."
        )
        self.check_is_fitted(attributes=["criterion_"], msg=msg)
        state_dict = _get_state_dict(f_criterion)
        self.criterion_.load_state_dict(state_dict)


def get_loss(self, y_pred, y_true, X=None, training=False):
    """Return the loss for this batch."""
    y_true = to_tensor(y_true, device=self.device)

    if isinstance(self.criterion_, nn.Module):
        self.criterion_.train(training)

    return self.criterion_(y_pred, y_true)


def fit_loop(self, X, y=None, epochs=None, **fit_params):

    self.check_data(X, y)
    epochs = epochs if epochs is not None else self.max_epochs

    dataset_train, dataset_valid = self.get_split_datasets(X, y, **fit_params)
    on_epoch_kwargs = {"dataset_train": dataset_train,
                       "dataset_valid": dataset_valid}

    start = 0
    if self.is_train_delta_epoch:

        # in case you load the model you want to only train the epoch difference
        start = len(self.history)

        # make sure that still run 1 epoch for model saving / notify
        start = min(start, epochs - 1)

        logger.info(f"Model was loaded,training only {start}-{epochs}")

    for epoch in range(start, epochs):
        self.notify("on_epoch_begin", **on_epoch_kwargs)

        self._single_epoch(dataset_train, training=True,
                           epoch=epoch, **fit_params)

        if dataset_valid is not None:
            self._single_epoch(dataset_valid, training=False,
                               epoch=epoch, **fit_params)

        self.notify("on_epoch_end", **on_epoch_kwargs)

    return self


def _single_epoch(self, dataset, training, epoch, **fit_params):
    """Computes a single epoch of train or validation."""

    is_placeholder_y = uses_placeholder_y(dataset)

    if training:
        prfx = "train"
        step_fn = self.train_step
    else:
        prfx = "valid"
        step_fn = self.validation_step

    batch_count = 0
    for data in self.get_iterator(dataset, training=training):
        Xi, yi = unpack_data(data)
        yi_res = yi if not is_placeholder_y else None
        self.notify("on_batch_begin", X=Xi, y=yi_res, training=training)
        step = step_fn(Xi, yi, **fit_params)
        self.history.record_batch(prfx + "_loss", step["loss"].item())
        self.history.record_batch(prfx + "_batch_size", get_len(Xi))
        self.notify("on_batch_end", X=Xi, y=yi_res, training=training, **step)
        batch_count += 1

    self.history.record(prfx + "_batch_count", batch_count)

    if hasattr(self.criterion_, "to_store"):
        for k, v in self.criterion_.to_store.items():
            with suppress(NotImplementedError):
                # pytorch raises NotImplementedError on wrong types
                self.history.record(prfx + "_" + k, (v[0] / v[1]).item())
        self.criterion_.to_store = dict()


doc_neural_net_clf = (
    """Wrapper around skorch.NeuralNetClassifier. Differences:

    Parameters
    ----------
    Notes
    -----
    - use by default crossentropy loss instead of NNLoss
    - enables storing of additional losses.

    Base documentation:
    """
    + skorch.NeuralNetClassifier.__doc__
)


class NeuralNetClassifier(skorch.NeuralNetClassifier):

    __doc__ = doc_neural_net_clf

    def __init__(
        self,
        *args,
        criterion=torch.nn.CrossEntropyLoss,
        is_train_delta_epoch=True,
        **kwargs,
    ):
        super().__init__(*args, criterion=criterion, **kwargs)

        self.is_train_delta_epoch = is_train_delta_epoch

    @property
    def _default_callbacks(self):
        _default_callbacks = dict(super()._default_callbacks)

        _default_callbacks["valid_acc"] = EpochScoring(
            "accuracy",
            name="valid_acc",
            lower_is_better=False,
            target_extractor=target_extractor,
        )

        return [(k, v) for k, v in _default_callbacks.items()]

    def predict_proba(self, X):
        """Return probability estimates for samples.

        Notes
        -----
        - output of model should be logits (softmax applied in this function)
        - If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using :func:`~skorch.NeuralNet.forward`
        instead.

        Returns
        -------
        y_proba : numpy ndarray
        
        """
        # output of model should be logits!
        logits = super().predict_proba(X)
        return softmax(logits, axis=1)

    fit_loop = fit_loop
    _single_epoch = _single_epoch
    get_loss = get_loss
    _get_params_for_optimizer = _get_params_for_optimizer
    save_params = save_params
    load_params = load_params


doc_neural_net_trnsf = (
    """Wrapper around skorch.NeuralNet for transforming data. Differences:

    Methods
    -------
    freeze:
        Freezes the model such that it cannot be fitted again.

    transform:
        Returns a numpy array containing all the first outputs, similarly to `.predict`. The main 
        difference is that it sets `module_.is_transform` to `True`. The correct behavior should thus
        be implemented in the module using `is_transform` flag.

    Notes
    -----
    - enables storing of additional losses.

    Base documentation:
    """
    + skorch.NeuralNet.__doc__
)


class NeuralNetTransformer(skorch.NeuralNet, TransformerMixin):
    __doc__ = doc_neural_net_trnsf

    def __init__(self, *args, is_train_delta_epoch=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_train_delta_epoch = is_train_delta_epoch

    def freeze(self, is_freeze=True):
        """Freezes (or unfreeze) the model such that it cannot be fitted again."""
        self._is_frozen = is_freeze
        return self

    def fit(self, X, y=None, **fit_params):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if self.verbose > 0:
                warnings.warn("Skipping fitting because froze etimator.")
            return self

        return super().fit(X, y=y, **fit_params)

    def transform(self, X):
        """Transform an input."""
        self.module_.is_transform = True
        self.module_.training = False
        X_transf = super().predict_proba(X)  # does not actually predict proba
        self.module_.is_transform = False
        self.module_.training = True
        return X_transf

    def predict_proba(self, X):
        """Return probability estimates for samples.

        Notes
        -----
        - output of model should be logits (softmax applied in this function)
        - If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using :func:`~skorch.NeuralNet.forward`
        instead.

        Returns
        -------
        y_proba : numpy ndarray
        
        """
        # output of model should be logits!
        logits = super().predict_proba(X)
        return softmax(logits, axis=1)

    fit_loop = fit_loop
    _single_epoch = _single_epoch
    get_loss = get_loss
    _get_params_for_optimizer = _get_params_for_optimizer
    save_params = save_params
    load_params = load_params