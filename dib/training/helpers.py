"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import random
import warnings

import numpy as np
import skorch
import torch
from skorch.callbacks import Callback

from dib.utils.helpers import cont_tuple_to_tuple_cont, ratio_to_int, set_seed, to_numpy


def target_extractor(targets, is_multi_target=False):
    """
    Helper function that extracts the targets for scoring. There can be multiple targets 
    for the case where you appended indices or distractors.
    """
    if isinstance(targets, (list, tuple)):
        if is_multi_target:
            targets = torch.stack(targets, axis=1)
        else:
            targets = targets[0]

    return to_numpy(targets)


def clone_trainer(trainer, is_reinit_besides_param=False):
    """Clone a trainer with optional possibility of reinitializing everything besides 
    parameters (e.g. optimizers.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer_new = copy.deepcopy(trainer)

    if is_reinit_besides_param:
        trainer_new.initialize_callbacks()
        trainer_new.initialize_criterion()
        trainer_new.initialize_optimizer()
        trainer_new.initialize_history()

    return trainer_new


class FixRandomSeed(Callback):
    """
    Callback to have a deterministic behavior.
    Credits: https://github.com/skorch-dev/skorch/issues/280
    """

    def __init__(self, seed=123, is_cudnn_deterministic=False, verbose=0):
        self.seed = seed
        self.is_cudnn_deterministic = is_cudnn_deterministic
        self.verbose = verbose

    def initialize(self):
        if self.seed is not None:
            if self.verbose > 0:
                print("setting random seed to: ", self.seed, flush=True)
            set_seed(self.seed)
        torch.backends.cudnn.deterministic = self.is_cudnn_deterministic


class Checkpoint(skorch.callbacks.Checkpoint):
    """
    Difference with default is that save criterion.
    """

    def __init__(self, *args, f_criterion="criterion.pt", **kwargs):
        super().__init__(*args, **kwargs)
        self.f_criterion = f_criterion

    def save_model(self, net):
        super().save_model(net)

        if self.f_criterion is not None:
            f = self._format_target(net, self.f_criterion, -1)
            self._save_params(f, net, "f_criterion", "criterion parameters")

    def get_formatted_files(self, net):
        idx = -1
        if self.event_name is not None and net.history:
            for i, v in enumerate(net.history[:, self.event_name]):
                if v:
                    idx = i
        return {
            "f_params": self._format_target(net, self.f_params, idx),
            "f_optimizer": self._format_target(net, self.f_optimizer, idx),
            "f_history": self.f_history_,
            # ONLY DIFF
            "f_pickle": self._format_target(net, self.f_pickle, idx),
            "f_criterion": self._format_target(net, self.f_criterion, idx),
        }
