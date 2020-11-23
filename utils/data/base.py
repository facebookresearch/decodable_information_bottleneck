"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from dib import UNLABELLED_CLASS
from dib.utils.helpers import tmp_seed

DIR = os.path.abspath(os.path.dirname(__file__))


class BaseDataset:
    """BaseDataset that should be inherited by the format-specific ones.
    
    Parameters
    ----------
    root : str, optional
        Root to the data directory.

    logger : logging.Logger, optional
        Logger

    is_return_index : bool, optional
        Whether to return the index in addition to the labels.

    is_random_targets : bool, optional
        Whether to use random targets for the dataset.

    seed : int, optional
        Random seed.
    """

    unlabelled_class = UNLABELLED_CLASS

    def __init__(
        self,
        root=os.path.join(DIR, "../../../../data/"),
        logger=logging.getLogger(__name__),
        is_return_index=False,
        is_random_targets=False,
        seed=123,
    ):
        self.seed = seed
        self.logger = logger
        self.root = root
        self.is_return_index = is_return_index
        self.is_random_targets = is_random_targets
        self._is_return_constant = False

    @property
    def map_target_position(self):
        """
        Return a dictionary that maps the type of target (e.g. "index") to its position in the 
        outputted target.
        """
        target_names = {"target": 0}

        if self._is_return_constant:
            target_names["constant"] = len(target_names)

        if self.is_return_index:
            target_names["index"] = len(target_names)

        return target_names

    def randomize_targets_(self):
        """Randomize the targets in place"""
        with tmp_seed(self.seed):
            idcs = list(range(len(self.targets)))
            random.shuffle(idcs)
            self.targets = self.targets[idcs]

    def rm_all_transformations_(self):
        """Completely remove transformation."""
        pass

    def make_test_(self):
        """Make the data a test set."""
        pass

    def append_(self, other):
        """Append a dataset to the current one."""
        self.data = np.append(self.data, other.data, axis=0)
        self.targets = np.append(self.targets, other.targets, axis=0)

    def train_test_split(self, size=0.1, is_stratify=True, is_test_size=True):
        """Split the dataset into train and test (without data augmentation).

        Parameters
        ----------
        size : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of
            the dataset to include in the test split. If int, represents the absolute
            size of the test dataset. 

        is_stratify : bool, optional
            Whether to stratify splits based on class label.

        is_test_size : bool, optional
            Whether size should be the test size or the training one.

        Returns
        ------- 
        train : BaseDataset
            Train dataset containing the complement of `test_size` examples.

        test : BaseDataset
            Test dataset containing `test_size` examples.
        """
        idcs_all = list(range(len(self)))
        stratify = self.targets if is_stratify else None
        idcs_train, indcs_test = train_test_split(
            idcs_all, stratify=stratify, test_size=size, random_state=self.seed
        )

        if not is_test_size:
            indcs_test, idcs_train = idcs_train, indcs_test

        train = self.clone()
        train.keep_indcs_(idcs_train)

        test = self.clone()
        test.keep_indcs_(indcs_test)
        test.make_test_()

        return train, test

    def drop_labels_(self, drop_size, is_stratify=True):
        """Drop part of the labels to make the dataset semisupervised.
        
        Parameters
        ----------
        drop_size : float or int or tuple, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the labels to 
            drop. If int, represents the number of labels to drop. 0 means keep all.

        is_stratify : bool, optional
            Whether to stratify splits based on class label.
        """
        if drop_size == 0:
            return

        self.logger.info(f"Dropping {drop_size} labels...")

        idcs_all = list(range(len(self)))
        stratify = self.targets if is_stratify else None
        idcs_label, idcs_unlabel = train_test_split(
            idcs_all, stratify=stratify, test_size=drop_size, random_state=self.seed
        )

        self.targets[idcs_unlabel] = self.unlabelled_class

    def balance_labels_(self):
        """
        Balances the number of labelled and unlabbeld data by updasmpling labeled. Only works if
        number of labelled data is smaller than unlabelled.
        """
        self.logger.info(f"Balancing the semi-supervised labels...")

        idcs_unlab = [i for i, t in enumerate(
            self.targets) if t == UNLABELLED_CLASS]
        idcs_lab = [i for i, t in enumerate(
            self.targets) if t != UNLABELLED_CLASS]

        assert len(idcs_unlab) > len(idcs_lab)

        resampled_idcs_lab = resample(
            idcs_lab,
            replace=True,
            n_samples=len(idcs_unlab),
            stratify=self.targets[idcs_lab],
            random_state=self.seed,
        )

        self.keep_indcs_(idcs_unlab + resampled_idcs_lab)

    def drop_unlabelled_(self):
        """Drop all the unlabelled examples."""
        self.logger.info(f"Drop all the unlabelled examples...")

        idcs_lab = [i for i, t in enumerate(
            self.targets) if t != UNLABELLED_CLASS]
        self.keep_indcs_(idcs_lab)

    def get_subset(self, size, is_stratify=True):
        """Return a subset of `size` that does not share its memory.
        
        Parameters
        ----------
        size : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of
            the dataset to include in the subset. If int, represents the absolute
            size of the subset. If -1, return all.

        is_stratify : bool, optional
            Whether to stratify splits based on class label.
        """
        if size == -1:
            return self

        subset, _ = self.train_test_split(
            size=size, is_stratify=is_stratify, is_test_size=False
        )
        return subset

    def clone(self):
        """Returns a deepcopy of the daatset."""
        return copy.deepcopy(self)

    def keep_indcs_(self, indcs):
        """Keep the given indices.
        
        Parameters
        ----------
        indcs : array-like int
            Indices to keep. If the multiplicity of the indices is larger than 1 then will duplicate
            the data.
        """
        self.data = self.data[indcs]
        self.targets = self.targets[indcs]

    def drop_features_(self, drop_size):
        """Drop part of the features (e.g. pixels in images).

        Parameters
        ----------
        drop_size : float or int or tuple, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
            drop. If int, represents the number of datapoints to drop. If tuple, same as before 
            but give bounds (min and max). 1 means drop all.
        """
        if drop_size == 0:
            return

        raise NotImplementedError(
            "drop_features_ not implemented for current dataset")

    def add_index(self, y, index):
        """Append the index to the targets (if needed)."""
        if self.is_return_index:
            y = tuple(y) + (index,)
        return y

    def set_const_target_(self, target):
        """Set a constant target `target` to all targets."""
        if self.targets.ndim == 1:
            self.targets = np.expand_dims(self.targets, 1)
        self.targets = np.append(
            self.targets, self.targets * 0 + target, axis=1)
        self._is_return_constant = True

    def sort_data_(self, by="targets"):
        """Sort the data by {"targets"}. E.g. the first |X|/|Y| examples will be from the same target."""
        targets = self.targets
        if self.targets.ndim > 1:
            targets = targets[:, 0]

        if by == "targets":

            idcs = list(np.argsort(targets))
            self.targets = self.targets[idcs]
            self.data = self.data[idcs]
        else:
            raise ValueError(f"Unkown by={by}")

    def count_targets(self):
        """Return a dictionary where the keys are the targets and values the number of each target."""
        targets, counts = np.unique(self.targets, return_counts=True)
        return {t: c for t, c in zip(targets, counts)}
