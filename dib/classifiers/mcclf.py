"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn as nn

from dib.utils.helpers import mean_p_logits

__all__ = ["MCTrnsfClassifier"]


class MCTrnsfClassifier(nn.Module):
    """Class that merges a pretrained stochastic transformer with a classifier. It does 
    this by sampling multiple times from the transformer and passing it to the classifier.

    Parameters
    ----------
    transformer : nn.Module
       Transformer that return the representation. It should have a property `is_transform.`

    Classifier : nn.Module
        Uninitialized classifier.

    z_dim : int, optional
        Output size of the transformer (input to classifier).

    n_classes : int, optional
        Number of output classes.

    n_test_samples : int, optional
        Number of Monte Carlo samples from the transformer during test. The transformer should have 
        an attribute `n_test_samples` and return all the samples in the first dimension (before batch).

    is_freeze_transformer : bool, optional
        Whether to freeze the transformer or train it.

    kwargs : dict, optional
        Additional arguments to the classifier. 
    """

    def __init__(
        self,
        transformer,
        Classifier,
        z_dim,
        n_classes,
        n_test_samples=1,
        is_freeze_transformer=True,
        **kwargs
    ):
        super().__init__()
        self.transformer = transformer
        self.is_freeze_transformer = is_freeze_transformer
        self.clf = Classifier(z_dim, n_classes, **kwargs)
        self.transformer.n_test_samples = n_test_samples

    def forward(self, X):

        self.transformer.is_transform = True

        if self.is_freeze_transformer:
            with torch.no_grad():
                Z = self.transformer(X)
        else:
            Z = self.transformer(X)

        out = mean_p_logits(self.clf(Z))  # average over samples

        return out
