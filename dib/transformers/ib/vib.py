"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import math
import random
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from skorch.utils import to_numpy
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.parallel import parallel_apply

from dib.predefined import MLP
from dib.utils.distributions import MultivariateNormalDiag
from dib.utils.helpers import (
    BaseRepIthDigits,
    Constant,
    Identity,
    Nchunk_iterable,
    ReturnNotTensor,
    get_idx_permuter,
    is_sorted,
    mean_p_logits,
    return_not_tensor,
    set_requires_grad,
    set_seed,
    tmp_seed,
    update_dict_copy,
)
from dib.utils.initialization import weights_init

from .dib import DIBLoss, IBEncoder
from .helpers import BASE_LOG, EPS_MIN

__all__ = ["VIBLoss"]
logger = logging.getLogger(__name__)


class VIBLoss(nn.Module):
    """VIB Loss.

    Parameters
    ----------
    beta : float or callable, optional
        Regularization weight. If callable, should return a float given `is_training`.
        Importanlty this only changes the gradients but not the model selection / loss plotting.
        This is 1000x the multiple of beta in usual VIB (to make it comparable with DIB).

    n_per_target : dict
        Number of examples for each target.

    ZYCriterion : nn.Module, optional
        Criterion to compute the loss of Q_zy.

    seed : int, optional
        Random seed.
    """

    def __init__(
        self, n_per_target, beta=1, ZYCriterion=nn.CrossEntropyLoss, seed=123, **kwargs
    ):
        super().__init__()
        self.n_per_target = n_per_target
        self.beta = beta if callable(beta) else Constant(beta)
        self.to_store = dict()
        self.zy_criterion = ZYCriterion()
        self.compute_entropies_()
        self._scale_beta = 1e-3
        self.seed = seed
        set_seed(self.seed)  # ensure seed is  same as for DIB

    compute_zy_loss = DIBLoss.compute_zy_loss
    _store = DIBLoss._store

    def compute_entropies_(self):
        # all of these assume uniform distribution
        n_per_target = np.array(list(self.n_per_target.values()))

        self.H_x = math.log(n_per_target.sum(), BASE_LOG)
        self.H_y = entropy(n_per_target, base=BASE_LOG)
        self.H_xCy = sum(
            math.log(N, BASE_LOG) * N / n_per_target.sum() for N in n_per_target
        )

    def compute_z_loss(self, p_zCx):
        """Compute loss of Z."""

        # H[Z|X]
        H_zCx = p_zCx.entropy().mean(0) / math.log(BASE_LOG)

        mean_0 = torch.zeros_like(p_zCx.base_dist.loc)
        std_1 = torch.ones_like(p_zCx.base_dist.scale)

        p_z = MultivariateNormalDiag(mean_0, std_1)
        kl = kl_divergence(p_zCx, p_z).mean(0) / math.log(BASE_LOG)

        # I[Z,X] \approx KL[p(Z|x) || r(z)]
        # would be equal if r(z) (the prior) was replaced with the marginal p(z)
        I_xz = kl

        self._store(H_zCx=H_zCx, I_xz=I_xz)

        curr_beta = self.beta(self.training) if self.beta(
            False) is not None else 0
        curr_beta = self._scale_beta * curr_beta

        return curr_beta * I_xz

    def forward(self, out, y):

        #! dirty trick to get back the non tensor outputs
        out = [el.out if isinstance(el, ReturnNotTensor) else el for el in out]
        y_pred, z_sample, p_zCx = out

        p_zCx_base = p_zCx.base_dist
        self._store(
            z_norm=z_sample.pow(2).mean(),
            z_mean_norm=p_zCx_base.loc.abs().mean(),
            z_std=p_zCx_base.scale.mean(),
        )

        z_loss = self.compute_z_loss(p_zCx)
        zy_loss = self.compute_zy_loss(y_pred, y)
        self._store(aux_loss=z_loss)

        return zy_loss + z_loss
