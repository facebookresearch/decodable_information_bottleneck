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
from dib.utils.distributions import MultivariateNormalDiag, entropy_labels
from dib.utils.helpers import (
    BaseRepIthDigits,
    Constant,
    CrossEntropyLossGeneralize,
    Identity,
    Nchunk_iterable,
    ReturnNotTensor,
    extract_target,
    get_idx_permuter,
    is_sorted,
    mean_p_logits,
    return_not_tensor,
    set_requires_grad,
    tmp_seed,
    update_dict_copy,
)
from dib.utils.initialization import weights_init

from .dib import IBEncoder
from .helpers import (
    BASE_LOG,
    CORR_GROUPS,
    EPS_MIN,
    N_CORR,
    NotEnoughHeads,
    detach,
    mean_p_logits_parallel,
    mean_std,
)
from .vib import VIBLoss

__all__ = ["ERMLoss"]
logger = logging.getLogger(__name__)


class ERMLoss(VIBLoss):
    """Empirical risk minimizer Loss.

    Parameters
    ----------
    ZYCriterion : nn.Module, optional
        Criterion to compute the loss of Q_zy.

    map_target_position : dict, optional
        Dictionary that maps the type of target (e.g. "index") to its position in the 
        target.
    """

    def forward(self, out, y):
        y_pred, z_sample, p_zCx = out

        if p_zCx.out is not None:
            p_zCx_base = p_zCx.out.base_dist
            self._store(
                z_norm=z_sample.pow(2).mean(),
                z_mean_norm=p_zCx_base.loc.abs().mean(),
                z_std=p_zCx_base.scale.mean(),
            )

        zy_loss = self.compute_zy_loss(y_pred, y)
        return zy_loss
