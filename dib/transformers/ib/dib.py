"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import logging
import math
import random
from functools import partial
from itertools import zip_longest

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from skorch.utils import to_numpy
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.parallel import parallel_apply

from dib.predefined import MLP
from dib.utils.distributions import MultivariateNormalDiag, label_distribution
from dib.utils.helpers import (
    BaseRepIthDigits,
    BatchNorm1dLast,
    Constant,
    CrossEntropyLossGeneralize,
    Identity,
    Nchunk_iterable,
    ReturnNotTensor,
    extract_target,
    get_idx_permuter,
    is_sorted,
    mean_p_logits,
    no_grad_modules,
    return_not_tensor,
    set_seed,
    tmp_seed,
    update_dict_copy,
    wrap_batchnorm,
)
from dib.utils.initialization import weights_init

from .helpers import (
    BASE_LOG,
    CORR_GROUPS,
    EPS_MIN,
    N_CORR,
    NotEnoughHeads,
    SingleInputModuleList,
    detach,
    mean_p_logits_parallel,
    mean_std,
)

try:
    import higher
except ImportError:
    pass

__all__ = [
    "DIBLoss",
    "IBEncoder",
    "DIBLossAltern",
    "DIBLossLinear",
    "DIBLossAlternLinear",
    "DIBLossAlternLinearExact",
    "DIBLossAlternHigher",
    "DIBLossZX",
]
logger = logging.getLogger(__name__)


class IBEncoder(nn.Module):
    """General class for *IB Encoders.

    Parameters
    ----------
    Encoder : nn.Module
        Uninitialized module that takes `x_shape` and `z_dim` as input.

    Q : nn.Module
        Functional family of the classifier. I.e. for sufficiency.

    x_shape : tuple, optional
        Size of the inputs.

    n_classes : int, optional
        Number of output classes.

    z_dim : int, optional
        Size of the representation.

    is_stochastic : bool, optional 
        Whether to use a stochastic encoder.

    n_test_samples : int, optional
        Number of samples of z to use if `is_stochastic` and testing.
        
    is_avg_trnsf : bool, optional   
        Whether to return the average representation or all of them. THe former is 
        useful for plotting.

    kwargs:
        Additional arguments to Encoder.

    Return
    ------
    if is_transform:
        z_sample : torch.tensor, shape = [n_samples, batch_size, z_dim]

    else :
        y_pred : torch.tensor, shape = [batch_size, n_classes]

        z_sample : torch.tensor, shape = [n_samples, batch_size, z_dim]

        p_zCx : MultivariateNormalDiag, shape = [batch_size, z_dim].
            Distribution of p(z|x) None when testing.
    """

    def __init__(
        self,
        Encoder,
        Q,
        x_shape=(1, 32, 32),
        n_classes=10,
        z_dim=256,
        is_stochastic=True,
        n_test_samples=12,
        is_avg_trnsf=False,
        is_limit_growth=False,
        is_wrap_batchnorm=False,
        **kwargs,
    ):

        super().__init__()
        self.is_transform = False
        self.z_dim = z_dim
        self.is_stochastic = is_stochastic
        self.n_test_samples = n_test_samples
        self._to_plot_activation = {}
        self.is_avg_trnsf = is_avg_trnsf
        self.n_classes = n_classes
        self.is_limit_growth = is_limit_growth
        self.is_wrap_batchnorm = is_wrap_batchnorm

        self.encoder = Encoder(
            x_shape=x_shape, z_dim=z_dim * 2 if is_stochastic else z_dim
        )

        if self.is_wrap_batchnorm:
            self.batch_norm = BatchNorm1dLast(num_features=z_dim, affine=False)

        self.Q_zy = Q(z_dim, n_classes)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):

        batch_size = X.size(0)
        n_samples = 1 if self.training else self.n_test_samples

        if self.is_stochastic:
            z_suff_stat = self.encoder(X)
            z_mean, z_std = z_suff_stat.view(batch_size, -1, 2).unbind(-1)

            if self.is_limit_growth:
                z_mean = torch.tanh(z_mean)
                z_std = torch.sigmoid(z_std)
            else:
                # -5 as in vib +  delta
                z_std = EPS_MIN + F.softplus(z_std - 5)

            p_zCx = MultivariateNormalDiag(z_mean, z_std)
            z_sample = p_zCx.rsample([n_samples])
        else:
            z_sample = self.encoder(X).unsqueeze(0)  # unsqueeze as if 1 sample
            p_zCx = None

        if self.is_wrap_batchnorm:
            z_sample = self.batch_norm(z_sample)

        if self.is_transform:
            if self.is_avg_trnsf:
                return z_sample.mean(0)
            else:
                return z_sample

        y_pred = mean_p_logits(self.Q_zy(z_sample))

        self.plot_activation_(dict(y_pred=y_pred, z_sample=z_sample.mean(0)))

        # by passes the issue that skorch thinks it's tensor
        return y_pred, z_sample, return_not_tensor(p_zCx)

    def plot_activation_(self, activations):
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


class DIBLoss(nn.Module):
    """DIB Loss.

    Parameters
    ----------
    Q : nn.Module
        Functional family for minimality. 

    n_per_target : dict
        Number of examples for each target.

    beta : float or callable, optional
        Regularization weight. If callable, should return a float given `is_training`.
        Importantly this only changes the gradients but not the model selection / loss plotting.

    n_per_head : list, optional
        Number of resampling of optimal nuisance. In practice what it is permutting the indices and 
        then applying nuisance. 

    n_classes : int, optional
        Number of output classes.

    z_dim : int, optional
        Size of the representation.

    conditional : {None, "H_Q[X|Z,Y]", "H_Q[X|Z]-H_Q[Y|Z]", "H_Q'[X|Z,Y]"}
        If None uses DIB. If `"H_Q[X|Z,Y]"` uses a different head for depending on the label, this is
        the most correct version of conditional IB but is computationally intensive. If `"H_Q[X|Z,Y]"` 
        approximate the previous method by giving the label Y as input to the heads, as a result the 
        heads have architecture Q' instead of the desired Q. `"H_Q[X|Z]-H_Q[Y|Z]"` index all X 
        differently for each labels, this is is only correct if all 
        possible labels are independent. 

    is_optimal_nuisance : bool, optional    
        Whether to use optimal nuisance instead of randomly hashed ones. I.e. uses the representation
        of X index in base |Y|. The number of heads will be the same regardless (to enable comparison).

    is_same_indices : bool, optional
        Whether to use the same indices for each `n_per_head`. 

    ZYCriterion : nn.Module, optional
        Criterion to compute the loss of Q_zy.

    map_target_position : dict, optional
        Dictionary that maps the type of target (e.g. "index") to its position in the 
        target.

    warm_Q_zx : int, optional
        Number of steps where warming up Q_zx (i.e. only backprop through them).

    z_norm_reg : float, optional
        Regularizer on the mean of the squared of the representations (when it is larger than 1).
        Note that we also use a hard boundary (factor of 10) when the norm mean squared us larger 
        than 10. This is crucial to get good results : adversarial training makes the representation 
        go to infinity without that.

    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        Q,
        n_per_target,
        beta=1,
        n_per_head=3,
        n_classes=10,
        z_dim=128,
        conditional=None,
        seed=123,
        is_optimal_nuisance=True,
        is_same_indices=False,
        ZYCriterion=CrossEntropyLossGeneralize,
        map_target_position={"target": 0, "index": 1},
        warm_Q_zx=0,
        z_norm_reg=0.0,
        weight_kl=None,  # dev
        is_zx_only=False,  # DEV
        is_use_y_as_n=False,  # DEV
        threshold_suff=float("inf"),  # DEV
        is_wrap_batchnorm=False,
        **kwargs,
    ):
        super().__init__()

        self.n_per_target = n_per_target
        self.beta = beta if callable(beta) else Constant(beta)
        self.n_per_head = n_per_head
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.conditional = conditional
        self.seed = seed
        self.is_optimal_nuisance = is_optimal_nuisance
        self.is_same_indices = is_same_indices
        self.map_target_position = map_target_position
        self.is_zx_only = is_zx_only  # DEV
        self.is_use_y_as_n = is_use_y_as_n  # DEV
        self.threshold_suff = float(threshold_suff)  # DEV
        self.warm_Q_zx = warm_Q_zx
        self.z_norm_reg = z_norm_reg
        self.weight_kl = weight_kl
        self.is_wrap_batchnorm = is_wrap_batchnorm

        # all the Q heads
        self.n_heads = self._get_n_heads()

        if self.is_wrap_batchnorm:
            Q = wrap_batchnorm(Q)

        self.Q_zx = self.get_Q_zx(Q)

        self.zy_criterion = ZYCriterion()
        self.to_store = dict()
        self.nuisances = []

        self.precompute_random_labelings_()
        self.compute_entropies_()
        self.compute_probabilities()
        self.reset_parameters()

        set_seed(
            self.seed
        )  # ensures same seed as VIB  (different due to init of heads)

    def get_Q_zx_helper(self, Q):
        """Return one set of classifiers from Z to {\Tilde{Y}_i}_i."""
        input_dim = (
            (self.z_dim +
             1) if self.conditional == "H_Q'[X|Z,Y]" else self.z_dim
        )
        return nn.ModuleList(
            [Q(input_dim, self.n_classes) for _ in range(self.n_heads)]
        )

    def get_Q_zx(self, Q):
        if self.conditional == "H_Q[X|Z,Y]":
            # different set of optimal (arg infimum) classifiers for each labels
            return nn.ModuleList(
                [self.get_Q_zx_helper(Q) for _ in range(self.n_classes)]
            )
        else:
            # same set of optimal classifiers regardless of label
            return self.get_Q_zx_helper(Q)

    def idcs_to_baseK_nuisance(self, i, idcs, base, n_nuisance):
        """Return the ith nuisance, using the base |Y| decomposition. Computations in numpy"""

        if not isinstance(self.nuisances, BaseRepIthDigits):
            self.nuisances = BaseRepIthDigits(base)
            self._max_idx = base ** n_nuisance

        if not np.all(idcs < self._max_idx):
            raise NotEnoughHeads(
                f"Max idx is base^heads={base}^{n_nuisance}={self._max_idx}. These idcs do not satisfy that : {idcs[(idcs >= self._max_idx)]}"
            )

        # ith base B expansion of the indices. E.g. for [494,7,58] and i=1 and base=10 would return [9,0,5]
        return self.nuisances(idcs, i)

    def _store(self, **to_store):
        for k, v in to_store.items():
            # value and count
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.to_store[k] = self.to_store.get(k, np.array([0.0, 0])) + np.array(
                [v, 1]
            )

    def precompute_random_labelings_(self):
        """Precompute the randomization of indices for different random labelings."""

        if not is_sorted([int(k) for k in self.n_per_target.keys()]):
            raise ValueError(
                f"The keys of `n_per_target` need to be sorted but ={self.n_per_target.keys()}"
            )

        if self.conditional is None or not self.is_optimal_nuisance:
            n_idcs = sum(self.n_per_target.values())
        else:
            n_idcs = list(self.n_per_target.values())

        n_permuters = self.n_per_head if self.is_optimal_nuisance else self.n_heads

        # precompute the permutations of indices
        if self.is_same_indices:
            self._rand_indcs = [get_idx_permuter(
                n_idcs, seed=self.seed)] * n_permuters
        else:
            self._rand_indcs = [
                get_idx_permuter(n_idcs, seed=self.seed + i) for i in range(n_permuters)
            ]

    def _get_n_heads(self):
        """Compute the number of heads that will be needed."""
        if self.conditional in ["H_Q[X|Z,Y]", "H_Q[X|Z]-H_Q[Y|Z]", "H_Q'[X|Z,Y]"]:
            # if using concateate then will use the same heads for each labels but concat the
            # to the representation to make sure that you don't remove label information from
            # the representation

            n_to_cover = max(self.n_per_target.values())

        elif self.conditional is None:
            # if not conditional then you need to cover all of |X| regardless of how many fall in each labels
            n_to_cover = sum(self.n_per_target.values())

        else:
            raise ValueError(f"Unkown conditional={self.conditional}")

        self.n_covering = math.ceil(math.log(n_to_cover, self.n_classes))

        n_heads = self.n_per_head * self.n_covering

        logger.info(f"nheads: {n_heads}")

        return n_heads

    def compute_probabilities(self):
        n_per_target = np.array(list(self.n_per_target.values()))
        n_idcs = n_per_target.sum()

        # list of p_{\Tilde{Y}_i}
        self.p_ni = []
        for i in range(self.n_heads):
            n_i = self.get_ith_nuisance(i, torch.arange(n_idcs).long())
            self.p_ni.append(label_distribution(n_i, self.n_classes))

        # list of list of p_{\Tilde{Y}_i|Y}
        self.p_niCY = []
        for n_for_target in n_per_target:
            p_niCy = []  # for a specific target
            for i in range(self.n_heads):
                n_i = self.get_ith_nuisance(
                    i, torch.arange(n_for_target).long())
                p_niCy.append(label_distribution(n_i, self.n_classes))
            self.p_niCY.append(p_niCy)

    def compute_entropies_(self):
        # all of these assume uniform distribution
        n_per_target = np.array(list(self.n_per_target.values()))
        n_idcs = n_per_target.sum()

        self.H_x = math.log(n_idcs, BASE_LOG)
        self.H_y = entropy(n_per_target, base=BASE_LOG)
        self.H_yCx = sum(
            math.log(N, BASE_LOG) * N / n_per_target.sum() for N in n_per_target
        )

    def reset_parameters(self):
        weights_init(self)
        self._counter_warm_Q_zx = 0

    def get_ith_nuisance(self, i, x_idx, is_torch=True):
        if not self.is_optimal_nuisance:
            if is_torch:
                # making sure you are on correct device
                self._rand_indcs[i] = self._rand_indcs[i].to(x_idx.device)
            n_i = self._rand_indcs[i][x_idx] % self.n_classes
            return n_i

        # i_rand_idx : which n_per_head (i.e. which group of g=random index). => if n_per_head `n_per_head=1` then `i_rand_idx=0`
        # i_in_rand_idx : index in the i_mc group => if n_per_head `n_per_head=1` then `i_in_rand_idx=i`
        i_rand_idx, i_in_rand_idx = divmod(i, self.n_covering)

        if is_torch:
            # making sure you are on correct device
            self._rand_indcs[i_rand_idx] = self._rand_indcs[i_rand_idx].to(
                x_idx.device)

        # to have a better approximation of H_Q[X|Z] we actually compute a (MC approx.)
        # expectation over all possible permutation of indices. Indeed, the indices could
        # have been given differently which can change the optimization process. Each mc approx
        # corresponds to an other possible such index labeling
        if max(x_idx) >= len(self._rand_indcs[i_rand_idx]):
            raise NotEnoughHeads(
                f"max(x_idx)={max(x_idx)}>{len(self._rand_indcs[i_rand_idx])}=len(self._rand_indcs[i_rand_idx])"
            )

        permuted_x_idx = self._rand_indcs[i_rand_idx][x_idx]

        n_i = self.idcs_to_baseK_nuisance(
            i_in_rand_idx, to_numpy(
                permuted_x_idx), self.n_classes, self.n_covering
        )

        if is_torch:
            n_i = torch.from_numpy(n_i).to(x_idx.device)

        return n_i

    def compute_zy_loss(self, y_pred, targets):
        """Compute loss for the classifier Z -> Y."""
        # H_Q[Y|Z]
        H_Q_yCz = self.zy_criterion(y_pred, targets) / math.log(BASE_LOG)

        #! H[Y] is the training one (even at test time)
        # DI_Q[Y;Z] = H[Y] - H_Q[Y|Z]
        DIQ_yz = self.H_y - H_Q_yCz

        self._store(H_Q_yCz=H_Q_yCz, DIQ_yz=DIQ_yz)

        return H_Q_yCz

    def _compute_H_Q_nCzy(self, z_sample, targets, is_cap):
        """Compute \sum_i \sum_y H_Q[\Tilde{Y}_i|z,y]"""

        ys = extract_target(targets, self.map_target_position)
        idcs = targets[self.map_target_position["index"]]

        H_Q_nCzy = 0
        DI_Q_nCzy = 0
        heads_delta_acc_nCy = 0
        present_labels = ys.unique()

        for present_label in present_labels:
            # select everything based on the label => conditional prediction
            selector_cond = ys == present_label
            x_idcs_cond = idcs[selector_cond]
            z_sample_cond = z_sample[:, selector_cond, :]
            Q_zx_cond = self.Q_zx[present_label]
            (
                H_Q_nCzy_cond,
                heads_delta_acc_n_cond,
                DI_Q_nCzy_cond,
            ) = self._compute_H_Q_nCz(
                z_sample_cond,
                x_idcs_cond,
                Q_zx_cond,
                is_cap,
                targets[self.map_target_position["target"]],
            )
            H_Q_nCzy = H_Q_nCzy + H_Q_nCzy_cond / len(present_labels)
            heads_acc_nCy = heads_delta_acc_nCy + heads_delta_acc_n_cond / len(
                present_labels
            )
            DI_Q_nCzy = DI_Q_nCzy + DI_Q_nCzy_cond / len(present_labels)

        return H_Q_nCzy, heads_delta_acc_nCy, DI_Q_nCzy

    def _compute_H_Q_ni(self, n_i, i, q_niCz):
        """Estimate the worst case cross entropy (i.e. predicting with the marginal)."""
        self.p_ni[i].logits = self.p_ni[i].logits.to(n_i.device)
        p_ni = self.p_ni[i].logits.repeat(len(n_i), 1)

        H_Q_ni = F.cross_entropy(p_ni, n_i) / math.log(BASE_LOG)
        # also return the accuracy of marginal
        marginal_acc_ni = accuracy_score(n_i.cpu(), p_ni.argmax(-1).cpu())

        return H_Q_ni, marginal_acc_ni

    def batch_predict_heads(self, z_sample, Q_zx):
        if len(Q_zx) == 0:
            return []

        #! NOT IN PARALLEL BECAUSE WAS NOT WORKING WELL : still has to see why can't parallelize
        return [mean_p_logits(Q_zxi(z_sample)) for Q_zxi in Q_zx]

    def _compute_H_Q_nCz(self, z_sample, x_idcs, Q_zx, is_cap, y):
        """Compute \sum_i H_Q[\Tilde{Y}_i|z]"""

        # for all heads, predict, and average across num. samples
        q_nCz = self.batch_predict_heads(z_sample, Q_zx)

        H_Q_nCz = 0
        DI_Q_nCz = 0
        heads_delta_acc_n = 0

        for i, q_niCz in enumerate(q_nCz):
            n_i = self.get_ith_nuisance(i, x_idcs)

            if self.is_use_y_as_n:
                n_i = y

            # H_Q[\Tilde{Y}_i|Z]
            H_Q_niCz = F.cross_entropy(q_niCz, n_i) / math.log(BASE_LOG)

            # H_Q[\Tilde{Y}_i]
            H_Q_ni, marginal_acc_ni = self._compute_H_Q_ni(n_i, i, q_niCz)

            if is_cap:
                # in case your loss is worst than marginal, then don't backprop through encoder
                # but still improve the head (because it means that the internal optim is not correct)
                # accessorily this will also ensure positivity of estimated decodable information
                if H_Q_niCz > H_Q_ni:
                    H_Q_niCz = H_Q_niCz * 0 + H_Q_ni

            # DI_Q[\Tilde{Y}_i <- Z] = H_Q_ni - H_Q_niCz
            DI_Q_niz = H_Q_ni - H_Q_niCz

            # H_Q[\Tilde{Y}|Z] = \sum_i H_Q[\Tilde{Y}_i|Z]
            # DI_Q[\Tilde{Y} <- Z] = \sum_i DI_Q[\Tilde{Y}_i <- Z]
            # only divide by self.n_per_head because want to sum all heads besides the n_per_head should avg
            H_Q_nCz = H_Q_nCz + H_Q_niCz / self.n_per_head
            DI_Q_nCz = DI_Q_nCz + DI_Q_niz / self.n_per_head

            # only save delta accuracy with marginal
            heads_acc_ni = accuracy_score(n_i.cpu(), q_niCz.argmax(-1).cpu())
            heads_delta_acc_ni = heads_acc_ni - marginal_acc_ni
            heads_delta_acc_n = heads_delta_acc_n + \
                heads_delta_acc_ni / len(q_nCz)

        return H_Q_nCz, heads_delta_acc_n, DI_Q_nCz

    def compute_zx_loss_helper(self, z_sample, y, is_cap, is_store=True):
        if self.conditional == "H_Q[X|Z,Y]":
            H_loss, head_delta_acc, DI_loss = self._compute_H_Q_nCzy(
                z_sample, y, is_cap
            )

        else:
            # if self.conditional == "H_Q'[X|Z,Y]" actually computing H_Q_nCzy
            H_loss, head_delta_acc, DI_loss = self._compute_H_Q_nCz(
                z_sample,
                y[self.map_target_position["index"]],
                self.Q_zx,
                is_cap,
                y[self.map_target_position["target"]],  # DEV
            )

        if is_store:
            # storing for plots
            if self.conditional is None:
                # H_Q[X|Z] := \sum_i  H_Q[\Tilde{Y}_i|Z]
                # I_Q[X<-Z] := \sum_i  H[\Tilde{Y}_i] - H_Q[\Tilde{Y}_i|Z]
                self._store(H_Q_xCz=H_loss,
                            h_delta_acc=head_delta_acc, DIQ_xz=DI_loss)

            elif "[X|Z,Y]" in self.conditional:
                # H_Q[X|Z,Y] := \sum_y \sum_{\Tilde{Y}_i \neq Y} H_Q[\Tilde{Y}_i|Z,y]
                # I_Q[X<-Z|Y] := \sum_{\Tilde{Y}_i \neq Y} H[\Tilde{Y}_i | Y] - H_Q[X|Z,Y]
                self._store(
                    H_Q_xCzy=H_loss, h_delta_acc=head_delta_acc, DIQ_xzCy=DI_loss
                )

            elif self.conditional == "H_Q[X|Z]-H_Q[Y|Z]":
                # H_Q[X|Z] - H_Q[Y|Z] := \sum_{\Tilde{Y}_i \neq Y}  H_Q[\Tilde{Y}_i|Z]
                # I_Q[X<-Z] - I_Q[Y<-Z] := \sum_{\Tilde{Y}_i \neq Y}  H[\Tilde{Y}_i] - H_Q[\Tilde{Y}_i|Z]
                self._store(
                    d_H_Q_xCz=H_loss, h_delta_acc=head_delta_acc, d_DIQ_xz=DI_loss
                )

        return H_loss

    def compute_zx_loss_encoder(self, z_sample, targets):
        """Compute all losses for the encoder Z -> X."""

        curr_beta = self.beta(self.training) if self.beta(
            False) is not None else 0

        # capping : due to minimax one solution is to get infinitely bad (no need if no minimax)
        is_cap = True if curr_beta >= 0 else False
        H_loss = self.compute_zx_loss_helper(
            z_sample, targets, is_cap=is_cap, is_store=True
        )

        # the representation should do poorly on predicting the base expansion of the index
        # Could use gradient reversal layer, but that would not allow the clamping to random loss
        zx_loss = -H_loss

        # add some regularization if mean square > 1. If not it explodes due to adversarial training
        z_norm = z_sample.pow(2).mean()
        if curr_beta >= 0 and self.z_norm_reg > 0 and z_norm > 1:

            z_norm_reg = self.z_norm_reg
            if z_norm > 10:  # hard boudnary at 10
                z_norm_reg = 100 * z_norm_reg

            # only if positive beta because this causes norm to explode
            zx_loss = zx_loss + z_norm_reg * z_norm

        zx_loss = curr_beta * zx_loss

        return zx_loss

    def compute_zx_loss(self, z_sample, targets):
        """Compute all losses Z -> X, for encoder and for the heads."""

        zx_loss_heads = self.compute_zx_loss_helper(
            z_sample.detach(),  # don't backprop through encoder
            targets,
            is_cap=False,
            is_store=False,
        )

        with no_grad_modules([self.Q_zx]):  # don't backprop through heads
            zx_loss_enc = self.compute_zx_loss_encoder(z_sample, targets)

        if self._counter_warm_Q_zx < self.warm_Q_zx:
            zx_loss_enc = detach(zx_loss_enc)  # for logging you still want

        # - detach to make sure that gradients flow but loss does not cancel when plotting
        # also note that `zx_loss_heads` IS NOT SCALED BY BETA
        zx_loss = zx_loss_enc + zx_loss_heads - detach(zx_loss_heads)

        return zx_loss

    def forward(self, out, targets):

        if self.training:
            self._counter_warm_Q_zx += 1

        y_pred, z_sample, p_zCx = out

        if p_zCx.out is not None:
            p_zCx_base = p_zCx.out.base_dist
            # z_norm : elementwise square, z_mean_norm : mean of absolute val, z_std : mean of standard dev
            self._store(
                z_norm=z_sample.pow(2).mean(),
                z_mean_norm=p_zCx_base.loc.abs().mean(),
                z_std=p_zCx_base.scale.mean(),
            )

        if self.conditional == "H_Q'[X|Z,Y]":
            target = (
                extract_target(targets, self.map_target_position)
                .unsqueeze(0)
                .unsqueeze(-1)
                .float()
            )
            target = torch.repeat_interleave(target, z_sample.size(0), dim=0)
            z_sample = torch.cat([z_sample, target], dim=-1)

        try:
            zx_loss = self.compute_zx_loss(z_sample, targets)
        except NotEnoughHeads as e:
            # if not training then don't raise exception (because the indexing might be off in which
            # case your predictor cannot comoute zx_loss). But you don't want to never compute this
            # loss as for evaluation we give the training data but self.training=False
            if self.training:
                raise e
            zx_loss = 0

        if self.weight_kl is not None:
            p_zCx = p_zCx.out
            mean_0 = torch.zeros_like(p_zCx.base_dist.loc)
            std_1 = torch.ones_like(p_zCx.base_dist.scale)
            p_z = MultivariateNormalDiag(mean_0, std_1)
            kl = kl_divergence(p_zCx, p_z).mean(0) / math.log(BASE_LOG)
            zx_loss = zx_loss + self.weight_kl * kl

        zy_loss = self.compute_zy_loss(y_pred, targets)

        if zy_loss > self.threshold_suff:  # DEV
            zx_loss = zx_loss * 0 + detach(zx_loss)

        if self._counter_warm_Q_zx <= self.warm_Q_zx:
            # still return loss but no grad
            zy_loss = zy_loss * 0 + detach(zy_loss)

        if self.is_zx_only:  # DEV
            zy_loss = 0 * zy_loss

        self._store(aux_loss=zx_loss)

        if not self.training:
            # when evaluating the loss should be log likelihood for checkpointing
            return zy_loss

        return zy_loss + zx_loss


class DIBLossZX(DIBLoss):
    def forward(self, z_sample, targets):
        z_sample = z_sample.unsqueeze(
            0
        )  # when only computing DIQ the samples will be squeezed

        if self.conditional == "H_Q'[X|Z,Y]":
            target = (
                extract_target(targets, self.map_target_position)
                .unsqueeze(0)
                .unsqueeze(-1)
                .float()
            )
            target = torch.repeat_interleave(target, z_sample.size(0), dim=0)
            z_sample = torch.cat([z_sample, target], dim=-1)

        try:
            zx_loss = self.compute_zx_loss(z_sample, targets)
        except NotEnoughHeads as e:
            # if not training then don't raise exception (because the indexing might be off in which
            # case your predictor cannot comoute zx_loss). But you don't want to never compute this
            # loss as for evaluation we give the training data but self.training=False
            if self.training:
                raise e
            zx_loss = 0

        self._store(aux_loss=zx_loss)

        return zx_loss


class DIBLossAltern(DIBLoss):
    def __init__(
        self,
        *args,
        Optimizer=partial(torch.optim.Adam, lr=0.001),
        altern_minimax=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer = Optimizer(self.Q_zx.parameters())
        self.altern_minimax = altern_minimax

    def optimize_heads(self, z_sample, targets):

        z_sample = z_sample.detach()  # don't backprop through encoder

        def closure():
            self.optimizer.zero_grad()
            zx_loss = self.compute_zx_loss_helper(
                z_sample, targets, is_cap=False, is_store=False
            )
            zx_loss.backward()
            return zx_loss

        for i in range(self.altern_minimax):
            self.optimizer.step(closure)

    def compute_zx_loss(self, z_sample, targets):
        """Compute all losses Z -> X, for encoder and for the heads."""
        if not self.training:
            # ensure that no leeking of test information when training on test
            Q_zx_old = copy.deepcopy(self.Q_zx)

        try:
            # make sure grad enabled even when testing (better estimation of H_Q[X|Z])
            with torch.enable_grad():
                self.optimize_heads(z_sample, targets)

            with no_grad_modules([self.Q_zx]):
                zx_loss_enc = self.compute_zx_loss_encoder(z_sample, targets)

        finally:
            if not self.training:
                self.Q_zx = Q_zx_old

        if self._counter_warm_Q_zx <= self.warm_Q_zx:
            zx_loss_enc = zx_loss_enc * 0 + detach(zx_loss_enc)

        return zx_loss_enc


class DIBLossAlternHigher(DIBLoss):
    def __init__(
        self,
        *args,
        Optimizer=partial(torch.optim.Adam, lr=0.001),
        altern_minimax=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.Optimizer = Optimizer
        self.altern_minimax = altern_minimax

    def batch_predict_heads(self, z_sample, Q_zx):
        if len(Q_zx) == 0:
            return []
        return [mean_p_logits(pred_i) for pred_i in Q_zx(z_sample)]

    def get_Q_zx_helper(self, Q):
        """Return one set of classifiers from Z to {\Tilde{Y}_i}_i."""
        input_dim = (
            (self.z_dim +
             1) if self.conditional == "H_Q'[X|Z,Y]" else self.z_dim
        )
        return SingleInputModuleList(
            [Q(input_dim, self.n_classes) for _ in range(self.n_heads)]
        )

    def get_Q_zx(self, Q):
        if self.conditional == "H_Q[X|Z,Y]":
            # different set of optimal (arg infimum) classifiers for each labels
            return SingleInputModuleList(
                [self.get_Q_zx_helper(Q) for _ in range(self.n_classes)]
            )
        else:
            # same set of optimal classifiers regardless of label
            return self.get_Q_zx_helper(Q)

    def compute_zx_loss(self, z_sample, targets):
        """Compute all losses Z -> X, for encoder and for the heads."""

        Q_zx_curr, self.Q_zx = self.Q_zx, None

        inner_opt = self.Optimizer(Q_zx_curr.parameters())

        try:
            with higher.innerloop_ctx(
                Q_zx_curr, inner_opt, track_higher_grads=self.training,
            ) as (fnet, diffopt):

                self.Q_zx = fnet  # setting temporary attribute for computations

                # make sure grad enabled even when testing (better estimation of H_Q[X|Z])
                with torch.enable_grad():
                    # Take a few gradient steps to find good heads
                    for _ in range(self.altern_minimax):
                        zx_loss = self.compute_zx_loss_helper(
                            z_sample, targets, is_cap=False, is_store=False
                        )
                        diffopt.step(zx_loss)

                zx_loss_enc = self.compute_zx_loss_encoder(z_sample, targets)

                if self.training:
                    # reloading your weights so that you can do warm start for next step
                    # not when testing (avoiding leakage of test data)
                    Q_zx_curr.load_state_dict(fnet.state_dict())

        finally:
            self.Q_zx = Q_zx_curr

        if self._counter_warm_Q_zx <= self.warm_Q_zx:
            zx_loss_enc = zx_loss_enc * 0 + detach(zx_loss_enc)

        return zx_loss_enc


class DIBLossLinear(DIBLoss):
    def batch_predict_heads(self, z_sample, Q_zx):
        outs = Q_zx(z_sample)
        return [mean_p_logits(el) for el in torch.chunk(outs, self.n_heads, dim=-1)]

    def get_Q_zx_helper(self, Q):
        """Return one set of classifiers from Z to {\Tilde{Y}_i}_i."""
        input_dim = (
            (self.z_dim +
             1) if self.conditional == "H_Q'[X|Z,Y]" else self.z_dim
        )
        return nn.Linear(input_dim, self.n_classes * self.n_heads)


class DIBLossAlternLinear(DIBLossAltern):
    batch_predict_heads = DIBLossLinear.batch_predict_heads
    get_Q_zx_helper = DIBLossLinear.get_Q_zx_helper


class DIBLossAlternLinearExact(DIBLoss):
    def __init__(self, *args, is_Q_zy=False, seed=123, **kwargs):
        self.Q = partial(
            SGDClassifier, loss="log", random_state=seed, warm_start=True, alpha=0.01,
        )  # small regularization for stability

        super().__init__(*args, seed=seed, **kwargs)

    def predict_head(self, z_sample, Q_z):
        W = torch.from_numpy(Q_z.coef_).float().to(z_sample.device).T
        b = torch.from_numpy(Q_z.intercept_).float().to(z_sample.device)
        return z_sample @ W + b

    def batch_predict_heads(self, z_sample, Q_zx):
        return [mean_p_logits(self.predict_head(z_sample, Q_zxi)) for Q_zxi in Q_zx]

    def get_Q_zx(self, Q):
        if self.conditional == "H_Q[X|Z,Y]":
            # different set of optimal (arg infimum) classifiers for each labels
            return [self.get_Q_zx_helper(Q) for _ in range(self.n_classes)]
        else:
            # same set of optimal classifiers regardless of label
            return self.get_Q_zx_helper(Q)

    def get_Q_zx_helper(self, Q):
        """Return one set of classifiers from Z to {\Tilde{Y}_i}_i."""
        return [self.Q() for _ in range(self.n_heads)]

    def optimize_heads(self, z_sample, targets):
        z_sample = np.squeeze(
            to_numpy(z_sample).astype(np.float64), axis=0
        )  # squeeze sample (assume one during training)
        x_idcs = to_numpy(targets[self.map_target_position["index"]])

        for i, Q_zxi in enumerate(self.Q_zx):
            n_i = self.get_ith_nuisance(i, x_idcs, is_torch=False)
            if len(np.unique(n_i)) > 1:
                # if not error + it's useless
                self.Q_zx[i] = Q_zxi.fit(z_sample, n_i)

    def compute_zx_loss(self, z_sample, targets):
        """Compute all losses Z -> X, for encoder and for the heads."""

        if self.training:
            self.optimize_heads(z_sample, targets)

        zx_loss_enc = self.compute_zx_loss_encoder(z_sample, targets)

        return zx_loss_enc
