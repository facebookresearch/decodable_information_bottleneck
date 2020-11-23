"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import torch
from torch.distributions import Categorical, Independent, Normal


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


class NoDistribution:
    def __init__(self, x):
        self.x = x

    def rsample(self):
        return self.x


def straight_through(soft_samples, f):
    """
    Take soft_samples and transform them with f(straight_through) 
    while keeping it differentiable.
    """
    detached = soft_samples.detach()
    # removes a detached version of the soft X and adds the real X
    # to emulate the fact that we add some non differentiable noise which just
    # hapens to make the variable as you want. I.e the total is still differentiable
    detached_res = f(detached)
    detached_diff = detached_res - detached
    res = detached_diff + soft_samples
    return res


def softmax_to_onehot(X, dim=1):
    """Moves a vector on the simplex to the closes vertex."""
    max_idx = torch.argmax(X, dim, keepdim=True)
    one_hot = torch.zeros_like(X)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def label_distribution(labels, n_classes):
    """Return a categorical distribution of the labels."""

    probs = torch.zeros(n_classes, device=labels.device)
    label, counts = labels.unique(return_counts=True)
    probs[label] = counts.float() / counts.sum()

    return Categorical(probs=probs)


def entropy_labels(labels, n_classes, base=math.exp(1)):
    """Computes the entropy of labels."""
    probs = label_distribution(labels, n_classes)
    return probs.entropy().mean(0) / math.log(base)


def rm_conditioning(p_yCx):
    """Remove the conditioning of a distributions p(Y|X) -> p(Y) by taking a Monte Carlo Expectation
    of all besides current index.
    
    Parameters
    ----------
    q_yCx : torch.Tensor or torch.Distributions
        Distribution to uncondition. Each batch should be from a sample of conditioning 
        random variable X. Note that this should already be in pbabilities, not logits.
    """
    #! here i'm actually removing the current index so the estimate is slighlty biased,
    #! to unbias should giv weight 1/N instead of weight 0 to yourself
    p_y = torch.zeros_like(p_yCx)

    if isinstance(p_yCx, torch.Tensor):
        batch_size = p_yCx.size(0)
        for batch_idx in range(batch_size):
            p_y[batch_idx] = p_yCx[
                list(range(0, batch_idx)) +
                list(range(batch_idx + 1, batch_size))
            ].mean(0)

        return p_y
