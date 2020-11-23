"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import glob
import logging
import math
import os
from functools import partial, partialmethod

import numpy as np
import pandas as pd
import scipy
import skorch
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, log_loss
from skorch.callbacks import Freezer, LRScheduler
from skorch.callbacks.scoring import ScoringBase
from skorch.dataset import unpack_data, uses_placeholder_y
from skorch.history import History
from torch.nn.utils.fusion import fuse_conv_bn_eval

from dib.predefined import MLP
from dib.training import NeuralNetTransformer
from dib.training.helpers import clone_trainer
from dib.training.trainer import _get_params_for_optimizer
from dib.transformers import BASE_LOG, DIBLoss, DIBLossZX
from dib.utils.helpers import (
    CrossEntropyLossGeneralize,
    extract_target,
    set_seed,
    to_numpy,
)
from utils.helpers import (
    SFFX_TOAGG,
    BatchNormConv,
    StoreVarGrad,
    add_noise_to_param_,
    batchnorms2convs_,
    clip_perturbated_param_,
    cont_tuple_to_tuple_cont,
    dict_none_toNaN,
    get_device,
    get_exponential_decay_gamma,
    get_float_value,
    merge_dicts,
    save_pattern,
    set_requires_grad,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass


__all__ = ["eval_loglike", "eval_clf"]

FILE_SCORE = "score.csv"
FILE_CLF_REP = "clf_report.csv"

logger = logging.getLogger(__name__)


def eval_corr_gen(trainer, dataset, mode="train"):
    """
    Evaluates a classifier for correlation with generalization using some of the best predictors
    of generalization from each section of "FANTASTIC GENERALIZATION MEASURES AND WHERE TO FIND THEM.
    Also does the usual classifier evaluation
    """
    # measure usual classification (i.e. how well generalizes)
    out_eval_clf = eval_clf(trainer, dataset)

    if mode == "test":
        # only do correlation measure if "train mode"
        return out_eval_clf

    trainer = clone_trainer(trainer)
    logger.info(f"len(dataset)={len(dataset)}")

    # Variance of gradients (for classifier and transformer)
    logger.info("var_grad")
    var_grad = get_var_grad(trainer, dataset)
    logger.info(logger)

    logger.info("d_H_Q_xCz")
    # before freezing the net
    d_H_Q_xCz = get_H_Q_xCz(
        trainer, dataset, "d_H_Q_xCz", conditional="H_Q[X|Z]-H_Q[Y|Z]"
    )

    logger.info("H_Q_xCz")
    # H_Q[X|Z]
    H_Q_xCz = get_H_Q_xCz(trainer, dataset, "H_Q_xCz")

    # H_Q+[X|Z]
    logger.info("d_H_Q+_xCz")
    d_H_Qp_xCz = get_H_Q_xCz(
        trainer,
        dataset,
        "H_Q_xCz",
        Q_zx=partial(
            MLP, hidden_size=2048, n_hidden_layers=trainer.module_.clf.n_hidden_layers
        ),
    )

    # H_Q-[X|Z]
    logger.info("d_H_Q-_xCz")
    d_H_Qm_xCz = get_H_Q_xCz(
        trainer,
        dataset,
        "H_Q_xCz",
        Q_zx=partial(
            MLP, hidden_size=2, n_hidden_layers=trainer.module_.clf.n_hidden_layers
        ),
    )

    # freezes all batchnorm layers by converting them to convolutions
    trainer.module_.eval()
    batchnorms2convs_(trainer.module_)

    # Entropy of the logits
    logger.info("entropy")
    y_pred_proba = trainer.predict_proba(dataset)
    y_pred_ent = scipy.stats.entropy(
        y_pred_proba, axis=1, base=BASE_LOG).mean()

    # Path Norm (for classifier and transformer)
    logger.info("path_norm")
    path_norm = get_path_norm(trainer, dataset)

    # Sharpness magnitude => max (relative) change in weights that cause less than 1 diff in log like
    logger.info("sharp_mag")
    sharp_mag = get_sharp_mag(trainer, dataset)

    return dict(
        y_pred_ent=y_pred_ent,
        path_norm=path_norm,
        var_grad=var_grad,
        sharp_mag=sharp_mag,
        H_Q_xCz=H_Q_xCz,
        d_H_Qp_xCz=d_H_Qp_xCz,
        d_H_Qm_xCz=d_H_Qm_xCz,
        d_H_Q_xCz=d_H_Q_xCz,
        **out_eval_clf,
    )


def eval_clf(trainer, dataset, **kwargs):
    """Evaluates a classifier on a dateset."""
    y_pred_proba = trainer.predict_proba(dataset)
    loglike = -log_loss(dataset.targets, y_pred_proba)
    y_pred = y_pred_proba.argmax(-1)
    accuracy = accuracy_score(dataset.targets, y_pred)
    top5_acc = top_n_accuracy_score(dataset.targets, y_pred_proba, n=5)

    return dict(accuracy=accuracy, top5_acc=top5_acc, loglike=loglike)


def eval_trnsf(trainer, dataset, **kwargs):
    """
    Evaluates a transformer on a dateset by returning everything in history that starts with `valid_`.
    Difference with `eval_clf` is that saves all temporary variables.
    """
    trainer.check_data(dataset, None)
    trainer.notify("on_epoch_begin", dataset_train=dataset,
                   dataset_valid=dataset)

    trainer._single_epoch(dataset, training=False, epoch=0)

    # don't call "on epoch end" because should not checkpoint (but still want scoring)
    for _, cb in trainer.callbacks_:
        # score everything on validation because you didn't train
        if isinstance(cb, ScoringBase) and not cb.on_train:
            cb.on_epoch_end(trainer, dataset_train=dataset,
                            dataset_valid=dataset)

    return {
        k.replace("valid_", ""): v
        for k, v in trainer.history[-1].items()
        if k.startswith("valid_")
        and k not in ["valid_batch_count"]
        and "_best" not in k
    }


def eval_trainer_log(
    trainer,
    dataset,
    csv_score_pattern,
    chckpnt_dirnames,
    hyperparameters,
    evaluator=eval_clf,
    tensorboard_dir=None,
    is_append=False,
    mode="test",
    epoch="last",
    file_score=FILE_SCORE,
    file_clf_rep=FILE_CLF_REP,
    **kwargs,
):
    """Evaluate a trainer and log it."""

    to_log = evaluator(trainer, dataset, mode=mode, **kwargs)

    # first save the header then actual
    if not is_append:
        save_pattern(chckpnt_dirnames, csv_score_pattern, file_score)

    for metric_name, score in to_log.items():
        save_pattern(
            chckpnt_dirnames,
            csv_score_pattern,
            file_score,
            formatting=dict(epoch=epoch, metric=metric_name,
                            mode=mode, score=score),
            logger=logger,
        )

    if tensorboard_dir is not None:
        with SummaryWriter(log_dir=tensorboard_dir + "hypopt/") as w:
            w.add_hparams(
                # tensorboard does not accept None
                dict_none_toNaN(hyperparameters),
                {f"hparam/{metric_name}": score},
            )


def eval_loglike(trainer, dataset, seed=123, **kwargs):
    """Return the log likelihood for each image in order."""
    set_seed(seed)  # make sure same order and indices for context and target
    trainer.module_.to(trainer.device)
    trainer.criterion_.is_return_all = True
    y_valid_is_ph = uses_placeholder_y(dataset)
    all_losses = []

    trainer.notify("on_epoch_begin", dataset_valid=dataset)
    for data in trainer.get_iterator(dataset, training=False):
        Xi, yi = unpack_data(data)
        yi_res = yi if not y_valid_is_ph else None
        trainer.notify("on_batch_begin", X=Xi, y=yi_res, training=False)
        step = trainer.validation_step(Xi, yi, **kwargs)
        trainer.notify("on_batch_end", X=Xi, y=yi_res, training=False, **step)
        all_losses.append(-step["loss"])  # use log likelihood instead of NLL

    trainer.criterion_.is_return_all = False
    return torch.cat(all_losses, dim=0).detach().cpu().numpy()


# credits : https://github.com/scikit-learn/scikit-learn/pull/8234
def top_n_accuracy_score(y_true, y_pred, n=5, normalize=True):
    """top N Accuracy classification score.
    
    For multiclass classification tasks, this metric returns the
    number of times that the correct class was among the top N classes
    predicted.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    
    y_pred : array-like, where for each sample, each row represents the
        likelihood of each possible label.
        The number of columns must be at least as large as the set of possible
        label values.

    normalize : bool, optional (default=True)
        If ``False``, return the number of top N correctly classified samples.
        Otherwise, return the fraction of top N correctly classified samples.
    
    Returns
    -------
    score : float
        If ``normalize == True``, return the proportion of top N correctly
        classified samples, (float), else it returns the number of top N
        correctly classified samples (int.)
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    
    See also
    --------
    accuracy_score
    
    Notes
    -----
    If n = 1, the result will be the same as the accuracy_score. If n is the
    same as the number of classes, this score will be perfect and meaningless.
    In cases where two or more classes are assigned equal likelihood, the
    result may be incorrect if one of those classes falls at the threshold, as
    one class must be chosen to be the nth class and the class chosen may not
    be the correct one.
    
    Examples
    --------
    >>> import numpy as np
    >>> y_pred = np.array([[0.1, 0.3, 0.4, 0.2],
    ...                     [0.4, 0.3, 0.2, 0.1],
    ...                     [0.2, 0.3, 0.4, 0.1],
    ...                     [0.8, 0.1, 0.025, 0.075]])
    >>> y_true = np.array([2, 2, 2, 1])
    >>> top_n_accuracy_score(y_true, y_pred, n=1)
    0.5
    >>> top_n_accuracy_score(y_true, y_pred, n=2)
    0.75
    >>> top_n_accuracy_score(y_true, y_pred, n=3)
    1.0
    >>> top_n_accuracy_score(y_true, y_pred, n=2, normalize=False)
    3
    """
    num_obs, num_labels = y_pred.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter


def _load_something(
    get_rows_cols_agg, pattern, base_dir="", mark_to_agg=lambda c: c + SFFX_TOAGG
):
    """Load the something from all the saved values .
    
    Parameters
    ----------
    get_rows_toaggreg : callable
        function that takes files, raw_files as input and return all the rows as well as the name 
        of columns over which to aggregate.

    pattern : str
        Pattern of files to load. Needs to start with `tmp_results`.

    base_dir : str, optional
        Base directory to prepend to pattern.
        
    aggregate : list of string, optional
        Aggregation methods. If singleton will show the results as a single level. If empty list,
        will not aggregate anything.

    mark_to_agg : callable, optional
        Function that marks columns to aggregate. 
    """

    raw_files = glob.glob(base_dir + pattern, recursive=True)

    # rm results/ and /score.csv and add experiment_ name
    files = [
        "/".join(
            col if i > 0 else f"experiment_{col}"
            for i, col in enumerate(f[len(base_dir):].split("/")[1:-1])
        )
        for f in raw_files
    ]

    rows, columns, toaggreg = get_rows_cols_agg(files, raw_files)

    # the first column (experiment will be wrong if underscores in the experiments )
    results = pd.DataFrame(rows, columns=columns)

    results = results.apply(pd.to_numeric, errors="ignore")
    results = results.apply(to_bool)

    def fn_rename(col):
        if col in toaggreg:
            return mark_to_agg(col)
        return col

    return results.rename(columns=fn_rename)


def to_bool(s):
    if s.dtypes != "object":
        return s
    return s.replace({"True": True, "False": False})


def load_results(
    pattern="tmp_results/**/score.csv",
    metrics=["test_acc", "test_loss"],
    metric_column_name="{mode}_{metric}",
    **kwargs,
):
    """Load the results from the folder.
    
    Parameters
    ----------
    pattern : str, optional
        Pattern of files to load. Needs to start with `tmp_results`.
        
    metrics : list of string, optional
        Precomputed metrics to load. E.g. ["test_acc","test_loss","test_top5_acc","train_loss"].

    metric_column_name : str, optional
        Name of the column containing the metric.
    """

    def get_rows_cols_agg(files, raw_files, metrics=metrics):

        rows = []
        for raw_file, file in zip(raw_files, files):
            # hyperparameters
            row = [file.split("/")[0][len("experiment") + 1:]]

            # hyperparameters
            row += [str_to_val(folder.split("_")[-1])
                    for folder in file.split("/")[1:]]

            # metrics
            score = pd.read_csv(raw_file)
            # only keep numeric
            score = score[pd.to_numeric(
                score["{score}"], errors="coerce").notnull()]
            score["{score}"] = pd.to_numeric(score["{score}"])
            score = score.pivot_table(
                columns=metric_column_name, values="{score}", index="{epoch}"
            )
            score = score.reindex(columns=metrics).reset_index()

            for _, r in score.iterrows():
                rows.append(row + list(r.values))

        columns = (
            ["experiment"]
            + ["_".join(folder.split("_")[:-1])
               for folder in files[0].split("/")][1:]
            + ["epochs"]
            + metrics
        )

        return rows, columns, metrics

    results = _load_something(get_rows_cols_agg, pattern=pattern, **kwargs)

    return results


def load_histories(
    pattern="tmp_results/**/transformer/last_epoch_history.json", **kwargs
):
    """Load the history of a model (validation and train variables saved at every epoch).
    
    Parameters
    ----------

    pattern : str, optional
        Pattern of files to load. Needs to start with `tmp_results`.
        
    kwargs : list of string, optional
        Additional arguments to `_load_something`. 
    """

    def is_plot(key, values):
        """Columns to plot from history."""
        if not (key.startswith("train") or key.startswith("valid")):
            return False
        if len(set(values)) == 1:
            return False
        if not all(isinstance(v, float) for v in values):
            return False
        return True

    def get_rows_cols_agg(files, raw_files):

        to_plot_col = set()
        rows = []
        for raw_file, file in zip(raw_files, files):
            const_row = dict()

            const_row["experiment"] = file.split(
                "/")[0][len("experiment") + 1:]

            # hyperparameters
            for col in file.split("/")[1:]:
                key = "_".join(col.split("_")[:-1])
                value = col.split("_")[-1]
                const_row[key] = str_to_val(value)

            history = History.from_file(raw_file)

            # prepare for line plots
            history_to_plot = {
                key: history[:, key]
                for key in history[0].keys()
                if is_plot(key, history[:, key])
            }

            # Checking same number epoch
            for i, (k, v) in enumerate(history_to_plot.items()):
                if i == 0:
                    old_k = k
                    old_len = len(v)

                if old_len != len(v):
                    raise ValueError(
                        f"Number of epochs not the same for (at least) {old_k} and {k}."
                    )

            for epoch, history_per_epoch in enumerate(
                cont_tuple_to_tuple_cont(history_to_plot)
            ):
                row = const_row.copy()
                row["epochs"] = epoch
                for key, value in history_per_epoch.items():
                    row[key] = value
                    to_plot_col.add(key)
                rows.append(row)

        return rows, None, list(to_plot_col)

    results = _load_something(get_rows_cols_agg, pattern=pattern, **kwargs)

    return results


def str_to_val(s):
    if s == "None":
        return None
    return s


def accuracy_filter_train(model, X, y, map_target_position, **kwargs):
    """
    Helper function that computes the accuracy but only the training examples.
    """
    target = to_numpy(extract_target(y, map_target_position))

    try:
        is_train = y[:, map_target_position["constant"]] == 1
        target = target[is_train]
        y_pred = model.predict_proba(X)[is_train]
        out = accuracy_score(target, y_pred.argmax(-1), **kwargs)
    except (IndexError, KeyError):
        out = accuracy(model, X, target)

    return out


def accuracy(model, X, y, **kwargs):
    """
    Compute the accuracy score of a Sklearn Classifier on a given dataset. 
    """
    y_pred = model.predict_proba(X)
    return accuracy_score(y, y_pred.argmax(-1), **kwargs)


def loglike(model, X, y, **kwargs):
    """Compute the loglikelihood (base e) score of a Sklearn Classifier on a given dataset."""
    y_pred_proba = model.predict_proba(X)
    return -log_loss(y, y_pred_proba, **kwargs)


def get_path_norm(trainer, dataset):
    """Compute the pathnorm as described in "FANTASTIC GENERALIZATION MEASURES AND WHERE TO FIND THEM".
    I.e. squares all parameters, foreward pass all ones and then take sqrt of output."""
    trainer = clone_trainer(trainer)
    # use the mean instead of sampling to make sure that not negative
    trainer.module_.transformer.is_use_mean = True

    # square all parameters
    with torch.no_grad():
        for name, W in trainer.module_.named_parameters():
            W.pow_(2)

    all_ones = dataset[0][0].unsqueeze(0).fill_(1)
    logits = trainer.forward(all_ones)[0]
    sum_logits = logits.sum().item()
    return sum_logits ** 0.5


def get_var_grad(trainer, dataset):
    """Compute the variance of the gradients."""
    trainer = clone_trainer(trainer)

    # compute also the gradients of the transformer
    trainer.module_.is_freeze_transformer = False

    # compute elementwise variance of parameters
    trainer.callbacks_.append(("store_grad", StoreVarGrad()))
    trainer.check_data(dataset, None)
    trainer.notify("on_epoch_begin", dataset_train=dataset,
                   dataset_valid=dataset)
    trainer._single_epoch(dataset, training=True, epoch=0)
    last_epoch = trainer.history[-1]["epoch"]

    # don't call "on epoch end" because should not checkpoint (but still want to compute variance)
    for _, cb in trainer.callbacks_:
        if isinstance(cb, StoreVarGrad):
            cb.on_epoch_end(trainer, dataset_train=dataset,
                            dataset_valid=dataset)
            var_grad = np.concatenate([v.flatten()
                                       for v in cb.var_grads.values()])

    return var_grad.mean()  # take mean over all parameters


class NegCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        # select label from the targets
        return -super().forward(input, target[0])


def get_sharp_mag(
    trainer,
    dataset,
    sigma_min=0,
    sigma_max=2,
    target_deviation=0.1,
    n_restart_perturbate=3,
    max_binary_search=50,
    is_relative=True,
):
    """
    Compute the sharpness magnitude 1/alpha'^2 described in [1].

    Notes
    -----
    - This is slightly different than [1] because the target deviation is on cross-entropy instead
    of accuracy (as we don't care about accuracy in our paper).

    Parameters
    ----------
    trainer : skorch.NeuralNet

    dataset : torch.utils.data.Dataset

    sigma_min : float, optional
        Minimum standard deviation of perturbation.

    sigma_max : float, optional
        Maximum standard deviation of perturbation.
    
    n_adv_perturbate : int, optional
        Number of steps to perform adversarial perturbation for. 

    n_restart_perturbate : int, optional
        Number of times restarting the perturbation (different initialization for adv perturbate).

    target_deviation : float, optional
        Maximum difference of log likelihood allowed.

    max_binary_search : int, optional
        Maximum number of binary search tries.

    References
    ----------
    [1] Jiang, Yiding, et al. "Fantastic Generalization Measures and Where to Find Them." 
    arXiv preprint arXiv:1912.02178 (2019).
    """
    trainer = clone_trainer(trainer)
    acc = accuracy(trainer, dataset, dataset.targets)

    # compute also the gradients of the transformer
    trainer.module_.is_freeze_transformer = False

    # reverses cross entropy to MAXIMIZE (adversarial)
    trainer.criterion = NegCrossEntropyLoss

    for bin_search in range(max_binary_search):
        sigma_min, sigma_max = get_sharp_mag_interval(
            trainer,
            acc,
            dataset,
            sigma_min,
            sigma_max,
            target_deviation,
            n_restart_perturbate,
            is_relative,
        )

        if sigma_min > sigma_max or math.isclose(sigma_min, sigma_max, rel_tol=1e-2):
            # if interval for binary search is very small stop
            break

    if bin_search == max_binary_search - 1:
        logger.info(
            f"Stopped early beacuase reached max_binary_search={max_binary_search}. [sigma_min,sigma_max]=[{sigma_min},{sigma_max}]"
        )

    return 1 / (sigma_max ** 2)


def get_sharp_mag_interval(
    unperturbated_trainer,
    unperturbated_acc,
    dataset,
    sigma_min,
    sigma_max,
    target_deviation,
    n_restart_perturbate,
    is_relative,
):
    sigma_new = (sigma_min + sigma_max) / 2
    worst_acc = math.inf

    unperturbated_params = {
        name: param.detach()
        for name, param in unperturbated_trainer.module_.named_parameters()
    }

    for _ in range(n_restart_perturbate):
        trainer = clone_trainer(unperturbated_trainer,
                                is_reinit_besides_param=True)

        # add half of the possible noise to give some space for gradient ascent
        add_noise_to_param_(
            trainer.module_, sigma=sigma_new / 2, is_relative=is_relative
        )

        for i, data in enumerate(trainer.get_iterator(dataset, training=True)):
            Xi, yi = unpack_data(data)
            step = trainer.train_step(Xi, yi)

            # clipping perturbation value of added parameters to |w_i * sigma| or |sigma|
            clip_perturbated_param_(
                trainer.module_,
                unperturbated_params,
                sigma_new,
                is_relative=is_relative,
            )

            if not torch.isfinite(step["loss"]) or step["loss"].abs() > (
                abs(unperturbated_acc) + 10 * target_deviation
            ):
                # if loss is very large for one batch then no need to finish this loop
                return sigma_min, sigma_new

        curr_acc = accuracy(trainer, dataset, dataset.targets)
        worst_acc = min(worst_acc, curr_acc)

    deviation = abs(curr_acc - worst_acc)

    if math.isclose(unperturbated_acc, worst_acc, rel_tol=1e-2):
        # if not deviation is nearly zero can stop
        return sigma_new, sigma_new

    if deviation > target_deviation:
        sigma_max = sigma_new
    else:
        sigma_min = sigma_new

    return sigma_min, sigma_max


def get_H_Q_xCz(
    trainer,
    dataset,
    select,
    n_per_head=1,
    batch_size=256,
    lr=1e-2,
    max_epochs=100,
    Q_zx=None,
    **kwargs,
):

    trainer = clone_trainer(trainer)  # ensure not changing (shouldn't)

    trainer.module_.transformer.is_transform = False  # DIB Loss expects pred of label
    model = trainer.module_.transformer
    z_dim = model.z_dim

    def get_Q(*args, **kwargs):
        Q = copy.deepcopy(trainer.module_.clf)
        Q.reset_parameters()  # shouldn't be needed
        return Q

    count_targets = dataset.count_targets()
    n_per_target = {str(k): int(v) for k, v in count_targets.items()}

    dib = partial(
        DIBLoss,
        Q_zx if Q_zx is not None else get_Q,
        n_per_target,
        n_per_head=n_per_head,
        n_classes=dataset.n_classes,
        z_dim=z_dim,
        map_target_position=dataset.map_target_position,
        ZYCriterion=partial(
            CrossEntropyLossGeneralize,
            map_target_position=dataset.map_target_position,
            gamma=0,
        ),
        **kwargs,
    )

    # making sure that training of the parameter of the criterion
    NeuralNetTransformer._get_params_for_optimizer = partialmethod(
        _get_params_for_optimizer, is_add_criterion=True
    )

    set_requires_grad(model, False)

    net = NeuralNetTransformer(
        module=model,
        criterion=dib,
        optimizer=torch.optim.Adam,
        lr=lr,
        max_epochs=max_epochs,
        train_split=None,
        batch_size=batch_size,
        device=trainer.device,
        iterator_valid__batch_size=batch_size * 2,
        callbacks=[
            LRScheduler(
                torch.optim.lr_scheduler.ExponentialLR,
                gamma=get_exponential_decay_gamma(100, max_epochs),
            )
        ],
    )

    net.fit(dataset, y=None)

    out = eval_trnsf(net, dataset)

    return out[select]


class HQxCz:
    def __init__(self, trainer, dataset):

        trainer = clone_trainer(trainer)  # ensure not changing (shouldn't)

        self.Q_zy = copy.deepcopy(trainer.module_.clf)
        trainer.module_ = trainer.module_.transformer
        # using squeezing n_z_samples to work with forward call (working in determinstic setting )
        trainer.module_.is_avg_trnsf = True
        self.trainer = trainer

        Z = trainer.forward(dataset, training=False, device=trainer.device)
        targets = torch.stack([torch.tensor(y) for _, y in dataset])
        self.trnsf_dataset = skorch.dataset.Dataset(
            Z, [targets[:, i] for i in range(targets.size(1))]
        )  # compute transformed data onces

        self.z_dim = self.trnsf_dataset[0][0].size(-1)

        count_targets = dataset.count_targets()
        self.n_per_target = {str(k): int(v) for k, v in count_targets.items()}
        self.n_classes = dataset.n_classes
        self.map_target_position = dataset.map_target_position

    def __call__(
        self, select, n_per_head=1, batch_size=256, lr=1e-2, max_epochs=100, **kwargs
    ):
        def get_Q(*args, **kwargs):
            Q = copy.deepcopy(self.Q_zy)
            Q.reset_parameters()  # shouldn't be needed
            return Q

        dib = partial(
            DIBLossZX,
            get_Q,
            self.n_per_target,
            n_per_head=n_per_head,
            n_classes=self.n_classes,
            z_dim=self.z_dim,
            map_target_position=self.map_target_position,
            ZYCriterion=partial(
                CrossEntropyLossGeneralize,
                map_target_position=self.map_target_position,
                gamma=0,
            ),
            **kwargs,
        )

        # making sure that training of the parameter of the criterion
        NeuralNetTransformer._get_params_for_optimizer = partialmethod(
            _get_params_for_optimizer, is_add_criterion=True
        )

        net = NeuralNetTransformer(
            module=nn.Identity,
            criterion=dib,
            optimizer=torch.optim.Adam,
            lr=lr,
            max_epochs=max_epochs,
            train_split=None,
            batch_size=batch_size,
            device=self.trainer.device,
            iterator_valid__batch_size=batch_size * 2,
            callbacks=[
                LRScheduler(
                    torch.optim.lr_scheduler.ExponentialLR,
                    gamma=get_exponential_decay_gamma(100, max_epochs),
                )
            ],
        )

        net.fit(self.trnsf_dataset, y=None)

        out = eval_trnsf(net, self.trnsf_dataset)

        return out[select]
