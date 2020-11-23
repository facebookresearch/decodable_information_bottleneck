"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
import shutil

import skorch
import torch
from skorch.callbacks import EarlyStopping, LoadInitState, ProgressBar
from skorch.helper import predefined_split

from dib.training.helpers import FixRandomSeed
from dib.utils.helpers import set_seed

from .helpers import TensorBoard, clean_end_run, get_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass


__all__ = ["train_load"]

MOD_SUMM_FILENAME = "model_summary.txt"
logger = logging.getLogger(__name__)


def train_load(
    Model,
    datasets,
    chckpnt_dirnames,
    is_train=True,
    is_continue_train=False,
    is_continue_best=False,
    patience=None,
    is_progressbar=False,
    checkpoint_epochs=None,
    load_epoch=None,
    tensorboard_dir=None,
    seed=123,
    device="cuda" if torch.cuda.is_available else "cpu",
    callbacks=[],
    clean_after_run="training",
    monitor_best="valid_loss_best",
    is_load_criterion=True,  # DEV,
    is_return_init=False,
    **kwargs,
):
    """Train or load the model.

    Parameters
    ----------
    Model : sklearn.base.BaseEstimator
        Uninitialized model to train. 

    datasets : dictionary of torch.utils.data.Dataset
        Dictionary of the `"train"`, `"valid"`, and `"test"`.

    chckpnt_dirnames : list of str, optional
        Directories where checkpoints will be saved. 

    patience : int, optional
        Patience for early stopping. Only if given a a validation set.

    is_train : bool, optional
        Whether to train rather than load a pretrained model. If False, reverts the order of 
        chckpnt_dirnames. Such that loads from first file.

    is_continue_train : bool, optional
        Whether to continue training from the last checkpoint of the previous run.

    is_continue_best : bool, optional  
        Whether to continue training from the best model rather than last. If `is_continue_best`
        continues from the first checkpoint directory (i.e. result dir), but last one if not 
        (i.e. tmp dir). 

    is_progressbar : bool, optional
        Whether to train with a progressbar.

    checkpoint_epochs : list of int, optional
        List of int saves at each of these epochs.

    tensorboard_dir : str, optional
        Directory for saving tensorboard logs.

    load_epoch : int, optional
        What epoch to load if not `is_retrain` and saved multiple epochs with a 
        suffix `_epoch{e}`. By default : last.

    device : str, optional  
        Device on which to run the model.

    seed : int, optional
        Pseudo random seed. 

    callbacks : list, optional
        Initial callbacks.

    clean_after_run : ["training","all",None], optional
        Cleans the directory. If "training" removes all the checkpoiting needed for training
        (last epoch models and all the optimizer). If "all" also removes the best_epoch model.

    monitor_best : {"valid_loss_best", "valid_acc_best", "train_loss_best", "last", int}, optional 
        What should be monitored for the best model. If int this is simply a given epoch.

    kwargs : 
        Additional arguments to the model.
    """
    set_seed(seed)

    logger.info(f"Using {chckpnt_dirnames} for checkpoint.")
    for chckpnt_dirname in chckpnt_dirnames:
        os.makedirs(chckpnt_dirname, exist_ok=True)

    if not is_train:
        # to load reverse file order
        chckpnt_dirnames = chckpnt_dirnames[::-1]

    callbacks = get_callbakcs(callbacks, chckpnt_dirnames, is_continue_train, is_continue_best,
                              checkpoint_epochs, datasets, monitor_best, patience, seed,
                              is_progressbar, tensorboard_dir, is_train)

    train_split = predefined_split(
        datasets["valid"]) if "valid" in datasets else None

    trainer = Model(
        callbacks=callbacks, train_split=train_split, device=device, **kwargs
    )

    if is_return_init:
        trainer.initialize()
        return trainer

    if is_train:
        trainer.fit(datasets["train"], y=None)

    trainer = load_trainer(trainer, datasets, chckpnt_dirnames,
                           load_epoch, monitor_best, is_load_criterion=is_load_criterion)

    with open(os.path.join(chckpnt_dirnames[0], MOD_SUMM_FILENAME), "w") as f:
        f.write(str(trainer.module_))

    clean_end_run(clean_after_run, chckpnt_dirnames)

    return trainer


def get_callbakcs(callbacks, chckpnt_dirnames, is_continue_train,
                  is_continue_best, checkpoint_epochs, datasets,
                  monitor_best, patience, seed, is_progressbar,
                  tensorboard_dir, is_train):
    for chckpnt_dirname in chckpnt_dirnames:
        chckpt_last = get_checkpoint(chckpnt_dirname, monitor="last")
        callbacks.append(chckpt_last)

    # loading from previous checkpoint to continue training
    if is_continue_train:
        if is_continue_best:
            chckpt_cont = get_checkpoint(
                chckpnt_dirnames[0], monitor=monitor_best)
        else:
            chckpt_cont = chckpt_last
        # will continue from last dirname
        load_state = LoadInitState(chckpt_cont)
        callbacks.append(load_state)

    # checkpoint from a given epoch
    if checkpoint_epochs is not None:
        for chckpnt_dirname in chckpnt_dirnames:
            callbacks.append(
                get_checkpoint(chckpnt_dirname, monitor=checkpoint_epochs)
            )

    # Nota Bene : the best checkpoint added will be the one logged with a "+"
    if "valid" in datasets:
        for chckpnt_dirname in chckpnt_dirnames:
            chckpt_best = get_checkpoint(chckpnt_dirname, monitor=monitor_best)
            callbacks.append(chckpt_best)

        if patience is not None:
            callbacks.append(EarlyStopping(patience=patience))

    if seed is not None:
        callbacks.append(FixRandomSeed(seed))

    if is_progressbar:
        callbacks.append(ProgressBar())

    if tensorboard_dir is not None and is_train:
        if os.path.exists(tensorboard_dir) and os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
        writer = SummaryWriter(tensorboard_dir)
        callbacks.append(TensorBoard(writer))

    return callbacks


def load_trainer(trainer, datasets, chckpnt_dirnames, load_epoch,
                 monitor_best, is_load_criterion=True):

    trainer.initialize()

    # loads from last dirname
    if load_epoch is not None:
        # checkpoint at a specific epoch
        chckpt = get_checkpoint(chckpnt_dirnames[-1], monitor=load_epoch)

    elif "valid" in datasets:
        # if the best checkpoint was saved then load it
        chckpt = get_checkpoint(chckpnt_dirnames[-1], monitor=monitor_best)

    else:
        # load from first dirname for mode="last"
        chckpt = get_checkpoint(chckpnt_dirnames[-1], monitor="last")

    trainer = load_chkpnt_model_(
        trainer, chckpt, is_load_criterion=is_load_criterion)

    return trainer


def load_chkpnt_model_(trainer, chckpt, is_load_criterion=True):
    # don't load optimizer
    trainer.load_params(f_history=chckpt.f_history_)
    trainer.load_params(
        f_params=chckpt.get_formatted_files(trainer)["f_params"])
    if is_load_criterion:
        trainer.load_params(
            f_criterion=chckpt.get_formatted_files(trainer)["f_criterion"])
    return trainer
