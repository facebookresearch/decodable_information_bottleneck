"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import copy
import logging
import math
import os
import subprocess
from functools import partial, partialmethod
from pathlib import Path

import hydra
import omegaconf
import skorch
import torch
import torchvision
from omegaconf import OmegaConf
from skorch.callbacks import Callback, EpochScoring, Freezer, LRScheduler, Unfreezer

from dib.classifiers import MCTrnsfClassifier
from dib.predefined import MLP
from dib.training import NeuralNetClassifier, NeuralNetTransformer
from dib.training.helpers import target_extractor
from dib.training.trainer import _get_params_for_optimizer
from dib.transformers import (
    DIBLoss,
    DIBLossAltern,
    DIBLossAlternHigher,
    DIBLossAlternLinear,
    DIBLossAlternLinearExact,
    DIBLossLinear,
    ERMLoss,
    IBEncoder,
    VIBLoss,
    get_img_encoder,
)
from dib.utils.helpers import Identity, set_seed
from utils.data import get_train_dev_test_datasets
from utils.evaluate import (
    FILE_CLF_REP,
    accuracy,
    accuracy_filter_train,
    eval_clf,
    eval_corr_gen,
    eval_trainer_log,
    eval_trnsf,
    loglike,
)
from utils.helpers import (
    CrossEntropyLossGeneralize,
    SetLR,
    StopAtThreshold,
    _scoring,
    _single_epoch_skipvalid,
    append_sffx,
    force_generalization,
    format_container,
    get_exponential_decay_gamma,
    hyperparam_to_path,
    invert_dict,
    save_pattern,
)
from utils.train import train_load

logger = logging.getLogger(__name__)

FILE_LOGS = "log.csv"
FILE_END = "end.txt"


@hydra.main(config_path="conf/config.yaml")
def main_cli(args):
    """Function only used from CLI => to keep main() usable in jupyter"""

    if args.is_nvidia_smi:
        subprocess.call(["nvidia-smi"])
        print()

    return main(args)


def main(args):
    """Main function for training and testing representations."""

    trainers_return = dict()
    datasets_return = dict()

    # ARGS
    update_config_(args)
    set_seed(args.seed)

    # DATASET
    datasets = get_datasets(args)
    datasets_trnsf = prepare_transformer_datasets(args, datasets)
    update_config_datasets_(args, datasets_trnsf)

    # TRANSFORMER (i.e. Encoder)
    Transformer = get_Transformer(args, datasets_trnsf)

    if args.is_precompute_trnsf:

        name = "transformer"
        trainers_return[name] = fit_evaluate_trainer(
            Transformer, args, name, datasets_trnsf, True
        )
        datasets_return[name] = prepare_return_datasets(datasets_trnsf)

    else:

        # loading the pretrained transformer
        transformer = fit_trainer(
            Transformer,
            args,
            datasets_trnsf,
            True,
            "transformer",
            is_load_criterion=False,
        )

        datasets = prepare_classification_datasets_(args, datasets)

        for Classifier, clf_name in gen_Classifiers_name(args, transformer, datasets):

            trainers_return[clf_name] = fit_evaluate_trainer(
                Classifier,
                args,
                clf_name,
                datasets,
                False,
                is_return_init=args.is_correlation_Bob,
            )
            datasets_return[clf_name] = prepare_return_datasets(datasets)

    if args.is_return:
        return trainers_return, datasets_return


def update_config_(args):
    """Update the configuration values based on other values."""

    # increment the seed at each run
    args.seed = args.seed + args.run

    # multiply the number of examples by a factor size. Used to have number of examples depending
    # on number of labels. Usually factor is 1.
    args.datasize.n_examples = args.datasize.factor * args.datasize.n_examples

    if args.datasize.n_examples_test == "train":
        # use same number of train and test examples
        args.datasize.n_examples_test = args.datasize.n_examples

    if args.is_precompute_trnsf and args.train.trnsf_kwargs.is_train:
        # if training transformer then paths need to agree
        assert args.paths["trnsf_dirnames"][0] == args.paths["chckpnt_dirnames"][0]

    # monitor training when you randomize the labels because validation does not mean anything
    if args.dataset.kwargs.is_random_targets:
        args.train.trnsf_kwargs.monitor_best = "train_loss_best"
        args.train.clf_kwargs.monitor_best = "train_loss_best"

    if not args.train.is_tensorboard:
        args.paths["tensorboard_curr_dir"] = None

    if args.experiment == "gap":
        # dib with Q++
        if args.model.name == "vib":
            args.model.loss.beta = args.model.loss.beta * 40

        elif args.model.name == "cdibL":
            args.model.loss.beta = args.model.loss.beta / 100

        elif args.model.name == "cdibS":
            args.model.loss.beta = args.model.loss.beta * 30

    if "dibL" in args.model.name:
        # dib with Q++
        args.model.Q_zx.hidden_size = args.model.Q_zy.hidden_size * 64

    if "dibS" in args.model.name:
        # dib with Q--
        args.model.Q_zx.hidden_size = args.model.Q_zy.hidden_size // 64

    if "dibXS" in args.model.name:
        # dib with Q------
        args.model.Q_zx.hidden_size = 1

    if "dibXL" in args.model.name:
        # dib with Q++++++++
        args.model.Q_zx.hidden_size = 8192

    short_long_monitor = dict(
        vloss="valid_loss_best", tloss="train_loss_best", vacc="valid_acc_best"
    )

    # use short version for name of file
    args.train.monitor_best = invert_dict(short_long_monitor).get(
        args.train.monitor_best, args.train.monitor_best
    )

    hyperparam_path = hyperparam_to_path(args.hyperparameters)
    args.paths.merge_with(
        OmegaConf.create(
            format_container(args.paths, dict(hyperparam_path=hyperparam_path))
        )
    )
    # every change that should not modify the name of the file should go below this
    # ----------------------------------------------------------------------------

    # use long version in code
    args.train.monitor_best = short_long_monitor.get(
        args.train.monitor_best, args.train.monitor_best
    )
    args.train.trnsf_kwargs.monitor_best = short_long_monitor.get(
        args.train.trnsf_kwargs.monitor_best, args.train.trnsf_kwargs.monitor_best
    )
    args.train.clf_kwargs.monitor_best = short_long_monitor.get(
        args.train.clf_kwargs.monitor_best, args.train.clf_kwargs.monitor_best
    )

    if not args.is_precompute_trnsf:
        logger.info("Not precomputing the transformer so setting train=False.")
        args.train.trnsf_kwargs.is_train = False
        args.train.kwargs.lr = args.train.lr_clf  # ! DEV
    else:
        if args.model.name == "wdecayBob":
            args.train.weight_decay = 1e-4

        if args.model.name == "dropoutBob":
            args.encoder.architecture.dropout = 0.5

    if not args.datasize.is_valid_all_epochs and "train" in args.train.monitor_best:
        # don't validate all epochs when validation >>> training and you only look at training
        rm_valid_epochs_()

    if args.model.is_joint:
        args.model.gamma_force_generalization = 1

    if "distractor" in args.clfs.name and not args.is_precompute_trnsf:
        args.dataset.is_use_distractor = True

    if "random" in args.clfs.name and not args.is_precompute_trnsf:
        # if you want random dataset for classifier then make sure you are not randomizing for encoder
        args.dataset.kwargs.is_random_targets = True
        args.train.clf_kwargs.monitor_best = "train_loss_best"  # don't monitor val

    if isinstance(args.train.kwargs.lr, str) and "|" in args.train.kwargs.lr:
        lr, lr_factor_zx = args.train.kwargs.lr.split("|")
        args.train.kwargs.lr = float(lr)
        args.train.lr_factor_zx = float(lr_factor_zx)

    if args.model.name == "vibL":
        # keep alice the same but increase bob view of alice
        # vib with better approx of I[Z,Y] Q++
        args.model.Q_zy.hidden_size = args.model.Q_zy.hidden_size * 16

    if args.model.name == "wdecay":
        args.train.weight_decay = 1e-4

    if "correlation" in args.experiment:
        if args.train.optim == "rmsprop":
            if args.train.weight_decay == 0.0005:
                args.train.weight_decay = 0.0003

        elif args.train.optim == "sgd":
            args.train.kwargs.lr = args.train.kwargs.lr * 50

    if "perminvcdib" in args.model.name:
        args.encoder.architecture.hidden_size = [1024]
        args.model.architecture.z_dim = 1024
        args.model.Q_zy.hidden_size = 256
        args.model.Q_zy.n_hidden_layers = 1


def add_none(a, b):
    if a is None or b is None:
        return None
    return a + b


def rm_valid_epochs_():
    """Don't validate every epoch."""
    NeuralNetTransformer._single_epoch = _single_epoch_skipvalid
    NeuralNetClassifier._single_epoch = _single_epoch_skipvalid
    skorch.callbacks.scoring.ScoringBase._scoring = _scoring


def get_datasets(args):
    """return a dictionary of train, test, valid, datasets."""
    logger.info("Loading the dataset ...")

    datasets = get_train_dev_test_datasets(
        args.dataset.name,
        args.dataset.type,
        valid_size=args.dataset.valid_size,
        **OmegaConf.to_container(args.dataset.kwargs, resolve=True),
    )

    # Subsetting dataset if needed
    datasets["train"] = datasets["train"].get_subset(
        size=args.datasize.n_examples)
    datasets["test"] = datasets["test"].get_subset(
        size=args.datasize.n_examples_test)

    if args.dataset.is_use_distractor:
        for dataset in datasets.values():
            dataset._switch_distractor_target()  # will only work if dataset has a distractor

    if args.dataset.train == "trainvalid":
        # for VIB MNIST experiment
        datasets["train"].append_(datasets["valid"])
        args.dataset.train = "train"

    datasets["train"], datasets["valid"], datasets["test"] = (
        datasets[args.dataset.train],
        datasets[args.dataset.valid],
        datasets[args.dataset.test],
    )

    return datasets


def get_Transformer(args, datasets):
    """Return the correct transformer."""
    logger.info("Instantiating the transformer ...")

    # Q used for sufficiency
    Q_zy = partial(
        MLP, **OmegaConf.to_container(args.model.Q_zy, resolve=True))

    # Q used for minimality
    Q_zx = partial(
        MLP, **OmegaConf.to_container(args.model.Q_zx, resolve=True))

    kwargs_loss = OmegaConf.to_container(args.model.loss, resolve=True)
    kwargs_loss["Q"] = Q_zx

    kwargs_trnsf = dict(Q=Q_zy)

    Losses = dict(
        VIBLoss=VIBLoss, ERMLoss=ERMLoss, DIBLossSklearn=DIBLossAlternLinearExact
    )

    is_linear = args.model.Q_zx.n_hidden_layers == 0
    altern_minimax = args.model.loss.altern_minimax
    kwargs = {}
    if altern_minimax > 0:
        if is_linear:
            Losses["DIBLoss"] = DIBLossAlternLinear
        else:
            Losses["DIBLoss"] = (
                DIBLossAlternHigher if args.model.loss.is_higher else DIBLossAltern
            )

    elif args.model.Loss == "DIBLoss":
        # in the case where doing joint training you need to give the parameters of the criterion
        # to the main (and only) optimizer
        NeuralNetTransformer._get_params_for_optimizer = partialmethod(
            _get_params_for_optimizer, is_add_criterion=True
        )
        Losses["DIBLoss"] = DIBLossLinear if is_linear else DIBLoss
        kwargs["optimizer__param_groups"] = [
            ("Q_zx*", {"lr": args.train.kwargs.lr * args.train.lr_factor_zx})
        ]

    return partial(
        NeuralNetTransformer,
        module=partial(
            partial(IBEncoder, **kwargs_trnsf),
            Encoder=partial(
                get_img_encoder(args.encoder.name),
                **OmegaConf.to_container(args.encoder.architecture, resolve=True),
            ),
            **OmegaConf.to_container(args.model.architecture, resolve=True),
        ),
        optimizer=get_optim(args),
        criterion=partial(
            Losses[args.model.Loss],
            ZYCriterion=partial(
                CrossEntropyLossGeneralize,
                gamma=args.model.gamma_force_generalization,
                map_target_position=datasets["train"].map_target_position,
            ),
            **kwargs_loss,
        ),
        callbacks__print_log__keys_ignored=args.keys_ignored,
        **kwargs,
    )


def fit_evaluate_trainer(Trainer, args, name, datasets, is_trnsf, **kwargs):
    """Fit and evaluate a single trainer."""

    file_after_train = get_file_after_train(args, name)
    if not get_is_already_trained(args, file_after_train, is_trnsf):

        trainer = fit_trainer(Trainer, args, datasets,
                              is_trnsf, name, **kwargs)

        if args.train.is_evaluate:
            evaluate_trainer(trainer, args, datasets, is_trnsf, name)

        Path(file_after_train).touch(exist_ok=True)

        return trainer


def get_file_after_train(args, name):
    """Return a placeholder file which is used to say whether the transformer has been precomputed."""
    chckpnt_paths = get_chckpnt_paths(args, name)
    return os.path.join(chckpnt_paths[0], FILE_END)


def get_is_already_trained(args, file_after_train, is_trnsf):
    """Whether the encoder is already precomputed."""
    if is_trnsf:
        is_skip = args.is_skip_trnsf_if_precomputed
    else:
        is_skip = args.is_skip_clf_if_precomputed

    if not args.is_return and is_skip and os.path.isfile(file_after_train):
        logger.info(f"Not training because {file_after_train} exists.")
        return True

    # making sure the placeholder doesn't exist if you will retrain the model
    with contextlib.suppress(FileNotFoundError):
        os.remove(file_after_train)

    return False


def prepare_transformer_datasets(args, datasets):
    """Return a transformer dataset (not inplace)."""
    # make sure don't change the dataset for eval and clf
    datasets = copy.deepcopy(datasets)

    # store the old training for evaluation
    datasets["train_unmodified"] = datasets["train"]

    gamma = args.model.gamma_force_generalization
    if gamma != 0:

        datasets["train"] = force_generalization(datasets)

        if not args.model.is_joint:
            if gamma == "zero":
                # trick to add test data even without using gamma
                gamma = 0

            # gamma is rescaled to depend on size of train and test (i.e. be relative) but not if have access
            # to joint (in which case you should really have access to all train and test not rescaled)
            gamma *= len(datasets["train"]) / len(datasets["test"])
            args.model.gamma_force_generalization = gamma

    return datasets


def update_config_datasets_(args, datasets):
    """Update the configuration values based on the datasets."""
    args.datasize.n_examples = len(datasets["train"])  # store as an integer
    steps_per_epoch = len(datasets["train"]) // args.datasize.batch_size
    args.model.loss.warm_Q_zx = steps_per_epoch * args.model.loss.warm_Q_zx

    count_targets = datasets["train"].count_targets()
    with omegaconf.open_dict(args):
        args.model.loss.n_per_target = {
            str(k): int(v) for k, v in count_targets.items()
        }


def fit_trainer(
    Trainer, args, datasets, is_trnsf, name, is_load_criterion=True, **kwargs
):
    """Fits the given trainer on the datasets."""

    logger.info(f"Fitting {name} ...")

    specific_kwargs = args.train["trnsf_kwargs" if is_trnsf else "clf_kwargs"]

    chckpnt_paths = get_chckpnt_paths(args, name)

    trainer = train_load(
        Trainer,
        datasets,
        # always save the transformer at the precomputed path
        chckpnt_dirnames=get_chckpnt_paths(args, name),
        tensorboard_dir=get_tensorboard_paths(args, name),
        is_load_criterion=is_load_criterion,
        callbacks=get_callbacks(args, datasets, is_trnsf=is_trnsf),
        **OmegaConf.to_container(args.train.kwargs, resolve=True),
        **OmegaConf.to_container(specific_kwargs, resolve=True),
        **kwargs,
    )

    if specific_kwargs.is_train:
        log_training(trainer, args.csv_score_pattern, chckpnt_paths)

    return trainer


def get_tensorboard_paths(args, name):
    """Return the paths for tensorboard"""
    return add_none(args.paths["tensorboard_curr_dir"], name)


def get_chckpnt_paths(args, name):
    """Return the paths for the classifiers checkpoint"""
    return append_sffx(args.paths["chckpnt_dirnames"], name)


def get_callbacks(args, datasets, is_trnsf):
    """Return the correct callbacks for training."""
    if is_trnsf:
        callbacks = [
            (
                "valid_acc",
                EpochScoring(
                    accuracy,  # cannot use "accuracy" because using a transformer rather than classifier
                    name="valid_acc",
                    lower_is_better=False,
                    target_extractor=target_extractor,
                ),
            ),
            (
                "valid_loglike",
                EpochScoring(
                    loglike,  # the actual loss also contains all regularization
                    name="valid_loglike",
                    lower_is_better=False,
                    target_extractor=target_extractor,
                ),
            ),
        ]
    else:
        callbacks = []

    callbacks += [
        (
            "train_acc",
            EpochScoring(
                partial(
                    accuracy_filter_train,
                    map_target_position=datasets["train"].map_target_position,
                ),
                name="train_acc",
                on_train=True,
                lower_is_better=False,
                target_extractor=partial(
                    target_extractor, is_multi_target=True),
            ),
        )
    ]

    callbacks += get_lr_schedulers(args, datasets, is_trnsf=is_trnsf)

    # callbacks += [skorch.callbacks.GradientNormClipping(gradient_clip_value=0.1)]

    if args.train.freezer.patterns is not None:
        callbacks += [
            Freezer(
                args.train.freezer.patterns,
                at=args.train.freezer.at
                if args.train.freezer.at is not None
                else return_True,
            )
        ]

    if args.train.unfreezer.patterns is not None:
        callbacks += [
            Unfreezer(args.train.unfreezer.patterns,
                      at=args.train.unfreezer.at)
        ]

    if args.train.ce_threshold is not None:
        callbacks += [StopAtThreshold(threshold=args.train.ce_threshold)]

    return callbacks


def return_True(*args):
    return True


def get_optim(args):
    if args.train.optim == "sgd":
        return partial(
            torch.optim.SGD, momentum=0.9, weight_decay=args.train.weight_decay
        )
    elif args.train.optim == "adam":
        return partial(torch.optim.Adam, weight_decay=args.train.weight_decay)
    elif args.train.optim == "adam":
        return partial(torch.optim.Adam, weight_decay=args.train.weight_decay)
    elif args.train.optim == "rmsprop":
        return partial(torch.optim.RMSprop, weight_decay=args.train.weight_decay)
    elif args.train.optim == "adagrad":
        return partial(torch.optim.Adagrad, weight_decay=args.train.weight_decay)
    elif args.train.optim == "adamw":
        return partial(
            torch.optim.AdamW, weight_decay=args.train.weight_decay, amsgrad=True
        )
    elif args.train.optim == "LBFGS":
        NeuralNetTransformer.train_step = train_step_set_optim
        return partial(
            torch.optim.LBFGS,
            line_search_fn="strong_wolfe",
            history_size=10,
            max_iter=7,
        )
    else:
        raise ValueError(f"Unkown optim={args.train.optim}")


def get_lr_schedulers(args, datasets, is_trnsf):

    if args.train.scheduling_mode == "decay":
        gamma = get_exponential_decay_gamma(
            args.train.scheduling_factor, args.train.trnsf_kwargs.max_epochs
        )

        lr_scheduler = [
            LRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=gamma)
        ]
    elif args.train.scheduling_mode == "plateau":
        lr_scheduler = [
            LRScheduler(
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                monitor="valid_loss",
                factor=0.2,
            )
        ]
    elif args.train.scheduling_mode == "biplateau":
        lr_scheduler = [
            LRScheduler(
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                monitor="valid_loss",
                factor=0.2,  # 0.1
                patience=3,  # 0.5
                verbose=True,
                threshold=0.01,
                min_lr=1e-5,
            ),
            # increase lr but max at 0.5
            LRScheduler(SetLR, lr_lambda=lambda _, lr, __: min(lr * 1.3, 0.5)),
            # dirty way for not increasing lr in case loss didn't improve
            LRScheduler(
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                monitor="valid_loss",
                factor=1 / 1.3,
                patience=1,
            ),
        ]
    elif args.train.scheduling_mode is None:
        lr_scheduler = []
    else:
        raise ValueError(
            f"Unkown scheduling_mode={args.train.scheduling_mode}")

    return lr_scheduler


def log_training(trainer, csv_score_pattern, chckpnt_dirnames):
    """Log training history (loss and accuracy) in a more readable format ."""
    save_pattern(chckpnt_dirnames, csv_score_pattern, FILE_LOGS)

    for h in trainer.history:
        if not math.isnan(h["valid_acc"]):
            for metric in ["acc", "loss"]:
                for mode in ["train", "valid"]:
                    try:
                        save_pattern(
                            chckpnt_dirnames,
                            csv_score_pattern,
                            FILE_LOGS,
                            formatting=dict(
                                epoch=h["epoch"],
                                metric=metric,
                                mode=mode,
                                score=h[f"{mode}_{metric}"],
                            ),
                        )
                    except KeyError as e:
                        logger.debug(
                            f"Skipping a loop because couldn't find key {e}")


def evaluate_trainer(trainer, args, datasets, is_trnsf, name, **kwargs):
    """Evaluate trainers on the train and test dataset."""

    logger.info(f"Evaluating {name} ...")

    if is_trnsf:
        evaluator = eval_trnsf
    else:
        evaluator = eval_corr_gen if args.is_correlation else eval_clf

    chckpnt_paths = get_chckpnt_paths(args, name)
    tensorboard_dir = get_tensorboard_paths(args, name)
    trainers = {"best": trainer}

    is_append = False
    for epoch, trainer in trainers.items():
        # Test Evaluation
        eval_trainer_log(
            trainer,
            datasets["test"],
            args.csv_score_pattern,
            chckpnt_paths,
            dict(args.hyperparameters),
            evaluator=evaluator,
            tensorboard_dir=tensorboard_dir,
            epoch=epoch,
            is_append=is_append,
            **kwargs,
        )
        # only create the field for the first time you log
        is_append = True

        # evaluation should be made on the training without any addition (e.g. anti generalization)
        train_data = datasets.get("train_unmodified", datasets["train"])

        # Train Evaluation
        eval_trainer_log(
            trainer,
            train_data,
            args.csv_score_pattern,
            chckpnt_paths,
            dict(args.hyperparameters),
            evaluator=evaluator,
            tensorboard_dir=None,
            is_append=is_append,
            mode="train",
            file_clf_rep="train_" + FILE_CLF_REP,
            epoch=epoch,
            **kwargs,
        )


def prepare_classification_datasets_(args, datasets):
    """Modify inplace the datasets before classification."""

    # store the old training for evaluation
    datasets["train_unmodified"] = datasets["train"]

    gamma = args.clfs.gamma_force_generalization
    if args.clfs.gamma_force_generalization != 0:
        datasets["train"] = force_generalization(datasets)

        args.clfs.gamma_force_generalization = gamma

    return datasets


def gen_Classifiers_name(args, transformer, datasets):
    """Generator of uninstantiated classifiers."""
    gamma = args.clfs.gamma_force_generalization

    data_weight = len(datasets["train"]) / len(datasets["test"])

    for n_hid in OmegaConf.to_container(args.clfs.nhiddens, resolve=True):
        for n_lay in OmegaConf.to_container(args.clfs.nlayers, resolve=True):
            for k_pru in OmegaConf.to_container(args.clfs.kprune, resolve=True):
                clf_name = (
                    f"clf_nhid_{n_hid}/clf_nlay_{n_lay}/clf_kpru_{k_pru}/gamma_{gamma}/"
                )

                Classifier = partial(
                    MLP, hidden_size=n_hid, n_hidden_layers=n_lay, k_prune=k_pru
                )

                kwargs = {}
                if not args.clfs.is_reinitialize:
                    kwargs["previous_mlp"] = transformer.module_.Q_zy

                Classifier = partial(
                    NeuralNetClassifier,
                    module=partial(
                        MCTrnsfClassifier,
                        transformer=transformer.module_,
                        Classifier=Classifier,
                        **OmegaConf.to_container(args.clfs.kwargs, resolve=True),
                        **kwargs,
                    ),
                    # don't use any regularization if you only care about training (e.g. Rademacher)
                    optimizer=get_optim(args),
                    criterion=partial(
                        CrossEntropyLossGeneralize,
                        gamma=gamma * data_weight,
                        map_target_position=datasets["train"].map_target_position,
                    ),
                )

                yield Classifier, clf_name


def prepare_return_datasets(datasets):
    """Prepares the datasets to return them"""
    datasets = copy.deepcopy(datasets)

    # removing modifications to trainign set such as adding the test set for anti generalizatiom
    if "train_unmodified" in datasets:
        datasets["train_modified"] = datasets["train"]
        datasets["train"] = datasets["train_unmodified"]

    return datasets


if __name__ == "__main__":
    main_cli()
