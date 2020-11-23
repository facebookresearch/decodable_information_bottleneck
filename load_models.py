"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
import string
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.preprocessing import minmax_scale

from aggregate import PRETTY_RENAMER
from dib.training.helpers import clone_trainer
from main import main as main_training
from utils.evaluate import accuracy, loglike
from utils.helpers import (PartialFormatMap, all_logging_disabled,
                           flip_nested_dict)
from utils.visualize import plot_2D_decision_boundary

logger = logging.getLogger(__name__)


class ModelsAnalyser:
    """Loader of pretrained models for analsis. Alalyse one set of enoders with their respective clfs.

    Parameters
    ----------
    save_dir : str
        Where to save all results.
        
    context_kwargs : dict, optional
        Context arguments for plotting.

    is_interactive : bool, optional
            Whether to plot interactively, useful in jupyter notebooks.

    prfx : str, optional
        Prefix for the filename to save.

    pretty_renamer : dict, optional
        Dictionary mapping string (keys) to human readable ones for nicer printing and plotting.

    dpi : int, optional
        Resolution of the figures
    """

    def __init__(
        self,
        save_dir,
        context_kwargs={"context": "talk"},
        is_interactive=False,
        prfx="",
        pretty_renamer=PRETTY_RENAMER,
        dpi=300,
    ):

        self.save_dir = save_dir
        self.is_interactive = is_interactive
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_context(**context_kwargs)
        self.trainers = {}
        self.datasets = {}
        self.prfx = prfx
        self.n_clfs = 0
        self.n_encs = 0
        self.pretty_renamer = pretty_renamer
        self.dpi = dpi

    def recolt_data(
        self,
        cfg,
        clf_patterns,
        encoders_param,
        encoders_vals,
        get_name_clf=lambda p: PRETTY_RENAMER[p.replace("/", "")],
        get_name_encoder=lambda p, v: PRETTY_RENAMER[" ".join(
            p.split(".")[-2:])]
        + f" {v}",
    ):
        """Recolts all the data.
        
        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Arguments to the `main` function to load the models.

        clf_patterns : list of str
            List of patterns that will be used to select the clfs. I.e substring to recognize a single 
            clf. Each pattern should match one and only one clf.

        encoders_param : str 
            Parameter for which to loop over for getting different encoders. I.e. field of `encoders_vals`.

        encoders_vals : list
            Values of encoders_param for which to iterate over. Effectively giving `param=values[i]` 
            to generate each encoders.

        get_name_clf : callable, optional
            Function that takes in the pattern and return the name of the clf. Used to give human
            readable names.

        get_name_encoder : callable, optional
            Function that maps (encoders_param, encoders_vals[i]) -> name_enc. Used to give human
            readable names.
        """
        self.loaded_trainers = dict()
        self.loaded_datasets = dict()

        for i, enc_val in enumerate(encoders_vals):
            # make sure not changed in place cfg
            curr_cfg = deepcopy(cfg)

            name_enc = get_name_encoder(encoders_param, enc_val)
            self.loaded_trainers[name_enc] = dict()
            self.loaded_datasets[name_enc] = dict()

            # set the the current encoder
            curr_cfg.merge_with_dotlist([f"{encoders_param}={enc_val}"])

            with all_logging_disabled():
                trainers, datasets = main_training(curr_cfg)

            for pattern in clf_patterns:
                name_clf = get_name_clf(pattern)

                key = [k for k in trainers.keys() if pattern in k]

                if len(key) == 0:
                    raise ValueError(f"No keys have {pattern} as pattern.")
                elif len(key) > 1:
                    raise ValueError(
                        f"Multiple keys have {pattern} as pattern.")

                key = key[0]

                self.loaded_trainers[name_enc][name_clf] = trainers[key]
                self.loaded_datasets[name_enc][name_clf] = datasets[key]

        len_trainers = list(set(len(t) for t in self.loaded_trainers.values()))
        len_datasets = list(set(len(d) for d in self.loaded_datasets.values()))

        if (
            (len(len_trainers) != 1)
            or (len(len_datasets) != 1)
            or (len_datasets[0] != len_trainers[0])
        ):
            raise ValueError(
                f"datasets and trainers wrong length : len_trainers={len_trainers}, len_datasets={len_datasets}"
            )

        self.n_encs = len(encoders_vals)
        self.n_clfs = len_trainers[0]

    def plot_reps_clfs(
        self,
        filename,
        get_title_top=lambda clf: clf,
        get_title="{normloglike:.0%} Log Like.",
        is_invert_yaxis=False,
        diagonal_color="tab:green",
        is_plot_test=False,
        **kwargs,
    ):
        """Return a list of rep_clf figures for the correct classifiers.
        
        Parameters
        ----------
        filename : str

        get_title_top : callable or str, optional
            Function that return the title of the top row. (x label if inverted)

        get_title : callable or str or "loglike", optional
            Function that takes in the pattern and return the title. `{acc`|`{loglike`|`{normloglike`|
            `{percloglike` | `{normacc` will be replaced by actual accuracy | loglike | normalized 
            loglike | normalized accuracy. Normalized is by clf.

        is_invert_yaxis : bool, optional
            Whether to invert the x and y axis.

        diagonal_color : str, optional
            Color of the text on the diagonal.

        is_plot_test : bool, optional
            Whether to plot some test datapoints.
        
        kwargs : 
            Additional arguments to `plot_2D_decision_boundary`. 
        """
        if isinstance(get_title_top, str):
            get_title_top_str = get_title_top
            def get_title_top(clf): return get_title_top_str

        if isinstance(get_title, str):
            get_title_str = (
                get_title if get_title != "loglike" else "{loglike:.2f} Log Like."
            )
            def get_title(clf): return get_title_str

        named_axes = dict()
        metrics = dict()

        F, axes = plt.subplots(
            self.n_encs,
            self.n_clfs,
            figsize=(4 * self.n_clfs, 4 * self.n_encs),
            squeeze=False,
        )

        # plotting
        for i, key_enc in enumerate(self.loaded_trainers.keys()):
            if is_invert_yaxis:
                i = self.n_encs - i - 1

            named_axes[key_enc], metrics[key_enc] = get_figs_rep_clfs(
                self.loaded_trainers[key_enc],
                self.loaded_datasets[key_enc],
                axes=axes[i, :],
                is_plot_test=is_plot_test,
                **kwargs,
            )

        # metrics will be of depth 3 with keys (metric, encoder, clf)
        metrics = flip_nested_dict(metrics)
        # each metric will be dataframe
        metrics = {k: pd.DataFrame(v) for k, v in metrics.items()}
        # add normalized metrics
        for k in [k for k in metrics.keys()]:
            metrics[f"norm{k}"] = pd.DataFrame(
                minmax_scale(metrics[k], axis=1),
                index=metrics[k].index,
                columns=metrics[k].columns,
            )
        # back to dict
        for k in [k for k in metrics.keys()]:
            metrics[k] = metrics[k].to_dict()

        # set all the titles
        for i, enc in enumerate(named_axes.keys()):
            unmodified_i = i
            if is_invert_yaxis:
                is_bottom = i == 0
                i = self.n_encs - i - 1
                def get_prfx(key): return ""
                # if reverse put title top at the bottom as an x label
                get_xlabel = (
                    (lambda key: get_title_top(key)
                     ) if is_bottom else (lambda key: "")
                )

            else:
                def get_prfx(key): return (
                    get_title_top(key) + "\n") if i == 0 else ""
                get_xlabel = None

            for j, clf in enumerate(named_axes[enc].keys()):
                if j == 0:
                    axes[i, j].set_ylabel(enc)

                if get_xlabel is not None:
                    axes[i, j].set_xlabel(get_xlabel(clf))

                title = get_prfx(clf) + get_title(clf)

                for metric, vals in metrics.items():
                    if "{" + metric in title:
                        title = title.format_map(
                            PartialFormatMap(**{metric: vals[enc][clf]})
                        )
                title_kwargs = (
                    dict(color=diagonal_color, fontweight="bold")
                    if unmodified_i == j and diagonal_color is not None
                    else {}
                )
                axes[i, j].set_title(title, **title_kwargs)

        if self.is_interactive:
            plt.show(axes)
        else:
            F.savefig(
                os.path.join(self.save_dir, f"{self.prfx}{filename}.png"), dpi=self.dpi
            )
            plt.close(F)


@hydra.main(config_path="conf/config.yaml", strict=True)
def main_cli(args):
    return main(args)


def main(args):
    main_cfg = deepcopy(args)
    del main_cfg["load_models"]

    logger.info(f"Loading models for {args.experiment} ...")
    analyser = ModelsAnalyser(**args.load_models.kwargs)

    logger.info(f"Recolting the data ..")
    # Omega conf dictionaries don't have all the properties that usual dictionaries have
    analyser.recolt_data(
        main_cfg, **OmegaConf.to_container(args.load_models.recolt_data, resolve=True)
    )

    for f in args.load_models.mode:

        logger.info(f"Mode {f} ...")

        if f is None:
            continue

        if f in args.load_models:
            kwargs = args.load_models[f]
        else:
            kwargs = {}

        getattr(analyser, f)(**kwargs)


# HELPERS
def plot_rep_clf(trainer, dataset, dataset_test=None, ax=None, **kwargs):
    """
    Plot the given representation and the decision boundaries of the classifiers. 
    """

    # encode the data
    transformer = clone_trainer(trainer)
    transformer.module_ = transformer.module_.transformer
    transformer.module_.is_transform = True
    transformer.module_.is_avg_trnsf = True
    X, y = get_encoded_X_y(transformer, dataset)

    if dataset_test is not None:
        test = get_encoded_X_y(transformer, dataset_test)
    else:
        test = None

    # prepare the classifier
    clf = clone_trainer(trainer)
    clf.module_ = clf.module_.clf

    acc = accuracy(clf, X, np.array(y))
    log_like = loglike(clf, X, np.array(y))

    metrics = dict(acc=acc, loglike=log_like)

    ax = plot_2D_decision_boundary(X, y, clf, test=test, ax=ax, **kwargs)

    return ax, metrics


def get_encoded_X_y(transformer, dataset):
    X = super(type(transformer), transformer).predict_proba(
        dataset).astype("float32")
    y = [i[1] for i in dataset]

    if isinstance(y[0], tuple):  # if multitarget
        y = [el[0] for el in y]

    return X, y


def get_figs_rep_clfs(trainers, datasets, is_plot_test=False, axes=None, **kwargs):
    """Return a list of rep_clf figures for the correct classifiers.
    
    Parameters
    ----------
    trainers : dict of skorch.NeuralNetClassifer with MCTrnsfClassifier module
        Trainers that will be used for encoding and classification. The keys of the dictionary 
        will be selected using clf_patterns.
        
    datasets : dict of datasets
        Each element of the dictionary is itself a dictionary of datasets, with a key 
        "train" for the trainign dataset. The keys should be the same as for trainers.

    is_plot_test : bool, optional
        Whether to plot some test datapoints.
        
    axes: list matplotlib.axes, optional
        List of axis on which to plot.
        
    kwargs : 
        Additional arguments to `plot_2D_decision_boundary`. 
    """

    out_axs = dict()
    all_metrics = dict()

    # collecting all the metrics and data before plotting so that you can use normalize metrics
    for i, clf in enumerate(trainers.keys()):

        ax = None if axes is None else axes[i]
        out_axs[clf], metrics = plot_rep_clf(
            trainers[clf],
            datasets[clf]["train"],
            dataset_test=datasets[clf]["test"] if is_plot_test else None,
            ax=ax,
            **kwargs,
        )

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, {})
            all_metrics[k][clf] = metrics[k]

    return out_axs, all_metrics


if __name__ == "__main__":
    main_cli()
