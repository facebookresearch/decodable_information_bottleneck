"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import minmax_scale

import hydra
from dib.transformers.ib.helpers import CORR_GROUPS
from omegaconf import OmegaConf
from utils.evaluate import load_histories, load_results
from utils.helpers import (
    SFFX_TOAGG,
    aggregate_table,
    replace_None_with_all,
    rm_const_col,
    update_prepending,
)
from utils.visualize.helpers import kwargs_log_xscale

logger = logging.getLogger(__name__)


class StrFormatter:
    """Defult dictionary that takes a key dependent function.

    Parameters
    ----------
    exact_match : dict, optional
        Dictionary of strings that will be replaced by exact match.

    subtring_replace : dict, optional
        Dictionary of substring that will be replaced if no exact_match. Order matters.
        Everything is title case at this point. None gets mapped to "".

    to_upper : list, optional   
        Words to upper case.
    
    """

    def __init__(self, exact_match={}, subtring_replace={}, to_upper=[]):
        self.exact_match = exact_match
        self.subtring_replace = subtring_replace
        self.to_upper = to_upper

    def __getitem__(self, key):
        if not isinstance(key, str):
            return key

        if key in self.exact_match:
            return self.exact_match[key]

        key = key.title()

        for match, replace in self.subtring_replace.items():
            if replace is None:
                replace = ""
            key = key.replace(match, replace)

        for w in self.to_upper:
            key = key.replace(w, w.upper())

        return key

    def update(self, new_dict):
        """Update the substring replacer dictionary with a new one (missing keys will be prepended)."""
        self.subtring_replace = update_prepending(
            self.subtring_replace, new_dict)


PRETTY_RENAMER = StrFormatter(
    exact_match={
        f"train_H_lin_xCz{SFFX_TOAGG}": r"$\mathrm{H}_{\mathcal{V}}[\mathrm{X}|\mathrm{Z}]$ Linear",
        f"train_H_q_xCz{SFFX_TOAGG}": r"$\mathrm{H}_{\mathcal{V}}[\mathrm{X}|\mathrm{Z}]$ Z-64-Y (Clf)",
        f"train_H_hid16_xCz{SFFX_TOAGG}": r"$\mathrm{H}_{\mathcal{V}}[\mathrm{X}|\mathrm{Z}]$ Z-1024-Y",
        f"train_path_norm{SFFX_TOAGG}": "Path Norm",
        f"train_var_grad{SFFX_TOAGG}": "Final Grad. Var.",
        f"train_y_pred_ent{SFFX_TOAGG}": "Entropy",
        f"train_sharp_mag{SFFX_TOAGG}": "Sharp. Mag.",
        f"path_norm": "Path Norm",
        f"var_grad": "Final Grad. Var.",
        f"y_pred_ent": "Entropy",
        f"sharp_mag": "Sharp. Mag.",
        "wdecay": "Weight Decay",
        "resnet": "Depth",
        "b": "Batch Size",
        "b8": 8,
        "b16": 16,
        "b32": 32,
        "b64": 64,
        "b128": 128,
        "b256": 256,
        "b512": 512,
        "b1024": 1024,
        "b2048": 2048,
        "b4096": 4096,
    },
    subtring_replace={
        SFFX_TOAGG.title(): "",
        "_": " ",
        "nodist": "",
        "Zdim": "Z Dim.",
        "Q Family": r"$\mathcal{V}$",
        "H Acc": "Head Acc",
        "Model": "Objective",
        "D Diq Xz Space": r"$\mathrm{I}_{\mathcal{V}}[\mathrm{Z} \rightarrow \mathrm{X} ]$"
        + "\n"
        + r"$- \mathrm{I}_{\mathcal{V}}[\mathrm{Z} \rightarrow \mathrm{Y}]$",
        "D Diq Xz": r"$\mathrm{I}_{\mathcal{V}}[\mathrm{Z} \rightarrow \mathrm{Dec(X,Y)}]$",
        "Diq Xz": r"$\mathrm{I}_{\mathcal{V}}[\mathrm{Z} \rightarrow \mathrm{Dec(X,Y)}]$",
        "Diq Xzcy": r"$\mathrm{I}_{\mathcal{V}}[\mathrm{Z} \rightarrow \mathrm{Dec(X,Y)} ]$",
        "D H Q Xcz": r"$\frac{1}{N} \sum_{\mathrm{N}_i} \mathrm{H}_{\mathcal{V}}[\mathrm{N}_i|\mathrm{Z}] $",
        "I ": r"$\mathrm{I}$",
        "Diq ": r"$\mathrm{I}_{\mathcal{V}}$",
        "H Qp ": r"$\mathrm{H}_{\mathcal{V}^+}$",
        "H Qm ": r"$\mathrm{H}_{\mathcal{V}^-}$",
        "H Q Bob": r"$\mathrm{H}_{\mathcal{V}_{Bob}}$",
        "H Q Alice": r"$\mathrm{H}_{\mathcal{V}_{Alice}}$",
        "H Q ": r"$\mathrm{H}_{\mathcal{V}}$",
        "H ": r"$\mathrm{H}$",
        "Xczy": r"$[\mathrm{X}|\mathrm{Z},\mathrm{Y}]$",
        "Xcz": r"$[\mathrm{X}|\mathrm{Z}]$",
        "Xzcy": r"$[\mathrm{X} \rightarrow \mathrm{Z} | \mathrm{Y}]$",
        "Ycz": r"$[\mathrm{Y}|\mathrm{Z}]$",
        "Yz": r"$[\mathrm{Z} \rightarrow \mathrm{Y}]$",
        "Xz": r"$[\mathrm{Z} \rightarrow \mathrm{X}]$",
        # when it is something like H_Q[X|Z] don't put train because we will only look at train
        "Train $": "$",
        "Q Zy": r"$\mathcal{V}_{Bob}$",
        "Q Zx": r"$\mathcal{V}_{Bob Adv.}$",
        "Clf ": r"$\mathcal{V}_{Alice}$ ",
        "Beta": r"$\beta$",
        "Star": r"$*$",
        "Loglike": "Log Like.",
        "Resnet": "ResNet",
        "Dibsameidcs": "Fixed Indexing",
        "Dibrand": "Rand. DIB",
        "Cdibexact": "Cond. DIB",
        "Cdibapprox": "Concat. CDIB",
        "Cdib": r"$\Delta$ DIB",
        "higher": " Unrolled",
        "Accuracy": "Acc.",
        "Acc": "Acc.",
        "Perc.": r"$\%$",
        " All": "",
        "Nlay": "Depth",
        "N Hidden Layers": "Depth",
        "Nhid": "Width",
        "Hidden Size": "Width",
        "Minimax": "# Inner Optim. Steps",
        "Kpru": "Non Zero Weights",
        "K Prune": "Non Zero Weights",
        "N. ": "# of ",
        "Mchead": r"# Indexing of $\mathcal{X}$",  # "# of Possible Labels",  #
        "..": ".",
    },
    to_upper=["Cifar10", "Cifar100", "Mnist",
              "Svhn", "Cifar10Mnist", "Dib", "Vib", ],
)


class Aggregator:
    """Result aggregator.

    Parameters
    ----------
    save_dir : str
        Where to save all results.
        
    context_kwargs : dict, optional
        Context arguments for plotting.

    is_return_plots : bool, optional
        Whether to return plots instead of saving them.

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
        is_return_plots=False,
        prfx="",
        pretty_renamer=PRETTY_RENAMER,
        dpi=300,
    ):
        self.save_dir = save_dir
        self.is_return_plots = is_return_plots
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_context(**context_kwargs)
        self.tables = {}
        self.table_names = {"results", "aux_trnsf", "histories"}
        self.prfx = prfx
        self.pretty_renamer = pretty_renamer
        self.dpi = dpi

    def recolt_data(
        self,
        pattern_results,
        pattern_histories,
        pattern_aux_trnsf,
        metrics=["test_accuracy", "test_loglike",
                 "train_accuracy", "train_loglike"],
        aux_trnsf=[
            "test_I_xz",
            "train_I_xz",
            "train_diF_zy",
            "test_diF_zy",
            "train_acc",
            "test_acc",
            "train_loss",
            "test_loss",
            "train_loglike",
            "test_loglike",
        ],
        metric_column_name="{mode}_{metric}",
        **kwargs,
    ):
        """Recolts all the data.
        
        Parameters
        ----------
        pattern_results : str
            Pattern for globbing results.

        pattern_histories: str
            Pattern for globbing histories.

        pattern_aux_trnsf: str
            Pattern for globbing auxiliary losses of the transformer.

        metrics : list of str, optional
            Metrics to aggregate

        aux_trnsf : list of str, optional
            Auxiliary transformer to aggregate.

        metric_column_name : str, optional
            Name of the column containing the metric.

        kwargs :
            Additional arguments to `load_results` and `pattern_histories`.
        """
        self.table_names = set()

        if pattern_results is not None:
            self.tables["results"] = load_results(
                pattern=pattern_results,
                metrics=metrics,
                metric_column_name=metric_column_name,
                **kwargs,
            )
            self.table_names.add("results")

        if pattern_aux_trnsf is not None:
            self.tables["aux_trnsf"] = load_results(
                pattern=pattern_aux_trnsf,
                metrics=aux_trnsf,
                metric_column_name=metric_column_name,
                **kwargs,
            )

            self.table_names.add("aux_trnsf")

        if pattern_histories is not None:
            self.tables["histories"] = load_histories(
                pattern=pattern_histories, **kwargs
            )
            self.table_names.add("histories")

    def load_data(self):
        """Load the pre-recolted data."""

        for k in self.table_names.copy():
            try:
                self.tables[k] = pd.read_csv(
                    os.path.join(self.save_dir, self.prfx + f"{k}.csv")
                )
            except FileNotFoundError:
                self.table_names.remove(k)

    def prettify(self, table):
        """Make the name and values in a dataframe prettier / human readable."""
        def renamer(x): return self.pretty_renamer[x]
        table = table.rename(columns=renamer)
        table = table.applymap(renamer)
        return table

    def prettify_kwargs(self, pretty_data, **kwargs):
        """Change the kwargs of plotting function sucxh that they can be used with `prettify(table)`."""
        return {
            # only prettify if part of the columns (not arguments to seaborn)
            k: self.pretty_renamer[v]
            if isinstance(v, str) and self.pretty_renamer[v] in pretty_data.columns
            else v
            for k, v in kwargs.items()
        }

    def subset(self, col_val):
        """Subset all tables by keeping only the given values in given columns.
        
        Parameters
        ----------
        col_val : dict
            A dictionary where the keys are the columns to subset and values are a list of values to keep.
        """
        for col, val in col_val.items():
            logger.debug("Keeping only val={val} for col={col}.")
            for k in self.table_names:
                self.tables[k] = self.tables[k][(
                    self.tables[k][col]).isin(val)]
                if self.tables[k].empty:
                    logger.info(f"Empty table after filtering {col}={val}")

    def save_tables(self):
        """Save all tables to csv : one with no constant columns and one with all columns."""
        for k, table in self.tables.items():
            self._save_table(table, self.prfx + k)
            self._save_table(aggregate_table(table), self.prfx + k + "_agg")

    def _save_table(self, table, file):
        """Save to csv a table with no constant columns and one with all columns."""
        res_no_const = rm_const_col(table)
        # add runs even if constant column
        if "run" in table.columns:
            res_no_const["run"] = table["run"]
        else:
            res_no_const["run_count"] = table["run_count"]

        res_no_const.to_csv(
            os.path.join(self.save_dir, file + "_noconst.csv"), index=False
        )
        table.to_csv(os.path.join(self.save_dir, file + ".csv"), index=False)

    def plot_metrics(self, x, is_plot_gaps=False, is_lines=True, **kwargs):
        """Plot a lineplot for each metric
        
        Parameters
        ----------
        x : str
            Column name of x axis.
        
        is_plot_gaps : bool, optional
            Whether to plot gaps (i.e. train_*-test_*) in addition to all aux_trnsf.

        is_lines : bool, optional
            Whether to plot lines instead of heatmaps.

        kwargs : 
            Additional arguments to `_plot_lines` or `_plot_heatmaps`.
        """
        # use _mean instead of SFFX_TOAGG because you had to aggregate for heatmaps
        sffx_values = SFFX_TOAGG if is_lines else "_mean"

        def gen_values_name(data):
            for col in data.columns:
                if not col.endswith(sffx_values):
                    continue

                yield col, col.replace(sffx_values, "")

        table = self.tables["results"]
        if is_plot_gaps:
            table = add_gaps(table.copy())

        if is_lines:

            dflt_kwargs = dict(markers=True, dashes=False)
            dflt_kwargs.update(kwargs)

            return self._plot_lines(gen_values_name, table, x, **dflt_kwargs)

        else:
            # has to remove kwargs that are only for lines
            kwargs = {k: v for k, v in kwargs.items() if k not in [
                "logbase_x"]}

            # has to aggregate the tables for heatmap because cannot vary
            table = aggregate_table(table)

            return self._plot_heatmaps(gen_values_name, table, x, **kwargs)

    def plot_aux_trnsf(self, x, is_plot_gaps=False, **kwargs):
        """Plot a lineplot for each data.
        
        Parameters
        ----------
        x : str
            Column name of x axis.

        is_plot_gaps : bool, optional
            Whether to plot gaps (i.e. train_*-test_*) in addition to all aux_trnsf.

        kwargs : 
            Additional arguments to `_plot_lines`.
        """

        def gen_values_name(data):
            for col in data.columns:
                if not col.endswith(SFFX_TOAGG):
                    continue

                yield col, col.replace(SFFX_TOAGG, "_trnsf")

        dflt_kwargs = dict(markers=True, dashes=False)
        dflt_kwargs.update(kwargs)

        table = self.tables["aux_trnsf"]
        if is_plot_gaps:
            table = add_gaps(table.copy())

        return self._plot_lines(gen_values_name, table, x, **dflt_kwargs)

    def plot_histories(self, **kwargs):
        """Plot all the values in the history, for each dataset."""

        def gen_values_name(data):
            for col in data.columns:
                if not col.endswith(SFFX_TOAGG):
                    continue

                yield col, "epochs_" + col.replace(SFFX_TOAGG, "")

        # by default don't add marker because many epochs => becomes hard to read
        kwargs["marker"] = kwargs.get("marker", ",")

        return self._plot_lines(
            gen_values_name, self.tables["histories"], "epochs", **kwargs
        )

    def plot_superpose(
        self,
        x,
        to_superpose,
        value_name,
        filename="superposed_{value_name}",
        is_trnsf=True,
        is_legend_out=False,
        is_no_legend_title=True,
        **kwargs,
    ):
        """Plot a single line figure with multiple lineplots.
        
        Parameters
        ----------
        x : str
            Column name of x axis.

        to_superpose : list of str or distionary
            List of column values that should be plotted on the figure. If dictionary, then the keys
            correspond to the columns to plot and the values correspond to how they should be called.

        value_name : str
            Name of the yaxis.

        filename : str, optional
            Name of the figure when saving. Can use {value_name} for interpolation.

        is_trnsf : bool, optional
            Whether to use `"aux_trnsf"` instead of `"results"` table.

        kwargs : 
            Additional arguments to `_plot_lines`.
        """

        def gen_values_name(data):
            yield value_name, filename.format(value_name=value_name)

        table = self.tables["aux_trnsf" if is_trnsf else "results"].copy()

        try:
            renamer = {(k + SFFX_TOAGG): v for k, v in to_superpose.items()}
            to_superpose = to_superpose.keys()
        except AttributeError:
            renamer = {}

        table = table.melt(
            id_vars=[c for c in table if SFFX_TOAGG not in c],
            value_vars=[to_sup + SFFX_TOAGG for to_sup in to_superpose],
            value_name=value_name,
            var_name="mode",
        )

        table["mode"] = table["mode"].replace(renamer)

        kwargs["hue"] = "mode"
        kwargs["markers"] = kwargs.get("markers", True)

        return self._plot_lines(
            gen_values_name,
            table,
            x,
            is_legend_out=is_legend_out,
            is_no_legend_title=is_no_legend_title,
            **kwargs,
        )

    def plot_generalization(self, x, is_trnsf=True, **kwargs):
        """Plot the train and test loss/accuracy to see the generalization gap."""
        acc = "acc" if is_trnsf else "accuracy"
        loss = "loglike"

        outs = []
        for metric in [acc, loss]:
            outs.append(
                self.plot_superpose(
                    x,
                    {f"train_{metric}": "train", f"test_{metric}": "test"},
                    metric,
                    filename="gen_{value_name}",
                    is_trnsf=is_trnsf,
                    **kwargs,
                )
            )

        if any(out is not None for out in outs):
            return outs

    def correlation_experiment(
        self,
        cause,
        correlation_groups=CORR_GROUPS,
        logbase_x=1,
        col_sep_plots=None,  # column for which to plot different correlation plot
        xticks=None,
        xticklabels=None,
        thresholds={"loglike": -0.02, "accuracy": 0.995},
        standard_probs=["path_norm", "var_grad", "y_pred_ent", "sharp_mag"],
        **kwargs,
    ):
        """
        Results for the correlation experiments. It will make a plot showing side by side the 
        generalization gap and H_q[X|Z]. It will also make a file correlation_[acc|loglike].csv
        containing all the correlation measures. 

        Note 
        ----
        - Tables should have been previously saved. If the `results` table are not saved 
        in `save_dir` they will be searched for in the parent directory.
        - `thresholds` only selects model that reach a certain training performance on th given 
        metrics. 
        """
        figs = []
        old_cause = cause
        self.load_data()

        # Load result from parent dir if not previously found
        if "results" not in self.tables:
            logger.info(
                f"`results.csv` not found in {self.save_dir} looking in parent dir."
            )
            save_dir = self.save_dir
            self.save_dir = str(Path(save_dir).parent)
            self.tables["results"] = pd.read_csv(
                os.path.join(self.save_dir, self.prfx + f"results.csv")
            )
            self.save_dir = save_dir

        # self.subset(dict(epochs=["best"]))  # only take the best model (not the last)

        for metric in ["loglike", "accuracy"]:
            cause = old_cause

            # keep only the columns to plot from results (the gap acc|loglike and probs)
            results = add_gaps(self.tables["results"])

            if metric in thresholds:
                n_all = len(results)
                to_keep = results[f"train_{metric}_toagg"] > thresholds[metric]
                results = results[to_keep]
                n_dropped = n_all - len(results)
                perc_dropped = n_dropped / n_all
                logger.info(
                    f"dropped {n_dropped} (perc {perc_dropped}) because smaller than threshold "
                )

            col_probes = [
                f"train_{probe}{SFFX_TOAGG}" for probe in standard_probs]

            results = results.drop(
                columns=[
                    c
                    for c in results.columns
                    if SFFX_TOAGG in c
                    and c not in [f"gap_{metric}{SFFX_TOAGG}"]
                    and c not in col_probes  # previous litterature probes
                    and "H_Q" not in c
                    and "_xCz" not in c
                ]
            )

            if results[cause].dtype == "object":
                # if the column of interest contains strings then cannot compute correlation.
                # besides if ther are some numeric values as suffix of the string. E.g. resnet18,
                # resnet50 will be transformed as 18,50 with cause=="resnet"
                not_numbers = results[cause].apply(
                    lambda x: split_alpha_numeric(x)[0])
                unique_not_number = not_numbers.unique()
                if len(unique_not_number) > 1:
                    raise ValueError(
                        f"`cause`={cause} is a string AND contains multiple different prefixes {unique_not_number}"
                    )
                results[cause] = results[cause].apply(
                    lambda x: split_alpha_numeric(x)[1]
                )
                results = results.rename(columns={cause: unique_not_number[0]})
                cause = unique_not_number[0]

            col_H_q_xCz = [
                c for c in results.columns if (SFFX_TOAGG in c) and "H_Q" in c
            ]

            # all the correlation probes
            all_probes = col_H_q_xCz + col_probes
            table = pd.melt(
                results,
                id_vars=[c for c in results.columns if c not in all_probes],
                value_vars=all_probes,
                var_name="Probe Type",
                value_name="Probe Value",
            )

            table = self.prettify(table)
            results = self.prettify(results)

            # KENDAL CORRELATION

            arrays = dict(
                corr=results.corr(method="kendall"),
                corr_pval=results.corr(
                    method=lambda x, y: scipy.stats.kendalltau(x, y)[1]
                ),
            )

            for k in arrays.keys():
                arrays[k] = arrays[k][self.pretty_renamer[f"gap_{metric}{SFFX_TOAGG}"]]
                arrays[k] = arrays[k].rename(
                    index={self.pretty_renamer[cause]: "Cause"}
                )
                arrays[k] = arrays[k][
                    ["Cause"] + [self.pretty_renamer[probe]
                                 for probe in all_probes]
                ]
                arrays[k]["Varying"] = self.pretty_renamer[cause]

                arrays[k].to_csv(
                    os.path.join(self.save_dir, f"{metric}_{k}.csv"), header=False
                )

            # PLOTTING
            sep_plots = (
                table[self.pretty_renamer[col_sep_plots]].unique()
                if col_sep_plots is not None
                else [None]
            )

            old_table = table.copy()
            for batch_probes in [[c] for c in standard_probs] + [col_H_q_xCz]:
                table = old_table.copy()
                for sep in sep_plots:

                    fig, axes = plt.subplots(
                        2,
                        len(table[self.pretty_renamer["data"]].unique()),
                        sharex=True,
                        figsize=(17, 9),
                    )

                    for i, data in enumerate(
                        table[self.pretty_renamer["data"]].unique()
                    ):
                        axes[0, i].set_title(data.upper())

                        if col_sep_plots is not None:
                            table_subset = table[
                                table[self.pretty_renamer[col_sep_plots]] == sep
                            ]
                        else:
                            table_subset = table

                        # only plot the proposed probes H_Q[X|Z]
                        table_subset = table_subset[
                            table_subset["Probe Type"].isin(
                                [self.pretty_renamer[c] for c in batch_probes]
                            )
                        ]

                        table_subset = table_subset[
                            table_subset[self.pretty_renamer["data"]] == data
                        ]

                        table_subset = table_subset.dropna(
                            how="any", subset=["Probe Value"]
                        )

                        if logbase_x != 1:
                            plt.xscale("symlog")

                        sns.lineplot(
                            data=table_subset,
                            x=self.pretty_renamer[cause],
                            y=self.pretty_renamer[f"gap_{metric}{SFFX_TOAGG}"],
                            ax=axes[0, i],
                            **kwargs,
                        )
                        sns.lineplot(
                            data=table_subset,
                            x=self.pretty_renamer[cause],
                            y="Probe Value",
                            style="Probe Type",
                            hue="Probe Type",
                            ax=axes[1, i],
                            **kwargs,
                        )

                        if xticks is not None:
                            axes[0, i].set_xticks(
                                list(range(len(xticks))), xticks)
                            axes[1, i].set_xticks(
                                list(range(len(xticks))), xticks)
                            if xticklabels is not None:
                                axes[1, i].set_xticklabels(xticklabels)

                    if self.is_return_plots:
                        figs.append(fig)
                    else:
                        sffx = f"_{sep}" if col_sep_plots is not None else ""
                        sffx += f"_{batch_probes[0]}"
                        fig.savefig(
                            os.path.join(
                                self.save_dir, f"{self.prfx}corr_{metric}{sffx}.png"
                            ),
                            dpi=self.dpi,
                        )
                        plt.close(fig)

        if self.is_return_plots:
            return figs

    def _plot_lines(
        self,
        gen_values_name,
        data,
        x,
        folder_col=None,
        row="data",
        is_merge_data_size=True,
        transformer_data="identity",
        logbase_x=1,
        xticks=None,
        xticklabels=None,
        is_legend_out=True,
        is_no_legend_title=False,
        set_kwargs={},
        x_rotate=0,
        cols_vary_only=["run"],
        sharey=False,
        **kwargs,
    ):
        """Lines plots.
        
        Parameters
        ----------
        gen_values_name : generator
            Generates 2 string, the column of y axis and the filename to save the plot.

        data : pd.DataFrame
            Dataframe used for plotting.

        x : str
            Column name of x axis.

        folder_col : str, optional
            Name of a column tha will be used to separate the plot into multiple subfolders.

        row : str, optional
            Column name of rows.

        is_merge_data_size : bool, optional
            Whether to merge the "data" and "data_size".

        transformer_data : callable or {"identity", "normalize_by_x_axis", "fillin_None_x_axis", "replace_None_zero"}, optional
            Transform the data given the name of the columns seaborn will condition on.

        logbase_x : int, optional
             Base of the x axis. If 1 no logscale. if `None` will automatically chose.

        xticks : list of int, optional
            Set manually x ticks.

        xticklabels : list of str or int, optional
            Set manually x ticks labels.

        is_legend_out : bool, optional
            Whether to put the legend outside of the figure.

        is_no_legend_title : bool, optional
            Whether to remove the legend title. If `is_legend_out` then will actually duplicate the 
            legend :/, the best in that case is to remove the test of the legend column .

        set_kwargs : dict, optional
            Additional arguments to `FacetGrid.set`. E.g. dict(xlim=(0,None)).

        x_rotate : int, optional
            By how much to rotate the x labels.

        cols_vary_only : list of str, optional
            Name of the columns that can vary when plotting (i.e over which to compute bootstrap CI).

        sharey : bool, optional
            Wether to share y axis.

        kwargs : 
            Additional arguemnts to `sns.relplot`.
        """

        if is_merge_data_size:
            data = data.copy()
            data["data"] = data[["data", "datasize"]].apply(
                lambda x: "_".join(x), axis=1
            )

        dflt_kwargs = dict(
            legend="full",
            row=row,
            kind="line",
            facet_kws={"sharey": sharey, "sharex": True,
                       "legend_out": is_legend_out},
            hue=kwargs.get("style", None),
        )

        dflt_kwargs.update(kwargs)
        dflt_kwargs["x"] = x
        dflt_kwargs["marker"] = dflt_kwargs.get("marker", "X")

        def _helper_plot_lines(data, save_dir):
            sns_plots = []

            data = get_transformer_data(transformer_data)(
                data, get_col_kwargs(data, **dflt_kwargs)
            )

            for y, filename in gen_values_name(data):
                if data[y].isna().all():
                    logger.info(f"Skipping {filename} because all nan.")
                    continue

                _assert_sns_vary_only_cols(data, dflt_kwargs, cols_vary_only)

                # replace `None` with "None" for string columns such that can see those
                data = data.copy()
                str_col = data.select_dtypes(include=object).columns
                data[str_col] = data[str_col].fillna(value="None")

                pretty_data = self.prettify(data)
                pretty_kwargs = self.prettify_kwargs(
                    pretty_data, y=y, **dflt_kwargs)
                sns_plot = sns.relplot(data=pretty_data, **pretty_kwargs)

                if x_rotate != 0:
                    # calling directly `set_xticklabels` on FacetGrid removes the labels sometimes
                    for axes in sns_plot.axes.flat:
                        axes.set_xticklabels(
                            axes.get_xticklabels(), rotation=x_rotate)

                if logbase_x != 1:
                    x_data = np.array(
                        sorted(pretty_data[pretty_kwargs["x"]].unique()))
                    plt.xscale(**kwargs_log_xscale(x_data, base=logbase_x))

                if is_no_legend_title:
                    #! not going to work well if is_legend_out (double legend)
                    for ax in sns_plot.fig.axes:
                        handles, labels = ax.get_legend_handles_labels()
                        if len(handles) > 1:
                            ax.legend(handles=handles[1:], labels=labels[1:])

                if xticks is not None:
                    sns_plot.set(xticks=xticks)
                    if xticklabels is not None:
                        sns_plot.set(xticklabels=xticklabels)
                    if xticks[0] > xticks[1]:
                        # dirty check to see if should reverse
                        for ax in sns_plot.axes.reshape(-1):
                            ax.invert_xaxis()

                sns_plot.set(**set_kwargs)

                if self.is_return_plots:
                    sns_plots.append(sns_plot)
                else:
                    sns_plot.fig.savefig(
                        os.path.join(save_dir, f"{self.prfx}{filename}.png"),
                        dpi=self.dpi,
                    )
                    plt.close(sns_plot.fig)

            if self.is_return_plots:
                return sns_plots

        return self._foldersplit_call(_helper_plot_lines, folder_col, data)

    def _foldersplit_call(self, fn, folder_col, data):
        """Split the dataset by the values in folder_col a nd call fn on each subfolder."""
        if folder_col is None:
            return fn(data, self.save_dir)

        else:
            out = []
            for curr_folder in data[folder_col].unique():
                curr_data = data[data[folder_col] == curr_folder]

                sub_dir = os.path.join(
                    self.save_dir, f"{folder_col}_{curr_folder}")
                os.makedirs(sub_dir, exist_ok=True)

                out.append(fn(curr_data, sub_dir))
            return out

    def _plot_heatmaps(
        self,
        gen_values_name,
        data,
        x,
        y,
        col=None,
        folder_col=None,
        row="data",
        is_merge_data_size=True,
        transformer_data="identity",
        normalize=None,
        is_percentage=False,
        cbar_label=None,
        **kwargs,
    ):
        """Lines plots.
        
        Parameters
        ----------
        gen_values_name : generator
            Generates 2 string, the column of values axis and the filename to save the plot.

        data : pd.DataFrame
            Dataframe used for plotting.

        x : str
            Column name of x axis of heatmaps.

        y : str
            Column name of y axis of heatmaps.

        col : str, optional
            Column name of columns.

        row : str, optional
            Column name of rows.

        folder_col : str, optional
            Name of a column tha will be used to separate the plot into multiple subfolders.

        is_merge_data_size : bool, optional
            Whether to merge the "data" and "data_size".

        normalize : ["row","col",None], optional
            Whether to normalize the values by row (single 1 per row), by column (single 1 per col) 
            or not to.

        is_percentage : bool, optional
            Whether to use percentage for the annotation of the heatmap.

        cbar_label : str, optional
            Name for the colorbar.

        kwargs : 
            Additional arguments to `sns.heatmap`.
        """

        if is_merge_data_size:
            data = data.copy()
            data["data"] = data[["data", "datasize"]].apply(
                lambda x: "_".join(x), axis=1
            )

        def fmt(x, pos): return "{:.0%}".format(x)
        dflt_kwargs = dict(annot=True, linewidths=0.5)
        if is_percentage:
            dflt_kwargs.update(
                dict(fmt=".0%", cbar_kws={"format": FuncFormatter(fmt)}))
        if cbar_label is not None:
            dflt_kwargs["cbar_kws"] = {
                "label": self.pretty_renamer[cbar_label]}
        dflt_kwargs.update(kwargs)

        def _draw_heatmap(x, y, values, **kwargs):
            data = kwargs.pop("data")
            d = data.pivot(index=y, columns=x, values=values)

            if normalize is None:
                pass
            elif normalize == "row":
                d = pd.DataFrame(
                    minmax_scale(d.values, axis=1), columns=d.columns, index=d.index
                )
            elif normalize == "col":
                d = pd.DataFrame(
                    minmax_scale(d.values, axis=0), columns=d.columns, index=d.index
                )
            else:
                raise ValueError(f"Unkown normalize={normalize}")

            ax = sns.heatmap(d, **kwargs)
            ax.invert_yaxis()
            for label in ax.get_yticklabels():
                label.set_rotation(0)
            # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

        def _helper_plot_heatmaps(data, save_dir):
            """Plot the results as heatmaps."""
            sns_plots = []
            data = get_transformer_data(transformer_data)(
                data,
                get_col_kwargs(data, **dict(row=row, col=col,
                                            x=x, y=y, **dflt_kwargs)),
            )

            for values, filename in gen_values_name(data):
                if data[values].isna().all():
                    logger.info(f"Skipping {filename} because all nan.")
                    continue

                pretty_data = self.prettify(data)
                pretty_kwargs = self.prettify_kwargs(
                    pretty_data, **dflt_kwargs)
                sns_plot = sns.FacetGrid(
                    pretty_data,
                    row=self.pretty_renamer[row],
                    col=self.pretty_renamer[col],
                    dropna=False,
                    sharex=True,
                    sharey=True,
                    aspect=1.5,
                    height=6,
                )
                sns_plot.map_dataframe(
                    _draw_heatmap,
                    self.pretty_renamer[x],
                    self.pretty_renamer[y],
                    values=self.pretty_renamer[values],
                    **pretty_kwargs,
                )
                sns_plot.fig.tight_layout()

                if self.is_return_plots:
                    logger.info(f"heatmap for {values}")
                    sns_plots.append(sns_plot)
                else:
                    sns_plot.fig.savefig(
                        os.path.join(save_dir, f"{self.prfx}{filename}.png"),
                        dpi=self.dpi,
                    )
                    plt.close(sns_plot.fig)

            if self.is_return_plots:
                return sns_plots

        return self._foldersplit_call(_helper_plot_heatmaps, folder_col, data)


@hydra.main(config_path="conf/aggregate.yaml", strict=False)
def main_cli(args):
    return main(args)


def main(args):

    logger.info(f"Aggregating {args.experiment} ...")

    PRETTY_RENAMER.update(args.kwargs.pop("pretty_renamer"))
    aggregator = Aggregator(pretty_renamer=PRETTY_RENAMER, **args.kwargs)

    if args.is_recolt:
        logger.info(f"Recolting the data ..")
        # Omega conf dictionaries don't have all the properties that usual dictionaries have
        aggregator.recolt_data(
            **OmegaConf.to_container(args.recolt_data, resolve=True))
    else:
        logger.info(f"Loading previously recolted data ..")
        aggregator.load_data()

    aggregator.subset(OmegaConf.to_container(
        args.col_val_subset, resolve=True))

    for f in args.mode:

        logger.info(f"Mode {f} ...")

        if f is None:
            continue

        if f in args:
            kwargs = args[f]
        else:
            kwargs = {}

        getattr(aggregator, f)(**kwargs)


# HELPERS
def add_gaps(df):
    """Add train-test gaps to dataframe."""
    for col in df.columns:
        if col.startswith("train_") and col.replace("train_", "test_") in df.columns:
            # gap = max(train - test, 0)
            gap_col = col.replace("train_", "gap_")
            df[gap_col] = df[col] - df[col.replace("train_", "test_")]
            df.loc[df[gap_col] < 0, gap_col] = 0
    return df


def group_normalize_by_max(x, subgroup_col):
    """
    Add a column `normalizer_{col}` for every `_toagg` columns wich contains the maximum of the means of 
    `subgroup_col`.
    """
    means_xs = x.groupby(
        [subgroup_col]
    ).mean()  # average over runs for all subgroup_col
    for col in x.columns:
        if SFFX_TOAGG in col:
            # normalizer is teh maximum average
            normalizer = means_xs[col].max()
            x[f"normalizer_{col}"] = normalizer
    return x


def split_alpha_numeric(s):
    """Take a string containing letters followed by numbers and spli them."""
    not_numbers = s.rstrip("0123456789")
    numbers = s[len(not_numbers):]
    return not_numbers, int(numbers)


def normalize_by_x_axis(data, col_kwargs):
    """
    Prepares the data by normalizing by the maximum (average over runs) style at every point of the 
    x axis before plotting on seaborn. Return the normalized data in %{col}.
    """
    # normalizer will be different for all seaborn plots => compute normalizer for each group separately
    col_groupby = [
        v
        for k, v in col_kwargs.items()
        if v != col_kwargs["hue"] and SFFX_TOAGG not in v
    ]

    # compute the normalizers
    df = data.groupby(col_groupby).apply(
        partial(group_normalize_by_max, subgroup_col=col_kwargs["hue"])
    )

    # apply the normalization
    for col in data.columns:
        if "_toagg" in col and "normalizer" not in col:
            df[f"%{col}"] = df[col] / df[f"normalizer_{col}"]

    df = df[[col for col in df.columns if "normalizer" not in col]]

    return df


def fillin_None_x_axis(data, col_kwargs):
    """Prepares the data by removing the NaN in values of X axis by duplicatign the entry where x 
    axis is all unique value. I.e enable the plloting of a lineplot which does not vary the x_axis
    as a straight horizaontal line.
    """
    return replace_None_with_all(data, col_kwargs["x"])


def replace_None_zero(data, col_kwargs):
    """Replace all missing values with 0."""
    return data.fillna(0)


def get_transformer_data(transformer_data):
    if isinstance(transformer_data, str):
        if transformer_data == "normalize_by_x_axis":
            return normalize_by_x_axis
        elif transformer_data == "fillin_None_x_axis":
            return fillin_None_x_axis
        elif transformer_data == "replace_None_zero":
            return replace_None_zero
        elif transformer_data == "identity":
            return lambda data, col_kwargs: data
        else:
            raise ValueError(f"Unkown transformer_data={transformer_data}")

    return transformer_data


def get_col_kwargs(data, **kwargs):
    """Return all arguments that are names of the columns of the data."""
    return {
        n: col
        for n, col in kwargs.items()
        if (isinstance(col, str) and col in data.columns)
    }


def _assert_sns_vary_only_cols(data, kwargs, cols_vary_only):
    """
    Make sure that the only columns that has not been conditioned over for plotting and has non
    unique values are in `cols_vary_only`. `disregard_col` are columns that can also 
    """
    return
    conditioned_df = data

    for col in get_col_kwargs(data, **kwargs).values():
        first_unique = conditioned_df[col].dropna().unique()[0]
        conditioned_df = conditioned_df[conditioned_df[col] == first_unique]

    # make sure only `col_vary_only` varies
    if len(conditioned_df[cols_vary_only].drop_duplicates()) != len(conditioned_df):

        conditioned_df = conditioned_df[
            conditioned_df[cols_vary_only[0]]
            == conditioned_df[cols_vary_only[0]].dropna().unique()[0]
        ]

        varying_columns = []
        for col in conditioned_df.columns:
            if len(conditioned_df[col].unique()) > 1 and col not in cols_vary_only:
                varying_columns.append(col)

        raise ValueError(
            f"Not only varying {cols_vary_only}. At least one of the following varies {varying_columns}."
        )


if __name__ == "__main__":
    main_cli()
