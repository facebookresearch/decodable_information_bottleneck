"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Script to use to change directory structure in `tmp_results/*` in case you change the default
directory structure (i.e. you change `hyperparameters` in hyperparameters.

Use `python add_hyperparam.py experiment=...` to update the directory structure of a given experiment.
If you don't provide the experiment, it will update everything.
"""

import glob
import logging
import os
import shutil

import hydra
from omegaconf import OmegaConf

from utils.helpers import format_container, hyperparam_to_path

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml")
def add_hyperparam(args):
    """Function that renames results from `experiment` in case you added a new hyperparameter to save.
    It can also rename all results by setting `experiment=*`.
    """

    dflt_hyperparameters = OmegaConf.to_container(
        args.hyperparameters, resolve=True)
    subfolders = glob.glob(
        os.path.join(args.paths.base_dir,
                     f"tmp_results/{args.experiment}/**/run_*/"),
        recursive=True,
    )

    for subfolder in subfolders:

        relative_dir = subfolder.split("tmp_results")[-1][1:-1]
        args.experiment = relative_dir.split("/")[0]
        args.trnsf_experiment = args.experiment

        hyperparam = {
            "_".join(group.split("_")[:-1]): group.split("_")[-1]
            for group in relative_dir[len(args.experiment) + 1:].split("/")
        }

        curr_dflt_hyperparameters = dflt_hyperparameters.copy()
        curr_dflt_hyperparameters.update(hyperparam)

        hyperparam_path = hyperparam_to_path(curr_dflt_hyperparameters)
        paths = format_container(args.paths, dict(
            hyperparam_path=hyperparam_path))

        # remove the run_ as it will always be 0 in paths
        new_subfolder = "run_".join(
            paths["chckpnt_dirnames"][0].split("run_")[:-1])
        # remove rtailing slash if exist
        new_subfolder = new_subfolder.rstrip("/")

        if new_subfolder == subfolder:
            continue
        else:
            os.makedirs(
                new_subfolder.rsplit("/", 1)[0], exist_ok=True
            )  # make sure all folder until last exist

        shutil.move(subfolder, new_subfolder)


if __name__ == "__main__":
    add_hyperparam()