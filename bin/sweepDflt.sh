####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Base for sweeping to ncompute only once

experiment="sweepDflt"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlp 
dataset=cifar100
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=null
train.kwargs.lr=1e-5
train.optim=adam
is_random_labels_clf=True
train.monitor_best=vacc
train.clf_kwargs.monitor_best=tloss 
model.architecture.zy_kwargs.k_prune=0 
model.architecture.zy_kwargs.hidden_size=64
model.architecture.zy_kwargs.n_hidden_layers=0
model=vanilla 
train.trnsf_kwargs.max_epochs=100
train.clf_kwargs.max_epochs=300
hydra.launcher.time=2500
clfs=sweepRectangles
$dev
"

kwargs_multi="
run=0,1,2
model.architecture.z_dim=2,16,1024
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi
