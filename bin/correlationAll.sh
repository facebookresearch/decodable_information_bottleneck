####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether dropout can be seen as decreasing the decodable mutual information. Note that no dropout on zx heads
# Hypothesis
#   - The generalization gap will get smaller with larger dropout, and the DI[Z->X] will get smaller.

experiment="correlationAll"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_augment=False
clfs=default
train.clf_kwargs.clean_after_run=training
train.monitor_best=last
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=2
train.trnsf_kwargs.max_epochs=300
train.clf_kwargs.max_epochs=100
hydra.launcher.time=1000
model=erm
datasize=all
dataset=cifar10
train.ce_threshold=0.01
run=0
$dev
"

kwargs_multi="
encoder=cnnninw2d2,cnnninw2d4,cnnninw2d8,cnnninw2d2,cnnninw4d4,cnnninw4d8,cnnninw8d2,cnnninw8d4,cnnninw8d8
train.kwargs.lr=1e-3,3e-4,1e-4
datasize=b32,b64,b128
model.dropout=0,0.25,0.5
model.architecture.z_dim=32,128,512
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

      
  done
fi

wait 

