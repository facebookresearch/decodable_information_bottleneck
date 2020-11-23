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

experiment="corrWidthnew"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_augment=False
model.architecture.z_dim=1024
clfs=default
train.clf_kwargs.clean_after_run=training
train.kwargs.lr=1e-3
train.monitor_best=last
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=2
train.trnsf_kwargs.max_epochs=300
train.clf_kwargs.max_epochs=100
hydra.launcher.time=1500
model=erm
datasize=all
$dev
"

kwargs_multi="
run=0,1,2,3,4
dataset=cifar10,svhn
encoder=width32,width128,width512,width2048,width8192
"



if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

      
  done
fi

wait 


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=encoder \
       plot_generalization.is_trnsf=True \
       plot_generalization.row=data \
       plot_aux_trnsf.x=encoder \
       plot_aux_trnsf.row=data \
       plot_histories.style=encoder \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]
