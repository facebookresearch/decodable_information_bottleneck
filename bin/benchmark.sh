####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : benchmarking to make sure get descent losses with your models

experiment="benchmark"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset.kwargs.is_augment=True 
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=0
hydra.launcher.time=200
datasize.max_epochs=200
model.architecture.z_dim=1024
model=erm
train.kwargs.lr=0.01
encoder=resnet18
train.monitor_best=vacc
train.weight_decay=5e-4
$dev
"

kwargs_multi="
run=0,1,2
dataset=cifar10,cifar100
train.scheduling_mode=decay,biplateau
"



if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
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
       plot_aux_trnsf.row=data \
       plot_aux_trnsf.x=schedule \
       plot_histories.col=schedule \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]

