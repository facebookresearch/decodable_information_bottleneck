####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : reproducing VIB experiment on permutation variant mnist

experiment="perminv"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset.kwargs.is_augment=False 
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=0
hydra.launcher.time=1000
datasize.max_epochs=200
model.architecture.z_dim=256
train.kwargs.lr=1e-4
encoder=mlp
train.optim=adam
train.scheduling_mode=decay
datasize.batch_size=100
dataset.train='trainvalid'
train.monitor_best=last
is_skip_trnsf_if_precomputed=False
$dev
"

kwargs_multi="
run=0,1,2,3,4
dataset=mnist,cifar10,cifar10mnist
model=cdibhigher,perminvcdib,perminvcdib01,perminvcdib10
"
#vib001,vib01,vib,dropout,erm,stochasticErm,batchnorm,wdecay,

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
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.x_rotate=90 \
       plot_histories.col=model \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]
