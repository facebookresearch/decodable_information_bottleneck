####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : DIB on the permutation invariant mnist

experiment="perminvmnistLBetas"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset.kwargs.is_augment=False 
model.Q_zy.hidden_size=1024
model.Q_zy.n_hidden_layers=1
hydra.launcher.time=500
datasize.max_epochs=200
model.architecture.z_dim=256
train.kwargs.lr=1e-4
encoder=mlp
train.optim=adam
train.scheduling_mode=decay
datasize.batch_size=100
train.kwargs.is_continue_best=True
dataset=mnist
train.monitor_best=last
dataset.train='trainvalid'
model.loss.n_per_head=1
$dev
"

kwargs_multi="
run=0,1,2,3,4
model=vib,cdib,cdibhigher
model.loss.beta=0.003,0.1,0.3,1,3,10
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
       plot_aux_trnsf.folder_col=chckpnt \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.col=model \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_histories.col=model \
       plot_histories.row=beta \
       plot_histories.folder_col=chckpnt \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]
