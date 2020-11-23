####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See how well we do as a regularizer when augmenting data.

experiment="RegularizerDIB"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar10
train.kwargs.lr=1e-4
model.Q_zy.n_hidden_layers=0
hydra.launcher.time=500
datasize.max_epochs=200
train.monitor_best=last
encoder=mlpl
$dev
"

kwargs_multi="
run=0,1,2
model=vib,simplecdib,simplevcdib,simplecdibexact
model.loss.beta=0.03,0.1,0.3,1,3,10
dataset.kwargs.is_augment=True,False
model.architecture.z_dim=1024,10
datasize=small,all
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
       plot_aux_trnsf.folder_col=augment \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.row=data \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.col=zdim \
       plot_histories.style=zdim \
       plot_histories.hue=augment \
       plot_histories.col=model \
       plot_histories.row=data \
       plot_histories.folder_col=beta \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]
