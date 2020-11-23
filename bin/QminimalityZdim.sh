####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Understand effect of z dim on vib and dib

experiment="QminimalityZdim"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpl
dataset=cifar10
dataset.kwargs.is_augment=False 
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
hydra.launcher.time=1000
clfs.gamma_force_generalization=-0.1
datasize.max_epochs=100
model.loss.altern_minimax=5
model.loss.n_per_head=3
model.loss.is_higher=True
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0.01,0.1,1,10
model.architecture.z_dim=2,10,100,1024
model=vib,cdib
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

    wait 

    python main.py $kwargs $kwargs_multi $kwargs1 -m &
      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi

wait 

params="col_val_subset.data=[cifar10]
col_val_subset.lr=[5e-5]
col_val_subset.model=[cdib,vib]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[2,10,100,1024]
col_val_subset.minimax=[5]
col_val_subset.mchead=[3]
col_val_subset.beta=[0.01,0.1,1,10]
"

params=$params" col_val_subset.enc_zy_nhid=[128] "

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=True \
       plot_generalization.logbase_x=10 \
       plot_generalization.col=zdim \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=zdim \
       plot_aux_trnsf.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.row=zdim \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[128] "

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=False \
       plot_generalization.logbase_x=10 \
       plot_generalization.col=zdim \
       plot_metrics.x=beta \
       plot_metrics.style=zdim \
       plot_metrics.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.row=zdim \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]